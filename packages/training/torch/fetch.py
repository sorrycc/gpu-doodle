"""Download a prefix of Google's simplified Quick Draw ndjson for every pinned class.

`full/simplified/<class>.ndjson` is the output of the geometry that
`packages/core/src/preprocess.ts` reproduces, so training on it keeps the
browser and the model in one coordinate space. Each file is 50 to 80 MB; only
the first `--bytes` are fetched with an HTTP Range request and cut at the last
complete line. A file prefix is not a random sample of the class, which the
data README states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
DEFAULT_BYTES = 20_000_000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path, limit: int) -> None:
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{limit - 1}"})
    temporary = target.with_name(target.name + ".part")
    with urllib.request.urlopen(request) as response, temporary.open("wb") as out:
        if response.status not in (200, 206):
            raise RuntimeError(f"{url}: HTTP {response.status}")
        while chunk := response.read(1 << 20):
            out.write(chunk)
    # The range cut the last line; drop it so every record parses.
    data = temporary.read_bytes()
    cut = data.rfind(b"\n")
    if cut < 0:
        raise RuntimeError(f"{url}: no complete line in the first {limit} bytes")
    temporary.write_bytes(data[: cut + 1])
    temporary.replace(target)


def inspect(path: Path, name: str) -> dict:
    records = recognized = 0
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["word"] != name:
                raise RuntimeError(f"{path}: record for {record['word']!r} in {name!r}")
            records += 1
            recognized += bool(record["recognized"])
    return {"records": records, "recognized": recognized}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=Path, default=DATA / "classes.json")
    parser.add_argument("--out", type=Path, default=DATA / "simplified")
    parser.add_argument("--manifest", type=Path, default=DATA / "manifest.json")
    parser.add_argument("--bytes", type=int, default=DEFAULT_BYTES)
    parser.add_argument("--update-pins", action="store_true")
    args = parser.parse_args()

    classes = json.loads(args.classes.read_text())
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
    files = manifest.get("files", {}) if manifest.get("format") == "simplified-ndjson" else {}
    args.out.mkdir(parents=True, exist_ok=True)
    failures = []

    for name in classes:
        url = BASE + urllib.parse.quote(name) + ".ndjson"
        target = args.out / f"{name}.ndjson"
        pinned = files.get(name)
        if target.exists() and pinned and digest(target) == pinned["sha256"]:
            print(json.dumps({"class": name, "status": "cached"}), flush=True)
            continue
        if not target.exists():
            download(url, target, args.bytes)
        actual = digest(target)
        if pinned and actual != pinned["sha256"] and not args.update_pins:
            failures.append(name)
            print(
                json.dumps({"class": name, "status": "digest-mismatch", "expected": pinned["sha256"], "actual": actual}),
                flush=True,
            )
            continue
        files[name] = {
            "url": url,
            "requestedBytes": args.bytes,
            "bytes": target.stat().st_size,
            "sha256": actual,
            "retrieved": date.today().isoformat(),
            **inspect(target, name),
        }
        print(json.dumps({"class": name, "status": "downloaded", **files[name]}), flush=True)

    manifest = {
        "source": "Google Quick, Draw! dataset, full/simplified ndjson, file prefix",
        "license": "CC BY 4.0",
        "attribution": "https://github.com/googlecreativelab/quickdraw-dataset",
        "format": "simplified-ndjson",
        "geometry": "aligned top-left, larger extent scaled to 255, resampled at 1 px, RDP epsilon 2.0 (see packages/core/src/preprocess.ts)",
        "sampling": "the first --bytes of each file cut at the last complete line; not a random sample of the class",
        "classes": classes,
        "files": {name: files[name] for name in classes if name in files},
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    if failures:
        print(f"Digest mismatch for {len(failures)} files; pass --update-pins to accept the new prefixes.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

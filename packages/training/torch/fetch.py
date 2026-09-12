"""Download the Sketch-RNN npz packages for the pinned class list and record their digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BASE = "https://storage.googleapis.com/quickdraw_dataset/sketchrnn/"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path) -> None:
    temporary = target.with_suffix(".npz.part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    temporary.replace(target)


def inspect(path: Path) -> dict:
    with np.load(path, encoding="latin1", allow_pickle=True) as package:
        return {split: int(len(package[split])) for split in ("train", "valid", "test")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=Path, default=DATA / "classes.json")
    parser.add_argument("--out", type=Path, default=DATA / "npz")
    parser.add_argument("--manifest", type=Path, default=DATA / "manifest.json")
    parser.add_argument("--update-pins", action="store_true")
    args = parser.parse_args()

    classes = json.loads(args.classes.read_text())
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
    files = manifest.get("files", {})
    args.out.mkdir(parents=True, exist_ok=True)
    failures = []

    for name in classes:
        url = BASE + urllib.parse.quote(name) + ".npz"
        target = args.out / f"{name}.npz"
        pinned = files.get(name)
        if target.exists() and pinned and digest(target) == pinned["sha256"]:
            print(json.dumps({"class": name, "status": "cached"}), flush=True)
            continue
        if not target.exists():
            download(url, target)
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
            "bytes": target.stat().st_size,
            "sha256": actual,
            "retrieved": date.today().isoformat(),
            "sequences": inspect(target),
        }
        print(json.dumps({"class": name, "status": "downloaded", **files[name]}), flush=True)

    manifest = {
        "source": "Google Quick, Draw! dataset, Sketch-RNN packages",
        "license": "CC BY 4.0",
        "attribution": "https://github.com/googlecreativelab/quickdraw-dataset",
        "format": "stroke-3 [dx, dy, pen_lift], Quick Draw 0-255 space, RDP epsilon 2.0",
        "classes": classes,
        "files": {name: files[name] for name in classes if name in files},
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    if failures:
        print(f"Digest mismatch for {len(failures)} files; pass --update-pins to accept the new archives.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

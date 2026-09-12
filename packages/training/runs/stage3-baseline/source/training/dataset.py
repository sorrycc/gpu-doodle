"""Turn the simplified ndjson prefixes into featurized splits and serve batches.

Geometry is never touched here: the ndjson is already the output of the four
steps in `packages/core/src/preprocess.ts`. This module only filters on
`recognized`, splits deterministically by `key_id`, delta-encodes strokes the
same way `toStroke3` does, and pads sequences into length buckets.

    uv run --project . python torch/dataset.py build [--name default]

writes `data/synth/<name>/{train,valid,test}.*.bin` plus a manifest. `Dataset`
reads one split back and `batch()` produces the eight per-point features:

    0 dx / 255          delta from the previous point (first row from the origin)
    1 dy / 255
    2 penLift           this point ends a stroke
    3 penDown           this point starts a stroke
    4 x / 255           absolute position, cumulative sum of the deltas
    5 y / 255
    6 strokeIndex / 16  clipped to 1
    7 last              this point ends the sequence
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FEATURES = 8
BUCKETS = (64, 128)
MAX_POINTS = BUCKETS[-1]
SPLITS = {"train": 90, "valid": 5, "test": 5}


def split_of(key_id: str) -> str:
    """Deterministic split by sha256 of the record's key, 90/5/5."""
    bucket = int(hashlib.sha256(key_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < SPLITS["train"]:
        return "train"
    if bucket < SPLITS["train"] + SPLITS["valid"]:
        return "valid"
    return "test"


def to_stroke3(strokes: list[list[list[int]]]) -> np.ndarray:
    """Delta-encode simplified strokes as int16 rows [dx, dy, penLift].

    Mirrors `toStroke3` in preprocess.ts: deltas from the previous point, the
    first row relative to the origin, penLift 1 on the last point of each stroke.
    """
    count = sum(len(stroke[0]) for stroke in strokes)
    rows = np.zeros((count, 3), dtype=np.int16)
    previous_x = previous_y = 0
    row = 0
    for stroke in strokes:
        xs, ys = stroke[0], stroke[1]
        if len(xs) != len(ys):
            raise ValueError("Stroke x and y arrays differ in length.")
        last = len(xs) - 1
        for index in range(len(xs)):
            rows[row, 0] = xs[index] - previous_x
            rows[row, 1] = ys[index] - previous_y
            rows[row, 2] = 1 if index == last else 0
            previous_x, previous_y = xs[index], ys[index]
            row += 1
    return rows


def build(name: str, classes_path: Path, source: Path, manifest_path: Path) -> dict:
    classes = json.loads(classes_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    directory = DATA / "synth" / name
    directory.mkdir(parents=True, exist_ok=True)
    rows = {split: [] for split in SPLITS}
    labels = {split: [] for split in SPLITS}
    lengths = {split: [] for split in SPLITS}
    counts = {split: Counter() for split in SPLITS}
    dropped = Counter()
    sources = {}
    for label, cls in enumerate(classes):
        path = source / f"{cls}.ndjson"
        pinned = manifest["files"][cls]["sha256"]
        sources[cls] = pinned
        with path.open("rb") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not record["recognized"]:
                    dropped["unrecognized"] += 1
                    continue
                encoded = to_stroke3(record["drawing"])
                if len(encoded) == 0:
                    dropped["empty"] += 1
                    continue
                if len(encoded) > MAX_POINTS:
                    dropped["too-long"] += 1
                    continue
                split = split_of(record["key_id"])
                rows[split].append(encoded)
                labels[split].append(label)
                lengths[split].append(len(encoded))
                counts[split][cls] += 1
    report = {
        "name": name,
        "classes": classes,
        "sources": sources,
        "sourceManifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "maxPoints": MAX_POINTS,
        "buckets": list(BUCKETS),
        "splitPercent": SPLITS,
        "dropped": dict(dropped),
        "splits": {},
        "format": {"rows": "int16[3] dx dy penLift", "labels": "uint8", "offsets": "uint32"},
    }
    for split in SPLITS:
        prefix = directory / split
        stacked = np.concatenate(rows[split]) if rows[split] else np.zeros((0, 3), np.int16)
        offsets = np.concatenate(([0], np.cumsum(lengths[split]))).astype(np.uint32)
        stacked.astype(np.int16).tofile(f"{prefix}.rows.bin")
        np.asarray(labels[split], dtype=np.uint8).tofile(f"{prefix}.labels.bin")
        offsets.tofile(f"{prefix}.offsets.bin")
        report["splits"][split] = {
            "sequences": len(lengths[split]),
            "points": int(offsets[-1]),
            "perClass": {cls: counts[split][cls] for cls in classes},
            "meanLength": float(np.mean(lengths[split])) if lengths[split] else 0,
        }
    (directory / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class Dataset:
    def __init__(self, prefix: Path):
        self.rows = np.fromfile(f"{prefix}.rows.bin", dtype=np.int16).reshape(-1, 3)
        self.labels = np.fromfile(f"{prefix}.labels.bin", dtype=np.uint8)
        self.offsets = np.fromfile(f"{prefix}.offsets.bin", dtype=np.uint32)
        self.lengths = np.diff(self.offsets).astype(np.int64)
        self.manifest = json.loads((prefix.parent / "manifest.json").read_text())
        self.classes = self.manifest["classes"]
        # Per-point derived columns, computed once: absolute position, stroke
        # index, pen-down flag. Everything is bookkeeping over the stored deltas.
        self.absolute = np.zeros((len(self.rows), 2), dtype=np.int32)
        self.stroke_index = np.zeros(len(self.rows), dtype=np.int16)
        self.pen_down = np.zeros(len(self.rows), dtype=np.int8)
        for start, end in zip(self.offsets[:-1], self.offsets[1:]):
            deltas = self.rows[start:end, :2].astype(np.int32)
            self.absolute[start:end] = np.cumsum(deltas, axis=0)
            lifts = self.rows[start:end, 2]
            down = np.ones(end - start, dtype=np.int8)
            down[1:] = lifts[:-1]
            self.pen_down[start:end] = down
            self.stroke_index[start:end] = np.concatenate(([0], np.cumsum(lifts[:-1])))

    def __len__(self) -> int:
        return len(self.lengths)

    def bucket(self, length: int) -> int:
        for size in BUCKETS:
            if length <= size:
                return size
        raise ValueError(f"Sequence of {length} points exceeds the largest bucket")

    def features_of(self, index: int, strokes: int | None = None) -> np.ndarray:
        """Float features for one sequence; `strokes` keeps only the first k strokes."""
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        if strokes is not None:
            lifts = np.flatnonzero(self.rows[start:end, 2])
            if len(lifts) > strokes:
                end = start + int(lifts[strokes - 1]) + 1
        size = end - start
        out = np.zeros((size, FEATURES), dtype=np.float32)
        out[:, 0] = self.rows[start:end, 0] / 255
        out[:, 1] = self.rows[start:end, 1] / 255
        out[:, 2] = self.rows[start:end, 2]
        out[:, 3] = self.pen_down[start:end]
        out[:, 4] = self.absolute[start:end, 0] / 255
        out[:, 5] = self.absolute[start:end, 1] / 255
        out[:, 6] = np.minimum(self.stroke_index[start:end] / 16, 1)
        out[-1, 7] = 1
        out[-1, 2] = 1
        return out

    def batch(self, indices: np.ndarray, device: str, strokes: int | None = None):
        import torch

        sequences = [self.features_of(int(index), strokes) for index in indices]
        length = self.bucket(max(len(sequence) for sequence in sequences))
        features = np.zeros((len(indices), length, FEATURES), dtype=np.float32)
        valid = np.zeros((len(indices), length), dtype=np.bool_)
        for destination, sequence in enumerate(sequences):
            features[destination, : len(sequence)] = sequence
            valid[destination, : len(sequence)] = True
        labels = self.labels[indices].astype(np.int64)
        return tuple(
            torch.from_numpy(value).to(device) for value in (features, valid, labels)
        )

    def batches(self, batch_size: int, rng: np.random.Generator | None = None, limit: int | None = None):
        batches = []
        lower = 0
        for upper in BUCKETS:
            indices = np.flatnonzero((self.lengths > lower) & (self.lengths <= upper))
            if rng is not None:
                rng.shuffle(indices)
            if limit is not None:
                indices = indices[: max(1, int(limit * len(indices) / len(self)))]
            batches.extend(
                indices[start : start + batch_size]
                for start in range(0, len(indices), batch_size)
            )
            lower = upper
        if rng is not None:
            rng.shuffle(batches)
        return batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--name", default="default")
    parser.add_argument("--classes", type=Path, default=DATA / "classes.json")
    parser.add_argument("--source", type=Path, default=DATA / "simplified")
    parser.add_argument("--manifest", type=Path, default=DATA / "manifest.json")
    args = parser.parse_args()
    report = build(args.name, args.classes, args.source, args.manifest)
    summary = {
        "name": report["name"],
        "dropped": report["dropped"],
        **{split: {"sequences": v["sequences"], "points": v["points"]} for split, v in report["splits"].items()},
    }
    print(json.dumps(summary))
    sys.exit(0)

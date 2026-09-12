"""Checks that hold the model and the loader to their documented contracts.

    uv run --project . python torch/test_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from dataset import FEATURES, split_of, to_stroke3
from model import CLASSES, HIDDEN, DoodleTagger, affine_scan, sequential_scan

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT.parent / "core" / "test" / "fixtures" / "quickdraw-cat-parity.json"


def check_scan():
    torch.manual_seed(0)
    gate = torch.sigmoid(torch.randn(4, 37, HIDDEN))
    candidate = torch.randn(4, 37, HIDDEN)
    difference = (affine_scan(gate, candidate) - sequential_scan(gate, candidate)).abs().max()
    assert difference < 1e-4, f"parallel and sequential scans differ by {difference}"


def check_parameters():
    model = DoodleTagger()
    assert model.parameter_count() == DoodleTagger.expected_parameters(), model.parameter_count()
    print(json.dumps({"parameters": model.parameter_count(), "classes": CLASSES}))


def check_padding_invariance():
    """Padding must not change the logits: the kernel only ever sees real points."""
    torch.manual_seed(1)
    model = DoodleTagger().eval()
    features = torch.randn(2, 40, FEATURES)
    valid = torch.zeros(2, 40, dtype=torch.bool)
    valid[0, :23] = True
    valid[1, :40] = True
    with torch.no_grad():
        padded = model(features, valid)
        model.reference_scan = True
        reference = model(features, valid)
        tight = model(features[:1, :23], valid[:1, :23])
    assert (padded - reference).abs().max() < 1e-4
    assert (padded[0] - tight[0]).abs().max() < 1e-4, "padding leaked into the output"


def check_stroke3():
    """The Python delta encoding must match `toStroke3` in preprocess.ts."""
    strokes = [[[10, 20, 20], [5, 5, 15]], [[30], [30]]]
    rows = to_stroke3(strokes)
    expected = np.array([[10, 5, 0], [10, 0, 0], [0, 10, 1], [10, 15, 1]], dtype=np.int16)
    assert (rows == expected).all(), rows
    if FIXTURE.exists():
        pairs = json.loads(FIXTURE.read_text())["pairs"]
        for pair in pairs:
            strokes = [[stroke["x"], stroke["y"]] for stroke in pair["simplified"]]
            rows = to_stroke3(strokes)
            absolute = np.cumsum(rows[:, :2].astype(np.int32), axis=0)
            flat_x = [x for stroke in pair["simplified"] for x in stroke["x"]]
            flat_y = [y for stroke in pair["simplified"] for y in stroke["y"]]
            assert (absolute[:, 0] == flat_x).all() and (absolute[:, 1] == flat_y).all()
            lifts = np.flatnonzero(rows[:, 2])
            assert len(lifts) == len(pair["simplified"])


def check_split():
    counts = {"train": 0, "valid": 0, "test": 0}
    for key in range(20000):
        counts[split_of(str(key))] += 1
    assert 0.88 < counts["train"] / 20000 < 0.92, counts
    assert split_of("5201136883597312") == split_of("5201136883597312")


if __name__ == "__main__":
    for check in (check_scan, check_parameters, check_padding_invariance, check_stroke3, check_split):
        check()
        print(f"ok {check.__name__}")
    sys.exit(0)

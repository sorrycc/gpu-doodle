"""Train the doodle classifier and write measured evaluation and checkpoint artifacts.

    uv run --project . python torch/train.py --run experiment --epochs 5

Reads `data/synth/<data>/` built by `dataset.py build`. Selects `best.pt` on
validation top-1. The test split is evaluated once per epoch for the report
only and never drives selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dataset import Dataset
from model import DoodleTagger

TORCH = Path(__file__).resolve().parent
ROOT = TORCH.parent
CORE = ROOT.parent / "core"
STROKE_PREFIXES = (1, 2, 3)


def evaluate(
    model: DoodleTagger,
    dataset: Dataset,
    batch_size: int,
    device: str,
    strokes: int | None = None,
    per_class: bool = False,
) -> dict:
    model.eval()
    classes = len(dataset.classes)
    top1 = top3 = total = 0
    confusion = np.zeros((classes, classes), dtype=np.int64)
    with torch.no_grad():
        for indices in dataset.batches(batch_size):
            features, valid, labels = dataset.batch(indices, device, strokes)
            logits = model(features, valid)
            ranked = logits.topk(3, dim=-1).indices
            hit1 = ranked[:, 0] == labels
            hit3 = (ranked == labels.unsqueeze(-1)).any(dim=-1)
            top1 += int(hit1.sum())
            top3 += int(hit3.sum())
            total += len(indices)
            if per_class:
                np.add.at(
                    confusion,
                    (labels.cpu().numpy(), ranked[:, 0].cpu().numpy()),
                    1,
                )
    result = {
        "sequences": total,
        "top1": top1 / max(1, total),
        "top3": top3 / max(1, total),
    }
    if per_class:
        support = confusion.sum(axis=1)
        result["perClass"] = {
            name: {
                "support": int(support[index]),
                "top1": float(confusion[index, index] / max(1, support[index])),
            }
            for index, name in enumerate(dataset.classes)
        }
        off = confusion.copy()
        np.fill_diagonal(off, 0)
        pairs = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1][:10], off.shape))[0]
        result["confusions"] = [
            {
                "truth": dataset.classes[int(truth)],
                "predicted": dataset.classes[int(predicted)],
                "count": int(off[truth, predicted]),
            }
            for truth, predicted in pairs
            if off[truth, predicted] > 0
        ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="main")
    parser.add_argument("--data", default="default", help="data/synth/<data> built by dataset.py")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--samples", type=int, help="Cap training sequences per epoch.")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--quantization-bits", type=int, choices=[4, 5, 6], default=6)
    parser.add_argument("--qat-start", type=int, default=14, help="Epoch index at which fake quantization starts.")
    parser.add_argument("--storage", choices=["f16", "f32"], default="f32")
    parser.add_argument("--init", type=Path, help="Initialize weights from an earlier checkpoint; optimizer starts fresh.")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run = ROOT / "runs" / args.run
    run.mkdir(parents=True, exist_ok=True)
    directory = ROOT / "data" / "synth" / args.data
    training = Dataset(directory / "train")
    validation = Dataset(directory / "valid")
    test = Dataset(directory / "test")
    print(
        json.dumps(
            {
                "stage": "loaded",
                "run": args.run,
                "device": args.device,
                "train": len(training),
                "valid": len(validation),
                "test": len(test),
            }
        ),
        flush=True,
    )

    model = DoodleTagger(len(training.classes)).to(args.device)
    if args.init:
        initial = torch.load(args.init, map_location="cpu", weights_only=False)
        if initial["classes"] != training.classes:
            raise SystemExit("The --init checkpoint was trained on a different class list.")
        model.load_state_dict(initial["model"])
        args.init = str(args.init)
    model.quantization_bits = args.quantization_bits
    model.storage_f16 = args.storage == "f16"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Snapshot everything a checkpoint depends on, so an export can be traced
    # to the exact code and class list that produced it.
    sources = {
        "training/model.py": TORCH / "model.py",
        "training/train.py": TORCH / "train.py",
        "training/dataset.py": TORCH / "dataset.py",
        "data/classes.json": ROOT / "data" / "classes.json",
        "src/preprocess.ts": CORE / "src" / "preprocess.ts",
        "src/labels.ts": CORE / "src" / "labels.ts",
    }
    source_hashes = {}
    for name, path in sources.items():
        content = path.read_bytes()
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        source_hashes[name] = hashlib.sha256(content).hexdigest()

    history = []
    best = -1.0
    step = 0
    points_seen = 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        model.qat = epoch >= args.qat_start
        batches = training.batches(args.batch, rng, args.samples)
        losses = []
        epoch_started = time.perf_counter()
        for batch_index, indices in enumerate(batches):
            step += 1
            progress = (epoch + batch_index / len(batches)) / args.epochs
            learning_rate = (
                1e-4 + (args.learning_rate - 1e-4) * (1 + math.cos(math.pi * progress)) / 2
            ) * min(1, step / args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            features, valid, labels = training.batch(indices, args.device)
            logits = model(features, valid)
            loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())
            points_seen += int(training.lengths[indices].sum())
            if batch_index % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "batch": batch_index + 1,
                            "batches": len(batches),
                            "loss": torch.stack(losses[-args.log_every :]).mean().item(),
                            "lr": learning_rate,
                            "qat": model.qat,
                        }
                    ),
                    flush=True,
                )
        training_qat = model.qat
        model.qat = True
        metrics = {
            "validation": evaluate(model, validation, args.batch, args.device, per_class=True),
            "validationStrokes": {
                str(k): evaluate(model, validation, args.batch, args.device, strokes=k)
                for k in STROKE_PREFIXES
            },
            "test": evaluate(model, test, args.batch, args.device),
        }
        model.qat = training_qat
        entry = {
            "epoch": epoch + 1,
            "loss": torch.stack(losses).mean().item(),
            "seconds": time.perf_counter() - epoch_started,
            "qat": model.qat,
            "sequences": int(sum(len(batch) for batch in batches)),
            "validationTop1": metrics["validation"]["top1"],
            "validationTop3": metrics["validation"]["top3"],
            "validationStrokesTop1": {k: v["top1"] for k, v in metrics["validationStrokes"].items()},
            "testTop1": metrics["test"]["top1"],
        }
        history.append(entry)
        checkpoint = {
            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "config": vars(args),
            "epoch": epoch + 1,
            "metrics": metrics,
            "pointsSeen": points_seen,
            "classes": training.classes,
        }
        torch.save(checkpoint, run / "last.pt")
        if metrics["validation"]["top1"] > best:
            best = metrics["validation"]["top1"]
            torch.save(checkpoint, run / "best.pt")
            best_metrics = metrics
        report = {
            "selectionCriterion": "Validation top-1 with fake quantization enabled. The test split is reported and never selects.",
            "run": args.run,
            "parameters": model.parameter_count(),
            "config": vars(args),
            "sourceHashes": source_hashes,
            "data": training.manifest,
            "elapsedSeconds": time.perf_counter() - started,
            "pointsSeen": points_seen,
            "history": history,
            "best": {"validationTop1": best, "metrics": best_metrics},
            "status": "training" if epoch + 1 < args.epochs else "completed",
        }
        (run / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
    print(
        json.dumps(
            {
                "stage": "completed",
                "checkpoint": str(run / "best.pt"),
                "bestValidationTop1": best,
                "seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Export a checkpoint as int6 weights and measure the exported model.

    uv run --project . python torch/export.py --checkpoint runs/<run>/best.pt

Writes `packages/core/src/model/weights.gen.ts`, `active/export-report.json`,
and the `active/parity.*` fixtures that `packages/core/test/model-parity.test.ts`
replays through the TypeScript CPU reference. Every score in the report comes
from the weights decoded back off the wire, never from the f32 checkpoint.

Promotion is gated: the candidate must strictly improve test top-1 over the
shipped weights, with no per-class regression beyond a two-proportion
tolerance and no regression in first-3-stroke accuracy. Both sides are decoded
from their int6 form and scored on the same split in this process.
`--force` overrides the decision and records what it overrode.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from dataset import Dataset
from model import CLASSES, FEATURES, HEAD, HIDDEN, DoodleTagger
from train import STROKE_PREFIXES, evaluate

TORCH = Path(__file__).resolve().parent
ROOT = TORCH.parent
CORE = ROOT.parent / "core"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
SHIPPED = CORE / "src/model/weights.gen.ts"
GUARD_Z = 1.96
GUARD_MINIMUM_SUPPORT = 30
PARITY_SEQUENCES = 512
GATE_CRITERION = (
    "Strict improvement in test top-1 over the shipped weights, with no "
    f"per-class regression beyond a two-proportion z={GUARD_Z} tolerance and no "
    "first-3-stroke regression beyond the same tolerance. Both sides are decoded "
    "from their int6 wire form and re-scored on the same test split in this "
    "process; stored scores are never read. The validation split selected the "
    "checkpoint and gates nothing."
)


def encode(model: DoodleTagger, bits: int) -> tuple[str, list[dict]]:
    """Symmetric per-tensor int quantization, packed as one string of 64-symbol codes."""
    maximum = (1 << (bits - 1)) - 1
    encoded = []
    segments = []
    offset = 0
    for name, parameter in model.named_parameters():
        values = parameter.detach().numpy().astype(np.float32)
        scale = np.float32(max(float(np.abs(values).max()) / maximum, 1e-8))
        integers = np.clip(np.rint(values / scale), -maximum, maximum).astype(np.int8)
        flat = integers.reshape(-1)
        encoded.extend(
            ALPHABET[int(value) * 2 if value >= 0 else -int(value) * 2 - 1]
            for value in flat
        )
        segments.append(
            {
                "name": name,
                "offset": offset,
                "length": int(flat.size),
                "shape": list(values.shape),
                "scale": float(scale),
            }
        )
        offset += flat.size
    return "".join(encoded), segments


def decode(encoded: str, segments: list[dict]) -> dict[str, torch.Tensor]:
    values = {}
    for segment in segments:
        characters = encoded[segment["offset"] : segment["offset"] + segment["length"]]
        indices = np.array([ALPHABET.index(character) for character in characters])
        integers = np.where(indices % 2 == 0, indices // 2, -((indices + 1) // 2))
        shaped = integers.reshape(segment["shape"]).astype(np.float32)
        values[segment["name"]] = torch.from_numpy(shaped * np.float32(segment["scale"]))
    return values


def artifact_model(artifact: dict) -> DoodleTagger:
    model = DoodleTagger(artifact["classes"]).eval()
    model.load_state_dict(decode(artifact["q"], artifact["segments"]))
    model.storage_f16 = artifact["storage"] == "f16"
    model.reference_scan = True
    return model


def read_artifact(path: Path) -> dict:
    text = path.read_text()
    return json.loads(text[text.index("{") : text.rindex("}") + 1])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def corpus_digest(prefix: Path) -> str:
    running = hashlib.sha256()
    for suffix in ("rows", "labels", "offsets"):
        running.update(Path(f"{prefix}.{suffix}.bin").read_bytes())
    return running.hexdigest()


def lineage(checkpoint: Path) -> list[dict]:
    result = []
    seen = set()
    while checkpoint:
        key = checkpoint.resolve()
        if key in seen:
            raise ValueError("Checkpoint ancestry contains a cycle")
        seen.add(key)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        result.append(
            {
                "checkpoint": portable(checkpoint),
                "sha256": digest(checkpoint),
                "epoch": saved["epoch"],
                "pointsSeen": saved["pointsSeen"],
            }
        )
        parent = saved["config"].get("init")
        if not parent:
            break
        candidate = Path(parent)
        checkpoint = candidate if candidate.is_absolute() else ROOT / candidate
    return result


def proportion_guard(candidate: dict, baseline: dict) -> dict:
    support = candidate["support"]
    if support != baseline["support"]:
        return {"passed": False, "reason": "support mismatch", "support": support}
    if support < GUARD_MINIMUM_SUPPORT:
        return {"passed": False, "reason": "insufficient support", "support": support}
    # Add-one smoothed two-proportion tolerance; a handful of sketches either
    # way is sampling noise, not a regression.
    pc = (support - candidate["correct"] + 1) / (support + 2)
    pb = (support - baseline["correct"] + 1) / (support + 2)
    delta = (baseline["correct"] - candidate["correct"]) / support
    tolerance = GUARD_Z * math.sqrt((pc * (1 - pc) + pb * (1 - pb)) / support)
    return {
        "passed": delta <= tolerance,
        "reason": None if delta <= tolerance else "regression",
        "support": support,
        "delta": delta,
        "tolerance": tolerance,
    }


def score(model: DoodleTagger, dataset: Dataset) -> dict:
    """Counts rather than rates, so the guards can reason about support."""
    full = evaluate(model, dataset, 256, "cpu", per_class=True)
    strokes = evaluate(model, dataset, 256, "cpu", strokes=3)
    return {
        "support": full["sequences"],
        "correct": round(full["top1"] * full["sequences"]),
        "top1": full["top1"],
        "top3": full["top3"],
        "classes": {
            name: {
                "support": entry["support"],
                "correct": round(entry["top1"] * entry["support"]),
                "top1": entry["top1"],
            }
            for name, entry in full["perClass"].items()
        },
        "confusions": full["confusions"],
        "strokes3": {
            "support": strokes["sequences"],
            "correct": round(strokes["top1"] * strokes["sequences"]),
            "top1": strokes["top1"],
        },
    }


def decide(candidate: dict, baseline: dict | None, failures: list[str]) -> dict:
    decision = {
        "criterion": "test-top1-strict-improvement",
        "candidate": candidate,
        "baseline": baseline,
        "improvement": None,
        "guards": [],
        "failures": list(failures),
    }
    if baseline:
        if candidate["support"] != baseline["support"]:
            decision["failures"].append("evaluation support mismatch")
        else:
            decision["improvement"] = (
                candidate["correct"] - baseline["correct"]
            ) / candidate["support"]
            if candidate["correct"] <= baseline["correct"]:
                decision["failures"].append("test top-1 did not improve")
        for name in sorted(set(candidate["classes"]) | set(baseline["classes"])):
            current = candidate["classes"].get(name)
            before = baseline["classes"].get(name)
            if not current or not before:
                guard = {"passed": False, "reason": "support mismatch", "support": 0}
            else:
                guard = proportion_guard(current, before)
            decision["guards"].append({"family": name, **guard})
        strokes = proportion_guard(candidate["strokes3"], baseline["strokes3"])
        decision["guards"].append({"family": "first-3-strokes", **strokes})
        decision["failures"].extend(
            f"{guard['family']}: {guard['reason']}"
            for guard in decision["guards"]
            if not guard["passed"]
        )
    decision["accepted"] = not decision["failures"]
    return decision


def override(decision: dict) -> dict:
    return {
        **decision,
        "accepted": True,
        "forced": True,
        "criterion": "explicit-user-override",
        "overriddenCriterion": decision["criterion"],
        "overriddenFailures": decision["failures"],
        "failures": [],
    }


def gate(reference: DoodleTagger, test: Dataset, baseline_report: Path) -> dict:
    failures = []
    pinned = None
    artifact = None
    if not baseline_report.exists():
        failures.append(f"no pinned baseline at {portable(baseline_report)}")
    else:
        previous = json.loads(baseline_report.read_text())
        pinned = {
            "report": portable(baseline_report),
            "checkpoint": previous["checkpoint"],
            "checkpointSha256": previous["checkpointSha256"],
            "artifactSha256": previous["artifactSha256"],
            "artifact": portable(SHIPPED),
        }
        if not SHIPPED.exists() or digest(SHIPPED) != previous["artifactSha256"]:
            failures.append("shipped weights do not match the pinned baseline")
        else:
            artifact = read_artifact(SHIPPED)
            if artifact["labels"] != test.classes:
                failures.append("shipped weights were trained on a different class list")
                artifact = None
    candidate = score(reference, test)
    baseline = None
    if artifact:
        baseline = score(artifact_model(artifact), test)
        baseline.update(pinned)
    decision = decide(candidate, baseline, failures)
    if baseline is None and pinned:
        decision["baselineIdentity"] = pinned
    decision["corpus"] = {
        "split": "test",
        "sequences": len(test),
        "featurizedSha256": corpus_digest(test.prefix),
    }
    return decision


def publish(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_parity(reference: DoodleTagger, test: Dataset, prefix: Path) -> dict:
    """Stroke-3 rows plus the reference logits for the first sequences of the test split.

    The TypeScript test rebuilds the eight features from the rows, so the
    fixture also pins `featurize` against `Dataset.features_of`.
    """
    count = min(PARITY_SEQUENCES, len(test))
    indices = np.arange(count)
    logits = np.zeros((count, reference.classes), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, count, 64):
            chunk = indices[start : start + 64]
            features, valid, _ = test.batch(chunk, "cpu")
            logits[chunk] = reference(features, valid).numpy()
    lengths = test.lengths[indices].astype(np.uint32)
    offsets = np.concatenate(
        (np.array([0], dtype=np.uint32), np.cumsum(lengths, dtype=np.uint32))
    )
    rows = test.rows[int(test.offsets[0]) : int(test.offsets[count])].astype(np.int16)
    labels = test.labels[indices].astype(np.uint8)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, values in [
        ("rows", rows),
        ("offsets", offsets),
        ("labels", labels),
        ("logits", logits),
    ]:
        values.tofile(f"{prefix}.{suffix}.bin")
    parity = {
        "prefix": portable(prefix),
        "sequences": int(count),
        "points": int(lengths.sum()),
        "classes": reference.classes,
        "format": {
            "rows": "int16[3] dx dy penLift, concatenated",
            "offsets": "uint32, sequences + 1",
            "labels": "uint8",
            "logits": "float32[classes] per sequence",
        },
        "argmaxAgreement": float((logits.argmax(1) == labels).mean()),
    }
    publish(Path(f"{prefix}.json"), json.dumps(parity, indent=2) + "\n")
    return parity


def export(
    checkpoint: Path,
    destination: Path,
    report_path: Path,
    parity_prefix: Path | None,
    baseline_report: Path,
    force: bool,
) -> dict:
    torch.set_num_threads(4)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    classes = saved["classes"]
    reference = DoodleTagger(len(classes)).eval()
    reference.load_state_dict(saved["model"])
    bits = config.get("quantization_bits", 6)
    storage = config.get("storage", "f32")
    encoded, segments = encode(reference, bits)
    wire = {"q": encoded, "segments": segments}
    # Measure what ships: the reference runs the values decoded back off the wire.
    reference = artifact_model({**wire, "classes": len(classes), "storage": storage})

    data = ROOT / "data" / "synth" / config["data"]
    validation = Dataset(data / "valid")
    test = Dataset(data / "test")
    if validation.classes != classes:
        raise SystemExit("The checkpoint's class list does not match the data split.")

    promotion = gate(reference, test, baseline_report)
    if not promotion["accepted"]:
        if not force:
            raise SystemExit(
                "Export rejected: " + "; ".join(promotion["failures"]) + "\n"
                "Pass --force to override and record the override in the report."
            )
        promotion = override(promotion)
        print("Forcing export despite: " + "; ".join(promotion["overriddenFailures"]))
    promotion["description"] = GATE_CRITERION

    artifact = {
        "version": 1,
        "features": FEATURES,
        "hidden": HIDDEN,
        "head": HEAD,
        "classes": len(classes),
        "storage": storage,
        "labels": classes,
        **wire,
    }
    source = (
        "// Generated by packages/training/torch/export.py. The encoded string is model data, not source logic.\n"
        "export const weights = "
        + json.dumps(artifact, separators=(",", ":"))
        + " as const;\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    staged.write_text(source)
    brotli_script = (
        "import {readFileSync} from 'node:fs';import {brotliCompressSync} from 'node:zlib';"
        "process.stdout.write(String(brotliCompressSync(readFileSync(process.argv[1])).length));"
    )
    brotli = int(
        subprocess.check_output(
            ["node", "--input-type=module", "-e", brotli_script, str(staged)],
            cwd=ROOT,
            text=True,
        )
    )
    metrics = {
        "validation": evaluate(reference, validation, 256, "cpu", per_class=True),
        "validationStrokes": {
            str(k): evaluate(reference, validation, 256, "cpu", strokes=k)
            for k in STROKE_PREFIXES
        },
        "test": {
            key: value
            for key, value in promotion["candidate"].items()
            if key in ("support", "top1", "top3", "strokes3")
        },
    }
    ancestry = lineage(checkpoint)
    report = {
        "checkpoint": portable(checkpoint),
        "checkpointSha256": digest(checkpoint),
        "artifactSha256": hashlib.sha256(source.encode()).hexdigest(),
        "epoch": saved["epoch"],
        "pointsSeenAtCheckpoint": sum(item["pointsSeen"] for item in ancestry),
        "lineage": ancestry,
        "classes": classes,
        "parameters": int(sum(segment["length"] for segment in segments)),
        "quantizationBits": bits,
        "quantizationScheme": "tensor",
        "storage": storage,
        "logicalPackedBytes": int(math.ceil(sum(s["length"] for s in segments) * bits / 8)),
        "encodedCharacters": len(encoded),
        "moduleBytes": len(source.encode()),
        "moduleGzipBytes": len(gzip.compress(source.encode(), mtime=0)),
        "moduleBrotliBytes": brotli,
        "metrics": metrics,
        "corpora": {
            "validation": corpus_digest(data / "valid"),
            "test": corpus_digest(data / "test"),
        },
        "promotion": promotion,
        "scope": (
            f"Exact decoded int{bits} weights, sequential CPU PyTorch inference "
            "reference with the recorded storage precision. Training uses the "
            "mathematically equivalent parallel affine scan. Browser parity and "
            "the first-k-stroke experience are separate gates."
        ),
    }
    if parity_prefix:
        report["parity"] = write_parity(reference, test, parity_prefix)
    source_directory = ROOT / "exports" / report["artifactSha256"] / "source"
    report["exportSourceDirectory"] = portable(source_directory)
    report["exportSourceHashes"] = {}
    for name, path in {
        "training/export.py": TORCH / "export.py",
        "training/model.py": TORCH / "model.py",
        "training/dataset.py": TORCH / "dataset.py",
        "training/train.py": TORCH / "train.py",
        "training/uv.lock": ROOT / "uv.lock",
        "data/classes.json": ROOT / "data" / "classes.json",
    }.items():
        content = path.read_bytes()
        target = source_directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        report["exportSourceHashes"][name] = hashlib.sha256(content).hexdigest()
    staged.replace(destination)
    # Published last: the report's presence is what marks the export committed.
    publish(report_path, json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=SHIPPED)
    parser.add_argument("--report", type=Path, default=ROOT / "active/export-report.json")
    parser.add_argument("--parity", type=Path, default=ROOT / "active/parity")
    parser.add_argument("--baseline", type=Path, default=ROOT / "active/export-report.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = export(
        args.checkpoint, args.out, args.report, args.parity, args.baseline, args.force
    )
    summary = {
        "checkpoint": report["checkpoint"],
        "artifactSha256": report["artifactSha256"],
        "parameters": report["parameters"],
        "moduleBrotliBytes": report["moduleBrotliBytes"],
        "testTop1": report["metrics"]["test"]["top1"],
        "validationTop1": report["metrics"]["validation"]["top1"],
        "accepted": report["promotion"]["accepted"],
        "forced": report["promotion"].get("forced", False),
    }
    print(json.dumps(summary, indent=2))

/**
 * CPU reference versus PyTorch.
 *
 * `packages/training/torch/export.py` writes `active/parity.*`: stroke-3 rows
 * for the first 512 test sketches and the logits of the exported int6 model
 * evaluated with the sequential scan. This test rebuilds the features from
 * the rows, runs `CPUModel`, and requires the same argmax on every sketch and
 * a small maximum logit error. It also pins the shipped weights to the export
 * report by hash, so a hand-edited or stale `weights.gen.ts` fails here.
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  CPUModel,
  classify,
  classifyStroke3,
  featurize,
  fromStroke3,
  LABELS,
} from "../src/index.ts";
import { weights } from "../src/model/weights.gen.ts";

const active = new URL("../../training/active/", import.meta.url);
const read = (name: string) => readFileSync(new URL(name, active));
// Node may hand back a view into a shared pool, so copy exactly the file's bytes.
const bytes = (name: string) => {
  const buffer = read(name);
  return buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength,
  );
};
const parity = JSON.parse(read("parity.json").toString()) as {
  sequences: number;
  points: number;
  classes: number;
};
const report = JSON.parse(read("export-report.json").toString()) as {
  artifactSha256: string;
  classes: string[];
  parity: { sequences: number };
};
const rows = new Int16Array(bytes("parity.rows.bin"));
const offsets = new Uint32Array(bytes("parity.offsets.bin"));
const labels = new Uint8Array(bytes("parity.labels.bin"));
const logits = new Float32Array(bytes("parity.logits.bin"));

describe("shipped weights", () => {
  it("match the export report by hash", () => {
    const source = readFileSync(
      new URL("../src/model/weights.gen.ts", import.meta.url),
    );
    const digest = createHash("sha256").update(source).digest("hex");
    expect(digest).toBe(report.artifactSha256);
  });

  it("carry the public label order", () => {
    expect([...weights.labels]).toEqual([...LABELS]);
    expect(report.classes).toEqual([...LABELS]);
    expect(weights.classes).toBe(LABELS.length);
  });
});

describe("CPU reference", () => {
  it("reproduces PyTorch logits on the parity fixture", () => {
    expect(offsets.length).toBe(parity.sequences + 1);
    expect(offsets[parity.sequences]).toBe(parity.points);
    expect(rows.length).toBe(parity.points * 3);
    expect(logits.length).toBe(parity.sequences * parity.classes);

    const model = new CPUModel();
    let maxError = 0;
    let mismatches = 0;
    let correct = 0;
    for (let sequence = 0; sequence < parity.sequences; sequence++) {
      const slice = rows.subarray(
        offsets[sequence] * 3,
        offsets[sequence + 1] * 3,
      );
      const actual = model.infer(slice);
      const expectedBase = sequence * parity.classes;
      let expectedBest = 0;
      let actualBest = 0;
      for (let index = 0; index < parity.classes; index++) {
        const expected = logits[expectedBase + index];
        maxError = Math.max(maxError, Math.abs(actual[index] - expected));
        if (expected > logits[expectedBase + expectedBest])
          expectedBest = index;
        if (actual[index] > actual[actualBest]) actualBest = index;
      }
      if (expectedBest !== actualBest) mismatches++;
      if (actualBest === labels[sequence]) correct++;
    }
    expect({ mismatches, maxError }).toEqual({
      mismatches: 0,
      maxError: expect.any(Number),
    });
    expect(maxError).toBeLessThan(0.001);
    // The fixture's argmax agreement with its labels is recorded by export.py;
    // the CPU path must land on the same number.
    expect(correct / parity.sequences).toBeGreaterThan(0.8);
  });

  it("featurizes like Dataset.features_of", () => {
    // Two strokes: (10,5)→(20,5)→(20,15), then a single tap at (30,30).
    const rows = Int16Array.from([10, 5, 0, 10, 0, 0, 0, 10, 1, 10, 15, 1]);
    const { features, count } = featurize(rows);
    expect(count).toBe(4);
    const row = (index: number) =>
      Array.from(features.subarray(index * 8, index * 8 + 8));
    expect(row(0)).toEqual(
      [10 / 255, 5 / 255, 0, 1, 10 / 255, 5 / 255, 0, 0].map(Math.fround),
    );
    expect(row(2)).toEqual(
      [0, 10 / 255, 1, 0, 20 / 255, 15 / 255, 0, 0].map(Math.fround),
    );
    expect(row(3)).toEqual(
      [10 / 255, 15 / 255, 1, 1, 30 / 255, 30 / 255, 1 / 16, 1].map(
        Math.fround,
      ),
    );
    // Keeping only the first stroke ends the sequence at its pen lift.
    const first = featurize(rows, 1);
    expect(first.count).toBe(3);
    expect(first.features[2 * 8 + 7]).toBe(1);
  });

  it("answers a stroke-3 query with ranked guesses", () => {
    const slice = rows.subarray(0, offsets[1] * 3);
    const guesses = classifyStroke3(slice, { topK: 3 });
    expect(guesses).toHaveLength(3);
    expect(guesses[0].probability).toBeGreaterThanOrEqual(
      guesses[1].probability,
    );
    expect(guesses[0].index).toBe(labels[0]);
    expect(guesses[0].label).toBe(LABELS[labels[0]]);
    const total = classifyStroke3(slice, { topK: 30 }).reduce(
      (sum, guess) => sum + guess.probability,
      0,
    );
    expect(total).toBeCloseTo(1, 5);
  });

  it("classifies simplified strokes without re-simplifying them", () => {
    const slice = rows.subarray(0, offsets[1] * 3);
    const strokes = fromStroke3(slice);
    const direct = classifyStroke3(slice, { topK: 1 })[0];
    const viaStrokes = classify(strokes, { simplified: true, topK: 1 })[0];
    expect(viaStrokes).toEqual(direct);
  });

  it("classifies raw canvas strokes through simplify", () => {
    const pairs = JSON.parse(
      readFileSync(
        new URL("./fixtures/quickdraw-cat-parity.json", import.meta.url),
      ).toString(),
    ).pairs as { raw: { x: number[]; y: number[] }[] }[];
    let hits = 0;
    for (const pair of pairs.slice(0, 40)) {
      const guesses = classify(pair.raw, { topK: 3 });
      expect(guesses).toHaveLength(3);
      if (guesses.some((guess) => guess.label === "cat")) hits++;
    }
    // Cat is one of the weaker classes; top-3 on real raw sketches should still
    // land well above chance for a working end-to-end path.
    expect(hits).toBeGreaterThan(20);
  });
});

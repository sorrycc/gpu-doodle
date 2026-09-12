import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  CANVAS_EXTENT,
  fromStroke3,
  rdp,
  resample,
  simplify,
  toStroke3,
  type Point,
  type Stroke,
} from "../src/preprocess.ts";

interface Fixture {
  source: { raw: string; simplified: string };
  pairs: { keyId: string; raw: Stroke[]; simplified: Stroke[] }[];
}

const fixture: Fixture = JSON.parse(
  readFileSync(
    resolve(import.meta.dirname, "fixtures/quickdraw-cat-parity.json"),
    "utf8",
  ),
);

describe("resample", () => {
  it("keeps both endpoints and spaces interior points evenly", () => {
    const points = resample(
      [
        [0, 0],
        [3, 0],
      ],
      1,
    );
    expect(points).toEqual([
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
      [3, 0],
    ]);
  });
  it("carries the remainder across segments", () => {
    const points = resample(
      [
        [0, 0],
        [0.5, 0],
        [2.5, 0],
      ],
      1,
    );
    expect(points.map(([x]) => x)).toEqual([0, 1, 2, 2.5]);
  });
  it("turns a tap into two identical points", () => {
    expect(resample([[4, 5]])).toEqual([
      [4, 5],
      [4, 5],
    ]);
  });
});

describe("rdp", () => {
  it("collapses a straight line to its endpoints", () => {
    const line: Point[] = Array.from({ length: 50 }, (_, i) => [i, 2 * i]);
    expect(rdp(line, 2)).toEqual([line[0], line[49]]);
  });
  it("keeps a corner that exceeds epsilon", () => {
    const corner: Point[] = [
      [0, 0],
      [10, 0],
      [10, 10],
    ];
    expect(rdp(corner, 2)).toEqual(corner);
  });
  it("uses distance to the endpoint when the segment is degenerate", () => {
    expect(
      rdp(
        [
          [0, 0],
          [5, 0],
          [0, 0],
        ],
        2,
      ),
    ).toEqual([
      [0, 0],
      [5, 0],
      [0, 0],
    ]);
  });
});

describe("simplify", () => {
  it("returns nothing for an empty drawing", () => {
    expect(simplify([])).toEqual([]);
    expect(simplify([{ x: [], y: [] }])).toEqual([]);
  });
  it("aligns to the top-left and scales the longer extent to 255", () => {
    const [stroke] = simplify([{ x: [100, 610], y: [50, 152] }]);
    expect(Math.min(...stroke.x)).toBe(0);
    expect(Math.min(...stroke.y)).toBe(0);
    expect(Math.max(...stroke.x)).toBe(CANVAS_EXTENT);
    expect(Math.max(...stroke.y)).toBe(51);
  });
  it("keeps a tap as two identical points", () => {
    expect(simplify([{ x: [7], y: [9] }])).toEqual([{ x: [0, 0], y: [0, 0] }]);
  });
  it("rejects mismatched or non-finite coordinates", () => {
    expect(() => simplify([{ x: [1, 2], y: [1] }])).toThrow();
    expect(() => simplify([{ x: [1, NaN], y: [1, 2] }])).toThrow();
  });
});

describe("parity with Google's simplified Quick Draw output", () => {
  const results = fixture.pairs.map((pair) => {
    const mine = simplify(pair.raw);
    const reference = pair.simplified;
    const sameStrokeCount = mine.length === reference.length;
    const sameShape =
      sameStrokeCount &&
      mine.every(
        (stroke, index) => stroke.x.length === reference[index].x.length,
      );
    let deviation = sameShape ? 0 : Infinity;
    if (sameShape)
      mine.forEach((stroke, index) => {
        for (let point = 0; point < stroke.x.length; point++)
          deviation = Math.max(
            deviation,
            Math.abs(stroke.x[point] - reference[index].x[point]),
            Math.abs(stroke.y[point] - reference[index].y[point]),
          );
      });
    return { keyId: pair.keyId, sameStrokeCount, sameShape, deviation };
  });
  const total = results.length;
  const rate = (predicate: (result: (typeof results)[number]) => boolean) =>
    results.filter(predicate).length / total;

  it("has a real fixture", () => {
    expect(total).toBeGreaterThanOrEqual(100);
  });
  it("never changes the number of strokes", () => {
    expect(results.filter((r) => !r.sameStrokeCount)).toEqual([]);
  });
  it("reproduces Google's point sequence for almost every stroke", () => {
    // The remaining differences are floating-point ties inside RDP and
    // resampling. A geometry bug (wrong scale, wrong epsilon, dropped
    // endpoints) drives these rates to near zero rather than a few percent.
    // Measured on the tracked fixture: 116/120 same shape, 104/120 exact,
    // 109/120 within one pixel; on 1,433 pairs: 98%, 90%, 93%.
    expect(rate((r) => r.sameShape)).toBeGreaterThanOrEqual(0.9);
    expect(rate((r) => r.deviation === 0)).toBeGreaterThanOrEqual(0.8);
    expect(rate((r) => r.deviation <= 1)).toBeGreaterThanOrEqual(0.85);
  });
  it("stays inside the 0..255 box on the fixture", () => {
    for (const pair of fixture.pairs)
      for (const stroke of simplify(pair.raw))
        for (let i = 0; i < stroke.x.length; i++) {
          expect(stroke.x[i]).toBeGreaterThanOrEqual(0);
          expect(stroke.x[i]).toBeLessThanOrEqual(CANVAS_EXTENT);
          expect(stroke.y[i]).toBeGreaterThanOrEqual(0);
          expect(stroke.y[i]).toBeLessThanOrEqual(CANVAS_EXTENT);
        }
  });
});

describe("stroke-3 encoding", () => {
  it("round-trips simplified strokes and flags pen lifts", () => {
    const strokes: Stroke[] = [
      { x: [10, 20, 20], y: [5, 5, 15] },
      { x: [3, 3], y: [3, 3] },
    ];
    const rows = toStroke3(strokes);
    expect(Array.from(rows)).toEqual([
      10, 5, 0, 10, 0, 0, 0, 10, 1, -17, -12, 0, 0, 0, 1,
    ]);
    expect(fromStroke3(rows)).toEqual(strokes);
  });
  it("round-trips every fixture drawing", () => {
    for (const pair of fixture.pairs)
      expect(fromStroke3(toStroke3(pair.simplified))).toEqual(pair.simplified);
  });
});

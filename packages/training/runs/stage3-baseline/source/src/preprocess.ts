/**
 * Quick Draw geometry, shared by the browser and the training pipeline.
 *
 * `simplify` reproduces the four steps Google applied to produce the
 * `full/simplified` dataset from `full/raw` sketches:
 *
 * 1. align the drawing to the top-left corner, so the minimum x and y are 0
 * 2. uniformly scale the drawing so its larger extent becomes 255
 * 3. resample every stroke at 1 pixel spacing
 * 4. simplify every stroke with Ramer–Douglas–Peucker at epsilon 2.0
 *
 * The parity test in `test/preprocess.test.ts` checks this implementation
 * against Google's own output for the same sketches. Nothing else in the
 * repository is allowed to re-implement any of these steps.
 */

/** One stroke in the Quick Draw ndjson shape: parallel x and y arrays. */
export interface Stroke {
  x: number[];
  y: number[];
}

export type Point = [number, number];

/** The larger extent of a simplified drawing, in pixels. */
export const CANVAS_EXTENT = 255;
/** Spacing between resampled points, in pixels of the scaled drawing. */
export const RESAMPLE_SPACING = 1;
/** Ramer–Douglas–Peucker tolerance, in pixels of the scaled drawing. */
export const RDP_EPSILON = 2;

function toPoints(stroke: Stroke): Point[] {
  if (stroke.x.length !== stroke.y.length)
    throw new Error("Stroke x and y arrays differ in length.");
  const points: Point[] = [];
  for (let index = 0; index < stroke.x.length; index++) {
    const x = stroke.x[index];
    const y = stroke.y[index];
    if (!Number.isFinite(x) || !Number.isFinite(y))
      throw new Error("Stroke coordinates must be finite numbers.");
    points.push([x, y]);
  }
  return points;
}

/**
 * Resample a polyline at a fixed spacing along each segment. The first and
 * last input points are always kept, so a one-point stroke becomes two
 * identical points, matching Google's output for taps.
 */
export function resample(
  points: readonly Point[],
  spacing: number = RESAMPLE_SPACING,
): Point[] {
  if (points.length === 0) return [];
  if (points.length === 1) return [points[0], points[0]];
  const output: Point[] = [points[0]];
  let carry = 0;
  for (let index = 1; index < points.length; index++) {
    const [ax, ay] = points[index - 1];
    const [bx, by] = points[index];
    const segment = Math.hypot(bx - ax, by - ay);
    if (segment === 0) continue;
    let distance = spacing - carry;
    while (distance <= segment) {
      const t = distance / segment;
      output.push([ax + (bx - ax) * t, ay + (by - ay) * t]);
      distance += spacing;
    }
    carry = segment - (distance - spacing);
  }
  output.push(points[points.length - 1]);
  return output;
}

/**
 * Ramer–Douglas–Peucker with the perpendicular distance to the infinite line
 * through the segment endpoints. A segment splits at its farthest point when
 * that distance exceeds epsilon. Iterative so long resampled strokes cannot
 * overflow the stack; the result equals the recursive formulation.
 */
export function rdp(
  points: readonly Point[],
  epsilon: number = RDP_EPSILON,
): Point[] {
  if (points.length < 3) return [...points];
  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack: [number, number][] = [[0, points.length - 1]];
  while (stack.length) {
    const [first, last] = stack.pop()!;
    if (last - first < 2) continue;
    const [x1, y1] = points[first];
    const [x2, y2] = points[last];
    const dx = x2 - x1;
    const dy = y2 - y1;
    const length = Math.hypot(dx, dy);
    let best = -1;
    let bestIndex = first;
    for (let index = first + 1; index < last; index++) {
      const [x, y] = points[index];
      const distance =
        length === 0
          ? Math.hypot(x - x1, y - y1)
          : Math.abs(dy * x - dx * y + x2 * y1 - y2 * x1) / length;
      if (distance > best) {
        best = distance;
        bestIndex = index;
      }
    }
    if (best > epsilon) {
      keep[bestIndex] = 1;
      stack.push([bestIndex, last], [first, bestIndex]);
    }
  }
  const output: Point[] = [];
  for (let index = 0; index < points.length; index++)
    if (keep[index]) output.push(points[index]);
  return output;
}

/**
 * Convert raw strokes in any coordinate space into Quick Draw's simplified
 * form: integer coordinates in 0..255, aligned to the top-left, resampled and
 * simplified. Empty strokes are dropped; an empty drawing returns no strokes.
 */
export function simplify(strokes: readonly Stroke[]): Stroke[] {
  const raw = strokes.map(toPoints).filter((points) => points.length > 0);
  if (raw.length === 0) return [];
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const points of raw)
    for (const [x, y] of points) {
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  const span = Math.max(maxX - minX, maxY - minY);
  const scale = span > 0 ? CANVAS_EXTENT / span : 1;
  return raw.map((points) => {
    const scaled = points.map(([x, y]): Point => [
      (x - minX) * scale,
      (y - minY) * scale,
    ]);
    const simplified = rdp(resample(scaled));
    return {
      x: simplified.map(([x]) => Math.round(x)),
      y: simplified.map(([, y]) => Math.round(y)),
    };
  });
}

/**
 * Encode simplified strokes as stroke-3 rows `[dx, dy, penLift]`.
 *
 * Deltas are taken from the previous point; the first row is relative to the
 * origin, so a cumulative sum reproduces the absolute coordinates. `penLift`
 * is 1 on the last point of each stroke and 0 elsewhere. The training loader
 * uses the same convention: delta encoding is bookkeeping, not geometry.
 */
export function toStroke3(strokes: readonly Stroke[]): Int16Array {
  let count = 0;
  for (const stroke of strokes) count += stroke.x.length;
  const rows = new Int16Array(count * 3);
  let previousX = 0;
  let previousY = 0;
  let row = 0;
  for (const stroke of strokes) {
    const last = stroke.x.length - 1;
    for (let index = 0; index <= last; index++) {
      const x = stroke.x[index];
      const y = stroke.y[index];
      rows[row * 3] = x - previousX;
      rows[row * 3 + 1] = y - previousY;
      rows[row * 3 + 2] = index === last ? 1 : 0;
      previousX = x;
      previousY = y;
      row++;
    }
  }
  return rows;
}

/** Inverse of `toStroke3`, for tests and debugging. */
export function fromStroke3(rows: ArrayLike<number>): Stroke[] {
  const strokes: Stroke[] = [];
  let current: Stroke = { x: [], y: [] };
  let x = 0;
  let y = 0;
  for (let row = 0; row * 3 < rows.length; row++) {
    x += rows[row * 3];
    y += rows[row * 3 + 1];
    current.x.push(x);
    current.y.push(y);
    if (rows[row * 3 + 2]) {
      strokes.push(current);
      current = { x: [], y: [] };
    }
  }
  if (current.x.length) strokes.push(current);
  return strokes;
}

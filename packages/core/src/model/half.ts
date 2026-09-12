/**
 * Round a number to the nearest IEEE half (f16), returned as a double. Used
 * only when the exported model recorded `storage: "f16"`; the shipped model
 * stores f32 intermediates, in which case `store` is `Math.fround`.
 */
const nativeHalf = (
  Math as typeof Math & { f16round?: (value: number) => number }
).f16round;

function roundEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

export function roundHalfFallback(value: number): number {
  if (!Number.isFinite(value) || value === 0) return value;
  const sign = value < 0 ? -1 : 1;
  const absolute = Math.abs(value);
  if (absolute >= 65520) return sign * Infinity;
  const step =
    absolute < 2 ** -14
      ? 2 ** -24
      : 2 ** (Math.floor(Math.log2(absolute)) - 10);
  return sign * roundEven(absolute / step) * step;
}

export function storeHalf(value: number): number {
  const single = Math.fround(value);
  return nativeHalf ? nativeHalf(single) : roundHalfFallback(single);
}

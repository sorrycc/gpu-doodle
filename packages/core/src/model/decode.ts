/** The shape of `weights.gen.ts`, written by `packages/training/torch/export.py`. */
export interface EncodedWeights {
  version: number;
  features: number;
  hidden: number;
  head: number;
  classes: number;
  storage: "f16" | "f32";
  labels: readonly string[];
  /** Every tensor, flattened row-major and concatenated, one 64-symbol code per value. */
  q: string;
  segments: readonly {
    name: string;
    offset: number;
    length: number;
    scale: number;
    shape: readonly number[];
  }[];
}

const alphabet =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/**
 * Decode the int6 wire form back to f32 tensors. Symbol `2k` encodes `+k`,
 * symbol `2k-1` encodes `-k`; every value is then scaled by its tensor's
 * per-tensor scale. The result is bit-identical to what `export.py` measured.
 */
export function decodeWeights(
  encoded: EncodedWeights,
): Map<string, Float32Array> {
  const codes = new Uint8Array(128).fill(255);
  for (let index = 0; index < alphabet.length; index++)
    codes[alphabet.charCodeAt(index)] = index;

  const values = new Float32Array(encoded.q.length);
  const tensors = new Map<string, Float32Array>();
  let offset = 0;
  for (const segment of encoded.segments) {
    if (
      segment.offset !== offset ||
      !Number.isFinite(segment.scale) ||
      segment.scale <= 0
    )
      throw new Error("Invalid model tensor metadata.");
    const expected = segment.shape.reduce((product, size) => product * size, 1);
    if (expected !== segment.length)
      throw new Error("Model tensor shape does not match its length.");
    const scale = Math.fround(segment.scale);
    for (let index = offset; index < offset + segment.length; index++) {
      const code = codes[encoded.q.charCodeAt(index)];
      if (code === undefined || code > 63)
        throw new Error("Invalid quantized model data.");
      const value = code & 1 ? -(code + 1) / 2 : code / 2;
      values[index] = Math.fround(value * scale);
    }
    tensors.set(segment.name, values.subarray(offset, offset + segment.length));
    offset += segment.length;
  }
  if (offset !== encoded.q.length)
    throw new Error("Model length does not match its tensor metadata.");
  return tensors;
}

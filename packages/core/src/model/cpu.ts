/**
 * Scalar CPU reference for the doodle classifier.
 *
 * Mirrors `DoodleTagger.forward` in `packages/training/torch/model.py` with
 * `reference_scan = True`, evaluated one sketch at a time in f32. The WGSL
 * kernel (stage 5) is checked against this file, and this file is checked
 * against PyTorch logits by `test/model-parity.test.ts`.
 */
import { decodeWeights, type EncodedWeights } from "./decode.ts";
import { storeHalf } from "./half.ts";
import { weights } from "./weights.gen.ts";

/** Per-point features, in the order `dataset.py` documents them. */
export const FEATURES = 8;
const EXTENT = 255;
const STROKE_INDEX_SCALE = 16;

export interface Featurized {
  /** `count * FEATURES` floats, row-major. */
  features: Float32Array;
  count: number;
}

/**
 * Derive the eight per-point features from stroke-3 rows (`[dx, dy, penLift]`,
 * as `toStroke3` produces). `strokes` keeps only the first k strokes, which is
 * how the guess-while-drawing metric is scored. Mirrors `Dataset.features_of`.
 */
export function featurize(
  rows: ArrayLike<number>,
  strokes?: number,
): Featurized {
  let count = Math.floor(rows.length / 3);
  if (strokes !== undefined && strokes >= 1) {
    let seen = 0;
    for (let row = 0; row < count; row++) {
      if (rows[row * 3 + 2]) {
        seen++;
        if (seen === strokes) {
          count = row + 1;
          break;
        }
      }
    }
  }
  const features = new Float32Array(count * FEATURES);
  let x = 0;
  let y = 0;
  let strokeIndex = 0;
  let previousLift = 1;
  for (let row = 0; row < count; row++) {
    const dx = rows[row * 3];
    const dy = rows[row * 3 + 1];
    const lift = rows[row * 3 + 2] ? 1 : 0;
    x += dx;
    y += dy;
    const base = row * FEATURES;
    features[base] = dx / EXTENT;
    features[base + 1] = dy / EXTENT;
    features[base + 2] = lift;
    features[base + 3] = previousLift;
    features[base + 4] = x / EXTENT;
    features[base + 5] = y / EXTENT;
    features[base + 6] = Math.min(strokeIndex / STROKE_INDEX_SCALE, 1);
    features[base + 7] = 0;
    strokeIndex += lift;
    previousLift = lift;
  }
  if (count > 0) {
    const last = (count - 1) * FEATURES;
    features[last + 2] = 1;
    features[last + 7] = 1;
  }
  return { features, count };
}

export function softmax(logits: ArrayLike<number>): Float32Array {
  const result = new Float32Array(logits.length);
  let best = -Infinity;
  for (let index = 0; index < logits.length; index++)
    best = Math.max(best, logits[index]);
  let sum = 0;
  for (let index = 0; index < logits.length; index++) {
    result[index] = Math.exp(logits[index] - best);
    sum += result[index];
  }
  for (let index = 0; index < logits.length; index++) result[index] /= sum;
  return result;
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

/** Row `row` of a `(rows, width)` matrix dotted with `input[offset..offset+width]`. */
function dot(
  matrix: Float32Array,
  row: number,
  width: number,
  input: Float32Array,
  offset: number,
): number {
  let sum = 0;
  const base = row * width;
  for (let index = 0; index < width; index++)
    sum += matrix[base + index] * input[offset + index];
  return sum;
}

export interface Trace {
  embedded: Float32Array;
  encoded: Float32Array;
  gate: Float32Array;
  candidate: Float32Array;
  forward: Float32Array;
  backward: Float32Array;
  combined: Float32Array;
  pooled: Float32Array;
  hidden: Float32Array;
}

export class CPUModel {
  readonly labels: readonly string[];
  readonly classes: number;
  readonly hidden: number;
  readonly head: number;
  private readonly store: (value: number) => number;
  private readonly tensors: Map<string, Float32Array>;
  private workspace: { capacity: number; buffers: Float32Array[] } = {
    capacity: 0,
    buffers: [],
  };
  /** Intermediate activations of the last `infer` call, when `trace` was requested. */
  trace?: Trace;

  constructor(encoded: EncodedWeights = weights) {
    if (encoded.features !== FEATURES)
      throw new Error("The model expects a different feature width.");
    this.labels = encoded.labels;
    this.classes = encoded.classes;
    this.hidden = encoded.hidden;
    this.head = encoded.head;
    this.store = encoded.storage === "f16" ? storeHalf : Math.fround;
    this.tensors = decodeWeights(encoded);
    for (const name of [
      "input_weight",
      "input_bias",
      "convolution",
      "encoder_bias",
      "gate_weight",
      "gate_bias",
      "candidate_weight",
      "candidate_bias",
      "combine_weight",
      "combine_bias",
      "head_weight",
      "head_bias",
      "output_weight",
      "output_bias",
    ])
      if (!this.tensors.has(name)) throw new Error(`Model is missing ${name}.`);
  }

  private tensor(name: string): Float32Array {
    return this.tensors.get(name)!;
  }

  private buffers(count: number): Float32Array[] {
    const hidden = this.hidden;
    if (this.workspace.capacity < count) {
      const capacity = Math.max(count, this.workspace.capacity * 2, 64);
      this.workspace = {
        capacity,
        buffers: Array.from(
          { length: 7 },
          () => new Float32Array(capacity * hidden),
        ),
      };
    }
    return this.workspace.buffers;
  }

  /** Logits for one sketch given stroke-3 rows. */
  infer(
    rows: ArrayLike<number>,
    options: { strokes?: number; trace?: boolean } = {},
  ): Float32Array {
    const { features, count } = featurize(rows, options.strokes);
    return this.inferFeatures(features, count, options.trace);
  }

  /** Logits for one sketch given `count` rows of the eight per-point features. */
  inferFeatures(
    features: Float32Array,
    count: number,
    trace = false,
  ): Float32Array {
    const hidden = this.hidden;
    const store = this.store;
    const logits = new Float32Array(this.classes);
    if (count === 0) {
      // No points: PyTorch sees an all-padding sequence, mean and max of nothing.
      // Return the bias path so the caller still gets a well-formed vector.
      const pooled = new Float32Array(hidden * 2).fill(0);
      for (let index = 0; index < hidden; index++) pooled[hidden + index] = -2;
      return this.headAndOutput(pooled, logits);
    }
    const [embedded, encoded, gate, candidate, forward, backward, combined] =
      this.buffers(count);

    const inputWeight = this.tensor("input_weight");
    const inputBias = this.tensor("input_bias");
    for (let token = 0; token < count; token++)
      for (let channel = 0; channel < hidden; channel++)
        embedded[token * hidden + channel] = store(
          dot(inputWeight, channel, FEATURES, features, token * FEATURES) +
            inputBias[channel],
        );

    const convolution = this.tensor("convolution");
    const encoderBias = this.tensor("encoder_bias");
    for (let token = 0; token < count; token++)
      for (let channel = 0; channel < hidden; channel++) {
        let value = encoderBias[channel];
        for (let tap = 0; tap < 5; tap++) {
          const neighbor = token + tap - 2;
          if (neighbor >= 0 && neighbor < count)
            value +=
              embedded[neighbor * hidden + channel] *
              convolution[tap * hidden + channel];
        }
        encoded[token * hidden + channel] = store(Math.tanh(value));
      }

    const gateWeight = this.tensor("gate_weight");
    const gateBias = this.tensor("gate_bias");
    const candidateWeight = this.tensor("candidate_weight");
    const candidateBias = this.tensor("candidate_bias");
    for (let token = 0; token < count; token++)
      for (let channel = 0; channel < hidden; channel++) {
        const offset = token * hidden;
        const g = store(
          sigmoid(
            dot(gateWeight, channel, hidden, encoded, offset) +
              gateBias[channel],
          ),
        );
        gate[offset + channel] = g;
        candidate[offset + channel] = store(
          (1 - g) *
            Math.tanh(
              dot(candidateWeight, channel, hidden, encoded, offset) +
                candidateBias[channel],
            ),
        );
      }

    for (let channel = 0; channel < hidden; channel++) {
      let state = 0;
      for (let token = 0; token < count; token++) {
        const index = token * hidden + channel;
        state = store(gate[index] * state + candidate[index]);
        forward[index] = state;
      }
      state = 0;
      for (let token = count - 1; token >= 0; token--) {
        const index = token * hidden + channel;
        state = store(gate[index] * state + candidate[index]);
        backward[index] = state;
      }
    }

    const combineWeight = this.tensor("combine_weight");
    const combineBias = this.tensor("combine_bias");
    const pooled = new Float32Array(hidden * 2);
    const maximum = pooled.subarray(hidden).fill(-2);
    const sums = new Float64Array(hidden);
    const width = hidden * 2;
    for (let token = 0; token < count; token++) {
      const offset = token * hidden;
      for (let channel = 0; channel < hidden; channel++) {
        let value = combineBias[channel];
        const base = channel * width;
        for (let index = 0; index < hidden; index++) {
          value += combineWeight[base + index] * forward[offset + index];
          value +=
            combineWeight[base + hidden + index] * backward[offset + index];
        }
        const activation = store(Math.tanh(encoded[offset + channel] + value));
        combined[offset + channel] = activation;
        sums[channel] += activation;
        if (activation > maximum[channel]) maximum[channel] = activation;
      }
    }
    for (let channel = 0; channel < hidden; channel++) {
      pooled[channel] = store(sums[channel] / count);
      pooled[hidden + channel] = store(maximum[channel]);
    }
    const result = this.headAndOutput(pooled, logits);
    if (trace) {
      const slice = (buffer: Float32Array) => buffer.slice(0, count * hidden);
      this.trace = {
        embedded: slice(embedded),
        encoded: slice(encoded),
        gate: slice(gate),
        candidate: slice(candidate),
        forward: slice(forward),
        backward: slice(backward),
        combined: slice(combined),
        pooled: pooled.slice(),
        hidden: this.lastHidden!.slice(),
      };
    }
    return result;
  }

  private lastHidden?: Float32Array;

  private headAndOutput(
    pooled: Float32Array,
    logits: Float32Array,
  ): Float32Array {
    const store = this.store;
    const headWeight = this.tensor("head_weight");
    const headBias = this.tensor("head_bias");
    const outputWeight = this.tensor("output_weight");
    const outputBias = this.tensor("output_bias");
    const hidden = new Float32Array(this.head);
    for (let unit = 0; unit < this.head; unit++)
      hidden[unit] = store(
        Math.tanh(
          dot(headWeight, unit, this.hidden * 2, pooled, 0) + headBias[unit],
        ),
      );
    for (let index = 0; index < this.classes; index++)
      logits[index] = Math.fround(
        dot(outputWeight, index, this.head, hidden, 0) + outputBias[index],
      );
    this.lastHidden = hidden;
    return logits;
  }
}

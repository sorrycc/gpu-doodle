export { LABELS, LABEL_COUNT, labelZh, type Label } from "./labels.ts";
export {
  CANVAS_EXTENT,
  RDP_EPSILON,
  RESAMPLE_SPACING,
  fromStroke3,
  rdp,
  resample,
  simplify,
  toStroke3,
  type Point,
  type Stroke,
} from "./preprocess.ts";
export {
  CPUModel,
  FEATURES,
  featurize,
  softmax,
  type Featurized,
  type Trace,
} from "./model/cpu.ts";
export { decodeWeights, type EncodedWeights } from "./model/decode.ts";

import { LABELS, labelZh, type Label } from "./labels.ts";
import { CPUModel, softmax } from "./model/cpu.ts";
import { simplify, toStroke3, type Stroke } from "./preprocess.ts";

export interface Guess {
  index: number;
  label: Label;
  labelZh: string;
  probability: number;
}

export interface ClassifyOptions {
  /** How many guesses to return, best first. Default 3. */
  topK?: number;
  /** Score only the first k strokes, as the guess-while-drawing metric does. */
  strokes?: number;
  /**
   * Set when the strokes are already in Quick Draw's simplified coordinate
   * space (for example, records from `full/simplified`). Raw pointer strokes
   * from a canvas must go through `simplify`, which is the default.
   */
  simplified?: boolean;
}

let shared: CPUModel | undefined;

function model(): CPUModel {
  return (shared ??= new CPUModel());
}

function rank(logits: Float32Array, topK: number): Guess[] {
  const probabilities = softmax(logits);
  const order = Array.from(probabilities.keys()).sort(
    (a, b) => probabilities[b] - probabilities[a],
  );
  return order.slice(0, Math.max(0, topK)).map((index) => {
    const label = LABELS[index];
    return {
      index,
      label,
      labelZh: labelZh[label],
      probability: probabilities[index],
    };
  });
}

/** Classify stroke-3 rows (`[dx, dy, penLift]`, as `toStroke3` produces). */
export function classifyStroke3(
  rows: ArrayLike<number>,
  options: ClassifyOptions = {},
): Guess[] {
  return rank(
    model().infer(rows, { strokes: options.strokes }),
    options.topK ?? 3,
  );
}

/**
 * Classify a drawing given as strokes of pixel coordinates. Raw strokes are
 * first simplified into Quick Draw's coordinate space, which is the only
 * geometry the model has ever seen. Runs on the CPU; stage 5 adds WebGPU.
 */
export function classify(
  strokes: readonly Stroke[],
  options: ClassifyOptions = {},
): Guess[] {
  const prepared = options.simplified ? strokes : simplify(strokes);
  return classifyStroke3(toStroke3(prepared), options);
}

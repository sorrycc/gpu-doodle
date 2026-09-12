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
export { GPUModel, MAX_POINTS } from "./model/gpu.ts";

import { LABELS, labelZh, type Label } from "./labels.ts";
import { CPUModel, featurize, softmax, type Featurized } from "./model/cpu.ts";
import { GPUModel } from "./model/gpu.ts";
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
 * geometry the model has ever seen. Runs on the CPU, which is the faster path
 * for one sketch at a time; `defineClassifier` adds WebGPU for batches.
 */
export function classify(
  strokes: readonly Stroke[],
  options: ClassifyOptions = {},
): Guess[] {
  const prepared = options.simplified ? strokes : simplify(strokes);
  return classifyStroke3(toStroke3(prepared), options);
}

export type Backend = "cpu" | "webgpu" | "auto";

export interface ClassifierOptions {
  /**
   * `"cpu"` never touches WebGPU. `"webgpu"` initializes the device up front
   * and disables the CPU fallback, so the caller owns the no-WebGPU case and a
   * mid-session device loss. `"auto"` (default) scores batches of at least
   * `gpuBatch` sketches on WebGPU once it is available and everything else on
   * the CPU, where dispatch and readback would dominate.
   */
  backend?: Backend;
  /** Batch size from which `"auto"` prefers WebGPU. Default 32. */
  gpuBatch?: number;
}

export interface Classifier {
  /** The backend the instance was defined with. */
  readonly backend: Backend;
  /** The backend that scored the most recent batch. */
  readonly lastBackend: "cpu" | "webgpu" | undefined;
  /** Classify many drawings of pixel strokes. */
  classifyMany(
    drawings: readonly (readonly Stroke[])[],
    options?: ClassifyOptions,
  ): Promise<Guess[][]>;
  /** Classify many drawings already delta-encoded as stroke-3 rows. */
  classifyStroke3Many(
    rowsList: readonly ArrayLike<number>[],
    options?: ClassifyOptions,
  ): Promise<Guess[][]>;
  dispose(): void;
}

/** Sketches per GPU dispatch; bounds the resident state buffer. */
const GPU_DISPATCH = 512;

/**
 * A reusable classifier with explicit backend selection. WebGPU only pays off
 * for batches, which is why `classify` stays on the CPU.
 */
export async function defineClassifier(
  options: ClassifierOptions = {},
): Promise<Classifier> {
  const backend = options.backend ?? "auto";
  const gpuBatch = Math.max(1, options.gpuBatch ?? 32);
  const cpu = new CPUModel();
  let gpu: GPUModel | undefined;
  let gpuFailed = false;
  let disposed = false;
  let lastBackend: "cpu" | "webgpu" | undefined;
  if (backend === "webgpu") gpu = await GPUModel.create();

  const inferGPU = async (inputs: Featurized[]): Promise<Float32Array[]> => {
    const results: Float32Array[] = new Array(inputs.length);
    for (let start = 0; start < inputs.length; start += GPU_DISPATCH) {
      const chunk = inputs.slice(start, start + GPU_DISPATCH);
      const logits = await gpu!.inferMany(chunk);
      logits.forEach((value, index) => {
        // An empty drawing has nothing to dispatch; the CPU bias path stands in.
        results[start + index] =
          value ?? cpu.inferFeatures(chunk[index].features, 0);
      });
    }
    return results;
  };

  const infer = async (
    rowsList: readonly ArrayLike<number>[],
    strokes?: number,
  ): Promise<Float32Array[]> => {
    if (disposed) throw new Error("The classifier is disposed.");
    const inputs = rowsList.map((rows) => featurize(rows, strokes));
    let useGPU = backend === "webgpu";
    if (backend === "auto" && inputs.length >= gpuBatch && !gpuFailed) {
      if (!gpu) {
        try {
          gpu = await GPUModel.create();
        } catch {
          gpuFailed = true;
        }
      }
      useGPU = gpu !== undefined;
    }
    if (useGPU) {
      try {
        const results = await inferGPU(inputs);
        lastBackend = "webgpu";
        return results;
      } catch (error) {
        if (backend === "webgpu") throw error;
        gpuFailed = true;
        gpu?.dispose();
        gpu = undefined;
      }
    }
    lastBackend = "cpu";
    return inputs.map((input) =>
      cpu.inferFeatures(input.features, input.count),
    );
  };

  return {
    backend,
    get lastBackend() {
      return lastBackend;
    },
    async classifyStroke3Many(rowsList, options = {}) {
      const logits = await infer(rowsList, options.strokes);
      return logits.map((value) => rank(value, options.topK ?? 3));
    },
    async classifyMany(drawings, options = {}) {
      const rowsList = drawings.map((strokes) =>
        toStroke3(options.simplified ? strokes : simplify(strokes)),
      );
      const logits = await infer(rowsList, options.strokes);
      return logits.map((value) => rank(value, options.topK ?? 3));
    },
    dispose() {
      disposed = true;
      gpu?.dispose();
      gpu = undefined;
    },
  };
}

/**
 * WebGPU host for the doodle classifier.
 *
 * One compute dispatch scores a batch of sketches: one workgroup per sketch,
 * one lane per hidden channel, the kernel in `kernel.wgsl`. Inputs are the
 * eight per-point features that `featurize` in `cpu.ts` produces, so the GPU
 * and CPU paths share one feature layout. The device, pipeline, weights and
 * grow-only buffers stay resident between calls; a lost device is recreated
 * once and the failed batch retried, as gpu-time does.
 */
import { shader } from "./shader.ts";
import { fullPrecision } from "./options.ts";
import { weights } from "./weights.gen.ts";
import { decodeWeights } from "./decode.ts";
import { FEATURES, type Featurized } from "./cpu.ts";

interface BufferSlot {
  buffer: GPUBuffer;
  capacity: number;
}

/** The longest sequence one workgroup scores; matches the training cap. */
export const MAX_POINTS = 128;

export class GPUModel {
  readonly classes = weights.classes;
  readonly hidden = weights.hidden;
  readonly stats = { submissions: 0, recoveries: 0 };
  private device?: GPUDevice;
  private pipeline?: GPUComputePipeline;
  private weightBuffer?: GPUBuffer;
  private buffers = new Map<string, BufferSlot>();
  private group?: GPUBindGroup;
  private groupKey = "";
  private generation = 0;
  private stateBytes = 4;
  private closed = false;
  private lost = false;
  private queue: Promise<unknown> = Promise.resolve();

  private constructor(private emulateF16: boolean) {}

  static async create(
    options: { emulateF16?: boolean } = {},
  ): Promise<GPUModel> {
    const runtime = new GPUModel(options.emulateF16 ?? false);
    await runtime.initialize();
    return runtime;
  }

  /** The live device, for tests that exercise loss and recovery. */
  get gpuDevice(): GPUDevice | undefined {
    return this.device;
  }

  private async initialize(): Promise<void> {
    if (this.closed) throw new Error("The GPU model is disposed.");
    const adapter = await globalThis.navigator?.gpu?.requestAdapter();
    if (!adapter) throw new Error("WebGPU is unavailable.");
    const nativeHalf =
      !fullPrecision && !this.emulateF16 && adapter.features.has("shader-f16");
    const device = await adapter.requestDevice({
      requiredFeatures: nativeHalf ? ["shader-f16"] : [],
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
    try {
      if (this.closed) throw new Error("The GPU model is disposed.");
      const module = device.createShaderModule({ code: shader(nativeHalf) });
      const compilation = await module.getCompilationInfo();
      const errors = compilation.messages.filter(
        (message) => message.type === "error",
      );
      if (errors.length) {
        throw new Error(
          errors
            .map((message) => `${message.lineNum}: ${message.message}`)
            .join("\n"),
        );
      }
      if (this.closed) throw new Error("The GPU model is disposed.");
      const pipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module, entryPoint: "classify" },
      });
      if (this.closed) throw new Error("The GPU model is disposed.");
      const decoded = decodeWeights(weights);
      const values = new Float32Array(weights.q.length);
      for (const segment of weights.segments)
        values.set(decoded.get(segment.name)!, segment.offset);
      const weightBuffer = device.createBuffer({
        size: values.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(weightBuffer, 0, values);

      this.device = device;
      this.pipeline = pipeline;
      this.weightBuffer = weightBuffer;
      this.stateBytes = nativeHalf ? 2 : 4;
      this.lost = false;
      this.buffers.clear();
      this.group = undefined;
      this.generation++;
      void device.lost.then(() => {
        if (this.device === device && !this.closed) {
          this.lost = true;
          this.device = undefined;
          this.buffers.clear();
        }
      });
    } catch (error) {
      device.destroy();
      throw error;
    }
  }

  /**
   * Logits for every featurized sketch, in input order. Calls are serialized
   * on one internal queue so that buffer reuse is safe. Empty sketches yield
   * `undefined`; the caller decides what an empty drawing means.
   */
  inferMany(inputs: Featurized[]): Promise<(Float32Array | undefined)[]> {
    const result = this.queue.then(async () => {
      try {
        return await this.run(inputs);
      } catch (error) {
        if (this.closed || !this.lost || this.stats.recoveries >= 1)
          throw error;
        // Retry the failed batch after one confirmed device loss. A shader,
        // validation or capacity error must not consume the recovery budget.
        return this.run(inputs);
      }
    });
    this.queue = result.catch(() => undefined);
    return result;
  }

  private buffer(name: string, size: number, usage: number): GPUBuffer {
    const device = this.device!;
    if (
      size > device.limits.maxBufferSize ||
      (usage & GPUBufferUsage.STORAGE &&
        size > device.limits.maxStorageBufferBindingSize)
    ) {
      throw new RangeError(
        "The inference batch exceeds the GPU buffer limits.",
      );
    }
    const existing = this.buffers.get(name);
    if (existing && existing.capacity >= size) return existing.buffer;
    existing?.buffer.destroy();
    let capacity = 256;
    while (capacity < size && capacity <= device.limits.maxBufferSize / 2)
      capacity *= 2;
    if (capacity < size) capacity = Math.ceil(size / 4) * 4;
    const buffer = device.createBuffer({ size: capacity, usage });
    this.buffers.set(name, { buffer, capacity });
    this.generation++;
    return buffer;
  }

  private async run(
    inputs: Featurized[],
  ): Promise<(Float32Array | undefined)[]> {
    if (this.closed) throw new Error("The GPU model is disposed.");
    if (this.lost) {
      if (this.stats.recoveries >= 1)
        throw new Error("The GPU device was lost again after recovery.");
      this.stats.recoveries++;
      await this.initialize();
    }
    const streams = inputs.filter((input) => input.count > 0);
    if (streams.some((input) => input.count > MAX_POINTS))
      throw new RangeError(`A sketch supports at most ${MAX_POINTS} points.`);
    const pointCount = streams.reduce((total, input) => total + input.count, 0);
    if (!pointCount) return inputs.map(() => undefined);

    const features = new Float32Array(pointCount * FEATURES);
    const streamTable = new Uint32Array(streams.length * 2);
    let offset = 0;
    streams.forEach((input, index) => {
      streamTable.set([offset, input.count], index * 2);
      features.set(
        input.features.subarray(0, input.count * FEATURES),
        offset * FEATURES,
      );
      offset += input.count;
    });

    const device = this.device!;
    const classes = this.classes;
    const logitBytes = streams.length * classes * 4;
    const stateSize = pointCount * this.hidden * 4 * this.stateBytes;
    const sizes = [
      features.byteLength,
      streamTable.byteLength,
      16,
      this.weightBuffer!.size,
      stateSize,
      logitBytes,
    ];
    const buffers = [
      this.buffer(
        "features",
        sizes[0],
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      ),
      this.buffer(
        "streams",
        sizes[1],
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      ),
      this.buffer(
        "parameters",
        sizes[2],
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      ),
      this.weightBuffer!,
      this.buffer("states", sizes[4], GPUBufferUsage.STORAGE),
      this.buffer(
        "logits",
        sizes[5],
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      ),
    ];
    const readback = this.buffer(
      "readback",
      logitBytes,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );
    const groupKey = `${this.generation}:${sizes.join(",")}`;
    if (!this.group || this.groupKey !== groupKey) {
      this.group = device.createBindGroup({
        layout: this.pipeline!.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({
          binding,
          resource: { buffer, size: sizes[binding] },
        })),
      });
      this.groupKey = groupKey;
    }

    device.queue.writeBuffer(buffers[0], 0, features);
    device.queue.writeBuffer(buffers[1], 0, streamTable);
    device.queue.writeBuffer(
      buffers[2],
      0,
      Uint32Array.of(pointCount, streams.length, 0, 0),
    );
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.group);
    const width = Math.min(65535, streams.length);
    pass.dispatchWorkgroups(width, Math.ceil(streams.length / width));
    pass.end();
    encoder.copyBufferToBuffer(buffers[5], 0, readback, 0, logitBytes);
    device.queue.submit([encoder.finish()]);
    this.stats.submissions++;
    await readback.mapAsync(GPUMapMode.READ);

    try {
      const mapped = new Float32Array(
        readback.getMappedRange(0, logitBytes),
        0,
        streams.length * classes,
      );
      let stream = 0;
      return inputs.map((input) => {
        if (input.count === 0) return undefined;
        const logits = mapped.slice(stream * classes, (stream + 1) * classes);
        stream++;
        return logits;
      });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
    }
  }

  dispose(): void {
    if (this.closed) return;
    this.closed = true;
    this.device?.destroy();
    this.device = undefined;
    this.pipeline = undefined;
    this.weightBuffer = undefined;
    this.group = undefined;
    this.buffers.clear();
  }
}

# gpu-doodle

A compact neural doodle classifier for the browser. A 27,742-parameter sequence model reads pen strokes and ranks 100 [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) categories, on the CPU for a single drawing and on WebGPU for batches. Zero runtime dependencies; the WGSL kernel and the int6 weights are inlined in one ES module.

```ts
import { classify } from "gpu-doodle";

// strokes are { x: number[], y: number[] } in canvas pixels
const guesses = classify(strokes, { topK: 3 });
// [{ index: 8, label: "cat", labelZh: "猫", probability: 0.91 }, ...]
```

Raw pointer strokes go through the same alignment, scaling, resampling and RDP simplification that produced the training data, so pass pixel coordinates as drawn. Pass `simplified: true` for strokes already in Quick Draw's simplified space. `strokes: k` scores only the first `k` strokes.

Batches can run on WebGPU through a reusable instance:

```ts
import { defineClassifier } from "gpu-doodle";

const classifier = await defineClassifier({ backend: "auto" });
const results = await classifier.classifyMany(drawings, { topK: 3 });
classifier.dispose();
```

`"auto"` scores batches of 32 or more on the GPU and everything else on the CPU. `"webgpu"` initializes the device up front and disables the CPU fallback, so the caller owns the no-WebGPU case and a mid-session device loss. `"cpu"` never touches WebGPU.

Training data: Google's Quick, Draw! dataset, CC BY 4.0. Model provenance and limitations are in the repository's `MODEL_CARD.md`. MIT.

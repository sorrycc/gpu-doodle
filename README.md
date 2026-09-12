# gpu-doodle

An experimental doodle classifier that runs in the browser on WebGPU. A tiny sequence model reads pen strokes and guesses what you are drawing while you draw it. Trained on the [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset) by Google (CC BY 4.0).

See `PLAN.md` for the design and `AGENTS.md` for the working rules.

## Development

```sh
pnpm install
pnpm data:fetch      # 20 MB prefixes of 30 full/simplified ndjson files (about 570 MB)
pnpm data:build      # filter, split by key_id hash, delta-encode
pnpm train -- --run experiment --epochs 20
pnpm export -- --checkpoint runs/experiment/best.pt
pnpm test
```

One sketch at a time runs on the CPU, which is the faster path for a single drawing:

```ts
import { classify } from "gpu-doodle";
// strokes are {x: number[], y: number[]} in canvas pixels
classify(strokes, { topK: 3 });
// [{ label: "cat", labelZh: "猫", probability: 0.91, index: 8 }, ...]
```

Batches can run on WebGPU through a reusable instance. `"auto"` scores batches of 32 or more on the GPU and everything else on the CPU; `"webgpu"` disables the CPU fallback, so the caller owns the no-WebGPU case.

```ts
import { defineClassifier } from "gpu-doodle";
const classifier = await defineClassifier({ backend: "auto" });
const guesses = await classifier.classifyMany(drawings, { topK: 3 });
classifier.dispose();
```

`pnpm test:browser` checks the WGSL kernel against the CPU reference on 10,000 test sketches in headless Chrome and against PyTorch logits on 512 of them.

## Demo

```sh
pnpm site:dev      # Vite dev server for apps/site
pnpm site:smoke    # headless Chrome: draw a circle, expect a ranked guess list
```

The page draws on a canvas, guesses while you draw, and has a prompt mode in the spirit of the original Quick, Draw!: twenty seconds to draw a named category, next round when the top guess matches. It runs the WebGPU kernel when the browser has one and falls back to the CPU path otherwise.

Stages done: workspace and data (1), shared preprocessing with a parity test against Google's simplified output (2), model and training loop (3), int6 export with a gated promotion and a CPU reference checked against PyTorch logits (4), the WGSL kernel with browser parity and backend selection (5), the site (6). The package build and size gate follow.

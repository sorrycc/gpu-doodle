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

The CPU path works end to end today:

```ts
import { classify } from "gpu-doodle";
// strokes are {x: number[], y: number[]} in canvas pixels
classify(strokes, { topK: 3 });
// [{ label: "cat", labelZh: "猫", probability: 0.91, index: 8 }, ...]
```

Stages done: workspace and data (1), shared preprocessing with a parity test against Google's simplified output (2), model and training loop (3), int6 export with a gated promotion and a CPU reference checked against PyTorch logits (4). The WGSL kernel, the site, and the size gate follow.

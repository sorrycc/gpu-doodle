# gpu-doodle

An experimental doodle classifier that runs in the browser on WebGPU. A tiny sequence model reads pen strokes and guesses what you are drawing while you draw it. Trained on the [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset) by Google (CC BY 4.0).

See `PLAN.md` for the design, `MODEL_CARD.md` for the shipped model's provenance, metrics and limitations, and `AGENTS.md` for the working rules.

## Development

```sh
pnpm install
pnpm data:fetch      # 20 MB prefixes of 100 full/simplified ndjson files (about 1.9 GB)
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

## Package build

```sh
pnpm build:core      # packages/core/dist: one ESM entry with the minified kernel and int6 weights inlined, plus declarations and size.json
pnpm size:gate       # same build, fails when the entry exceeds 50,000 Brotli bytes
pnpm check:package   # npm pack, install into a throwaway consumer, type-check and run it
```

The shipped entry is 26,185 Brotli bytes (51,638 minified); the model ranks 100 Quick Draw categories at 89.4% test top-1. The WGSL file never ships: the build splices the model constants into it, minifies it with wgslender and inlines the string next to the weight table.

## Demo

Live at https://sorrycc.github.io/gpu-doodle/. `.github/workflows/pages.yml` rebuilds it on every push to `main`, with the Vite base set to `/gpu-doodle/`.

```sh
pnpm site:dev      # builds packages/core, then the Vite dev server for apps/site
pnpm site:smoke    # headless Chrome: draw a circle, expect a ranked guess list
pnpm site:og       # regenerate public/og.png, the 1200×630 link-preview image
```

The site consumes the built package, the same bundle npm ships, so `site:dev`, `site:build` and `site:smoke` build `packages/core` first.

`index.html` carries static Open Graph and Twitter card tags for link previews. Crawlers do not run the language script, so they are in Chinese with an English alternate. The image they point at, `apps/site/public/og.png`, is a committed screenshot from `pnpm site:og` (draws a cat, waits for the guess list); rerun it after a UI change. The smoke test checks the tag names a 1200×630 PNG that exists.

The page draws on a canvas, guesses while you draw, and has a prompt mode in the spirit of the original Quick, Draw!: twenty seconds to draw a named category, next round when the top guess matches. It runs the WebGPU kernel when the browser has one and falls back to the CPU path otherwise.

## Repository

- `packages/core`: publishable browser package, WGSL kernel, CPU reference, shared preprocessing, build
- `packages/training`: data fetch and build, PyTorch training, evaluation, gated export, provenance
- `apps/site`: the demo, in Chinese or English following the system language, with a switch

## License

MIT. Training data is Google's Quick, Draw! dataset, CC BY 4.0; the site, this README and `MODEL_CARD.md` credit it.

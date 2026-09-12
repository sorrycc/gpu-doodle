# gpu-doodle model card

## Model

The package embeds the checkpoint from run `c100-v1`, epoch 19, artifact SHA-256 `79ed75e801a4f66d68613f7f73739ec303388685180ab39da736ca1b67cf0436`. It has 32,292 parameters: an 8-feature linear input projection to 64 channels, a depthwise convolution of width 5, a gated affine scan run forward and backward, a combine layer, mean and max pooling, a 64-unit head, and 100 class logits. Weights use 6-bit symmetric per-tensor quantization with f32 intermediates. The weight module is 19,636 Brotli bytes; the whole published entry, kernel and weights inlined, is 26,185 Brotli bytes against a 50,000-byte release limit.

The model reads a drawing as a sequence of points in Quick Draw's simplified coordinate space: per point Δx, Δy, pen-lift and pen-down flags, absolute position, stroke index, and an end-of-sequence flag. Sequences longer than 128 points are dropped at training time; the runtime accepts any length but the GPU path caps at 128.

## Intended use

Guessing which of 100 categories a person is doodling, on the client, while they draw. The demo, prompt-style drawing games, and sketch-input widgets are the target. It supports exactly the 100 categories listed in `packages/core/src/labels.ts` and always returns a ranking, so a drawing of anything else gets a confident wrong answer.

## Training data

Google's [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) dataset, CC BY 4.0. Training reads the first 20 MB of each category's `full/simplified` ndjson file, which is a file prefix and not a random sample of the category. After dropping unrecognized drawings and drawings over 128 points:

| Split | Drawings  |
| ----- | --------- |
| train | 3,726,793 |
| valid | 206,276   |
| test  | 207,034   |

Per category between 24,371 and 70,650 records (18,077 to 60,973 in train), an imbalance of up to 3.4×, with no rebalancing in this version. The 100 categories are the first model's 30 plus 70 more, chosen to avoid silhouettes the first model already confused (no second bird, cup, or car). Splits are deterministic by SHA-256 of `key_id`; `dataset.py` never touches geometry, only filtering, splitting, and delta encoding. `packages/training/data/manifest.json` pins each prefix's digest.

## Training

20 epochs over the full train split, batch 512, AdamW at 3e-3 with 500 warmup steps and cosine decay, label smoothing 0.05, gradient clipping 1.0, 6-bit quantization-aware training from epoch 15. About 380 seconds per epoch on Apple Silicon (MPS), 7,442 seconds in total. Selection uses valid top-1 only; epoch 19 was best. Run configuration, per-epoch history and the source snapshot are in `packages/training/runs/c100-v1/`. The 30-class predecessor (`stage3-baseline`, 5 epochs, test top-1 93.6%) stays recorded under `runs/stage3-baseline/`.

## Evaluation

All numbers below are for the decoded int6 weights, which is what ships.

| Metric                              | Value                 |
| ----------------------------------- | --------------------- |
| test top-1 / top-3                  | 89.4% / 97.1%         |
| valid top-1 / top-3                 | 89.4% / 97.1%         |
| valid top-1 after 1 / 2 / 3 strokes | 31.3% / 54.5% / 69.8% |
| test top-1 after 3 strokes          | 69.6%                 |

Weakest categories on valid: dog 57.5%, frog 63.2%, bird 67.0%, whale 75.8%, cat 76.8%. Strongest: t-shirt 97.2%, ladder 96.8%, star 96.6%, rainbow 96.3%, snowman 96.0%. Largest confusions: hammer as axe (208), axe as hammer (199), dog as elephant (154), whale as fish (144), light bulb as hot air balloon (131).

Parity between implementations:

| Comparison                                     | Result                                      |
| ---------------------------------------------- | ------------------------------------------- |
| CPU TypeScript vs PyTorch, 512 test drawings   | 0 argmax mismatches, max logit error 8.1e-6 |
| WebGPU vs CPU TypeScript, 10,000 test drawings | 0 argmax mismatches, max logit error 1.0e-5 |
| WebGPU vs PyTorch, 512 test drawings           | 0 argmax mismatches, max logit error 8.1e-6 |

Timing in headless Chrome 148 on Apple Silicon: 10,000 drawings in 165 ms on WebGPU and 5,167 ms on the CPU path. A single drawing takes well under a millisecond on the CPU, which is why `classify` never dispatches to the GPU.

## Limitations

- Accuracy is measured on Quick Draw's own distribution. The test split comes from the same file prefixes as training, drawn by the same population, on the same canvas size. Accuracy on other input devices or drawing styles is unmeasured.
- 100 categories, no "none of these" output. Anything outside the list is misclassified with confidence.
- The first stroke is often ambiguous; a circle is a moon, a clock, a sun, or an apple until later strokes arrive.
- Categories that share silhouettes (hammer and axe; dog, elephant, cat, bird, frog; whale and fish; light bulb and hot air balloon) are the main error source. Going from 30 to 100 categories cost about four points of top-1 and ten points of first-3-stroke accuracy at the same model size.
- Class imbalance in the prefixes is not corrected.
- Input must go through the package's own `simplify` (alignment, scaling to 255, 1 px resampling, RDP ε=2). Geometry that skips it is out of distribution.

## Reproducibility

`packages/training/active/export-report.json` records the checkpoint hash, artifact hash, quantization, metrics, promotion decision, corpus digests and source hashes. `active/parity.*` holds the 512-drawing PyTorch logits the CPU and browser tests check against. The training run's sources are snapshotted under `runs/c100-v1/source/`, the export's under `exports/<artifactSha256>/source/`. The `.pt` checkpoint is not tracked, so re-export requires the local file.

## Promotion

Export is gated. A candidate ships only if it strictly improves test top-1 over the shipped weights, no category regresses beyond a two-proportion tolerance at z = 1.96, and first-3-stroke accuracy does not regress. Both sides are decoded from int6 and scored in the same process. The first export had no baseline and was forced. The 100-class export was forced as well because the pinned baseline was the 30-class artifact; its report records `forced: true` and the overridden reason ("shipped weights were trained on a different class list"). Later 100-class exports gate against it automatically.

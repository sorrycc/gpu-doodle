# gpu-doodle model card

## Model

The package embeds the checkpoint from run `stage3-baseline`, epoch 5, artifact SHA-256 `0d4f8a7796370fe6102c3b45e1c29787ffb757ce3f1ccde3562eaaed2562b1c6`. It has 27,742 parameters: an 8-feature linear input projection to 64 channels, a depthwise convolution of width 5, a gated affine scan run forward and backward, a combine layer, mean and max pooling, a 64-unit head, and 30 class logits. Weights use 6-bit symmetric per-tensor quantization with f32 intermediates. The weight module is 16,269 Brotli bytes; the whole published entry, kernel and weights inlined, is 22,238 Brotli bytes against a 50,000-byte release limit.

The model reads a drawing as a sequence of points in Quick Draw's simplified coordinate space: per point Δx, Δy, pen-lift and pen-down flags, absolute position, stroke index, and an end-of-sequence flag. Sequences longer than 128 points are dropped at training time; the runtime accepts any length but the GPU path caps at 128.

## Intended use

Guessing which of 30 categories a person is doodling, on the client, while they draw. The demo, prompt-style drawing games, and sketch-input widgets are the target. It supports exactly the 30 categories listed in `packages/core/src/labels.ts` and always returns a ranking, so a drawing of anything else gets a confident wrong answer.

## Training data

Google's [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) dataset, CC BY 4.0. Training reads the first 20 MB of each category's `full/simplified` ndjson file, which is a file prefix and not a random sample of the category. After dropping unrecognized drawings and drawings over 128 points:

| Split | Drawings  |
| ----- | --------- |
| train | 1,120,607 |
| valid | 62,248    |
| test  | 62,040    |

Per category between 30,408 and 70,650 records, an imbalance of up to 2.5×, with no rebalancing in this version. Splits are deterministic by SHA-256 of `key_id`; `dataset.py` never touches geometry, only filtering, splitting, and delta encoding. `packages/training/data/manifest.json` pins each prefix's digest.

## Training

5 epochs over the full train split, batch 512, AdamW at 3e-3 with 500 warmup steps and cosine decay, label smoothing 0.05, gradient clipping 1.0, 6-bit quantization-aware training from epoch 3. About 97 seconds per epoch on Apple Silicon (MPS). The curve was still rising at epoch 5; a 20-epoch run is the expected next model. Selection uses valid top-1 only. Run configuration, per-epoch history and the source snapshot are in `packages/training/runs/stage3-baseline/`.

## Evaluation

All numbers below are for the decoded int6 weights, which is what ships.

| Metric                              | Value                 |
| ----------------------------------- | --------------------- |
| test top-1 / top-3                  | 93.6% / 98.7%         |
| valid top-1 / top-3                 | 93.8% / 98.6%         |
| valid top-1 after 1 / 2 / 3 strokes | 45.0% / 68.4% / 81.1% |
| test top-1 after 3 strokes          | 80.9%                 |

Weakest categories on valid: dog 74.7%, bird 79.4%, elephant 85.6%, cat 86.3%, banana 89.0%. Strongest: mountain 98.7%, bicycle 98.0%, t-shirt 98.0%, clock 97.5%, house 97.3%. Largest confusions: banana as moon (196), dog as elephant (162), elephant as dog (115), flower as tree (94), cat as dog (87).

Parity between implementations:

| Comparison                                     | Result                                      |
| ---------------------------------------------- | ------------------------------------------- |
| CPU TypeScript vs PyTorch, 512 test drawings   | 0 argmax mismatches, max logit error 4.8e-6 |
| WebGPU vs CPU TypeScript, 10,000 test drawings | 0 argmax mismatches, max logit error 6.4e-6 |
| WebGPU vs PyTorch, 512 test drawings           | 0 argmax mismatches, max logit error 5.2e-6 |

Timing in headless Chrome 148 on Apple Silicon: 10,000 drawings in 137 ms on WebGPU and 5,760 ms on the CPU path. A single drawing takes well under a millisecond on the CPU, which is why `classify` never dispatches to the GPU.

## Limitations

- Accuracy is measured on Quick Draw's own distribution. The test split comes from the same file prefixes as training, drawn by the same population, on the same canvas size. Accuracy on other input devices or drawing styles is unmeasured.
- 30 categories, no "none of these" output. Anything outside the list is misclassified with confidence.
- The first stroke is often ambiguous; a circle is a moon, a clock, a sun, or an apple until later strokes arrive.
- Categories that share silhouettes (dog, cat, elephant, rabbit; banana, moon) are the main error source. Changing the category list would help more than a bigger model.
- Class imbalance in the prefixes is not corrected.
- Input must go through the package's own `simplify` (alignment, scaling to 255, 1 px resampling, RDP ε=2). Geometry that skips it is out of distribution.

## Reproducibility

`packages/training/active/export-report.json` records the checkpoint hash, artifact hash, quantization, metrics, promotion decision, corpus digests and source hashes. `active/parity.*` holds the 512-drawing PyTorch logits the CPU and browser tests check against. The training run's sources are snapshotted under `runs/stage3-baseline/source/`, the export's under `exports/<artifactSha256>/source/`. The `.pt` checkpoint is not tracked, so re-export requires the local file.

## Promotion

Export is gated. A candidate ships only if it strictly improves test top-1 over the shipped weights, no category regresses beyond a two-proportion tolerance at z = 1.96, and first-3-stroke accuracy does not regress. Both sides are decoded from int6 and scored in the same process. The first export had no baseline and was forced; the report records `forced: true` and the overridden reason.

# Training data

`classes.json` is the ordered class list; its order is the model's output index and must match `packages/core/src/labels.ts`.

`npz/<class>.npz` (git-ignored) are Google's Sketch-RNN packages from the Quick, Draw! dataset, downloaded by `torch/fetch.py` from `https://storage.googleapis.com/quickdraw_dataset/sketchrnn/<class>.npz`. Each holds `train` (70,000), `valid` (2,500), and `test` (2,500) sketches in stroke-3 format: rows of `[dx, dy, pen_lift]`.

**These are not the `full/simplified` coordinates.** Checked against `full/raw` in stage 2 (221 of 400 raw sketches matched): the npz geometry is the raw device coordinates with RDP (epsilon 2.0) applied at raw scale, no alignment, no scaling to 255, no resampling, and the first point dropped so deltas start from the first point. Extents vary with the capture device (median largest extent 341 over 5,000 cat sketches, maximum 1,459). The browser pipeline in `packages/core/src/preprocess.ts` reproduces `full/simplified`, so training on these files as-is would train on a different distribution. See `PLAN.md`, stage 3, for the decision.

`manifest.json` records the URL, byte size, sha256, and retrieval date of every file. `fetch.py` refuses to overwrite a file whose digest no longer matches the manifest; update the pin deliberately.

Quick Draw is CC BY 4.0. Anything shipped from a model trained on it owes Google attribution.

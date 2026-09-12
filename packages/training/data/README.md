# Training data

`classes.json` is the ordered class list; its order is the model's output index and must match `packages/core/src/labels.ts`.

`npz/<class>.npz` (git-ignored) are Google's Sketch-RNN packages from the Quick, Draw! dataset, downloaded by `torch/fetch.py` from `https://storage.googleapis.com/quickdraw_dataset/sketchrnn/<class>.npz`. Each holds `train` (70,000), `valid` (2,500), and `test` (2,500) sketches in stroke-3 format: rows of `[dx, dy, pen_lift]` in the 0–255 Quick Draw coordinate space, already aligned, scaled, resampled, and RDP-simplified (epsilon 2.0) by Google's pipeline.

`manifest.json` records the URL, byte size, sha256, and retrieval date of every file. `fetch.py` refuses to overwrite a file whose digest no longer matches the manifest; update the pin deliberately.

Quick Draw is CC BY 4.0. Anything shipped from a model trained on it owes Google attribution.

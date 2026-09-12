# Training data

`classes.json` is the ordered class list; its order is the model's output index and must match `packages/core/src/labels.ts`.

## `simplified/<class>.ndjson` (git-ignored)

The first 20 MB of Google's `full/simplified/<class>.ndjson` from the Quick, Draw! dataset, fetched by `torch/fetch.py` with an HTTP Range request from `https://storage.googleapis.com/quickdraw_dataset/full/simplified/<class>.ndjson` and cut at the last complete line. Each record carries `word`, `key_id`, `recognized`, and `drawing` as strokes of parallel integer x and y arrays in 0..255.

These files are the output of the geometry that `packages/core/src/preprocess.ts` reproduces (align top-left, scale the larger extent to 255, resample at 1 px, RDP epsilon 2.0). Training reads them without touching geometry, so the browser and the model share one coordinate space. The parity test in `packages/core/test/preprocess.test.ts` is what makes that claim checkable.

**A file prefix is not a random sample of the class.** Google's files are ordered by collection time, so the prefix is roughly the earliest 15,000 sketches per class. Numbers quoted from this data describe that sample.

`manifest.json` records the URL, requested and actual bytes, sha256, record and recognized counts, and retrieval date of every file. `fetch.py` refuses to overwrite a file whose digest no longer matches; update the pin deliberately with `--update-pins`.

## `synth/<name>/` (git-ignored)

Built by `torch/dataset.py build` from the ndjson prefixes:

- records with `recognized: false` are dropped;
- sketches longer than 128 points are dropped and counted (`dropped.too-long` in the manifest);
- the rest are split 90/5/5 by the sha256 of `key_id`, so the split is stable across rebuilds and independent of file order;
- strokes are delta-encoded exactly like `toStroke3` in `preprocess.ts`: int16 rows of `[dx, dy, penLift]`, the first row relative to the origin, `penLift` 1 on the last point of each stroke.

Each split is `<split>.rows.bin` (int16 × 3), `<split>.labels.bin` (uint8), `<split>.offsets.bin` (uint32). `manifest.json` records per-class counts, the source digests, and the sha256 of the download manifest the build read.

The earlier stage-1 download of `sketchrnn/<class>.npz` was removed in stage 3: those packages are RDP at raw device scale with no alignment, scaling, or resampling, which is not the browser's coordinate space.

Quick Draw is CC BY 4.0. Anything shipped from a model trained on it owes Google attribution.

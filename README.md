# gpu-doodle

An experimental doodle classifier that runs in the browser on WebGPU. A tiny sequence model reads pen strokes and guesses what you are drawing while you draw it. Trained on the [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset) by Google (CC BY 4.0).

See `PLAN.md` for the design and `AGENTS.md` for the working rules.

## Development

```sh
pnpm install
pnpm data:fetch      # downloads packages/training/data/npz (about 450 MB)
```

Stage 1 only sets up the workspace and the data fetch. Model, kernel, and site follow in later stages.

/**
 * Real WebGPU versus the CPU reference and the PyTorch fixture.
 *
 * Serves the workspace through a Vite dev server, opens headless Chrome with
 * Playwright, and scores the first 10,000 sketches of the `test` split on the
 * GPU and on the CPU. Every sketch must land on the same argmax with a small
 * maximum logit error; the first 512 are also compared with the PyTorch logits
 * in `packages/training/active/`. The test then destroys the device and checks
 * the documented one-time recovery, and finally runs `defineClassifier` end to
 * end with both backends. Results go to `packages/training/results/parity-gpu.json`.
 *
 *   pnpm test:browser
 *
 * Needs `pnpm data:build` to have written the `data/synth/<name>/test.*.bin`
 * split whose digest the export report records.
 */
import { chromium } from "playwright";
import { createServer } from "vite";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

const workspaceRoot = resolve(import.meta.dirname, "../../..");
const trainingRoot = `${workspaceRoot}/packages/training`;
const SEQUENCES = 10_000;

const report = JSON.parse(
  await readFile(`${trainingRoot}/active/export-report.json`, "utf8"),
);
// The export report pins the digest of the test split it scored; pick the
// built split whose digest matches, so the labels are in the model's index space.
const corpusDigest = async (prefix: string) => {
  const hash = createHash("sha256");
  for (const suffix of ["rows", "labels", "offsets"])
    hash.update(await readFile(`${prefix}.${suffix}.bin`));
  return hash.digest("hex");
};
const synthRoot = `${trainingRoot}/data/synth`;
let synthName: string | undefined;
for (const entry of existsSync(synthRoot) ? await readdir(synthRoot) : []) {
  if (!existsSync(`${synthRoot}/${entry}/test.rows.bin`)) continue;
  if (
    (await corpusDigest(`${synthRoot}/${entry}/test`)) === report.corpora.test
  ) {
    synthName = entry;
    break;
  }
}
if (!synthName) {
  throw new Error(
    `No data/synth/<name>/test split matches the export report's test corpus digest; run \`pnpm data:fetch && pnpm data:build\` for the shipped class list first.`,
  );
}
const synthPath = `/packages/training/data/synth/${synthName}`;

const server = await createServer({
  configFile: false,
  root: workspaceRoot,
  server: { host: "127.0.0.1", port: 0 },
  logLevel: "error",
});
await server.listen();
const browser = await chromium.launch({
  channel: "chrome",
  headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
});
try {
  const page = await browser.newPage();
  page.on("console", (message) => {
    if (message.type() === "error") console.error("[browser]", message.text());
  });
  await page.route("**/runner.html", (route) =>
    route.fulfill({
      contentType: "text/html",
      body: "<!doctype html><title>Model parity</title>",
    }),
  );
  await page.goto(server.resolvedUrls!.local[0] + "runner.html");

  const result = await page.evaluate(
    async ({ SEQUENCES, synth }) => {
      const root = "/packages/core/src";
      const { GPUModel } = await import(root + "/model/gpu.ts");
      const { CPUModel, featurize } = await import(root + "/model/cpu.ts");
      const binary = async (path: string) => {
        const response = await fetch(path);
        if (!response.ok) throw new Error(`Cannot fetch ${path}`);
        return response.arrayBuffer();
      };
      const active = "/packages/training/active";
      const rows = new Int16Array(await binary(`${synth}/test.rows.bin`));
      const offsets = new Uint32Array(
        await binary(`${synth}/test.offsets.bin`),
      );
      const labels = new Uint8Array(await binary(`${synth}/test.labels.bin`));
      const parity = await (await fetch(`${active}/parity.json`)).json();
      const parityRows = new Int16Array(
        await binary(`${active}/parity.rows.bin`),
      );
      const parityOffsets = new Uint32Array(
        await binary(`${active}/parity.offsets.bin`),
      );
      const parityLogits = new Float32Array(
        await binary(`${active}/parity.logits.bin`),
      );
      const classes = parity.classes as number;
      const sequences = Math.min(SEQUENCES, offsets.length - 1);

      // The parity fixture is the head of the test split; confirm before trusting it.
      for (let sequence = 0; sequence < parity.sequences; sequence++) {
        const a = rows.subarray(
          offsets[sequence] * 3,
          offsets[sequence + 1] * 3,
        );
        const b = parityRows.subarray(
          parityOffsets[sequence] * 3,
          parityOffsets[sequence + 1] * 3,
        );
        if (
          a.length !== b.length ||
          a.some((value, index) => value !== b[index])
        )
          throw new Error(
            `Parity fixture ${sequence} differs from the test split`,
          );
      }

      const cpu = new CPUModel();
      const gpu = await GPUModel.create();
      const inputs = Array.from({ length: sequences }, (_, sequence) =>
        featurize(
          rows.subarray(offsets[sequence] * 3, offsets[sequence + 1] * 3),
        ),
      );
      let maxError = 0;
      let pythonMaxError = 0;
      let mismatches = 0;
      let pythonMismatches = 0;
      let correct = 0;
      let points = 0;
      const started = performance.now();
      let gpuMilliseconds = 0;
      let cpuMilliseconds = 0;
      try {
        for (let start = 0; start < sequences; start += 512) {
          const batch = inputs.slice(start, start + 512);
          const gpuStarted = performance.now();
          const actual = await gpu.inferMany(batch);
          gpuMilliseconds += performance.now() - gpuStarted;
          for (let index = 0; index < batch.length; index++) {
            const sequence = start + index;
            const cpuStarted = performance.now();
            const expected = cpu.inferFeatures(
              batch[index].features,
              batch[index].count,
            );
            cpuMilliseconds += performance.now() - cpuStarted;
            const logits = actual[index]!;
            points += batch[index].count;
            let bestGPU = 0;
            let bestCPU = 0;
            let bestPython = 0;
            for (let label = 0; label < classes; label++) {
              maxError = Math.max(
                maxError,
                Math.abs(logits[label] - expected[label]),
              );
              if (logits[label] > logits[bestGPU]) bestGPU = label;
              if (expected[label] > expected[bestCPU]) bestCPU = label;
              if (sequence < parity.sequences) {
                const python = parityLogits[sequence * classes + label];
                pythonMaxError = Math.max(
                  pythonMaxError,
                  Math.abs(logits[label] - python),
                );
                if (python > parityLogits[sequence * classes + bestPython])
                  bestPython = label;
              }
            }
            if (bestGPU !== bestCPU) mismatches++;
            if (sequence < parity.sequences && bestGPU !== bestPython)
              pythonMismatches++;
            if (bestGPU === labels[sequence]) correct++;
          }
        }
        const elapsed = performance.now() - started;

        // Destroy a real device and exercise the documented one-time recovery.
        const device = gpu.gpuDevice!;
        const lost = device.lost;
        device.destroy();
        await lost;
        const recovered = (await gpu.inferMany([inputs[0]]))[0]!;
        const expected = cpu.inferFeatures(inputs[0].features, inputs[0].count);
        let recoveryError = 0;
        for (let label = 0; label < classes; label++)
          recoveryError = Math.max(
            recoveryError,
            Math.abs(recovered[label] - expected[label]),
          );
        // An empty drawing never reaches the GPU.
        const empty = await gpu.inferMany([featurize([])]);
        return {
          sequences,
          points,
          pythonSequences: parity.sequences,
          maxError,
          pythonMaxError,
          mismatches,
          pythonMismatches,
          top1: correct / sequences,
          gpuMilliseconds,
          cpuMilliseconds,
          elapsedMilliseconds: elapsed,
          recoveryError,
          recoveries: gpu.stats.recoveries,
          submissions: gpu.stats.submissions,
          emptyIsUndefined: empty[0] === undefined,
        };
      } finally {
        gpu.dispose();
      }
    },
    { SEQUENCES, synth: synthPath },
  );

  const packaged = await page.evaluate(async (synth) => {
    const indexPath = "/packages/core/src/index.ts";
    const { defineClassifier, fromStroke3 } = await import(indexPath);
    const rows = new Int16Array(
      await (await fetch(`${synth}/test.rows.bin`)).arrayBuffer(),
    );
    const offsets = new Uint32Array(
      await (await fetch(`${synth}/test.offsets.bin`)).arrayBuffer(),
    );
    const drawings = Array.from({ length: 1000 }, (_, sequence) =>
      fromStroke3(
        rows.subarray(offsets[sequence] * 3, offsets[sequence + 1] * 3),
      ),
    );
    const cpu = await defineClassifier({ backend: "cpu" });
    const gpu = await defineClassifier({ backend: "webgpu" });
    const auto = await defineClassifier({ backend: "auto", gpuBatch: 32 });
    try {
      const [a, b] = await Promise.all([
        cpu.classifyMany(drawings, { simplified: true, topK: 1 }),
        gpu.classifyMany(drawings, { simplified: true, topK: 1 }),
      ]);
      let mismatches = 0;
      for (let index = 0; index < drawings.length; index++)
        if (a[index][0].index !== b[index][0].index) mismatches++;
      await auto.classifyMany(drawings.slice(0, 8), { simplified: true });
      const small = auto.lastBackend;
      await auto.classifyMany(drawings.slice(0, 64), { simplified: true });
      const large = auto.lastBackend;
      // Scoring only the first stroke must still return a full ranking.
      const partial = await gpu.classifyMany([drawings[0]], {
        simplified: true,
        strokes: 1,
        topK: 100,
      });
      const total = partial[0].reduce(
        (sum: number, guess: { probability: number }) =>
          sum + guess.probability,
        0,
      );
      return {
        sequences: drawings.length,
        mismatches,
        gpuBackend: gpu.lastBackend,
        autoSmall: small,
        autoLarge: large,
        partialSumsToOne: Math.abs(total - 1) < 1e-4,
      };
    } finally {
      cpu.dispose();
      gpu.dispose();
      auto.dispose();
    }
  }, synthPath);

  await mkdir(`${trainingRoot}/results`, { recursive: true });
  await writeFile(
    `${trainingRoot}/results/parity-gpu.json`,
    JSON.stringify(
      {
        model: report.artifactSha256,
        browser: browser.version(),
        classifier: packaged,
        ...result,
      },
      null,
      2,
    ) + "\n",
  );
  console.log(JSON.stringify(result, null, 2));
  console.log("Classifier:", packaged);
  if (
    result.sequences < SEQUENCES ||
    result.mismatches ||
    result.pythonMismatches ||
    result.maxError > 0.001 ||
    result.pythonMaxError > 0.001 ||
    result.recoveryError > 0.001 ||
    result.recoveries !== 1 ||
    !result.emptyIsUndefined ||
    packaged.mismatches ||
    packaged.gpuBackend !== "webgpu" ||
    packaged.autoSmall !== "cpu" ||
    packaged.autoLarge !== "webgpu" ||
    !packaged.partialSumsToOne
  )
    throw new Error(
      "Browser parity, device recovery or backend selection failed; see packages/training/results/parity-gpu.json",
    );
} finally {
  await browser.close();
  await server.close();
}

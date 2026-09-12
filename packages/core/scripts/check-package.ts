/**
 * Pack the built package, install it into a throwaway consumer, type-check
 * the public declarations under --strict and run a classification through
 * the installed bundle. The caller builds dist/ first.
 */
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createRequire } from "node:module";

const packageRoot = resolve(import.meta.dirname, "..");
const typescriptBin = createRequire(import.meta.url).resolve(
  "typescript/bin/tsc",
);

const temporary = await mkdtemp(join(tmpdir(), "gpu-doodle-consumer-"));
try {
  const packed = JSON.parse(
    // --ignore-scripts keeps prepack's build output off stdout, which would
    // otherwise corrupt --json.
    execFileSync(
      "npm",
      ["pack", "--ignore-scripts", "--json", "--pack-destination", temporary],
      { cwd: packageRoot, encoding: "utf8" },
    ),
  );
  // Assert the exact shipped file list. A declaration that references a
  // pruned sibling typechecks as `any` under skipLibCheck; only a contents
  // check catches that.
  const shipped = packed[0].files.map((entry: { path: string }) => entry.path);
  const expected = [
    "LICENSE",
    "README.md",
    "dist/index.d.ts",
    "dist/index.js",
    "dist/labels.d.ts",
    "dist/model/cpu.d.ts",
    "dist/model/decode.d.ts",
    "dist/model/gpu.d.ts",
    "dist/preprocess.d.ts",
    "dist/size.json",
    "package.json",
  ];
  const actual = [...shipped].sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(
      `Package contents changed.\n  expected: ${expected.join(", ")}\n  actual:   ${actual.join(", ")}`,
    );
  }

  await writeFile(
    join(temporary, "package.json"),
    JSON.stringify({ private: true, type: "module" }),
  );
  execFileSync(
    "npm",
    [
      "install",
      "--ignore-scripts",
      "--no-audit",
      "--no-fund",
      join(temporary, packed[0].filename),
    ],
    { cwd: temporary, stdio: "pipe" },
  );
  await writeFile(
    join(temporary, "consumer.ts"),
    `
import { classify, defineClassifier, simplify, LABELS, type Guess, type Stroke } from "gpu-doodle";
// A circle of 48 points, the shape the site's smoke test draws.
const circle: Stroke = { x: [], y: [] };
for (let step = 0; step <= 48; step++) {
  const angle = (step / 48) * Math.PI * 2;
  circle.x.push(200 + 120 * Math.cos(angle));
  circle.y.push(200 + 120 * Math.sin(angle));
}
const guesses: Guess[] = classify([circle], { topK: 3 });
if (guesses.length !== 3) throw new Error("classify did not return three guesses.");
const total = classify([circle], { topK: 30 }).reduce((sum, guess) => sum + guess.probability, 0);
if (Math.abs(total - 1) > 1e-4) throw new Error("Probabilities do not sum to one: " + total);
for (const guess of guesses) {
  if (LABELS[guess.index] !== guess.label || !guess.labelZh) throw new Error("Guess label does not match its index.");
}
if (simplify([circle]).length !== 1) throw new Error("simplify dropped the stroke.");
const classifier = await defineClassifier({ backend: "cpu" });
const [batch] = await classifier.classifyMany([[circle]], { topK: 3 });
if (classifier.lastBackend !== "cpu") throw new Error("The cpu backend did not run on the CPU.");
if (batch[0].label !== guesses[0].label || Math.abs(batch[0].probability - guesses[0].probability) > 1e-6)
  throw new Error("defineClassifier disagrees with classify.");
classifier.dispose();
console.log("Installed package and public declarations passed: " + guesses[0].label + " " + guesses[0].probability.toFixed(3));
`,
  );
  execFileSync(
    "node",
    [
      typescriptBin,
      "consumer.ts",
      "--strict",
      "--skipLibCheck",
      "--target",
      "es2022",
      "--module",
      "nodenext",
    ],
    { cwd: temporary, stdio: "inherit" },
  );
  execFileSync("node", ["consumer.js"], { cwd: temporary, stdio: "inherit" });
} finally {
  await rm(temporary, { recursive: true, force: true });
}

/**
 * Package build: one standalone ESM entry with the specialized, minified WGSL
 * kernel and the int6 weight table inlined, plus the public declarations and
 * a size report. Mirrors gpu-time's build.
 *
 *   node --experimental-strip-types scripts/build.ts [--report-only] [--outdir dist] [--weights path]
 *
 * Without --report-only the script fails when the entry exceeds the Brotli
 * budget, which is how `pnpm size:gate` blocks CI.
 */
import { build } from "esbuild";
import { minify as minifyJavaScript } from "terser";
import { mkdir, readFile, writeFile, readdir, rm } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { brotliCompressSync, constants, gzipSync } from "node:zlib";
import { initialize, minify } from "wgslender";
import { buildShader } from "../src/model/shader-source.ts";
import { weights as sourceWeights } from "../src/model/weights.gen.ts";
import type { EncodedWeights } from "../src/model/decode.ts";
import { resolve as resolvePath, join } from "node:path";
import { createRequire } from "node:module";

const packageRoot = resolvePath(import.meta.dirname, "..");
const typescriptBin = createRequire(import.meta.url).resolve(
  "typescript/bin/tsc",
);

const argument = (name: string) => {
  const index = process.argv.indexOf(name);
  if (index < 0) return undefined;
  const value = process.argv[index + 1];
  if (!value || value.startsWith("--"))
    throw new Error(`Missing value for ${name}.`);
  return value;
};
const modelPath = argument("--weights");
const outputDirectory = resolvePath(
  packageRoot,
  argument("--outdir") ?? "dist",
);
const weights: EncodedWeights = modelPath
  ? (await import(resolvePath(modelPath))).weights
  : sourceWeights;

await initialize();
const source = await readFile(
  join(packageRoot, "src/model/kernel.wgsl"),
  "utf8",
);
// f32 storage needs one shader; f16 storage needs a variant with and without
// native half support, chosen at device creation.
const variants = weights.storage === "f32" ? [false] : [false, true];
const shaders = variants.map((nativeHalf) => {
  const result = minify(buildShader(source, weights, nativeHalf), {
    keepNames: ["classify"],
    mangleExternalBindings: true,
  });
  if (result.errors.length) throw new Error(JSON.stringify(result.errors));
  return result.code;
});

await rm(outputDirectory, { recursive: true, force: true });
await mkdir(outputDirectory, { recursive: true });
await build({
  entryPoints: [join(packageRoot, "src/index.ts")],
  outdir: outputDirectory,
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  minify: true,
  legalComments: "none",
  define: {
    GPU_DOODLE_STORAGE: JSON.stringify(weights.storage ?? "f32"),
  },
  plugins: [
    {
      name: "compiled-shader",
      setup(builder) {
        // Ship only the fields the runtime reads. `labels` lives in labels.ts
        // and `version` is export metadata; both would only cost bytes.
        builder.onLoad({ filter: /[/\\]model[/\\]weights\.gen\.ts$/ }, () => ({
          contents: `export const weights = ${JSON.stringify({
            features: weights.features,
            hidden: weights.hidden,
            head: weights.head,
            classes: weights.classes,
            storage: weights.storage,
            q: weights.q,
            segments: weights.segments.map((segment) => ({
              name: segment.name,
              offset: segment.offset,
              length: segment.length,
              shape: segment.shape,
              scale: segment.scale,
            })),
          })};`,
          loader: "js",
        }));
        // The specialized kernel replaces the `?raw` import and the splice.
        builder.onLoad({ filter: /[/\\]model[/\\]shader\.ts$/ }, () => ({
          contents: `export function shader(nativeHalf) { return ${
            shaders.length === 1
              ? JSON.stringify(shaders[0])
              : `nativeHalf ? ${JSON.stringify(shaders[1])} : ${JSON.stringify(shaders[0])}`
          }; }`,
          loader: "js",
        }));
      },
    },
  ],
});

// gpu-time found that aggressive variable collapsing shrank the bundle but
// slowed the scalar CPU path in Chrome; keep the same conservative settings.
const entryPath = `${outputDirectory}/index.js`;
const result = await minifyJavaScript(await readFile(entryPath, "utf8"), {
  module: true,
  compress: {
    passes: 3,
    ...(process.argv.includes("--min-size")
      ? {}
      : { sequences: false, collapse_vars: false, reduce_vars: false }),
  },
  mangle: true,
  format: { comments: false },
});
if (!result.code) throw new Error("No minified output for index.js.");
await writeFile(entryPath, result.code);

execFileSync(
  "node",
  [
    typescriptBin,
    "-p",
    join(packageRoot, "tsconfig.build.json"),
    "--outDir",
    outputDirectory,
  ],
  { stdio: "inherit" },
);

// Only declarations reachable from index.d.ts ship. The kernel, the weight
// table and the shader splice are build-time internals; their declarations
// would otherwise embed a second copy of the weight string.
const publicTypes = new Set([
  "index.d.ts",
  "labels.d.ts",
  "preprocess.d.ts",
  "model/cpu.d.ts",
  "model/decode.d.ts",
  "model/gpu.d.ts",
]);
for (const entry of await readdir(outputDirectory, { recursive: true })) {
  const name = String(entry).replaceAll("\\", "/");
  if (!name.endsWith(".d.ts")) continue;
  const path = `${outputDirectory}/${name}`;
  if (!publicTypes.has(name)) {
    await rm(path);
    continue;
  }
  // Source imports carry explicit `.ts` extensions for Node's type stripping;
  // consumers resolve the declarations by `.js` specifiers.
  const declaration = await readFile(path, "utf8");
  await writeFile(
    path,
    declaration.replaceAll(/(from\s+"\.[^"]*)\.ts"/g, '$1.js"'),
  );
}

const entry = await readFile(entryPath);
const files = [
  {
    file: "index.js",
    bytes: entry.byteLength,
    gzipBytes: gzipSync(entry, { level: 9 }).byteLength,
    brotliBytes: brotliCompressSync(entry, {
      params: { [constants.BROTLI_PARAM_QUALITY]: 11 },
    }).byteLength,
  },
];
const limitBytes = 50_000;
const withinBudget = files[0].brotliBytes <= limitBytes;
await writeFile(
  `${outputDirectory}/size.json`,
  JSON.stringify(
    {
      method:
        "Entire minified ESM entry point, gzip level 9 and Brotli quality 11. The entry is standalone: kernel and int6 weights inlined, no runtime dependencies.",
      limitBytes,
      withinBudget,
      model: {
        parameters: weights.q.length,
        storage: weights.storage ?? "f32",
        classes: weights.classes,
      },
      files,
    },
    null,
    2,
  ) + "\n",
);
console.table(files);
if (!withinBudget) {
  console.error(`Main entry exceeds the ${limitBytes}-byte Brotli budget.`);
  if (!process.argv.includes("--report-only")) process.exitCode = 1;
}

import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

const workspace = fileURLToPath(new URL("../..", import.meta.url));

export default defineConfig({
  resolve: {
    // The site runs against the package source. `gpu-doodle` publishes
    // `dist/`, which the stage 7 build emits; until then the alias keeps the
    // demo, the parity test, and the training export on the same code.
    alias: {
      "gpu-doodle": fileURLToPath(
        new URL("../../packages/core/src/index.ts", import.meta.url),
      ),
    },
  },
  server: { fs: { allow: [workspace] } },
  build: { target: "es2022" },
});

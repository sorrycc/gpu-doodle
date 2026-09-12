import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

const workspace = fileURLToPath(new URL("../..", import.meta.url));

export default defineConfig({
  // The site consumes the built package (`packages/core/dist`), the same
  // bundle npm ships. `pnpm site:dev` builds it first. Excluding it from
  // dependency pre-bundling keeps a rebuilt dist from being served stale.
  optimizeDeps: { exclude: ["gpu-doodle"] },
  server: { fs: { allow: [workspace] } },
  build: { target: "es2022" },
});

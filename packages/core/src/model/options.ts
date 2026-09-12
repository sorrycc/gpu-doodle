import { weights } from "./weights.gen.ts";

// The package build replaces this constant; development reads the weights.
declare const GPU_DOODLE_STORAGE: "f16" | "f32";
export const fullPrecision =
  (typeof GPU_DOODLE_STORAGE === "undefined"
    ? weights.storage
    : GPU_DOODLE_STORAGE) === "f32";

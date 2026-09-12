import source from "./kernel.wgsl?raw";
import { weights } from "./weights.gen.ts";
import { buildShader } from "./shader-source.ts";

/** The specialized kernel; the package build replaces this with a minified string. */
export function shader(nativeHalf: boolean): string {
  return buildShader(source, weights, nativeHalf);
}

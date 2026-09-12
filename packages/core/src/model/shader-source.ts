import type { EncodedWeights } from "./decode.ts";

/**
 * Resolve model constants before shader compilation or build-time
 * minification. The kernel names every tensor offset and every dimension
 * symbolically so that the WGSL file never has to change when the model does.
 */
export function buildShader(
  source: string,
  model: EncodedWeights,
  nativeHalf: boolean,
): string {
  const roundBody =
    model.storage === "f32"
      ? "return value;"
      : nativeHalf
        ? "return f32(f16(value));"
        : "return unpack2x16float(pack2x16float(vec2<f32>(value, 0.0))).x;";
  const offsets = model.segments
    .map(
      (segment) =>
        `const ${segment.name.toUpperCase()}_OFFSET: u32 = ${segment.offset}u;`,
    )
    .join("\n");
  const dimensions = [
    `const HIDDEN: u32 = ${model.hidden}u;`,
    `const HIDDEN2: u32 = ${model.hidden * 2}u;`,
    `const HEAD: u32 = ${model.head}u;`,
    `const CLASSES: u32 = ${model.classes}u;`,
    `const FEATURES: u32 = ${model.features}u;`,
  ].join("\n");
  return `${nativeHalf ? "enable f16;\n" : ""}${offsets}
${dimensions}
${source.replaceAll("STATE_TYPE", nativeHalf ? "f16" : "f32").replace("ROUND_BODY", roundBody)}`;
}

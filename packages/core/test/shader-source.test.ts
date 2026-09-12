/**
 * The kernel is specialized at load time. This checks the splice without a
 * GPU: every placeholder is resolved, every tensor offset is named, and the
 * two storage modes produce the rounding bodies the CPU reference mirrors.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { buildShader } from "../src/model/shader-source.ts";
import { weights } from "../src/model/weights.gen.ts";

const source = readFileSync(
  new URL("../src/model/kernel.wgsl", import.meta.url),
  "utf8",
);

describe("shader specialization", () => {
  it("resolves every placeholder and names every tensor", () => {
    const code = buildShader(source, weights, false);
    expect(code).not.toMatch(/ROUND_BODY|STATE_TYPE/);
    for (const segment of weights.segments)
      expect(code).toContain(
        `const ${segment.name.toUpperCase()}_OFFSET: u32 = ${segment.offset}u;`,
      );
    expect(code).toContain(`const HIDDEN: u32 = ${weights.hidden}u;`);
    expect(code).toContain(`const HIDDEN2: u32 = ${weights.hidden * 2}u;`);
    expect(code).toContain(`const HEAD: u32 = ${weights.head}u;`);
    expect(code).toContain(`const CLASSES: u32 = ${weights.classes}u;`);
    expect(code).toContain(`const FEATURES: u32 = ${weights.features}u;`);
    expect(code).toContain("fn rounded(value: f32) -> f32 { return value; }");
    expect(code).toContain("array<f32>;");
    expect(code).not.toContain("enable f16;");
  });

  it("rounds through f16 when the model stores half intermediates", () => {
    const half = { ...weights, storage: "f16" as const };
    const native = buildShader(source, half, true);
    expect(native.startsWith("enable f16;\n")).toBe(true);
    expect(native).toContain("return f32(f16(value));");
    expect(native).toContain("array<f16>;");
    const emulated = buildShader(source, half, false);
    expect(emulated).toContain("pack2x16float");
    expect(emulated).not.toContain("enable f16;");
  });

  it("keeps the kernel's only entry point named classify", () => {
    expect(source.match(/@compute/g)).toHaveLength(1);
    expect(source).toContain("fn classify(");
  });
});

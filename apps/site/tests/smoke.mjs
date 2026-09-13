/**
 * First-load smoke test for the demo: serve the site with Vite, open it in
 * headless Chrome, draw a circle with the mouse, and require a ranked guess
 * list with no console errors. Also exercises the prompt mode once, and
 * checks in a dark-scheme page that the ink stays light on the dark canvas.
 *
 *   pnpm --filter @gpu-doodle/site test:smoke
 */
import { chromium } from "playwright";
import { createServer } from "vite";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  configFile: `${root}/vite.config.ts`,
  root,
  server: { host: "127.0.0.1", port: 0 },
  logLevel: "error",
});
await server.listen();
const browser = await chromium.launch({
  channel: "chrome",
  headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
});
const errors = [];
try {
  const page = await browser.newPage({
    viewport: { width: 1100, height: 900 },
  });
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.goto(server.resolvedUrls.local[0]);
  await page.waitForFunction(
    () => !document.getElementById("backend").textContent.includes("初始化"),
    null,
    { timeout: 15_000 },
  );

  const box = await page.locator("#pad").boundingBox();
  const centerX = box.x + box.width / 2;
  const centerY = box.y + box.height / 2;
  const radius = box.width * 0.3;
  await page.mouse.move(centerX + radius, centerY);
  await page.mouse.down();
  for (let step = 1; step <= 40; step++) {
    const angle = (step / 40) * Math.PI * 2;
    await page.mouse.move(
      centerX + radius * Math.cos(angle),
      centerY + radius * Math.sin(angle),
    );
  }
  await page.mouse.up();
  await page.waitForFunction(
    () =>
      document.querySelectorAll("#guesses li").length >= 3 &&
      document.getElementById("timeline").textContent !== "",
    null,
    { timeout: 5_000 },
  );

  const afterCircle = await page.evaluate(() => ({
    backend: document.getElementById("backend").textContent,
    timing: document.getElementById("timing").textContent,
    headline: document.getElementById("headline").textContent,
    strokes: document.getElementById("strokes").textContent,
    guesses: [...document.querySelectorAll("#guesses li")].map(
      (item) => item.textContent,
    ),
    timeline: document.getElementById("timeline").textContent,
  }));

  await page.click("#play");
  // The canvas reset reaches the stroke counter through the rAF-coalesced
  // result path, one frame after the prompt block appears.
  await page.waitForFunction(
    () =>
      document.querySelector(".game-live").hidden === false &&
      document.getElementById("strokes").textContent === "0 笔",
    null,
    { timeout: 2_000 },
  );
  const prompt = await page.evaluate(() => ({
    target: document.getElementById("target-en").textContent,
    strokes: document.getElementById("strokes").textContent,
    countdown: document.getElementById("countdown").textContent,
  }));
  if (!prompt.target) throw new Error("Prompt mode showed no target.");
  if (prompt.strokes !== "0 笔")
    throw new Error("Prompt mode did not clear the canvas.");
  await page.click("#stop");

  await page.close();

  // Dark scheme: the stroke must be drawn in the light `--ink`, not the dark
  // fallback. A self-referencing `--ink` on `#pad` once made every stroke
  // `#111`, invisible on the dark canvas. Sample the pixel under a short
  // horizontal stroke through the centre.
  const dark = await browser.newPage({
    viewport: { width: 1100, height: 900 },
    colorScheme: "dark",
  });
  dark.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  dark.on("pageerror", (error) => errors.push(String(error)));
  await dark.goto(server.resolvedUrls.local[0]);
  const darkBox = await dark.locator("#pad").boundingBox();
  const darkY = darkBox.y + darkBox.height / 2;
  await dark.mouse.move(darkBox.x + darkBox.width * 0.3, darkY);
  await dark.mouse.down();
  await dark.mouse.move(darkBox.x + darkBox.width * 0.7, darkY, { steps: 10 });
  await dark.mouse.up();
  const sampleInk = () =>
    dark.evaluate(() => {
      const canvas = document.getElementById("pad");
      const ratio = window.devicePixelRatio || 1;
      const context = canvas.getContext("2d");
      const x = Math.round((canvas.clientWidth / 2) * ratio);
      const y = Math.round((canvas.clientHeight / 2) * ratio);
      const [r, g, b, a] = context.getImageData(x, y, 1, 1).data;
      const ink = getComputedStyle(canvas).getPropertyValue("--ink").trim();
      return { ink, pixel: [r, g, b, a], luminance: (r + g + b) / 3 };
    });
  const darkInk = await sampleInk();
  // Flip the scheme with the stroke still on the canvas: the sketchpad's
  // `prefers-color-scheme` listener must repaint it in the light-scheme ink.
  // The change event dispatches on the next rendering frame, after computed
  // styles already show the new value, so wait on the pixel, not on `--ink`.
  await dark.emulateMedia({ colorScheme: "light" });
  await dark.waitForFunction(
    () => {
      const canvas = document.getElementById("pad");
      const ratio = window.devicePixelRatio || 1;
      const [r, g, b] = canvas
        .getContext("2d")
        .getImageData(
          Math.round((canvas.clientWidth / 2) * ratio),
          Math.round((canvas.clientHeight / 2) * ratio),
          1,
          1,
        ).data;
      return (r + g + b) / 3 < 60;
    },
    null,
    { timeout: 2_000 },
  );
  const flippedInk = await sampleInk();
  await dark.close();

  const result = { afterCircle, prompt, darkInk, flippedInk, errors };
  console.log(JSON.stringify(result, null, 2));
  if (errors.length) throw new Error("Console errors during smoke test.");
  if (afterCircle.strokes !== "1 笔") throw new Error("Stroke count is wrong.");
  if (!/ms$/.test(afterCircle.timing)) throw new Error("No timing shown.");
  if (darkInk.ink !== "#f2f2f0")
    throw new Error(
      `Dark scheme --ink resolved to ${JSON.stringify(darkInk.ink)}.`,
    );
  if (darkInk.pixel[3] === 0 || darkInk.luminance < 200)
    throw new Error(`Dark scheme stroke is not light: ${darkInk.pixel}.`);
  if (flippedInk.pixel[3] === 0 || flippedInk.luminance > 60)
    throw new Error(
      `Stroke was not repainted after the scheme flip: ${flippedInk.pixel}.`,
    );
} finally {
  await browser.close();
  await server.close();
}

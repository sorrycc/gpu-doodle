/**
 * Render the link-preview image `public/og.png` (1200×630) that the `og:image`
 * tag in `index.html` points at. Serves the site with Vite like the smoke
 * test, opens it in headless Chrome as a Chinese, light-scheme page, draws a
 * few strokes on the pad, waits for the guess list and screenshots the
 * viewport. The image is committed, not built in CI, so a different guess on
 * another machine never churns the file; rerun this when the UI changes.
 *
 *   pnpm site:og
 */
import { chromium } from "playwright";
import { createServer } from "vite";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const output = `${root}/public/og.png`;
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
try {
  const page = await browser.newPage({
    viewport: { width: 1200, height: 630 },
    deviceScaleFactor: 1,
    locale: "zh-CN",
    colorScheme: "light",
  });
  await page.goto(server.resolvedUrls.local[0]);
  await page.waitForSelector("#backend[data-backend]", { timeout: 15_000 });
  // Hide the pointer hint and the cursor so the pad reads as a finished
  // sketch.
  await page.addStyleTag({ content: ".hint { display: none !important }" });

  const box = await page.locator("#pad").boundingBox();
  const point = (x, y) => [box.x + box.width * x, box.y + box.height * y];
  const stroke = async (points) => {
    await page.mouse.move(...points[0]);
    await page.mouse.down();
    for (const [x, y] of points.slice(1))
      await page.mouse.move(x, y, { steps: 4 });
    await page.mouse.up();
  };
  const arc = (cx, cy, rx, ry, from, to, steps = 36) =>
    Array.from({ length: steps + 1 }, (_, i) => {
      const a = from + ((to - from) * i) / steps;
      return point(cx + rx * Math.cos(a), cy + ry * Math.sin(a));
    });
  // A cat: face, two ears, eyes, nose, whiskers.
  await stroke(arc(0.5, 0.55, 0.28, 0.28, 0, Math.PI * 2));
  await stroke([point(0.3, 0.38), point(0.27, 0.15), point(0.44, 0.28)]);
  await stroke([point(0.7, 0.38), point(0.73, 0.15), point(0.56, 0.28)]);
  await stroke(arc(0.4, 0.5, 0.03, 0.03, 0, Math.PI * 2, 12));
  await stroke(arc(0.6, 0.5, 0.03, 0.03, 0, Math.PI * 2, 12));
  await stroke([
    point(0.47, 0.6),
    point(0.53, 0.6),
    point(0.5, 0.65),
    point(0.47, 0.6),
  ]);
  await stroke([point(0.35, 0.62), point(0.15, 0.58)]);
  await stroke([point(0.35, 0.67), point(0.15, 0.7)]);
  await stroke([point(0.65, 0.62), point(0.85, 0.58)]);
  await stroke([point(0.65, 0.67), point(0.85, 0.7)]);
  await page.waitForFunction(
    () => document.querySelectorAll("#guesses li").length >= 3,
    null,
    { timeout: 5_000 },
  );
  const top = await page.evaluate(() =>
    document.querySelector("#guesses li").textContent.trim(),
  );
  await page.screenshot({ path: output, type: "png" });
  console.log(`Wrote ${output} (top guess: ${top})`);
} finally {
  await browser.close();
  await server.close();
}

/**
 * First-load smoke test for the demo: serve the site with Vite, open it in
 * headless Chrome, draw a circle with the mouse, and require a ranked guess
 * list with no console errors. Also exercises the prompt mode once, checks
 * that a Chinese-locale page renders Chinese and that a language choice
 * survives a reload, and checks in a dark-scheme page that the ink stays
 * light on the dark canvas. Finally checks that the `og:image` tag points
 * at a 1200×630 PNG that exists under `public/`, so a renamed or missing
 * preview image fails here instead of showing up as a bare link.
 *
 * Playwright's default locale is en-US, so the first page is the English UI;
 * the assertions use `data-*` hooks where the text depends on the language.
 *
 *   pnpm --filter @gpu-doodle/site test:smoke
 */
import { chromium } from "playwright";
import { createServer } from "vite";
import { fileURLToPath } from "node:url";
import { readFileSync, statSync } from "node:fs";

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
  await page.waitForSelector("#backend[data-backend]", { timeout: 15_000 });
  const english = await page.evaluate(() => ({
    lang: document.documentElement.lang,
    title: document.title,
    headline: document.getElementById("headline").textContent.trim(),
    pending: document.documentElement.classList.contains("i18n-pending"),
  }));

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
      document.getElementById("strokes").dataset.strokes === "0",
    null,
    { timeout: 2_000 },
  );
  const prompt = await page.evaluate(() => ({
    target: document.getElementById("target-name").textContent,
    strokes: document.getElementById("strokes").dataset.strokes,
    countdown: document.getElementById("countdown").textContent,
  }));
  if (!prompt.target) throw new Error("Prompt mode showed no target.");
  if (prompt.strokes !== "0")
    throw new Error("Prompt mode did not clear the canvas.");
  await page.click("#stop");

  await page.close();

  // Chinese system language: the page must come up in Chinese without any
  // stored choice. Then pick English from the switch and reload; the choice
  // must win over the system language.
  const chinese = await browser.newPage({
    viewport: { width: 1100, height: 900 },
    locale: "zh-CN",
  });
  chinese.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  chinese.on("pageerror", (error) => errors.push(String(error)));
  await chinese.goto(server.resolvedUrls.local[0]);
  await chinese.waitForSelector("#backend[data-backend]", { timeout: 15_000 });
  const zh = await chinese.evaluate(() => ({
    lang: document.documentElement.lang,
    title: document.title,
    headline: document.getElementById("headline").textContent.trim(),
    firstLabel: document.querySelector("#labels li").textContent,
  }));
  await chinese.selectOption("#language", "en");
  const switched = await chinese.evaluate(() => ({
    lang: document.documentElement.lang,
    headline: document.getElementById("headline").textContent.trim(),
    firstLabel: document.querySelector("#labels li").textContent,
  }));
  await chinese.reload();
  await chinese.waitForSelector("#backend[data-backend]", { timeout: 15_000 });
  const persisted = await chinese.evaluate(() => ({
    lang: document.documentElement.lang,
    headline: document.getElementById("headline").textContent.trim(),
    language: document.getElementById("language").value,
  }));
  await chinese.close();

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

  // Theme switch: with the system in light mode, picking "dark" must set
  // `data-theme`, repaint the stroke already on the canvas in the light ink,
  // and survive a reload.
  const themed = await browser.newPage({
    viewport: { width: 1100, height: 900 },
    colorScheme: "light",
  });
  themed.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  themed.on("pageerror", (error) => errors.push(String(error)));
  await themed.goto(server.resolvedUrls.local[0]);
  await themed.waitForSelector("#backend[data-backend]", { timeout: 15_000 });
  const themedBox = await themed.locator("#pad").boundingBox();
  const themedY = themedBox.y + themedBox.height / 2;
  await themed.mouse.move(themedBox.x + themedBox.width * 0.3, themedY);
  await themed.mouse.down();
  await themed.mouse.move(themedBox.x + themedBox.width * 0.7, themedY, {
    steps: 10,
  });
  await themed.mouse.up();
  await themed.selectOption("#theme", "dark");
  const sampleThemed = () =>
    themed.evaluate(() => {
      const canvas = document.getElementById("pad");
      const ratio = window.devicePixelRatio || 1;
      const [r, g, b, a] = canvas
        .getContext("2d")
        .getImageData(
          Math.round((canvas.clientWidth / 2) * ratio),
          Math.round((canvas.clientHeight / 2) * ratio),
          1,
          1,
        ).data;
      return {
        theme: document.documentElement.dataset.theme ?? "",
        ink: getComputedStyle(canvas).getPropertyValue("--ink").trim(),
        pixel: [r, g, b, a],
        luminance: (r + g + b) / 3,
        select: document.getElementById("theme").value,
      };
    });
  const darkOverride = await sampleThemed();
  await themed.reload();
  await themed.waitForSelector("#backend[data-backend]", { timeout: 15_000 });
  const darkPersisted = await sampleThemed();
  await themed.close();

  // Link preview: the static head must name an image that ships with the
  // site and matches the declared size. PNG stores width and height as
  // big-endian u32 at bytes 16 and 20 of the IHDR chunk.
  const html = readFileSync(`${root}/index.html`, "utf8");
  const imageTag = html.match(/property="og:image"\s+content="([^"]+)"/);
  const declaredWidth = Number(
    html.match(/property="og:image:width" content="(\d+)"/)?.[1],
  );
  const declaredHeight = Number(
    html.match(/property="og:image:height" content="(\d+)"/)?.[1],
  );
  const imageFile = imageTag && new URL(imageTag[1]).pathname.split("/").pop();
  const imageBytes = imageFile
    ? readFileSync(`${root}/public/${imageFile}`)
    : null;
  const preview = {
    url: imageTag?.[1] ?? "",
    file: imageFile ?? "",
    declared: [declaredWidth, declaredHeight],
    actual: imageBytes
      ? [imageBytes.readUInt32BE(16), imageBytes.readUInt32BE(20)]
      : null,
    bytes: imageFile ? statSync(`${root}/public/${imageFile}`).size : 0,
  };

  const result = {
    preview,
    english,
    afterCircle,
    prompt,
    zh,
    switched,
    persisted,
    darkInk,
    flippedInk,
    darkOverride,
    darkPersisted,
    errors,
  };
  console.log(JSON.stringify(result, null, 2));
  if (errors.length) throw new Error("Console errors during smoke test.");
  if (english.lang !== "en" || english.headline !== "Draw something on the pad")
    throw new Error("en-US locale did not render the English UI.");
  if (english.pending) throw new Error("Body is still hidden after render.");
  if (afterCircle.strokes !== "1 stroke")
    throw new Error("Stroke count is wrong.");
  if (zh.lang !== "zh-CN" || zh.headline !== "在画板上画点什么")
    throw new Error("zh-CN locale did not render the Chinese UI.");
  if (!/^\S+ airplane$/.test(zh.firstLabel))
    throw new Error(
      `Chinese label list lacks the English name: ${zh.firstLabel}`,
    );
  if (switched.lang !== "en" || switched.firstLabel !== "airplane")
    throw new Error("Switching to English did not re-render the page.");
  if (persisted.lang !== "en" || persisted.language !== "en")
    throw new Error("Language choice did not survive a reload.");
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
  if (darkOverride.theme !== "dark" || darkOverride.ink !== "#f2f2f0")
    throw new Error("Theme switch did not override the light scheme.");
  if (darkOverride.pixel[3] === 0 || darkOverride.luminance < 200)
    throw new Error(
      `Stroke was not repainted after the theme switch: ${darkOverride.pixel}.`,
    );
  if (darkPersisted.theme !== "dark" || darkPersisted.select !== "dark")
    throw new Error("Theme choice did not survive a reload.");
  if (
    !/^https:\/\/sorrycc\.github\.io\/gpu-doodle\/[^/]+\.png$/.test(preview.url)
  )
    throw new Error(`og:image is not an absolute site URL: ${preview.url}`);
  if (
    !preview.actual ||
    preview.actual[0] !== preview.declared[0] ||
    preview.actual[1] !== preview.declared[1] ||
    preview.declared[0] !== 1200 ||
    preview.declared[1] !== 630
  )
    throw new Error(
      `og:image ${preview.file} is ${preview.actual}, declared ${preview.declared}.`,
    );
} finally {
  await browser.close();
  await server.close();
}

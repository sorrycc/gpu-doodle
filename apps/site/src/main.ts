import "./style.css";
import {
  classify,
  defineClassifier,
  LABELS,
  type Classifier,
  type Guess,
  type Label,
  type Stroke,
} from "gpu-doodle";
import { Sketchpad } from "./sketchpad.ts";
import { Game, ROUND_SECONDS, type GameState } from "./game.ts";
import {
  detectLocale,
  isLocale,
  LANGUAGE_TAGS,
  persistLocale,
  Translator,
  type Key,
  type Locale,
} from "./i18n.ts";
import { applyTheme, isTheme, readTheme } from "./theme.ts";

const TOP_K = 5;

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing #${id}`);
  return found as T;
}

const ui = {
  pad: element<HTMLCanvasElement>("pad"),
  undo: element<HTMLButtonElement>("undo"),
  clear: element<HTMLButtonElement>("clear"),
  strokes: element<HTMLSpanElement>("strokes"),
  headline: element<HTMLParagraphElement>("headline"),
  guesses: element<HTMLOListElement>("guesses"),
  timeline: element<HTMLParagraphElement>("timeline"),
  backend: element<HTMLSpanElement>("backend"),
  timing: element<HTMLSpanElement>("timing"),
  labelSummary: element<HTMLElement>("label-summary"),
  labels: element<HTMLUListElement>("labels"),
  game: element<HTMLElement>("game"),
  play: element<HTMLButtonElement>("play"),
  skip: element<HTMLButtonElement>("skip"),
  stop: element<HTMLButtonElement>("stop"),
  targetName: element<HTMLElement>("target-name"),
  targetAlt: element<HTMLSpanElement>("target-alt"),
  countdown: element<HTMLSpanElement>("countdown"),
  score: element<HTMLSpanElement>("score"),
  gameStatus: element<HTMLParagraphElement>("game-status"),
  language: element<HTMLSelectElement>("language"),
  theme: element<HTMLSelectElement>("theme"),
};

const i18n = new Translator(detectLocale());

// Everything the page shows is re-rendered from these on a language switch.
let lastGuesses: Guess[] = [];
let lastStrokes = 0;
let lastElapsed = 0;
let lastGame: GameState | undefined;
let backendNote: Key = "backend.init";
const timeline: Label[] = [];

/** Static strings: `data-i18n` text, `data-i18n-aria` labels, head, label list. */
function renderStatic(): void {
  const { locale } = i18n;
  document.documentElement.lang = LANGUAGE_TAGS[locale];
  document.title = i18n.t("title");
  document
    .querySelector('meta[name="description"]')
    ?.setAttribute("content", i18n.t("description"));
  for (const node of document.querySelectorAll<HTMLElement>("[data-i18n]"))
    node.textContent = i18n.t(node.dataset.i18n as Key);
  for (const node of document.querySelectorAll<HTMLElement>("[data-i18n-aria]"))
    node.setAttribute("aria-label", i18n.t(node.dataset.i18nAria as Key));
  ui.language.setAttribute("aria-label", i18n.t("controls.language"));
  ui.theme.setAttribute("aria-label", i18n.t("controls.theme"));
  ui.language.value = locale;
  ui.labelSummary.textContent = i18n.t("labels.summary", {
    count: LABELS.length,
  });
  ui.labels.replaceChildren(
    ...LABELS.map((label) => {
      const item = document.createElement("li");
      item.textContent = i18n.labelText(label);
      return item;
    }),
  );
}

function setLocale(locale: Locale): void {
  i18n.use(locale);
  renderStatic();
  render(lastGuesses, lastStrokes, lastElapsed);
  if (lastGame) renderGame(lastGame);
}

// One sketch at a time is fastest on the CPU; the demo still runs WebGPU when
// it can because seeing the kernel work is the point of the page. A failed
// device or a lost one drops to the CPU path and says so.
let classifier: Classifier | undefined;
let backend: "cpu" | "webgpu" = "cpu";

async function initialize(): Promise<void> {
  if (!navigator.gpu) {
    backendNote = "backend.missing";
    return;
  }
  try {
    classifier = await defineClassifier({ backend: "webgpu" });
    backend = "webgpu";
    backendNote = "backend.webgpu";
  } catch {
    backendNote = "backend.failed";
  }
}

function fallbackToCPU(reason: Key): void {
  classifier?.dispose();
  classifier = undefined;
  backend = "cpu";
  backendNote = reason;
}

async function score(strokes: readonly Stroke[]): Promise<Guess[]> {
  if (backend === "webgpu" && classifier) {
    try {
      const [guesses] = await classifier.classifyMany([strokes], {
        topK: TOP_K,
      });
      return guesses;
    } catch {
      fallbackToCPU("backend.lost");
    }
  }
  return classify(strokes, { topK: TOP_K });
}

let ticket = 0;
let scheduled = false;
let latest: { strokes: readonly Stroke[]; live: boolean } | undefined;

function schedule(strokes: readonly Stroke[], live: boolean): void {
  latest = { strokes, live };
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => {
    scheduled = false;
    const request = latest;
    latest = undefined;
    if (request) void run(request.strokes, request.live);
  });
}

async function run(strokes: readonly Stroke[], live: boolean): Promise<void> {
  const id = ++ticket;
  const started = performance.now();
  const guesses = await score(strokes);
  const elapsed = performance.now() - started;
  // A pen-up result is keyed by stroke count, so it stays valid even when a
  // newer live request has already superseded it for display.
  if (!live) {
    timeline.length = strokes.length;
    if (strokes.length > 0 && guesses[0])
      timeline[strokes.length - 1] = guesses[0].label;
  }
  if (id !== ticket) return;
  render(guesses, strokes.length, elapsed);
  game.observe(guesses, strokes.length);
}

function render(guesses: Guess[], strokes: number, elapsed: number): void {
  lastGuesses = guesses;
  lastStrokes = strokes;
  lastElapsed = elapsed;
  ui.strokes.textContent = i18n.strokes(strokes);
  ui.strokes.dataset.strokes = String(strokes);
  ui.undo.disabled = strokes === 0;
  ui.clear.disabled = strokes === 0;
  const top = guesses[0];
  if (strokes === 0 || !top) {
    ui.headline.textContent = i18n.t("headline.empty");
    ui.guesses.replaceChildren();
    ui.timeline.textContent = "";
    ui.timing.textContent = "";
  } else {
    const name = i18n.labelName(top.label);
    ui.headline.textContent =
      top.probability >= 0.6
        ? i18n.t("headline.sure", { name })
        : top.probability >= 0.3
          ? i18n.t("headline.maybe", { name })
          : i18n.t("headline.unsure");
    ui.guesses.replaceChildren(
      ...guesses.map((guess) => {
        const item = document.createElement("li");
        const bar = document.createElement("span");
        bar.className = "bar";
        bar.style.width = `${Math.max(1, guess.probability * 100).toFixed(1)}%`;
        const text = document.createElement("span");
        text.className = "label";
        text.textContent = i18n.labelText(guess.label);
        const percent = document.createElement("span");
        percent.className = "percent";
        percent.textContent = `${(guess.probability * 100).toFixed(0)}%`;
        item.append(bar, text, percent);
        return item;
      }),
    );
    ui.timeline.textContent = timeline
      .map((label, index) => `${index + 1} ${i18n.labelName(label)}`)
      .join(" · ");
    ui.timing.textContent = `${elapsed.toFixed(1)} ms`;
  }
  ui.backend.textContent = i18n.t(backendNote);
}

const game = new Game({
  onReset: () => pad.clear(),
  onState: (state) => renderGame(state),
});

function renderGame(state: GameState): void {
  lastGame = state;
  const idle = ui.game.querySelector<HTMLElement>(".game-idle")!;
  const live = ui.game.querySelector<HTMLElement>(".game-live")!;
  idle.hidden = state.phase !== "idle";
  live.hidden = state.phase === "idle";
  ui.game.dataset.phase = state.phase;
  if (state.phase === "idle" || !state.target) return;
  ui.targetName.textContent = i18n.labelName(state.target);
  ui.targetAlt.textContent = i18n.labelAlt(state.target);
  ui.countdown.textContent = i18n.t("countdown", {
    s: state.phase === "drawing" ? state.remaining.toFixed(1) : ROUND_SECONDS,
  });
  ui.score.textContent = i18n.t("score", {
    solved: state.solved,
    rounds: state.rounds,
  });
  const { outcome } = state;
  ui.gameStatus.textContent =
    outcome.kind === "solved"
      ? i18n.t("game.solved", {
          strokes: outcome.strokes,
          seconds: outcome.seconds.toFixed(1),
        })
      : outcome.kind === "timeout"
        ? i18n.t("game.timeout", { name: i18n.labelText(state.target) })
        : "";
  ui.skip.textContent = i18n.t(state.phase === "drawing" ? "skip" : "next");
}

const pad = new Sketchpad(ui.pad, {
  onChange: (strokes, live) => schedule(strokes, live),
});

ui.undo.addEventListener("click", () => pad.undo());
ui.clear.addEventListener("click", () => pad.clear());
ui.play.addEventListener("click", () => game.start());
ui.skip.addEventListener("click", () => game.skip());
ui.stop.addEventListener("click", () => game.stop());
ui.language.addEventListener("change", () => {
  const { value } = ui.language;
  if (!isLocale(value)) return;
  persistLocale(value);
  setLocale(value);
  // A `?lang=` in the URL would beat the choice on the next load; drop it.
  if (new URLSearchParams(location.search).has("lang"))
    history.replaceState(null, "", location.pathname + location.hash);
});
ui.theme.addEventListener("change", () => {
  const { value } = ui.theme;
  if (!isTheme(value)) return;
  applyTheme(value);
  // `--ink` changed without a `prefers-color-scheme` event; repaint by hand.
  pad.repaint();
});
window.addEventListener("keydown", (event) => {
  if (event.metaKey || event.ctrlKey) {
    if (event.key === "z") {
      event.preventDefault();
      pad.undo();
    }
    return;
  }
  if (event.key === "Escape") pad.clear();
});

ui.theme.value = readTheme();
renderStatic();
render([], 0, 0);
// The head script hid the body while the static markup was in the wrong
// language; every string is in place now.
document.documentElement.classList.remove("i18n-pending");
void initialize().then(() => {
  ui.backend.textContent = i18n.t(backendNote);
  ui.backend.dataset.backend = backend;
  if (pad.strokes.length) schedule(pad.strokes, false);
});

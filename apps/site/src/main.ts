import "./style.css";
import {
  classify,
  defineClassifier,
  LABELS,
  labelZh,
  type Classifier,
  type Guess,
  type Stroke,
} from "gpu-doodle";
import { Sketchpad } from "./sketchpad.ts";
import { Game, ROUND_SECONDS, type GameState } from "./game.ts";

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
  labels: element<HTMLUListElement>("labels"),
  game: element<HTMLDivElement>("game"),
  play: element<HTMLButtonElement>("play"),
  skip: element<HTMLButtonElement>("skip"),
  stop: element<HTMLButtonElement>("stop"),
  targetZh: element<HTMLElement>("target-zh"),
  targetEn: element<HTMLSpanElement>("target-en"),
  countdown: element<HTMLSpanElement>("countdown"),
  score: element<HTMLSpanElement>("score"),
  gameStatus: element<HTMLParagraphElement>("game-status"),
};

for (const label of LABELS) {
  const item = document.createElement("li");
  item.textContent = `${labelZh[label]} ${label}`;
  ui.labels.append(item);
}

// One sketch at a time is fastest on the CPU; the demo still runs WebGPU when
// it can because seeing the kernel work is the point of the page. A failed
// device or a lost one drops to the CPU path and says so.
let classifier: Classifier | undefined;
let backend: "cpu" | "webgpu" = "cpu";
let backendNote = "";

async function initialize(): Promise<void> {
  if (!navigator.gpu) {
    backendNote = "浏览器没有 WebGPU，用 CPU";
    return;
  }
  try {
    classifier = await defineClassifier({ backend: "webgpu" });
    backend = "webgpu";
    backendNote = "WebGPU";
  } catch {
    backendNote = "WebGPU 初始化失败，用 CPU";
  }
}

function fallbackToCPU(reason: string): void {
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
      fallbackToCPU("WebGPU 设备丢失，已切到 CPU");
    }
  }
  return classify(strokes, { topK: TOP_K });
}

const timeline: string[] = [];
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
      timeline[strokes.length - 1] = labelZh[guesses[0].label];
  }
  if (id !== ticket) return;
  render(guesses, strokes.length, elapsed);
  game.observe(guesses, strokes.length);
}

function render(guesses: Guess[], strokes: number, elapsed: number): void {
  ui.strokes.textContent = `${strokes} 笔`;
  ui.undo.disabled = strokes === 0;
  ui.clear.disabled = strokes === 0;
  const top = guesses[0];
  if (strokes === 0 || !top) {
    ui.headline.textContent = "在画板上画点什么";
    ui.guesses.replaceChildren();
    ui.timeline.textContent = "";
    ui.timing.textContent = "";
  } else {
    const name = `${labelZh[top.label]}`;
    ui.headline.textContent =
      top.probability >= 0.6
        ? `我猜是 ${name}`
        : top.probability >= 0.3
          ? `可能是 ${name}？`
          : "还看不出来…";
    ui.guesses.replaceChildren(
      ...guesses.map((guess) => {
        const item = document.createElement("li");
        const bar = document.createElement("span");
        bar.className = "bar";
        bar.style.width = `${Math.max(1, guess.probability * 100).toFixed(1)}%`;
        const text = document.createElement("span");
        text.className = "label";
        text.textContent = `${labelZh[guess.label]} ${guess.label}`;
        const percent = document.createElement("span");
        percent.className = "percent";
        percent.textContent = `${(guess.probability * 100).toFixed(0)}%`;
        item.append(bar, text, percent);
        return item;
      }),
    );
    ui.timeline.textContent = timeline
      .map((label, index) => `${index + 1} ${label}`)
      .join(" · ");
    ui.timing.textContent = `${elapsed.toFixed(1)} ms`;
  }
  ui.backend.textContent = backendNote;
}

const game = new Game({
  onReset: () => pad.clear(),
  onState: (state) => renderGame(state),
});

function renderGame(state: GameState): void {
  const idle = ui.game.querySelector<HTMLElement>(".game-idle")!;
  const live = ui.game.querySelector<HTMLElement>(".game-live")!;
  idle.hidden = state.phase !== "idle";
  live.hidden = state.phase === "idle";
  ui.game.dataset.phase = state.phase;
  if (state.phase === "idle") return;
  ui.targetZh.textContent = state.targetZh;
  ui.targetEn.textContent = state.target ?? "";
  ui.countdown.textContent =
    state.phase === "drawing"
      ? `${state.remaining.toFixed(1)} s`
      : `${ROUND_SECONDS} s`;
  ui.score.textContent = `猜中 ${state.solved} / ${state.rounds}`;
  ui.gameStatus.textContent = state.message;
  ui.skip.textContent = state.phase === "drawing" ? "跳过" : "下一题";
}

const pad = new Sketchpad(ui.pad, {
  onChange: (strokes, live) => schedule(strokes, live),
});

ui.undo.addEventListener("click", () => pad.undo());
ui.clear.addEventListener("click", () => pad.clear());
ui.play.addEventListener("click", () => game.start());
ui.skip.addEventListener("click", () => game.skip());
ui.stop.addEventListener("click", () => game.stop());
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

render([], 0, 0);
void initialize().then(() => {
  ui.backend.textContent = backendNote;
  if (pad.strokes.length) schedule(pad.strokes, false);
});

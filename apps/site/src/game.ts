import { LABELS, labelZh, type Guess, type Label } from "gpu-doodle";

export const ROUND_SECONDS = 20;
const NEXT_DELAY_MS = 1400;

export type Phase = "idle" | "drawing" | "solved" | "timeout";

export interface GameState {
  phase: Phase;
  target: Label | undefined;
  targetZh: string;
  remaining: number;
  solved: number;
  rounds: number;
  message: string;
}

export interface GameEvents {
  onState(state: GameState): void;
  /** The game asks the page to wipe the canvas before a new prompt. */
  onReset(): void;
}

/**
 * The prompt mode of the original Quick, Draw!: name a category, give the
 * player twenty seconds, advance when the model's top guess matches. The
 * clock runs on `setInterval`; the page feeds every guess through `observe`.
 */
export class Game {
  private phase: Phase = "idle";
  private target: Label | undefined;
  private deadline = 0;
  private solved = 0;
  private rounds = 0;
  private message = "";
  private ticker: ReturnType<typeof setInterval> | undefined;
  private advance: ReturnType<typeof setTimeout> | undefined;

  constructor(
    private readonly events: GameEvents,
    private readonly random: () => number = Math.random,
  ) {}

  get active(): boolean {
    return this.phase !== "idle";
  }

  start(): void {
    this.solved = 0;
    this.rounds = 0;
    this.next();
  }

  skip(): void {
    if (this.phase === "idle") return;
    this.next();
  }

  stop(): void {
    this.clearTimers();
    this.phase = "idle";
    this.target = undefined;
    this.message = "";
    this.emit();
  }

  /** Called with every fresh guess list; `strokes` is the current stroke count. */
  observe(guesses: readonly Guess[], strokes: number): void {
    if (this.phase !== "drawing" || !this.target || strokes === 0) return;
    const top = guesses[0];
    if (!top || top.label !== this.target) return;
    this.clearTimers();
    this.phase = "solved";
    this.solved++;
    const seconds = ROUND_SECONDS - this.remaining();
    this.message = `猜中了！第 ${strokes} 笔，用时 ${seconds.toFixed(1)} 秒`;
    this.emit();
    this.advance = setTimeout(() => this.next(), NEXT_DELAY_MS);
  }

  private next(): void {
    this.clearTimers();
    const previous = this.target;
    let candidate: Label;
    do {
      candidate = LABELS[Math.floor(this.random() * LABELS.length)];
    } while (candidate === previous && LABELS.length > 1);
    this.target = candidate;
    this.rounds++;
    this.phase = "drawing";
    this.message = "";
    this.deadline = performance.now() + ROUND_SECONDS * 1000;
    this.events.onReset();
    this.emit();
    this.ticker = setInterval(() => this.tick(), 100);
  }

  private tick(): void {
    if (this.phase !== "drawing") return;
    if (this.remaining() <= 0) {
      this.clearTimers();
      this.phase = "timeout";
      this.message = `时间到，答案是 ${labelZh[this.target!]} ${this.target}`;
    }
    this.emit();
  }

  private remaining(): number {
    return Math.max(0, (this.deadline - performance.now()) / 1000);
  }

  private clearTimers(): void {
    if (this.ticker !== undefined) clearInterval(this.ticker);
    if (this.advance !== undefined) clearTimeout(this.advance);
    this.ticker = undefined;
    this.advance = undefined;
  }

  private emit(): void {
    this.events.onState({
      phase: this.phase,
      target: this.target,
      targetZh: this.target ? labelZh[this.target] : "",
      remaining: this.phase === "drawing" ? this.remaining() : 0,
      solved: this.solved,
      rounds: this.rounds,
      message: this.message,
    });
  }
}

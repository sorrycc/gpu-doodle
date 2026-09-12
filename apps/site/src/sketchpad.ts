import type { Stroke } from "gpu-doodle";

export interface SketchpadEvents {
  /** Fires on every pointer move while drawing (`live`) and once on pen up. */
  onChange(strokes: readonly Stroke[], live: boolean): void;
}

/**
 * A square drawing surface that records strokes as `{x, y}` arrays in CSS
 * pixels, the shape `classify` accepts. The model never sees these numbers
 * directly: `simplify` in the core package rescales and simplifies them.
 */
export class Sketchpad {
  readonly strokes: Stroke[] = [];
  private readonly context: CanvasRenderingContext2D;
  private pointerId: number | undefined;
  private lineWidth = 6;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly events: SketchpadEvents,
  ) {
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas 2D is unavailable.");
    this.context = context;
    canvas.style.touchAction = "none";
    canvas.addEventListener("pointerdown", this.down);
    canvas.addEventListener("pointermove", this.move);
    canvas.addEventListener("pointerup", this.up);
    canvas.addEventListener("pointercancel", this.up);
    canvas.addEventListener("lostpointercapture", this.up);
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  get drawing(): boolean {
    return this.pointerId !== undefined;
  }

  undo(): void {
    if (this.drawing || this.strokes.length === 0) return;
    this.strokes.pop();
    this.redraw();
    this.events.onChange(this.strokes, false);
  }

  clear(): void {
    if (this.drawing) return;
    this.strokes.length = 0;
    this.redraw();
    this.events.onChange(this.strokes, false);
  }

  private resize(): void {
    const ratio = window.devicePixelRatio || 1;
    const size = this.canvas.clientWidth;
    if (size === 0) return;
    const pixels = Math.round(size * ratio);
    if (this.canvas.width !== pixels || this.canvas.height !== pixels) {
      this.canvas.width = pixels;
      this.canvas.height = pixels;
    }
    this.redraw();
  }

  private position(event: PointerEvent): [number, number] {
    const bounds = this.canvas.getBoundingClientRect();
    return [event.clientX - bounds.left, event.clientY - bounds.top];
  }

  private readonly down = (event: PointerEvent): void => {
    if (this.drawing || event.button !== 0) return;
    event.preventDefault();
    this.pointerId = event.pointerId;
    this.canvas.setPointerCapture(event.pointerId);
    const [x, y] = this.position(event);
    this.strokes.push({ x: [x], y: [y] });
    this.redraw();
    this.events.onChange(this.strokes, true);
  };

  private readonly move = (event: PointerEvent): void => {
    if (event.pointerId !== this.pointerId) return;
    event.preventDefault();
    const stroke = this.strokes[this.strokes.length - 1];
    const samples =
      typeof event.getCoalescedEvents === "function"
        ? event.getCoalescedEvents()
        : [];
    const ratio = window.devicePixelRatio || 1;
    this.context.save();
    this.context.scale(ratio, ratio);
    this.applyStyle();
    for (const sample of samples.length ? samples : [event]) {
      const [x, y] = this.position(sample);
      const previousX = stroke.x[stroke.x.length - 1];
      const previousY = stroke.y[stroke.y.length - 1];
      stroke.x.push(x);
      stroke.y.push(y);
      this.context.beginPath();
      this.context.moveTo(previousX, previousY);
      this.context.lineTo(x, y);
      this.context.stroke();
    }
    this.context.restore();
    this.events.onChange(this.strokes, true);
  };

  private readonly up = (event: PointerEvent): void => {
    if (event.pointerId !== this.pointerId) return;
    this.pointerId = undefined;
    if (this.canvas.hasPointerCapture(event.pointerId))
      this.canvas.releasePointerCapture(event.pointerId);
    this.events.onChange(this.strokes, false);
  };

  private applyStyle(): void {
    const ink = getComputedStyle(this.canvas).getPropertyValue("--ink").trim();
    this.context.strokeStyle = ink || "#111";
    this.context.fillStyle = this.context.strokeStyle;
    this.context.lineWidth = this.lineWidth;
    this.context.lineCap = "round";
    this.context.lineJoin = "round";
  }

  private redraw(): void {
    const ratio = window.devicePixelRatio || 1;
    this.context.save();
    this.context.setTransform(1, 0, 0, 1, 0, 0);
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.context.scale(ratio, ratio);
    this.applyStyle();
    for (const stroke of this.strokes) {
      if (stroke.x.length === 1) {
        this.context.beginPath();
        this.context.arc(
          stroke.x[0],
          stroke.y[0],
          this.lineWidth / 2,
          0,
          Math.PI * 2,
        );
        this.context.fill();
        continue;
      }
      this.context.beginPath();
      this.context.moveTo(stroke.x[0], stroke.y[0]);
      for (let index = 1; index < stroke.x.length; index++)
        this.context.lineTo(stroke.x[index], stroke.y[index]);
      this.context.stroke();
    }
    this.context.restore();
  }
}

import { labelZh, type Label } from "gpu-doodle";

export type Locale = "zh" | "en";
export const LOCALES: readonly Locale[] = ["zh", "en"];
export const LOCALE_STORAGE_KEY = "gpu-doodle:lang";

/** The BCP 47 tag written to `<html lang>` for each locale. */
export const LANGUAGE_TAGS: Record<Locale, string> = {
  zh: "zh-CN",
  en: "en",
};

// Keys are shared by both dictionaries; `{name}`-style placeholders are
// filled by `t`. The head of `index.html` carries a copy of the title and
// description so the first paint is already in the right language.
const zh = {
  title: "gpu-doodle · 你画我猜",
  description:
    "一个三万多参数的小模型在你的浏览器里边画边猜。WebGPU 推理，不上传任何数据。",
  "hero.body":
    "画点什么，一个三万多参数的小模型在你的浏览器里边画边猜。推理跑在 WebGPU 上，笔画不离开这台设备。",
  "controls.language": "语言",
  "controls.theme": "主题",
  "theme.system": "跟随系统",
  "theme.light": "浅色",
  "theme.dark": "深色",
  "pad.label": "画板",
  undo: "撤销一笔",
  clear: "清空",
  "strokes.one": "{n} 笔",
  "strokes.other": "{n} 笔",
  hints: "快捷键：⌘/Ctrl+Z 撤销一笔，Esc 清空",
  play: "来一题，20 秒内画出它",
  "prompt.draw": "请画",
  countdown: "{s} s",
  score: "猜中 {solved} / {rounds}",
  skip: "跳过",
  next: "下一题",
  stop: "结束",
  "game.solved": "猜中了！第 {strokes} 笔，用时 {seconds} 秒",
  "game.timeout": "时间到，答案是 {name}",
  "headline.empty": "在画板上画点什么",
  "headline.sure": "我猜是 {name}",
  "headline.maybe": "可能是 {name}？",
  "headline.unsure": "还看不出来…",
  "backend.init": "正在初始化模型…",
  "backend.webgpu": "WebGPU",
  "backend.missing": "浏览器没有 WebGPU，用 CPU",
  "backend.failed": "WebGPU 初始化失败，用 CPU",
  "backend.lost": "WebGPU 设备丢失，已切到 CPU",
  "labels.summary": "它认识的 {count} 样东西",
  "footer.dataBefore": "训练数据来自 Google 的",
  "footer.dataLink": "Quick, Draw! 数据集",
  "footer.dataAfter":
    "（CC BY 4.0）。模型只见过这 100 类，画别的它也会硬猜一个。",
  "footer.codeBefore": "代码与模型：gpu-doodle，MIT。配方来自",
  "footer.codeJoin": "与",
  "footer.codeAfter": "。",
};

const en: Record<Key, string> = {
  title: "gpu-doodle · Guess the doodle",
  description:
    "A tiny model with just over thirty thousand parameters guesses your doodle while you draw. WebGPU inference, nothing uploaded.",
  "hero.body":
    "Draw something and a model with just over thirty thousand parameters guesses along as you go, right in your browser. Inference runs on WebGPU; your strokes never leave this device.",
  "controls.language": "Language",
  "controls.theme": "Theme",
  "theme.system": "System",
  "theme.light": "Light",
  "theme.dark": "Dark",
  "pad.label": "Drawing pad",
  undo: "Undo stroke",
  clear: "Clear",
  "strokes.one": "{n} stroke",
  "strokes.other": "{n} strokes",
  hints: "Shortcuts: ⌘/Ctrl+Z undo, Esc clear",
  play: "Play a round: draw the prompt in 20 s",
  "prompt.draw": "Draw",
  countdown: "{s} s",
  score: "Got {solved} / {rounds}",
  skip: "Skip",
  next: "Next",
  stop: "Quit",
  "game.solved": "Got it! Stroke {strokes}, {seconds} s",
  "game.timeout": "Time's up, it was {name}",
  "headline.empty": "Draw something on the pad",
  "headline.sure": "I think it's {name}",
  "headline.maybe": "Maybe {name}?",
  "headline.unsure": "Can't tell yet…",
  "backend.init": "Loading the model…",
  "backend.webgpu": "WebGPU",
  "backend.missing": "No WebGPU in this browser, using CPU",
  "backend.failed": "WebGPU failed to start, using CPU",
  "backend.lost": "WebGPU device lost, switched to CPU",
  "labels.summary": "The {count} things it knows",
  "footer.dataBefore": "Training data comes from Google's",
  "footer.dataLink": "Quick, Draw! dataset",
  "footer.dataAfter":
    " (CC BY 4.0). The model has only seen these 100 categories; draw anything else and it will still pick one.",
  "footer.codeBefore": "Code and model: gpu-doodle, MIT. Recipe from",
  "footer.codeJoin": "and",
  "footer.codeAfter": ".",
};

export type Key = keyof typeof zh;

const dictionaries: Record<Locale, Record<Key, string>> = { zh, en };

export function isLocale(value: unknown): value is Locale {
  return value === "zh" || value === "en";
}

/**
 * Query string, then the persisted choice, then the first system language
 * that is Chinese or English. Anything else reads English. The inline script
 * in `index.html` mirrors this order so `<html lang>` is right before paint.
 */
export function detectLocale(): Locale {
  const query = new URLSearchParams(location.search).get("lang");
  if (isLocale(query)) return query;
  try {
    const stored = localStorage.getItem(LOCALE_STORAGE_KEY);
    if (isLocale(stored)) return stored;
  } catch {
    // Storage can be blocked; fall through to the system language.
  }
  const tags = navigator.languages?.length
    ? navigator.languages
    : [navigator.language];
  for (const tag of tags) {
    if (/^zh\b/i.test(tag)) return "zh";
    if (/^en\b/i.test(tag)) return "en";
  }
  return "en";
}

export function persistLocale(locale: Locale): void {
  try {
    localStorage.setItem(LOCALE_STORAGE_KEY, locale);
  } catch {
    // Without storage the choice lasts for the page only.
  }
}

export class Translator {
  private plurals: Intl.PluralRules;

  constructor(public locale: Locale) {
    this.plurals = new Intl.PluralRules(LANGUAGE_TAGS[locale]);
  }

  use(locale: Locale): void {
    this.locale = locale;
    this.plurals = new Intl.PluralRules(LANGUAGE_TAGS[locale]);
  }

  t(key: Key, params: Record<string, string | number> = {}): string {
    return dictionaries[this.locale][key].replace(
      /\{(\w+)\}/g,
      (match, name: string) => (name in params ? String(params[name]) : match),
    );
  }

  /** `{n} 笔` in Chinese, `1 stroke` / `2 strokes` in English. */
  strokes(n: number): string {
    const form =
      this.plurals.select(n) === "one" ? "strokes.one" : "strokes.other";
    return this.t(form, { n });
  }

  /** The primary name of a category: Chinese in `zh`, the label itself in `en`. */
  labelName(label: Label): string {
    return this.locale === "zh" ? labelZh[label] : label;
  }

  /** The secondary name shown beside the primary one, or nothing. */
  labelAlt(label: Label): string {
    return this.locale === "zh" ? label : "";
  }

  /** Primary and secondary name joined, for lists and messages. */
  labelText(label: Label): string {
    const alt = this.labelAlt(label);
    return alt ? `${this.labelName(label)} ${alt}` : this.labelName(label);
  }
}

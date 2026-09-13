export type Theme = "system" | "light" | "dark";
export const THEMES: readonly Theme[] = ["system", "light", "dark"];
export const THEME_STORAGE_KEY = "gpu-doodle:theme";

export function isTheme(value: unknown): value is Theme {
  return value === "system" || value === "light" || value === "dark";
}

/** The persisted override, or `system` when there is none or storage is blocked. */
export function readTheme(): Theme {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    // Storage can be blocked; follow the system.
  }
  return "system";
}

/**
 * `data-theme` on `<html>` overrides the `prefers-color-scheme` rules in
 * `style.css`; removing it hands control back to the system. The inline
 * script in `index.html` applies the same attribute before first paint.
 */
export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  if (theme === "system") delete root.dataset.theme;
  else root.dataset.theme = theme;
  try {
    if (theme === "system") localStorage.removeItem(THEME_STORAGE_KEY);
    else localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Without storage the choice lasts for the page only.
  }
}

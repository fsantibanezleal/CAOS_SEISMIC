export type Theme = "light" | "dark";

const STORAGE_KEY = "caos.seismic.theme";

/** Read the persisted theme, falling back to the OS preference, then dark. */
export function readTheme(): Theme {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // storage disabled — fall through to media query
  }
  if (
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  ) {
    return "dark";
  }
  // The product's canonical look is the dark-technical palette; default to it
  // when nothing else is known.
  return "dark";
}

/** Apply the theme to <html data-theme> and persist it. */
export function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // ignore — storage disabled
  }
}

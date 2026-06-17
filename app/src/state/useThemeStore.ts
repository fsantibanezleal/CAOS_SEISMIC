import { create } from "zustand";

import { applyTheme, readTheme, type Theme } from "@/lib/theme";

type ThemeState = {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
};

/** Single source of truth for the active theme; mirrors the inline pre-paint
 *  script in index.html so the store and the DOM never drift. */
export const useThemeStore = create<ThemeState>((set, get) => ({
  theme: readTheme(),
  setTheme: (theme) => {
    applyTheme(theme);
    set({ theme });
  },
  toggleTheme: () => {
    const next: Theme = get().theme === "dark" ? "light" : "dark";
    applyTheme(next);
    set({ theme: next });
  },
}));

import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Moon, Sun } from "lucide-react";

import { useThemeStore } from "@/state/useThemeStore";

/**
 * Dedicated light/dark toggle, mirroring the sister LDA-HSI app but wired to this
 * product's zustand theme store and plain-CSS `.icon-btn` treatment (no Tailwind).
 *
 * Shows the icon of the theme you would switch TO (Sun while dark, Moon while light)
 * so the affordance reads clearly. The store already applies `data-theme` to <html>
 * and persists the choice; the effect here is a belt-and-braces re-sync in case the
 * inline pre-paint script and the React store ever drift.
 */
export function ThemeToggle() {
  const { t } = useTranslation();
  const theme = useThemeStore((s) => s.theme);
  const toggleTheme = useThemeStore((s) => s.toggleTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  return (
    <button
      type="button"
      className="icon-btn"
      onClick={toggleTheme}
      aria-label={t("header.toggleTheme")}
      title={t("header.toggleTheme")}
    >
      {theme === "dark" ? <Sun size={18} aria-hidden="true" /> : <Moon size={18} aria-hidden="true" />}
      <span className="sr-only">
        {theme === "dark" ? t("header.lightThemeShort") : t("header.darkThemeShort")}
      </span>
    </button>
  );
}

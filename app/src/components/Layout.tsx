import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { NavLink } from "react-router-dom";

import { EXTERNAL_LINKS } from "@/lib/links";
import { ROUTES } from "@/lib/routes";
import { APP_BRANCH, APP_BUILD_TIME, APP_COMMIT_SHA, APP_VERSION } from "@/lib/version";
import { ensureLanguageLoaded, persistLanguage, type Language } from "@/i18n/config";
import { useThemeStore } from "@/state/useThemeStore";

/**
 * The application shell: a sticky header (brand + the six-route nav + theme toggle +
 * language toggle + external links) and a footer (attribution / credits + the always-on
 * honest disclaimer + build provenance).
 *
 * Styling uses the dark/light CSS variables in src/styles/globals.css (the dark-technical
 * palette) via the `.site-header`, `.main-nav`, `.site-footer` etc. classes already defined
 * there. No CSS framework — plain classes only.
 *
 * The honest framing is structural, not optional: the footer disclaimer renders on every
 * page and states plainly that this is a forecaster, never a predictor, and never an alarm.
 */
export interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const { t, i18n } = useTranslation();
  const theme = useThemeStore((s) => s.theme);
  const toggleTheme = useThemeStore((s) => s.toggleTheme);

  const currentLang = (i18n.resolvedLanguage ?? i18n.language ?? "en").slice(0, 2) as Language;
  const nextLang: Language = currentLang === "en" ? "es" : "en";

  async function switchLanguage(): Promise<void> {
    await ensureLanguageLoaded(nextLang);
    await i18n.changeLanguage(nextLang);
    persistLanguage(nextLang);
  }

  // The label shows the language you would switch TO (so the affordance reads clearly).
  const otherLangLabel = nextLang.toUpperCase();

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="header-inner">
          <NavLink to="/" className="brand" aria-label={t("product.name")}>
            <span className="brand-mark">◆</span>
            <span>{t("product.name")}</span>
          </NavLink>

          <nav className="main-nav" aria-label={t("product.name")}>
            {ROUTES.map((r) => (
              <NavLink
                key={r.id}
                to={r.path}
                end={r.path === "/"}
                className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
              >
                {t(r.labelKey)}
              </NavLink>
            ))}
          </nav>

          <div className="header-actions">
            <button
              type="button"
              className="icon-btn"
              onClick={toggleTheme}
              aria-label={t("header.toggleTheme")}
              title={t("header.toggleTheme")}
            >
              <span aria-hidden="true">{theme === "dark" ? "☾" : "☀"}</span>
              <span className="sr-only">
                {theme === "dark" ? t("header.darkThemeShort") : t("header.lightThemeShort")}
              </span>
            </button>

            <button
              type="button"
              className="icon-btn"
              onClick={() => void switchLanguage()}
              aria-label={t("header.toggleLanguage")}
              title={t("header.toggleLanguage")}
            >
              <span aria-hidden="true">{otherLangLabel}</span>
            </button>

            <span className="header-sep" aria-hidden="true" />

            <a
              className="icon-btn"
              href={EXTERNAL_LINKS.github}
              target="_blank"
              rel="noreferrer noopener"
              title={t("header.github")}
            >
              <span aria-hidden="true">{"</>"}</span>
              <span className="sr-only">{t("header.github")}</span>
            </a>
            <a
              className="icon-btn"
              href={EXTERNAL_LINKS.personal}
              target="_blank"
              rel="noreferrer noopener"
              title={t("header.personal")}
            >
              <span aria-hidden="true">⌂</span>
              <span className="sr-only">{t("header.personal")}</span>
            </a>
            <a
              className="icon-btn"
              href={EXTERNAL_LINKS.portfolio}
              target="_blank"
              rel="noreferrer noopener"
              title={t("header.portfolio")}
            >
              <span aria-hidden="true">▦</span>
              <span className="sr-only">{t("header.portfolio")}</span>
            </a>
          </div>
        </div>
      </header>

      <main className="page">{children}</main>

      <footer className="site-footer">
        <div className="footer-inner">
          <p className="disclaimer">{t("disclaimer.short")}</p>

          <div className="footer-meta">
            <span>{t("footer.attribution")}</span>
            <span aria-hidden="true">·</span>
            <span>{t("footer.complement")}</span>
          </div>

          <div className="footer-meta">
            <a href={EXTERNAL_LINKS.github} target="_blank" rel="noreferrer noopener">
              {t("header.github")}
            </a>
            <span aria-hidden="true">·</span>
            <a href={EXTERNAL_LINKS.personal} target="_blank" rel="noreferrer noopener">
              {t("header.personal")}
            </a>
            <span aria-hidden="true">·</span>
            <a href={EXTERNAL_LINKS.portfolio} target="_blank" rel="noreferrer noopener">
              {t("header.portfolio")}
            </a>
            <span aria-hidden="true">·</span>
            <span className="faint">{t("footer.license")}</span>

            <span className="footer-build">
              <span>
                {t("footer.version")} {APP_VERSION}
              </span>
              <span>
                {t("footer.commit")} {APP_COMMIT_SHA}
              </span>
              <span>
                {t("footer.build")} {APP_BUILD_TIME}
              </span>
              <span>{APP_BRANCH}</span>
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}

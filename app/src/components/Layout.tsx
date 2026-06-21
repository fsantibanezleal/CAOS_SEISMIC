import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { NavLink } from "react-router-dom";
import { Activity, Briefcase, Github, Globe, Info } from "lucide-react";

import { ThemeToggle } from "@/components/ThemeToggle";
import { LanguageToggle } from "@/components/LanguageToggle";
import { ArchitectureModal } from "@/components/ArchitectureModal";
import { EXTERNAL_LINKS } from "@/lib/links";
import { ROUTES } from "@/lib/routes";
import { APP_BRANCH, APP_BUILD_TIME, APP_COMMIT_SHA, APP_VERSION } from "@/lib/version";

/**
 * The application shell: a sticky, backdrop-blurred header (brand + the six-route nav +
 * external icon-links + dedicated theme/language toggles) and a footer (attribution /
 * credits + the always-on honest disclaimer + build provenance).
 *
 * Visual pattern mirrors the sister LDA-HSI app — lucide-react icons (never unicode
 * glyphs), an icon-button hover-opacity treatment, accent-soft active nav, and a vertical
 * separator before the toggles — but is implemented with this product's plain-CSS classes
 * in src/styles/globals.css (no Tailwind). The brand mark is a seismic `Activity` glyph in
 * the accent colour.
 *
 * The honest framing is structural, not optional: the footer disclaimer renders on every
 * page and states plainly that this is a forecaster, never a predictor, and never an alarm.
 */
export interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const { t } = useTranslation();
  const [archOpen, setArchOpen] = useState(false);

  return (
    <div className="app-shell">
      {archOpen ? <ArchitectureModal onClose={() => setArchOpen(false)} /> : null}
      <header className="site-header">
        <div className="header-inner">
          <NavLink to="/" className="brand" aria-label={t("product.name")}>
            <Activity size={18} aria-hidden="true" className="brand-mark" />
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
              className="icon-btn arch-open"
              onClick={() => setArchOpen(true)}
              aria-label={t("arch.open")}
              title={t("arch.open")}
            >
              <Info size={18} aria-hidden="true" />
              <span className="sr-only">{t("arch.open")}</span>
            </button>
            <a
              className="icon-btn"
              href={EXTERNAL_LINKS.github}
              target="_blank"
              rel="noreferrer noopener"
              aria-label={t("header.github")}
              title={t("header.github")}
            >
              <Github size={18} aria-hidden="true" />
              <span className="sr-only">{t("header.github")}</span>
            </a>
            <a
              className="icon-btn"
              href={EXTERNAL_LINKS.personal}
              target="_blank"
              rel="noreferrer noopener"
              aria-label={t("header.personal")}
              title={t("header.personal")}
            >
              <Globe size={18} aria-hidden="true" />
              <span className="sr-only">{t("header.personal")}</span>
            </a>
            <a
              className="icon-btn"
              href={EXTERNAL_LINKS.portfolio}
              target="_blank"
              rel="noreferrer noopener"
              aria-label={t("header.portfolio")}
              title={t("header.portfolio")}
            >
              <Briefcase size={18} aria-hidden="true" />
              <span className="sr-only">{t("header.portfolio")}</span>
            </a>

            <span className="header-sep" aria-hidden="true" />

            <LanguageToggle />
            <ThemeToggle />
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

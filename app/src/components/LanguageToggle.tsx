import { useTranslation } from "react-i18next";
import { Languages } from "lucide-react";

import {
  SUPPORTED_LANGUAGES,
  ensureLanguageLoaded,
  persistLanguage,
  type Language,
} from "@/i18n/config";

/**
 * Dedicated language toggle, mirroring the sister LDA-HSI app but wired to this
 * product's i18n config (EN bundled, ES lazy-loaded) and plain-CSS `.icon-btn`.
 *
 * The visible chip shows the CURRENT language code; pressing it lazy-loads the other
 * bundle (so English visitors never pay for the Spanish payload), switches, and
 * persists the choice.
 */
export function LanguageToggle() {
  const { i18n, t } = useTranslation();
  const current = ((i18n.resolvedLanguage ?? i18n.language ?? "en").slice(0, 2)) as Language;
  const next: Language = current === "en" ? "es" : "en";

  const swap = async (): Promise<void> => {
    if (!SUPPORTED_LANGUAGES.includes(next)) return;
    await ensureLanguageLoaded(next);
    await i18n.changeLanguage(next);
    persistLanguage(next);
  };

  return (
    <button
      type="button"
      className="icon-btn"
      onClick={() => {
        void swap();
      }}
      aria-label={t("header.toggleLanguage")}
      title={t("header.toggleLanguage")}
    >
      <Languages size={18} aria-hidden="true" />
      <span className="lang-code">{current}</span>
    </button>
  );
}

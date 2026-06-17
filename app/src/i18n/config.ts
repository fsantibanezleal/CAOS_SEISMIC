import i18n from "i18next";
import { initReactI18next } from "react-i18next";

// EN is the source of truth and is bundled statically so first paint never
// blocks on a fetch. ES is loaded lazily, only when the user opts in via the
// LanguageToggle, so English visitors never pay for the Spanish bundle.
import en from "@/i18n/en.json";

export const SUPPORTED_LANGUAGES = ["en", "es"] as const;
export type Language = (typeof SUPPORTED_LANGUAGES)[number];

const STORAGE_KEY = "caos.seismic.lang";

let esLoaded = false;
async function loadEs(): Promise<void> {
  if (esLoaded) return;
  const es = await import("@/i18n/es.json");
  i18n.addResourceBundle("es", "translation", es.default, true, true);
  esLoaded = true;
}

/** Ensure the bundle for `lng` is present before switching to it. */
export async function ensureLanguageLoaded(lng: Language): Promise<void> {
  if (lng === "es") await loadEs();
}

function readPersistedLanguage(): Language {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "en" || saved === "es") return saved;
  } catch {
    // ignore — storage disabled
  }
  // Hard default English: the site is research/methodology in English by intent;
  // users opt into Spanish and the choice persists. We do NOT auto-detect
  // navigator.language so the first-visit experience is deterministic.
  return "en";
}

const initialLang = readPersistedLanguage();

void i18n
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      // es: added lazily via ensureLanguageLoaded("es")
    },
    lng: "en",
    fallbackLng: "en",
    supportedLngs: [...SUPPORTED_LANGUAGES],
    defaultNS: "translation",
    interpolation: { escapeValue: false },
  })
  .then(async () => {
    if (initialLang === "es") {
      await loadEs();
      await i18n.changeLanguage("es");
    }
  });

/** Persist the chosen language. Call after a successful changeLanguage. */
export function persistLanguage(lng: Language): void {
  try {
    localStorage.setItem(STORAGE_KEY, lng);
  } catch {
    // ignore — storage disabled
  }
}

export default i18n;

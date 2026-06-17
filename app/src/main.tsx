import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// Side-effecting i18n init (creates and configures the i18next singleton, registers the EN
// bundle, and lazy-loads ES if persisted). Must run before any component calls useTranslation.
import "@/i18n/config";

import { applyTheme, readTheme } from "@/lib/theme";
import "@/styles/globals.css";
import AppRouter from "@/router";

// Reconcile the theme with the pre-paint inline script in index.html: the inline script set
// <html data-theme> before first paint to avoid a flash; we re-apply the resolved theme here
// so the persisted value and the DOM attribute never drift (the zustand store reads the same
// source). Idempotent — applyTheme just sets the attribute + persists.
applyTheme(readTheme());

const rootEl = document.getElementById("root");
if (!rootEl) {
  throw new Error('Root element "#root" not found in index.html');
}

createRoot(rootEl).render(
  <StrictMode>
    <AppRouter />
  </StrictMode>,
);

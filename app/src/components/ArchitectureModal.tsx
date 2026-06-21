import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { X } from "lucide-react";

import { ARCH_TABS } from "@/lib/architecture-tabs";
import type { Language } from "@/i18n/config";

/**
 * Architecture / "How it works" modal (ADR-0058). A centered dialog (Esc-to-close, focus-trapped,
 * role="dialog") with a tab strip; each tab pairs ONE hand-authored theme-aware SVG with a bilingual
 * explanation at complete depth. CAOS_SEISMIC is a non-shell app, so this implements the pattern directly.
 *
 * The SVG is FETCHED + INLINED (dangerouslySetInnerHTML) — an <img> would not inherit the app's CSS
 * variables, so the diagram would not follow the light/dark theme. Fetched via import.meta.env.BASE_URL so
 * it resolves under a Pages subpath.
 */

function svgUrl(file: string): string {
  let base = "/";
  try {
    const env = (import.meta as unknown as { env?: { BASE_URL?: string } }).env;
    if (env?.BASE_URL) base = env.BASE_URL;
  } catch {
    /* non-Vite */
  }
  return `${base.replace(/\/+$/, "")}/svg/tech/${file}`;
}

export interface ArchitectureModalProps {
  onClose: () => void;
}

export function ArchitectureModal({ onClose }: ArchitectureModalProps) {
  const { t, i18n } = useTranslation();
  const lang = ((i18n.resolvedLanguage ?? "en").slice(0, 2) as Language) ?? "en";
  const [active, setActive] = useState(0);
  const [svgMarkup, setSvgMarkup] = useState<string>("");
  const dialogRef = useRef<HTMLDivElement | null>(null);

  const tab = ARCH_TABS[active] ?? ARCH_TABS[0]!;

  // Fetch + inline the active tab's themed SVG.
  useEffect(() => {
    let cancelled = false;
    setSvgMarkup("");
    fetch(svgUrl(tab.svg), { cache: "no-cache" })
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`${r.status}`))))
      .then((txt) => !cancelled && setSvgMarkup(txt))
      .catch(() => !cancelled && setSvgMarkup(""));
    return () => {
      cancelled = true;
    };
  }, [tab.svg]);

  // Esc to close; focus the dialog on open.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    dialogRef.current?.focus();
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  return (
    <div className="arch-overlay" onClick={onClose} role="presentation">
      <div
        className="arch-modal"
        role="dialog"
        aria-modal="true"
        aria-label={t("arch.title")}
        tabIndex={-1}
        ref={dialogRef}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="arch-head">
          <div>
            <h2>{t("arch.title")}</h2>
            <p className="arch-sub">{t("arch.subtitle")}</p>
          </div>
          <button type="button" className="arch-close icon-btn" onClick={onClose} aria-label={t("arch.close")}>
            <X size={18} aria-hidden="true" />
          </button>
        </header>

        <div className="arch-tabs" role="tablist" aria-label={t("arch.title")}>
          {ARCH_TABS.map((tb, i) => (
            <button
              key={tb.id}
              type="button"
              role="tab"
              aria-selected={i === active}
              className={i === active ? "arch-tab on" : "arch-tab"}
              onClick={() => setActive(i)}
            >
              <span className="arch-tab-n">{i + 1}</span>
              {tb.label[lang] ?? tb.label.en}
            </button>
          ))}
        </div>

        <div className="arch-body">
          <div className="arch-diagram">
            {svgMarkup ? (
              <div className="arch-svg" dangerouslySetInnerHTML={{ __html: svgMarkup }} />
            ) : (
              <p className="muted">{t("common.loading")}</p>
            )}
          </div>
          <div className="arch-text">
            {(tab.body[lang] ?? tab.body.en).map((p, i) => (
              <p key={i}>{p}</p>
            ))}
          </div>
        </div>

        <footer className="arch-foot muted small">{t("arch.footer")}</footer>
      </div>
    </div>
  );
}

export default ArchitectureModal;

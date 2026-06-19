import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import type { Language } from "@/i18n/config";

/**
 * Route — Experiments / the route we are taking. An honest, append-only journey of the experiments
 * (including the dead ends and the negative results), read from the committed `data/experiments.json`
 * (mirrors the repo experiment register). The point is that the *path* — not just the final numbers —
 * is visible: what we tried, what it showed, and why we changed direction. Bilingual via the `lang` prop.
 */

interface ExperimentText {
  title: string;
  summary: string;
  finding: string;
}
interface ExperimentEntry {
  id: string;
  date: string;
  status: "done" | "running" | "pending" | string;
  phase: string;
  en: ExperimentText;
  es: ExperimentText;
}
interface ExperimentJourney {
  note_en: string;
  note_es: string;
  experiments: ExperimentEntry[];
}

function resolveBase(): string {
  try {
    const env = (import.meta as unknown as { env?: { BASE_URL?: string } }).env;
    if (env?.BASE_URL) return env.BASE_URL;
  } catch {
    /* non-Vite env */
  }
  return "/";
}

export default function Experiments() {
  const { t, i18n } = useTranslation();
  const lang = ((i18n.resolvedLanguage ?? "en").slice(0, 2) as Language) ?? "en";
  const [data, setData] = useState<ExperimentJourney | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const base = resolveBase().replace(/\/+$/, "");
    fetch(`${base}/data/experiments.json`, { cache: "no-cache" })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then((d: ExperimentJourney) => !cancelled && setData(d))
      .catch((e: unknown) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, []);

  const note = useMemo(() => (data ? (lang === "es" ? data.note_es : data.note_en) : ""), [data, lang]);
  const statusLabel = (s: string) =>
    s === "running"
      ? lang === "es" ? "en curso" : "running"
      : s === "pending"
        ? lang === "es" ? "pendiente" : "pending"
        : lang === "es" ? "hecho" : "done";

  return (
    <article className="page-body prose">
      <header className="page-head">
        <h1>{t("exp.title")}</h1>
        <p className="lede">{t("exp.lede")}</p>
      </header>

      {error ? <p className="error-note">{t("exp.error", { message: error })}</p> : null}
      {!data && !error ? <p className="muted">{t("common.loading")}</p> : null}

      {data ? (
        <>
          <p className="callout honest">{note}</p>

          <ol className="exp-timeline">
            {data.experiments.map((e) => {
              const x = lang === "es" ? e.es : e.en;
              return (
                <li key={e.id} className={`exp-item status-${e.status}`}>
                  <div className="exp-meta">
                    <span className="exp-id mono">{e.id}</span>
                    <span className="exp-date mono">{e.date}</span>
                    <span className={`exp-status badge ${e.status}`}>{statusLabel(e.status)}</span>
                    <span className="exp-phase tag">{e.phase}</span>
                  </div>
                  <h3 className="exp-title">{x.title}</h3>
                  <p className="exp-summary">{x.summary}</p>
                  <p className="exp-finding">
                    <span className="exp-finding-key">{lang === "es" ? "Hallazgo" : "Finding"}:</span>{" "}
                    {x.finding}
                  </p>
                </li>
              );
            })}
          </ol>

          <p className="muted small">{t("exp.footer")}</p>
        </>
      ) : null}
    </article>
  );
}

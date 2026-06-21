import { Suspense, lazy, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { loadOutlook, type OutlookArtifact, type OutlookEvidence, type OutlookViewEvidence } from "@/data/outlook";

const OutlookFieldMap = lazy(() => import("@/components/outlook/OutlookFieldMap"));

/**
 * Route — 30-day OUTLOOK. The honest longer-horizon surface: the geodetic-context neural background
 * measurably beats ETAS at 30 days (E11/E14), the one horizon where a covariate helps. The 1-7 day
 * operational product stays ETAS (Monitoring) — the geodetic context does NOT help there, and we say so.
 */

const REGION_NAMES: Record<string, string> = {
  global: "Global", JP: "Japan", CL: "Chile", "US-CA": "California", NZ: "New Zealand",
  IT: "Italy", GR: "Greece", TR: "Türkiye", MX: "Mexico", PE: "Peru", ID: "Indonesia",
};

export default function Outlook() {
  const { t, i18n } = useTranslation();
  const lang = (i18n.resolvedLanguage ?? "en").slice(0, 2);
  const [data, setData] = useState<{ artifact: OutlookArtifact; evidence: OutlookEvidence | null } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadOutlook()
      .then((d) => !cancelled && setData(d))
      .catch((e: unknown) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, []);

  const a = data?.artifact;
  const ev = data?.evidence;
  const summary = ev?._summary;
  const topZones = a
    ? [...a.field].sort((x, y) => y.n30 - x.n30).slice(0, 8)
    : [];

  return (
    <article className="page-body prose">
      <header className="page-head">
        <h1>{t("outlook.title")}</h1>
        <p className="lede">{t("outlook.lede")}</p>
      </header>

      <p className="callout honest">{t("outlook.honest")}</p>

      {error ? <p className="error-note">{t("outlook.error", { message: error })}</p> : null}
      {!data && !error ? <p className="muted">{t("common.loading")}</p> : null}

      {a ? (
        <>
          <div className="outlook-stats">
            <div className="stat">
              <span className="stat-num">{a.issued_at.slice(0, 10)}</span>
              <span className="stat-lab">{t("outlook.issued")}</span>
            </div>
            <div className="stat">
              <span className="stat-num">{a.n_total_30d.toFixed(0)}</span>
              <span className="stat-lab">{t("outlook.expected30")}</span>
            </div>
            <div className="stat">
              <span className="stat-num">M≥{a.m_threshold}</span>
              <span className="stat-lab">{t("outlook.threshold")}</span>
            </div>
          </div>

          <Suspense fallback={<p className="muted">{t("common.loading")}</p>}>
            <OutlookFieldMap field={a.field} />
          </Suspense>

          {ev ? (
            <section>
              <h2>{t("outlook.evidenceTitle")}</h2>
              <p>{t("outlook.evidenceLede")}</p>
              <table className="bench-table">
                <thead>
                  <tr>
                    <th>{t("outlook.view")}</th>
                    <th>{t("outlook.meanIgpe")}</th>
                    <th>{t("outlook.windowsPos")}</th>
                    <th>{t("outlook.nEq")}</th>
                  </tr>
                </thead>
                <tbody>
                  {["global", "JP", "CL", "US-CA", "NZ"].map((vid) => {
                    const v = ev[vid] as OutlookViewEvidence | undefined;
                    if (!v || typeof v.mean_igpe_vs_etas === "undefined") return null;
                    const pos = (v.mean_igpe_vs_etas ?? 0) > 0;
                    return (
                      <tr key={vid} className={vid === "global" ? "row-global" : ""}>
                        <td>{REGION_NAMES[vid] ?? vid}</td>
                        <td style={{ color: pos ? "#2e9e5b" : "#c0564a" }}>
                          {v.mean_igpe_vs_etas == null ? "—" : (v.mean_igpe_vs_etas > 0 ? "+" : "") + v.mean_igpe_vs_etas.toFixed(4)}
                        </td>
                        <td>{v.windows_positive}</td>
                        <td>{v.n_eq}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {summary ? <p className="muted small">{t("outlook.verdict")}: {summary.verdict}</p> : null}
            </section>
          ) : null}

          <section>
            <h2>{t("outlook.topZonesTitle")}</h2>
            <ol className="outlook-zones">
              {topZones.map((z, i) => (
                <li key={i}>
                  <span className="mono">{z.lat.toFixed(1)}°, {z.lon.toFixed(1)}°</span>
                  {" — "}
                  {(z.p30 * 100).toFixed(1)}% {lang === "es" ? "prob. 30d" : "30d prob."}
                  <span className="muted small"> (N={z.n30.toFixed(3)})</span>
                </li>
              ))}
            </ol>
          </section>

          <p className="muted small">{t("outlook.footer")}</p>
        </>
      ) : null}
    </article>
  );
}

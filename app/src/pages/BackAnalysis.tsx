import { useEffect, useMemo, useState } from "react";
import { Trans, useTranslation } from "react-i18next";

import { Callout } from "@/components/content/Callout";
import { Cite, ReferenceList } from "@/components/content/Cite";
import { ExpectedVsObserved } from "@/components/charts/ExpectedVsObserved";
import { ReliabilityDiagram } from "@/components/charts/ReliabilityDiagram";
import {
  loadBackAnalysis,
  type BackAnalysisCell,
  type BackAnalysisRegion,
  type BackAnalysisReport,
  type ComparisonTest,
  type ConsistencyBlock,
} from "@/data/backanalysis";
import type { Language } from "@/i18n/config";
import type { CitationId } from "@/lib/citations";

/**
 * Route 5 — Back-analysis (web-app-spec.md §6, evaluation-plan.md §7/§9).
 *
 * Renders the retrospective, pseudo-prospective CSEP results per region × horizon from the
 * committed report JSON (via the static data loader — sample data until real pyCSEP output
 * lands). Per cell it shows:
 *   - the N/M/S/L/CL consistency tests, gridded AND catalog-based (Poisson grid over-rejection
 *     during sequences is annotated and paired with the catalog result);
 *   - the T/W comparison vs smoothed-seismicity AND ETAS baselines, with the IGPE CI in nats —
 *     INCLUDING the honest failures where the model does NOT beat ETAS;
 *   - a reliability diagram per horizon;
 *   - the expected-vs-observed time series.
 *
 * The Ridgecrest worked example teaches that a single outcome neither validates nor
 * invalidates a probabilistic forecast.
 */

const BA_REFS: CitationId[] = [
  "savran2022",
  "schorlemmer2007",
  "zechar2010",
  "rhoades2011",
  "kagan2017",
  "serafini2025",
  "savran2020",
  "mizrahi2024",
];

export default function BackAnalysis() {
  const { t, i18n } = useTranslation();
  const lang = ((i18n.resolvedLanguage ?? "en").slice(0, 2) as Language) ?? "en";

  const [report, setReport] = useState<BackAnalysisReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeRegion, setActiveRegion] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    loadBackAnalysis()
      .then((r) => {
        if (cancelled) return;
        setReport(r);
        setActiveRegion(r.regions[0]?.region_id ?? "");
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const region = useMemo(
    () => report?.regions.find((r) => r.region_id === activeRegion) ?? report?.regions[0],
    [report, activeRegion],
  );

  return (
    <article className="page-body prose">
      <header className="page-head">
        <h1>{t("ba.title")}</h1>
        <p className="lede">{t("ba.lede")}</p>
      </header>

      <Callout tone="honest">
        <Trans i18nKey="ba.honesty" components={{ b: <strong /> }} />
      </Callout>

      {/* Ridgecrest worked example */}
      <Callout tone="note" title={t("ba.ridgecrest.title")}>
        <Trans i18nKey="ba.ridgecrest.body" components={{ cite: <Cite id="savran2020" />, b: <strong /> }} />
      </Callout>

      {error ? <p className="error-note">{t("ba.error", { message: error })}</p> : null}
      {!report && !error ? <p className="muted">{t("common.loading")}</p> : null}

      {report ? (
        <>
          {report.sample ? <p className="sample-banner">{t("ba.sampleBanner")}</p> : null}

          <p className="muted small">
            {lang === "es" ? report.tooling.note_es : report.tooling.note_en}{" "}
            <Cite id="savran2022" />
          </p>

          {/* Region selector */}
          <div className="region-tabs" role="tablist" aria-label={t("ba.regionSelector")}>
            {report.regions.map((r) => (
              <button
                key={r.region_id}
                role="tab"
                type="button"
                aria-selected={r.region_id === (region?.region_id ?? "")}
                className={r.region_id === (region?.region_id ?? "") ? "region-tab active" : "region-tab"}
                onClick={() => setActiveRegion(r.region_id)}
              >
                {lang === "es" ? r.name_es : r.name_en}
              </button>
            ))}
          </div>

          {region ? <RegionReport region={region} lang={lang} /> : null}
        </>
      ) : null}

      <ReferenceList ids={BA_REFS} heading={t("common.references")} />
    </article>
  );
}

function RegionReport({ region, lang }: { region: BackAnalysisRegion; lang: Language }) {
  const { t } = useTranslation();
  return (
    <section className="region-report">
      <header className="region-meta">
        <h2>{lang === "es" ? region.name_es : region.name_en}</h2>
        <p className="muted">{lang === "es" ? region.rationale_es : region.rationale_en}</p>
        <ul className="meta-list">
          <li>
            <span className="meta-key">{t("ba.meta.catalog")}</span> {region.catalog}
          </li>
          <li>
            <span className="meta-key">{t("ba.meta.train")}</span> {region.train_period[0]} → {region.train_period[1]}
          </li>
          <li>
            <span className="meta-key">{t("ba.meta.test")}</span> {region.test_period[0]} → {region.test_period[1]}
          </li>
          <li>
            <span className="meta-key">{t("ba.meta.mmin")}</span> M ≥ {region.m_min}
          </li>
        </ul>
      </header>

      {region.cells.map((cell) => (
        <HorizonCell key={cell.horizon_days} cell={cell} />
      ))}
    </section>
  );
}

function HorizonCell({ cell }: { cell: BackAnalysisCell }) {
  const { t } = useTranslation();
  return (
    <section className="card horizon-cell">
      <header className="cell-head">
        <h3>{t("ba.cell.horizon", { days: cell.horizon_days })}</h3>
        {cell.poisson_over_rejected ? (
          <span className="badge over-reject" title={t("ba.cell.overRejectTip")}>
            {t("ba.cell.overReject")}
          </span>
        ) : null}
      </header>

      <div className="cell-grid">
        {/* Consistency: gridded + catalog */}
        <div className="cell-block">
          <h4>{t("ba.consistency.title")}</h4>
          <div className="consistency-pair">
            <ConsistencyTable
              block={cell.consistency_gridded}
              label={t("ba.consistency.gridded")}
            />
            <ConsistencyTable
              block={cell.consistency_catalog}
              label={t("ba.consistency.catalog")}
            />
          </div>
        </div>

        {/* Comparison vs baselines */}
        <div className="cell-block">
          <h4>{t("ba.comparison.title")}</h4>
          <table className="cmp-table">
            <thead>
              <tr>
                <th scope="col">{t("ba.comparison.baseline")}</th>
                <th scope="col">{t("ba.comparison.igpe")}</th>
                <th scope="col">{t("ba.comparison.tci")}</th>
                <th scope="col">{t("ba.comparison.w")}</th>
                <th scope="col">{t("ba.comparison.skill")}</th>
              </tr>
            </thead>
            <tbody>
              {cell.comparison.map((c) => (
                <ComparisonRow key={c.baseline} c={c} />
              ))}
            </tbody>
          </table>
          <p className="muted small">{t("ba.comparison.note")}</p>
        </div>

        {/* Reliability diagram */}
        <div className="cell-block">
          <h4>{t("ba.reliability.title")}</h4>
          <ReliabilityDiagram points={cell.reliability} title={t("ba.reliability.title")} />
        </div>

        {/* Expected vs observed */}
        <div className="cell-block wide">
          <h4>{t("ba.evo.title")}</h4>
          <ExpectedVsObserved
            data={cell.expected_vs_observed}
            title={t("ba.evo.title")}
            expectedLabel={t("ba.evo.expected")}
            observedLabel={t("ba.evo.observed")}
          />
        </div>

        {/* Secondary scoring metrics */}
        <div className="cell-block wide">
          <h4>{t("ba.scoring.title")}</h4>
          <dl className="scoring-grid">
            <div>
              <dt>{t("ba.scoring.brier")}</dt>
              <dd className="mono">{cell.scoring.brier.toFixed(4)}</dd>
            </div>
            <div>
              <dt>{t("ba.scoring.log")}</dt>
              <dd className="mono">{cell.scoring.log_score.toFixed(4)}</dd>
            </div>
            <div>
              <dt>{t("ba.scoring.crps")}</dt>
              <dd className="mono">{cell.scoring.crps.toFixed(4)}</dd>
            </div>
            <div>
              <dt>{t("ba.scoring.ass")}</dt>
              <dd className="mono">{cell.scoring.area_skill_score.toFixed(3)}</dd>
            </div>
          </dl>
        </div>
      </div>
    </section>
  );
}

function ConsistencyTable({ block, label }: { block: ConsistencyBlock; label: string }) {
  const { t } = useTranslation();
  const rows: { key: string; test: { quantile: number; pass: boolean; quantile2?: number } }[] = [
    { key: "N", test: block.N },
    { key: "M", test: block.M },
    { key: "S", test: block.S },
    { key: "L", test: block.L },
    { key: "CL", test: block.CL },
  ];
  return (
    <div className="consistency-table-wrap">
      <p className="consistency-label">{label}</p>
      <table className="consistency-table">
        <tbody>
          {rows.map(({ key, test }) => (
            <tr key={key} className={test.pass ? "pass" : "fail"}>
              <th scope="row" title={t(`csep.name.${key}`)}>
                {t(`csep.short.${key}`)}
              </th>
              <td className="mono">
                {test.quantile.toFixed(2)}
                {test.quantile2 !== undefined ? ` / ${test.quantile2.toFixed(2)}` : ""}
              </td>
              <td className={`status-cell ${test.pass ? "ok" : "no"}`}>
                {test.pass ? t("csep.statusLabel.pass") : t("csep.statusLabel.fail")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ComparisonRow({ c }: { c: ComparisonTest }) {
  const { t } = useTranslation();
  const sign = c.igpe_nats > 0 ? "+" : "";
  return (
    <tr className={c.skill ? "skill-yes" : "skill-no"}>
      <th scope="row">{t(`ba.comparison.baselineName.${c.baseline}`, c.baseline)}</th>
      <td className="mono">
        {sign}
        {c.igpe_nats.toFixed(3)}
      </td>
      <td className="mono">
        [{c.t_ci[0].toFixed(3)}, {c.t_ci[1].toFixed(3)}]
      </td>
      <td className="mono">{c.w_pvalue.toFixed(3)}</td>
      <td>
        <span className={`skill-flag ${c.skill ? "yes" : "no"}`}>
          {c.skill ? t("ba.comparison.skillYes") : t("ba.comparison.skillNo")}
        </span>
      </td>
    </tr>
  );
}

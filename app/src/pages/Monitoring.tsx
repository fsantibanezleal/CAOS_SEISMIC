import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { getHexagonAreaAvg, UNITS } from "h3-js";

import { Callout } from "@/components/content/Callout";
import { CalibrationBadge } from "@/components/charts/CalibrationBadge";
import { CsepPanel } from "@/components/charts/CsepPanel";
import { ReliabilityDiagram } from "@/components/charts/ReliabilityDiagram";
import { CellDetail } from "@/components/monitoring/CellDetail";
import { FieldLegend } from "@/components/monitoring/FieldLegend";
import { MonitoringControls } from "@/components/monitoring/MonitoringControls";
import { MonitoringField } from "@/components/monitoring/MonitoringField";
import { NoMapSummary } from "@/components/monitoring/NoMapSummary";
import { StalenessBanner } from "@/components/monitoring/StalenessBanner";
import {
  availableHorizons,
  availableThresholds,
  forecastClient,
  selectField,
} from "@/data/client";
import type { Bound, CellSelection, ForecastArtifact } from "@/data/types";

/**
 * Route 6 — Monitoring (the adversarial focus; web-app-spec.md §7–§8 implemented exactly).
 *
 * Default = a WORLD probability FIELD (NOT alarm dots) rendered via MapLibre GL JS + deck.gl
 * (MapboxOverlay interleaved, H3HexagonLayer) on a perceptually-uniform SEQUENTIAL colormap —
 * never a red traffic-light ramp. The map + deck.gl bundle is lazy-loaded / code-split behind
 * this route (MonitoringField).
 *
 * Honest controls, all always visible:
 *   - horizon selector (1d/2d/7d) — a probability with no horizon is meaningless;
 *   - magnitude-threshold selector (M*);
 *   - bounds triad (Optimistic P10 · Expected median · Pessimistic P90), recolouring the SAME
 *     field, default Expected, with the persistent "plausible bad case, not a prediction" caption;
 *   - a no-map SVG summary, offered as a first-class alternative AND the no-WebGL fallback;
 *   - baseline comparison shown as ratio AND absolute (in the summary + the cell drill-down);
 *   - a numeric legend with cell area + horizon + threshold (never ordinal low/medium/high);
 *   - a calibration badge (green/amber/red = MODEL QUALITY only — the ONLY place red appears)
 *     + an expandable CSEP panel + the reliability diagram;
 *   - a staleness / last-run banner + a coverage mask (blank ≠ safe).
 *
 * The artifact is read via the static data client (no per-request inference). Until real daily
 * artifacts exist the bundled SAMPLE is served by the same code path.
 */
export default function Monitoring() {
  const { t } = useTranslation();

  const [artifact, setArtifact] = useState<ForecastArtifact | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Controls state
  const [horizon, setHorizon] = useState<number>(1);
  const [threshold, setThreshold] = useState<number>(5.0);
  const [bound, setBound] = useState<Bound>("expected"); // default lands on Expected
  const [noMap, setNoMap] = useState<boolean>(false);
  const [mapMode, setMapMode] = useState<"hexbins" | "heatmap">("hexbins");
  const [picked, setPicked] = useState<CellSelection | null>(null);
  const [showCsep, setShowCsep] = useState<boolean>(false);

  // Load the latest artifact once.
  useEffect(() => {
    let cancelled = false;
    forecastClient
      .loadLatest()
      .then((a) => {
        if (cancelled) return;
        setArtifact(a);
        const hs = availableHorizons(a);
        const ts = availableThresholds(a);
        if (hs.length && !hs.includes(horizon)) setHorizon(hs[0]!);
        if (ts.length && !ts.includes(threshold)) setThreshold(ts[0]!);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const horizons = useMemo(() => (artifact ? availableHorizons(artifact) : [1, 2, 7]), [artifact]);
  const thresholds = useMemo(
    () => (artifact ? availableThresholds(artifact) : [5.0, 6.0, 7.0]),
    [artifact],
  );

  // The flat per-cell selection for the active (horizon, threshold, bound) slice.
  const cells = useMemo<CellSelection[]>(
    () => (artifact ? selectField(artifact, horizon, threshold, bound) : []),
    [artifact, horizon, threshold, bound],
  );

  // Mean cell area (km²) for the artifact's display H3 resolution — drives the legend.
  const cellAreaKm2 = useMemo(() => {
    const res = artifact?.grid?.resolution;
    if (typeof res !== "number") return 0;
    try {
      // Mean hexagon area (km²) for the display H3 resolution — drives the legend caption.
      return getHexagonAreaAvg(res, UNITS.km2);
    } catch {
      return 0;
    }
  }, [artifact]);

  // Region-aware initial centre for the map.
  const center = useMemo<[number, number]>(() => {
    const b = artifact?.region?.bbox;
    if (!b) return [-71, -31];
    return [(b.lon_min + b.lon_max) / 2, (b.lat_min + b.lat_max) / 2];
  }, [artifact]);

  const degraded = artifact?.staleness?.ok === false;
  const isSample = Boolean(artifact?.provenance?.sample);
  const boundLabel = t(`monitoring.bound.${bound}.label`);

  return (
    <article className="page-body monitoring">
      <header className="page-head">
        <h1>{t("monitoring.title")}</h1>
        <p className="lede">{t("monitoring.lede")}</p>
      </header>

      {/* Honest framing — the field is "elevated relative to its own baseline", not an alarm. */}
      <Callout tone="honest">{t("monitoring.framing")}</Callout>

      {error ? <p className="error-note">{t("monitoring.error", { message: error })}</p> : null}
      {!artifact && !error ? <p className="muted">{t("common.loading")}</p> : null}

      {artifact ? (
        <>
          <StalenessBanner staleness={artifact.staleness} sample={isSample} />

          <div className="monitoring-topbar">
            <div className="region-name">
              <span className="muted">{t("monitoring.region")}:</span>{" "}
              <strong>{t("monitoring.regionName", { name: artifact.region.name_en })}</strong>
            </div>
            <button
              type="button"
              className="calibration-trigger"
              onClick={() => setShowCsep((v) => !v)}
              aria-expanded={showCsep}
            >
              <CalibrationBadge csep={artifact.calibration.csep} />
              <span className="faint">{showCsep ? t("monitoring.hideCsep") : t("monitoring.showCsep")}</span>
            </button>
          </div>

          {showCsep ? (
            <section className="csep-section card">
              <h2>{t("monitoring.csepTitle")}</h2>
              <div className="csep-layout">
                <CsepPanel calibration={artifact.calibration} />
                {artifact.calibration.reliability.length > 0 ? (
                  <div className="csep-reliability">
                    <h3>{t("monitoring.reliabilityTitle")}</h3>
                    <ReliabilityDiagram
                      points={artifact.calibration.reliability}
                      title={t("monitoring.reliabilityTitle")}
                    />
                  </div>
                ) : null}
              </div>
            </section>
          ) : null}

          <MonitoringControls
            horizons={horizons}
            horizon={horizon}
            onHorizon={setHorizon}
            thresholds={thresholds}
            threshold={threshold}
            onThreshold={setThreshold}
            bound={bound}
            onBound={setBound}
            noMap={noMap}
            onToggleNoMap={setNoMap}
            mapMode={mapMode}
            onMapMode={setMapMode}
          />

          {/* Persistent pessimistic-bound caption (only relevant on the P90 view, but always
              honest to surface near the controls). */}
          {bound === "hi" ? (
            <p className="bound-caption">{t("monitoring.bound.hi.caption")}</p>
          ) : null}

          <div className="monitoring-stage">
            <div className="stage-main">
              {noMap ? (
                <NoMapSummary
                  cells={cells}
                  horizonDays={horizon}
                  mThreshold={threshold}
                  boundLabel={boundLabel}
                />
              ) : (
                <MonitoringField
                  cells={cells}
                  coverageMask={artifact.coverage_mask}
                  center={center}
                  zoom={4}
                  degraded={degraded}
                  mode={mapMode}
                  onPick={setPicked}
                />
              )}
            </div>

            <aside className="stage-side">
              <FieldLegend
                horizonDays={horizon}
                mThreshold={threshold}
                cellAreaKm2={cellAreaKm2}
                boundLabel={boundLabel}
              />
              {!noMap && picked ? (
                <CellDetail
                  cell={picked}
                  horizonDays={horizon}
                  mThreshold={threshold}
                  activeBound={bound}
                  onClose={() => setPicked(null)}
                />
              ) : null}
              {artifact.coverage_mask.length > 0 ? (
                <p className="coverage-note muted small">
                  {t("monitoring.coverageNote", { count: artifact.coverage_mask.length })}
                </p>
              ) : null}
            </aside>
          </div>

          <p className="muted small provenance-note">
            {t("monitoring.provenanceNote", {
              schema: artifact.schema_version,
              issued: artifact.issued_at,
            })}
          </p>
        </>
      ) : null}
    </article>
  );
}

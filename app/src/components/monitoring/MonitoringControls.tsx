import { useTranslation } from "react-i18next";

import type { Bound } from "@/data/types";

/**
 * The honest, always-visible Monitoring controls (web-app-spec.md §7.3):
 *
 *  - HORIZON selector (1d / 2d / 7d) — always visible; a probability with no horizon is
 *    meaningless. Switching horizon recolours the field.
 *  - MAGNITUDE-THRESHOLD selector (M*) — the field encodes P(≥1 event ≥ M*).
 *  - BOUNDS triad — Optimistic (P10) · Expected (median) · Pessimistic (P90), a segmented
 *    control recolouring the SAME field. Default lands on Expected, with the persistent
 *    caption "the pessimistic view is a plausible bad case, not a prediction".
 *  - BASELINE-comparison mode — show the ratio-to-baseline AND the absolute expected count.
 *
 * None of these is a traffic-light; all are neutral segmented controls. The component is
 * purely presentational — state lives in the parent (Monitoring page).
 */

export interface MonitoringControlsProps {
  horizons: number[];
  horizon: number;
  onHorizon: (h: number) => void;

  thresholds: number[];
  threshold: number;
  onThreshold: (m: number) => void;

  bound: Bound;
  onBound: (b: Bound) => void;

  /** Whether the no-map summary (SVG) is shown instead of the WebGL field. */
  noMap: boolean;
  onToggleNoMap: (v: boolean) => void;
}

const BOUNDS: Bound[] = ["lo", "expected", "hi"];

export function MonitoringControls(props: MonitoringControlsProps) {
  const { t } = useTranslation();

  return (
    <div className="monitoring-controls">
      {/* Horizon */}
      <fieldset className="ctl-group">
        <legend>{t("monitoring.ctl.horizon")}</legend>
        <div className="segmented" role="group" aria-label={t("monitoring.ctl.horizon")}>
          {props.horizons.map((h) => (
            <button
              key={h}
              type="button"
              className={h === props.horizon ? "seg active" : "seg"}
              aria-pressed={h === props.horizon}
              onClick={() => props.onHorizon(h)}
            >
              {t("monitoring.ctl.days", { count: h })}
            </button>
          ))}
        </div>
      </fieldset>

      {/* Magnitude threshold */}
      <fieldset className="ctl-group">
        <legend>{t("monitoring.ctl.threshold")}</legend>
        <div className="segmented" role="group" aria-label={t("monitoring.ctl.threshold")}>
          {props.thresholds.map((m) => (
            <button
              key={m}
              type="button"
              className={m === props.threshold ? "seg active" : "seg"}
              aria-pressed={m === props.threshold}
              onClick={() => props.onThreshold(m)}
            >
              {t("monitoring.ctl.mstar", { m: m.toFixed(1) })}
            </button>
          ))}
        </div>
      </fieldset>

      {/* Bounds triad */}
      <fieldset className="ctl-group">
        <legend>{t("monitoring.ctl.bounds")}</legend>
        <div className="segmented" role="group" aria-label={t("monitoring.ctl.bounds")}>
          {BOUNDS.map((b) => (
            <button
              key={b}
              type="button"
              className={b === props.bound ? "seg active" : "seg"}
              aria-pressed={b === props.bound}
              onClick={() => props.onBound(b)}
              title={t(`monitoring.bound.${b}.tip`)}
            >
              {t(`monitoring.bound.${b}.label`)}
            </button>
          ))}
        </div>
      </fieldset>

      {/* View: WebGL field vs no-map summary */}
      <fieldset className="ctl-group">
        <legend>{t("monitoring.ctl.view")}</legend>
        <div className="segmented" role="group" aria-label={t("monitoring.ctl.view")}>
          <button
            type="button"
            className={!props.noMap ? "seg active" : "seg"}
            aria-pressed={!props.noMap}
            onClick={() => props.onToggleNoMap(false)}
          >
            {t("monitoring.ctl.viewMap")}
          </button>
          <button
            type="button"
            className={props.noMap ? "seg active" : "seg"}
            aria-pressed={props.noMap}
            onClick={() => props.onToggleNoMap(true)}
          >
            {t("monitoring.ctl.viewSummary")}
          </button>
        </div>
      </fieldset>
    </div>
  );
}

import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import { legendStops, type ScaleOptions } from "@/components/monitoring/colormap";

/**
 * Numeric legend for the probability field (web-app-spec.md §7.3).
 *
 * MANDATORY content: the legend states the HORIZON and the MAGNITUDE THRESHOLD (a probability
 * with neither is meaningless), the CELL AREA in km² (so "0.02 expected events / cell / 24 h"
 * is interpretable), and a NUMERIC ramp — never ordinal low/medium/high, never a traffic-light.
 * The ramp swatches are sampled from the same perceptually-uniform sequential colormap the
 * field uses.
 *
 * The legend always shows the active BOUND (optimistic/expected/pessimistic) so the reader
 * knows which surface they are looking at.
 */
export interface FieldLegendProps {
  horizonDays: number;
  mThreshold: number;
  /** Mean cell area in km² for the current grid resolution. */
  cellAreaKm2: number;
  /** Active bound label (already translated). */
  boundLabel: string;
  scale?: ScaleOptions;
}

export function FieldLegend({ horizonDays, mThreshold, cellAreaKm2, boundLabel, scale }: FieldLegendProps) {
  const { t } = useTranslation();
  const stops = useMemo(() => legendStops(6, scale), [scale]);

  const fmtPct = (v: number): string => {
    if (v < 0.001) return `${(v * 100).toFixed(3)}%`;
    if (v < 0.01) return `${(v * 100).toFixed(2)}%`;
    return `${(v * 100).toFixed(1)}%`;
  };

  return (
    <div className="field-legend" aria-label={t("monitoring.legend.aria")}>
      <p className="legend-title">
        {t("monitoring.legend.title", {
          bound: boundLabel,
          days: horizonDays,
          m: mThreshold.toFixed(1),
        })}
      </p>

      <div
        className="legend-ramp"
        role="img"
        aria-label={t("monitoring.legend.rampAria")}
      >
        {stops.map((s, i) => (
          <span key={i} className="legend-swatch" style={{ backgroundColor: s.css }} title={fmtPct(s.value)} />
        ))}
      </div>
      <div className="legend-scale">
        {stops.map((s, i) => (
          <span key={i} className="legend-tick">
            {fmtPct(s.value)}
          </span>
        ))}
      </div>

      <p className="legend-caption muted">
        {t("monitoring.legend.caption", {
          area: Math.round(cellAreaKm2).toLocaleString(),
          days: horizonDays,
          m: mThreshold.toFixed(1),
        })}
      </p>
      <p className="legend-honesty">{t("monitoring.legend.notSafe")}</p>
    </div>
  );
}

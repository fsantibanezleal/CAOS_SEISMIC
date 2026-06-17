import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import { valueToCss, type ScaleOptions } from "@/components/monitoring/colormap";
import type { CellSelection } from "@/data/types";

/**
 * No-map summary view (web-app-spec.md §7.1, §7.2, §8.3). This is a FIRST-CLASS, honesty-first
 * alternative to the WebGL field — and the no-WebGL / accessibility fallback. It uses zero
 * map-library bytes: a ranked horizontal-bar table of the most-elevated cells, each coloured
 * on the SAME perceptually-uniform sequential ramp the map uses, with the MANDATORY baseline
 * companion shown BOTH ways:
 *   - ratio R = p / baseline (e.g. "≈ 4× the usual rate") — the "elevated" notion, never an alarm;
 *   - absolute expected probability p (so a 10× ratio on a near-zero baseline still reads
 *     "still very unlikely").
 *
 * Cells are sorted by the active bound's value. The header restates horizon + threshold +
 * bound. Renders from the same artifact slice the map consumes (`CellSelection[]`).
 */
export interface NoMapSummaryProps {
  cells: CellSelection[];
  horizonDays: number;
  mThreshold: number;
  boundLabel: string;
  scale?: ScaleOptions;
  /** Max rows to show (the long tail of near-baseline cells is summarized). */
  topN?: number;
}

function fmtPct(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (v < 0.001) return `${(v * 100).toFixed(3)}%`;
  if (v < 0.01) return `${(v * 100).toFixed(2)}%`;
  return `${(v * 100).toFixed(1)}%`;
}

function fmtRatio(r: number): string {
  if (!Number.isFinite(r)) return "∞×";
  if (r >= 100) return `${Math.round(r)}×`;
  if (r >= 10) return `${r.toFixed(0)}×`;
  return `${r.toFixed(1)}×`;
}

export function NoMapSummary({
  cells,
  horizonDays,
  mThreshold,
  boundLabel,
  scale,
  topN = 24,
}: NoMapSummaryProps) {
  const { t } = useTranslation();

  const { rows, max, restCount } = useMemo(() => {
    const sorted = [...cells].sort((a, b) => b.value - a.value);
    const top = sorted.slice(0, topN);
    const maxV = top[0]?.value ?? 1;
    return { rows: top, max: maxV || 1, restCount: Math.max(0, sorted.length - top.length) };
  }, [cells, topN]);

  if (cells.length === 0) {
    return <p className="muted">{t("monitoring.summary.empty")}</p>;
  }

  return (
    <div className="no-map-summary">
      <p className="summary-head">
        {t("monitoring.summary.head", {
          bound: boundLabel,
          days: horizonDays,
          m: mThreshold.toFixed(1),
        })}
      </p>

      <table className="summary-table">
        <thead>
          <tr>
            <th scope="col">{t("monitoring.summary.cell")}</th>
            <th scope="col">{t("monitoring.summary.probability")}</th>
            <th scope="col">{t("monitoring.summary.baseline")}</th>
            <th scope="col" title={t("monitoring.summary.ratioTip")}>
              {t("monitoring.summary.ratio")}
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => {
            const widthPct = Math.max(2, (c.value / max) * 100);
            return (
              <tr key={c.cell}>
                <th scope="row" className="mono cell-key" title={c.cell}>
                  {c.cell.slice(0, 9)}…
                </th>
                <td className="bar-cell">
                  <span className="cell-bar-track">
                    <span
                      className="cell-bar"
                      style={{ width: `${widthPct}%`, backgroundColor: valueToCss(c.value, scale) }}
                    />
                  </span>
                  <span className="mono cell-val">{fmtPct(c.value)}</span>
                </td>
                <td className="mono muted">{fmtPct(c.baseline)}</td>
                <td className="mono">{fmtRatio(c.ratioToBaseline)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {restCount > 0 ? (
        <p className="muted small">{t("monitoring.summary.rest", { count: restCount })}</p>
      ) : null}
      <p className="muted small">{t("monitoring.summary.baselineNote")}</p>
    </div>
  );
}

import { useTranslation } from "react-i18next";

import type { Bound, CellSelection } from "@/data/types";

/**
 * Per-cell drill-down panel shown when a hexagon is picked on the field
 * (web-app-spec.md §7.1 drill-down, §7.3 baseline companion).
 *
 * It surfaces the honesty triad for the picked cell — the optimistic (P10), expected (median),
 * and pessimistic (P90) bounds together — plus the MANDATORY baseline companion BOTH ways:
 * the ratio-to-baseline AND the absolute expected probability, so an "N× elevated" number on a
 * near-zero baseline still reads "still very unlikely". The expected event count (rate) is
 * shown alongside. No alarm language, no "danger" colour — the panel is neutral.
 */
export interface CellDetailProps {
  cell: CellSelection | null;
  horizonDays: number;
  mThreshold: number;
  /** The currently-active bound, highlighted in the triad. */
  activeBound: Bound;
  onClose: () => void;
}

function pct(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (v < 0.001) return `${(v * 100).toFixed(3)}%`;
  if (v < 0.01) return `${(v * 100).toFixed(2)}%`;
  return `${(v * 100).toFixed(1)}%`;
}

function ratio(r: number): string {
  if (!Number.isFinite(r)) return "∞×";
  if (r >= 10) return `${r.toFixed(0)}×`;
  return `${r.toFixed(1)}×`;
}

export function CellDetail({ cell, horizonDays, mThreshold, activeBound, onClose }: CellDetailProps) {
  const { t } = useTranslation();
  if (!cell) return null;

  return (
    <aside className="cell-detail card" aria-label={t("monitoring.detail.aria")}>
      <header className="detail-head">
        <h3 className="mono">{cell.cell}</h3>
        <button type="button" className="icon-btn" onClick={onClose} aria-label={t("monitoring.detail.close")}>
          ✕
        </button>
      </header>

      <p className="detail-context">
        {t("monitoring.detail.context", { days: horizonDays, m: mThreshold.toFixed(1) })}
      </p>

      <div className="triad">
        <div className={`triad-item ${activeBound === "lo" ? "active" : ""}`}>
          <span className="triad-label">{t("monitoring.bound.lo.label")}</span>
          <span className="triad-val mono">{pct(cell.lo)}</span>
        </div>
        <div className={`triad-item ${activeBound === "expected" ? "active" : ""}`}>
          <span className="triad-label">{t("monitoring.bound.expected.label")}</span>
          <span className="triad-val mono">{pct(cell.p)}</span>
        </div>
        <div className={`triad-item ${activeBound === "hi" ? "active" : ""}`}>
          <span className="triad-label">{t("monitoring.bound.hi.label")}</span>
          <span className="triad-val mono">{pct(cell.hi)}</span>
        </div>
      </div>

      <dl className="detail-list">
        <div>
          <dt>{t("monitoring.detail.absolute")}</dt>
          <dd className="mono">{pct(cell.p)}</dd>
        </div>
        <div>
          <dt>{t("monitoring.detail.baseline")}</dt>
          <dd className="mono">{pct(cell.baseline)}</dd>
        </div>
        <div>
          <dt>{t("monitoring.detail.ratio")}</dt>
          <dd className="mono">{ratio(cell.ratioToBaseline)}</dd>
        </div>
        <div>
          <dt>{t("monitoring.detail.rate")}</dt>
          <dd className="mono">{cell.rate.toFixed(4)}</dd>
        </div>
      </dl>

      <p className="detail-honesty muted small">{t("monitoring.detail.honesty")}</p>
    </aside>
  );
}

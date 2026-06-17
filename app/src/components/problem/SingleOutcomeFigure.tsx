import { useId } from "react";
import { useTranslation } from "react-i18next";

/**
 * Figure (c) — "a single outcome neither validates nor invalidates a probabilistic forecast",
 * built on the 2019 Ridgecrest worked example (Savran et al. 2020).
 *
 * After the 4 July 2019 M6.4, UCERF3-ETAS gave ~3% chance of a larger event in the first week;
 * the M7.1 struck ~34 h later. The figure draws a unit probability bar split into the forecast
 * 3% "larger-event" slice and the 97% "no-larger-event" slice, then marks that the realised
 * outcome fell in the small slice — which is exactly what a well-calibrated ~3% forecast permits
 * (3% != 0%). The honest reading is rendered as the takeaway: skill is judged over MANY forecasts
 * by calibration, never by one hit or miss.
 *
 * Theme-aware via the `.soc-*` classes in globals.css; copy is i18n (`problem.fig.soc.*`).
 */
export function SingleOutcomeFigure() {
  const { t } = useTranslation();
  const uid = useId();

  const W = 560;
  const H = 220;
  const barX = 40;
  const barY = 70;
  const barW = 480;
  const barH = 46;

  const pSmall = 0.03; // ~3% forecast of a larger event
  const wSmall = Math.max(barW * pSmall, 14); // keep the slice visible even though it is tiny
  const xSplit = barX + (barW - wSmall);

  return (
    <svg
      className="soc-svg"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-labelledby={`${uid}-t ${uid}-d`}
      preserveAspectRatio="xMidYMid meet"
    >
      <title id={`${uid}-t`}>{t("problem.fig.soc.svgTitle")}</title>
      <desc id={`${uid}-d`}>{t("problem.fig.soc.svgDesc")}</desc>

      <defs>
        <marker
          id={`${uid}-arrow`}
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="7"
          markerHeight="7"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" className="soc-arrowhead" />
        </marker>
      </defs>

      <text x={W / 2} y={36} textAnchor="middle" className="soc-head">
        {t("problem.fig.soc.head")}
      </text>

      {/* The unit probability bar: 97% no-larger-event | 3% larger-event. */}
      <rect x={barX} y={barY} width={barW - wSmall} height={barH} className="soc-bar-large" />
      <rect x={xSplit} y={barY} width={wSmall} height={barH} className="soc-bar-small" />
      <rect x={barX} y={barY} width={barW} height={barH} className="soc-bar-frame" />

      {/* 97% slice label (inside) */}
      <text x={barX + (barW - wSmall) / 2} y={barY + barH / 2 + 4} textAnchor="middle" className="soc-bar-label">
        {t("problem.fig.soc.noLarger")}
      </text>

      {/* 3% slice callout (above, with a leader) */}
      <line x1={xSplit + wSmall / 2} y1={barY - 6} x2={xSplit + wSmall / 2} y2={barY - 26} className="soc-leader" />
      <text x={xSplit + wSmall / 2} y={barY - 32} textAnchor="middle" className="soc-small-label">
        {t("problem.fig.soc.larger")}
      </text>

      {/* The realised outcome arrow → into the 3% slice. */}
      <line
        x1={xSplit + wSmall / 2}
        y1={barY + barH + 30}
        x2={xSplit + wSmall / 2}
        y2={barY + barH + 6}
        className="soc-outcome-line"
        markerEnd={`url(#${uid}-arrow)`}
      />
      <text x={xSplit + wSmall / 2} y={barY + barH + 46} textAnchor="middle" className="soc-outcome-label">
        {t("problem.fig.soc.outcome")}
      </text>

      {/* The honest takeaway line. */}
      <text x={W / 2} y={H - 10} textAnchor="middle" className="soc-takeaway">
        {t("problem.fig.soc.takeaway")}
      </text>
    </svg>
  );
}

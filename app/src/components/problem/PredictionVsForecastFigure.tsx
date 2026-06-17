import { useId } from "react";
import { useTranslation } from "react-i18next";

/**
 * Figure (a) — prediction (deterministic) vs forecast (probability in (0,1)).
 *
 * A side-by-side contrast that makes the ICEF distinction (Jordan et al. 2011) visual:
 *  - LEFT  ("prediction"): a binary alarm dial pinned to a single yes/no state — exactly the UI
 *    this product never renders. Drawn in the "bad"/faint palette and explicitly crossed out.
 *  - RIGHT ("forecast"):  a continuous probability axis in the open interval (0, 1) with an
 *    uncertainty band and a marker sitting LOW but elevated above its baseline tick — the only
 *    thing this product ever emits.
 *
 * The SVG is theme-aware (it reads the palette through the `.pvf-*` classes in globals.css), so
 * it follows light/dark automatically. All copy is i18n (`problem.fig.pvf.*`); the SVG carries an
 * accessible <title>/<desc>.
 */
export function PredictionVsForecastFigure() {
  const { t } = useTranslation();
  const uid = useId();

  // viewBox geometry: two stacked panels.
  const W = 640;
  const H = 250;
  const colW = 300;
  const gap = 40;
  const xL = 0;
  const xR = colW + gap;

  // Right panel probability axis (the (0,1) interval), in panel-local coords.
  const axisX0 = xR + 24;
  const axisX1 = xR + colW - 16;
  const axisY = 150;
  const axisLen = axisX1 - axisX0;
  const px = (p: number) => axisX0 + p * axisLen; // p in [0,1]

  const baseP = 0.05; // long-term baseline tick
  const lo = 0.12;
  const exp = 0.2; // expected ~ elevated but still low
  const hi = 0.31;

  return (
    <svg
      className="pvf-svg"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-labelledby={`${uid}-t ${uid}-d`}
      preserveAspectRatio="xMidYMid meet"
    >
      <title id={`${uid}-t`}>{t("problem.fig.pvf.svgTitle")}</title>
      <desc id={`${uid}-d`}>{t("problem.fig.pvf.svgDesc")}</desc>

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
          <path d="M 0 0 L 10 5 L 0 10 z" className="pvf-arrowhead" />
        </marker>
      </defs>

      {/* ── LEFT: prediction (deterministic, rejected) ───────────────────── */}
      <g className="pvf-panel pvf-rejected">
        <rect x={xL} y={20} width={colW} height={H - 40} rx={10} className="pvf-frame" />
        <text x={xL + colW / 2} y={46} textAnchor="middle" className="pvf-panel-title">
          {t("problem.fig.pvf.predTitle")}
        </text>
        <text x={xL + colW / 2} y={66} textAnchor="middle" className="pvf-panel-sub">
          {t("problem.fig.pvf.predSub")}
        </text>

        {/* Binary alarm dial: YES | NO, the needle slammed to one side. */}
        <g>
          <rect x={xL + 50} y={100} width={90} height={48} rx={6} className="pvf-binary pvf-binary-on" />
          <text x={xL + 95} y={129} textAnchor="middle" className="pvf-binary-label">
            {t("problem.fig.pvf.yes")}
          </text>
          <rect x={xL + 160} y={100} width={90} height={48} rx={6} className="pvf-binary pvf-binary-off" />
          <text x={xL + 205} y={129} textAnchor="middle" className="pvf-binary-label off">
            {t("problem.fig.pvf.no")}
          </text>
        </g>

        <text x={xL + colW / 2} y={188} textAnchor="middle" className="pvf-note">
          {t("problem.fig.pvf.predNote")}
        </text>

        {/* The big cross: this product never renders this. */}
        <line x1={xL + 16} y1={32} x2={xL + colW - 16} y2={H - 52} className="pvf-cross" />
        <line x1={xL + colW - 16} y1={32} x2={xL + 16} y2={H - 52} className="pvf-cross" />
      </g>

      {/* ── RIGHT: forecast (probability in (0,1)) ───────────────────────── */}
      <g className="pvf-panel pvf-accepted">
        <rect x={xR} y={20} width={colW} height={H - 40} rx={10} className="pvf-frame pvf-frame-on" />
        <text x={xR + colW / 2} y={46} textAnchor="middle" className="pvf-panel-title on">
          {t("problem.fig.pvf.foreTitle")}
        </text>
        <text x={xR + colW / 2} y={66} textAnchor="middle" className="pvf-panel-sub">
          {t("problem.fig.pvf.foreSub")}
        </text>

        {/* Continuous (0,1) axis with an open interval. */}
        <line
          x1={axisX0}
          y1={axisY}
          x2={axisX1}
          y2={axisY}
          className="pvf-axis"
          markerEnd={`url(#${uid}-arrow)`}
        />
        {/* Open endpoints 0 and 1 (hollow circles → open interval). */}
        <circle cx={axisX0} cy={axisY} r={4} className="pvf-open" />
        <circle cx={axisX1} cy={axisY} r={4} className="pvf-open" />
        <text x={axisX0} y={axisY + 22} textAnchor="middle" className="pvf-axis-tick">
          0
        </text>
        <text x={axisX1} y={axisY + 22} textAnchor="middle" className="pvf-axis-tick">
          1
        </text>

        {/* Uncertainty band lo–hi + expected marker. */}
        <rect
          x={px(lo)}
          y={axisY - 16}
          width={px(hi) - px(lo)}
          height={32}
          rx={4}
          className="pvf-band"
        />
        <line x1={px(exp)} y1={axisY - 22} x2={px(exp)} y2={axisY + 22} className="pvf-marker" />
        <text x={px(exp)} y={axisY - 30} textAnchor="middle" className="pvf-marker-label">
          {t("problem.fig.pvf.expected")}
        </text>

        {/* Baseline tick. */}
        <line x1={px(baseP)} y1={axisY - 10} x2={px(baseP)} y2={axisY + 10} className="pvf-baseline" />
        <text x={px(baseP)} y={axisY + 36} textAnchor="middle" className="pvf-baseline-label">
          {t("problem.fig.pvf.baseline")}
        </text>

        <text x={xR + colW / 2} y={210} textAnchor="middle" className="pvf-note on">
          {t("problem.fig.pvf.foreNote")}
        </text>
      </g>
    </svg>
  );
}

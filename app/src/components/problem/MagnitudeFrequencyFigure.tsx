import { useId } from "react";
import { useTranslation } from "react-i18next";

/**
 * Figure (b) — the Gutenberg–Richter magnitude–frequency law as a log-linear plot.
 *
 * Plots log10 N(>= M) against magnitude M. The defining feature is the straight descending line
 * of slope -b (the "b-value", here b ~= 1): a one-unit increase in magnitude divides the event
 * count by ~10. Scattered markers around the line evoke a real catalog's empirical points (denser
 * and noisier at small M, sparse at large M). This is the population statistic that lets the
 * forecast carry a magnitude term — and the same scale-invariance (no characteristic event size)
 * that is the empirical fingerprint of self-organized criticality and the reason the *timing* of
 * the next large event is not encoded here.
 *
 * Theme-aware via the `.mfd-*` classes in globals.css; copy is i18n (`problem.fig.mfd.*`).
 */
export function MagnitudeFrequencyFigure() {
  const { t } = useTranslation();
  const uid = useId();

  const W = 460;
  const H = 320;
  const padL = 56;
  const padR = 24;
  const padT = 28;
  const padB = 52;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // Data domain: magnitude 2..7 on x; log10 N from 4 (1e4) down to 0 (1e0) on y.
  const mMin = 2;
  const mMax = 7;
  const logMax = 4; // log10 N at M = mMin
  const logMin = 0; // log10 N at M = mMax

  const sx = (m: number) => padL + ((m - mMin) / (mMax - mMin)) * plotW;
  const sy = (logN: number) => padT + (1 - (logN - logMin) / (logMax - logMin)) * plotH;

  // G–R line: log10 N = a - b M, with b = 1 and chosen so it spans (mMin, logMax)..(mMax, logMin).
  const b = 1;
  const a = logMax + b * mMin; // => log10 N(mMin) = logMax
  const grLine = (m: number) => a - b * m;

  const xTicks = [2, 3, 4, 5, 6, 7];
  const yTicks = [0, 1, 2, 3, 4]; // powers of ten

  // Synthetic catalog scatter around the line (deterministic, no RNG → stable render).
  const scatter: Array<{ m: number; d: number }> = [
    { m: 2.1, d: 0.18 }, { m: 2.4, d: -0.22 }, { m: 2.7, d: 0.1 }, { m: 3.0, d: -0.12 },
    { m: 3.2, d: 0.2 }, { m: 3.5, d: -0.08 }, { m: 3.8, d: 0.14 }, { m: 4.1, d: -0.18 },
    { m: 4.4, d: 0.22 }, { m: 4.7, d: -0.1 }, { m: 5.0, d: 0.16 }, { m: 5.3, d: -0.24 },
    { m: 5.6, d: 0.2 }, { m: 5.9, d: 0.3 }, { m: 6.2, d: -0.28 }, { m: 6.6, d: 0.34 },
  ];

  return (
    <svg
      className="mfd-svg"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-labelledby={`${uid}-t ${uid}-d`}
      preserveAspectRatio="xMidYMid meet"
    >
      <title id={`${uid}-t`}>{t("problem.fig.mfd.svgTitle")}</title>
      <desc id={`${uid}-d`}>{t("problem.fig.mfd.svgDesc")}</desc>

      {/* Plot frame */}
      <rect x={padL} y={padT} width={plotW} height={plotH} className="mfd-frame" />

      {/* Gridlines + ticks */}
      {xTicks.map((m) => (
        <g key={`x${m}`} className="mfd-grid">
          <line x1={sx(m)} y1={padT} x2={sx(m)} y2={padT + plotH} />
          <text x={sx(m)} y={padT + plotH + 18} textAnchor="middle" className="mfd-tick">
            {m}
          </text>
        </g>
      ))}
      {yTicks.map((l) => (
        <g key={`y${l}`} className="mfd-grid">
          <line x1={padL} y1={sy(l)} x2={padL + plotW} y2={sy(l)} />
          <text x={padL - 8} y={sy(l) + 4} textAnchor="end" className="mfd-tick">
            10{supDigit(l)}
          </text>
        </g>
      ))}

      {/* Empirical catalog scatter */}
      {scatter.map((pt, i) => (
        <circle key={i} cx={sx(pt.m)} cy={sy(grLine(pt.m) + pt.d)} r={3.4} className="mfd-point" />
      ))}

      {/* The G–R log-linear fit: slope -b */}
      <line
        x1={sx(mMin)}
        y1={sy(grLine(mMin))}
        x2={sx(mMax)}
        y2={sy(grLine(mMax))}
        className="mfd-line"
      />

      {/* Slope annotation: a magnitude step of +1 → ÷10 in count. */}
      <g className="mfd-slope">
        <line x1={sx(4)} y1={sy(grLine(4))} x2={sx(5)} y2={sy(grLine(4))} strokeDasharray="3 3" />
        <line x1={sx(5)} y1={sy(grLine(4))} x2={sx(5)} y2={sy(grLine(5))} strokeDasharray="3 3" />
        <text x={sx(5) + 6} y={(sy(grLine(4)) + sy(grLine(5))) / 2 + 4} className="mfd-slope-label">
          {t("problem.fig.mfd.slope")}
        </text>
      </g>

      {/* Axis labels */}
      <text x={padL + plotW / 2} y={H - 12} textAnchor="middle" className="mfd-axis">
        {t("problem.fig.mfd.xAxis")}
      </text>
      <text
        x={16}
        y={padT + plotH / 2}
        textAnchor="middle"
        className="mfd-axis"
        transform={`rotate(-90 16 ${padT + plotH / 2})`}
      >
        {t("problem.fig.mfd.yAxis")}
      </text>
    </svg>
  );
}

/** Render a small integer as a unicode superscript for the 10^n y-axis labels. */
function supDigit(n: number): string {
  const map: Record<string, string> = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴" };
  return String(n)
    .split("")
    .map((c) => map[c] ?? c)
    .join("");
}

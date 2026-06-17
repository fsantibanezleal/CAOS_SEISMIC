import { useId } from "react";

import type { ReliabilityPoint } from "@/data/types";

/**
 * Reliability (calibration) diagram — the product's central credibility artifact
 * (evaluation-plan.md §6.4, web-app-spec.md §7.3). Plots observed frequency against forecast
 * probability; a well-calibrated forecast sits on the diagonal ("when we said X%, it happened
 * ~X%"). Point area scales with the per-bin sample count `n` so the eye weights the
 * well-supported bins (the quiet, cold-start bins dominate the diagram — their honesty is the
 * product's honesty).
 *
 * Hand-rolled SVG — zero charting-library cost, follows the theme via CSS variables, and works
 * with no WebGL (the accessibility / no-map path). Both axes are on a shared [0,1] linear
 * scale by default; a perfectly-calibrated model lies on y = x.
 */
export interface ReliabilityDiagramProps {
  points: ReliabilityPoint[];
  /** Square plot side in px (viewBox units). */
  size?: number;
  /** Accessible title. */
  title?: string;
}

export function ReliabilityDiagram({ points, size = 280, title }: ReliabilityDiagramProps) {
  const uid = useId();
  const pad = 38;
  const plot = size - pad * 2;
  const maxN = Math.max(1, ...points.map((p) => p[2]));

  // Linear [0,1] scales. Forecast prob on x, observed freq on y (y inverted in SVG).
  const sx = (v: number) => pad + v * plot;
  const sy = (v: number) => pad + (1 - v) * plot;
  const r = (n: number) => 3 + 6 * Math.sqrt(n / maxN);

  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <svg
      className="chart reliability"
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-labelledby={`${uid}-t`}
      preserveAspectRatio="xMidYMid meet"
    >
      <title id={`${uid}-t`}>{title ?? "Reliability diagram"}</title>

      {/* Plot frame */}
      <rect x={pad} y={pad} width={plot} height={plot} className="chart-frame" />

      {/* Gridlines + tick labels */}
      {ticks.map((tk) => (
        <g key={tk} className="chart-grid">
          <line x1={sx(tk)} y1={pad} x2={sx(tk)} y2={pad + plot} />
          <line x1={pad} y1={sy(tk)} x2={pad + plot} y2={sy(tk)} />
          <text x={sx(tk)} y={pad + plot + 16} textAnchor="middle" className="chart-tick">
            {tk}
          </text>
          <text x={pad - 8} y={sy(tk) + 4} textAnchor="end" className="chart-tick">
            {tk}
          </text>
        </g>
      ))}

      {/* Perfect-calibration diagonal (y = x) */}
      <line
        x1={sx(0)}
        y1={sy(0)}
        x2={sx(1)}
        y2={sy(1)}
        className="chart-diagonal"
        strokeDasharray="4 4"
      />

      {/* Observed-vs-forecast points; area ∝ sample count n */}
      {points.map((p, i) => (
        <circle
          key={i}
          cx={sx(p[0])}
          cy={sy(p[1])}
          r={r(p[2])}
          className="chart-point"
        >
          <title>{`forecast ${(p[0] * 100).toFixed(1)}% · observed ${(p[1] * 100).toFixed(1)}% · n=${p[2]}`}</title>
        </circle>
      ))}

      {/* Axis labels */}
      <text x={pad + plot / 2} y={size - 4} textAnchor="middle" className="chart-axis">
        forecast probability
      </text>
      <text
        x={12}
        y={pad + plot / 2}
        textAnchor="middle"
        className="chart-axis"
        transform={`rotate(-90 12 ${pad + plot / 2})`}
      >
        observed frequency
      </text>
    </svg>
  );
}

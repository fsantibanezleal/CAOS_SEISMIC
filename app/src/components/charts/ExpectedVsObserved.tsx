import { useId } from "react";

/**
 * Expected-vs-observed event-count time series (evaluation-plan.md §9; web-app-spec.md §6).
 *
 * Grouped bars per period: the model's expected count next to the observed count, so an
 * aftershock-sequence spike (observed ≫ expected) is visible — the motivation for the
 * over-dispersion-honest catalog-based tests. Hand-rolled SVG, theme-driven, no library, and
 * works with no WebGL.
 */
export interface EvoPoint {
  period: string;
  expected: number;
  observed: number;
}

export interface ExpectedVsObservedProps {
  data: EvoPoint[];
  width?: number;
  height?: number;
  title?: string;
  expectedLabel?: string;
  observedLabel?: string;
}

export function ExpectedVsObserved({
  data,
  width = 560,
  height = 240,
  title,
  expectedLabel = "expected",
  observedLabel = "observed",
}: ExpectedVsObservedProps) {
  const uid = useId();
  const padL = 40;
  const padR = 12;
  const padT = 16;
  const padB = 40;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const maxV = Math.max(1, ...data.flatMap((d) => [d.expected, d.observed]));
  const yMax = niceMax(maxV);
  const groupW = plotW / Math.max(1, data.length);
  const barW = Math.max(3, (groupW - 6) / 2);

  const sy = (v: number) => padT + (1 - v / yMax) * plotH;
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(yMax * f * 10) / 10);

  return (
    <svg
      className="chart evo"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-labelledby={`${uid}-t`}
      preserveAspectRatio="xMidYMid meet"
    >
      <title id={`${uid}-t`}>{title ?? "Expected vs observed counts"}</title>

      {/* Y gridlines + ticks */}
      {yTicks.map((tk) => (
        <g key={tk} className="chart-grid">
          <line x1={padL} y1={sy(tk)} x2={padL + plotW} y2={sy(tk)} />
          <text x={padL - 6} y={sy(tk) + 4} textAnchor="end" className="chart-tick">
            {tk}
          </text>
        </g>
      ))}

      {/* Grouped bars */}
      {data.map((d, i) => {
        const x0 = padL + i * groupW + 3;
        return (
          <g key={d.period}>
            <rect
              x={x0}
              y={sy(d.expected)}
              width={barW}
              height={padT + plotH - sy(d.expected)}
              className="bar-expected"
            >
              <title>{`${d.period} · ${expectedLabel} ${d.expected}`}</title>
            </rect>
            <rect
              x={x0 + barW}
              y={sy(d.observed)}
              width={barW}
              height={padT + plotH - sy(d.observed)}
              className="bar-observed"
            >
              <title>{`${d.period} · ${observedLabel} ${d.observed}`}</title>
            </rect>
            {/* sparse x labels (every other) to avoid crowding */}
            {i % 2 === 0 ? (
              <text
                x={x0 + barW}
                y={height - 22}
                textAnchor="middle"
                className="chart-tick small"
              >
                {d.period.slice(2)}
              </text>
            ) : null}
          </g>
        );
      })}

      {/* Legend */}
      <g className="chart-legend" transform={`translate(${padL}, ${height - 8})`}>
        <rect x={0} y={-9} width={10} height={10} className="bar-expected" />
        <text x={16} y={0} className="chart-tick">
          {expectedLabel}
        </text>
        <rect x={90} y={-9} width={10} height={10} className="bar-observed" />
        <text x={106} y={0} className="chart-tick">
          {observedLabel}
        </text>
      </g>
    </svg>
  );
}

/** Round a max value up to a "nice" axis bound. */
function niceMax(v: number): number {
  const pow = Math.pow(10, Math.floor(Math.log10(v)));
  const n = v / pow;
  const step = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
  return step * pow;
}

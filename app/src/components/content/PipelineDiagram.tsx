import { useTranslation } from "react-i18next";

/**
 * Inline SVG flow diagram of the daily offline pipeline (web-app-spec.md §5.2).
 *
 * A single top-to-bottom pipeline in the dark-technical palette: five stacked boxes
 * connected by downward arrows, with arrow labels between stages and one caption underneath.
 * Colours read from the CSS theme variables (so the diagram follows light/dark) via
 * `currentColor`-style fills set through inline `style` on theme-aware classes; the SVG
 * itself uses CSS custom properties resolved at paint.
 *
 * The five stages (verbatim intent from §5.2):
 *  1. Data feeds (external, read-only, public) — USGS ComCat / ISC / IRIS-EarthScope FDSN.
 *  2. Offline daily job (single VPS, CPU) — ingest → QC/dedup → Mc & magnitude
 *     homogenization → declustering (dual-catalog) → fit/condition ETAS (+ R-J fallback) →
 *     simulate ensemble → per-cell rate field → optimistic/expected/pessimistic bounds →
 *     rolling CSEP stats.
 *  3. Artifact (committed/served static, gzipped) — forecast_YYYY-MM-DD.json(.gz) + H3 rate
 *     arrays for {1d,2d,7d} × {lo,exp,hi}; baseline; metadata; N/S/M/L summary; coverage mask.
 *  4. API (FastAPI, thin, stateless, read-only) — /forecast/latest, /forecast/{date},
 *     /region/{iso}, /calibration; ORJSON + GZip.
 *  5. SPA (Vite + React + TS) — the six pages incl. this diagram and the Monitoring field.
 *
 * All labels are translated (i18n `impl.diagram.*`); the SVG has an accessible title/desc.
 */

interface StageDef {
  titleKey: string;
  bodyKey: string;
}

const STAGES: StageDef[] = [
  { titleKey: "impl.diagram.feeds.title", bodyKey: "impl.diagram.feeds.body" },
  { titleKey: "impl.diagram.job.title", bodyKey: "impl.diagram.job.body" },
  { titleKey: "impl.diagram.artifact.title", bodyKey: "impl.diagram.artifact.body" },
  { titleKey: "impl.diagram.api.title", bodyKey: "impl.diagram.api.body" },
  { titleKey: "impl.diagram.spa.title", bodyKey: "impl.diagram.spa.body" },
];

/** Arrow labels rendered on the connectors between successive stages. */
const ARROW_KEYS = [
  "impl.diagram.arrow.pull",
  "impl.diagram.arrow.write",
  "impl.diagram.arrow.serve",
  "impl.diagram.arrow.render",
];

export function PipelineDiagram() {
  const { t } = useTranslation();

  // Layout geometry (viewBox units). Boxes are wide and short; arrows + labels sit between.
  const boxW = 520;
  const boxH = 96;
  const gap = 58; // vertical gap for the arrow + its label
  const xPad = 40;
  const totalW = boxW + xPad * 2;
  const stageTop = (i: number) => 24 + i * (boxH + gap);
  const totalH = stageTop(STAGES.length - 1) + boxH + 40;
  const cx = totalW / 2;

  return (
    <figure className="pipeline-figure">
      <svg
        className="pipeline-svg"
        viewBox={`0 0 ${totalW} ${totalH}`}
        role="img"
        aria-labelledby="pipeline-title pipeline-desc"
        preserveAspectRatio="xMidYMid meet"
      >
        <title id="pipeline-title">{t("impl.diagram.svgTitle")}</title>
        <desc id="pipeline-desc">{t("impl.diagram.svgDesc")}</desc>

        <defs>
          <marker
            id="pipeline-arrow"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" className="pipeline-arrowhead" />
          </marker>
        </defs>

        {STAGES.map((stage, i) => {
          const y = stageTop(i);
          const x = (totalW - boxW) / 2;
          return (
            <g key={stage.titleKey} className="pipeline-stage">
              <rect
                x={x}
                y={y}
                width={boxW}
                height={boxH}
                rx={10}
                ry={10}
                className="pipeline-box"
              />
              <text x={cx} y={y + 28} textAnchor="middle" className="pipeline-box-title">
                {t(stage.titleKey)}
              </text>
              <text x={cx} y={y + 52} textAnchor="middle" className="pipeline-box-body">
                {wrapToTspans(t(stage.bodyKey), cx, y + 52, 64)}
              </text>
            </g>
          );
        })}

        {/* Connectors + arrow labels between successive stages. */}
        {ARROW_KEYS.map((key, i) => {
          const y1 = stageTop(i) + boxH;
          const y2 = stageTop(i + 1);
          const midY = (y1 + y2) / 2;
          return (
            <g key={key} className="pipeline-connector">
              <line
                x1={cx}
                y1={y1 + 2}
                x2={cx}
                y2={y2 - 4}
                className="pipeline-arrow-line"
                markerEnd="url(#pipeline-arrow)"
              />
              <text x={cx + 14} y={midY + 4} textAnchor="start" className="pipeline-arrow-label">
                {t(key)}
              </text>
            </g>
          );
        })}
      </svg>

      <figcaption className="pipeline-caption">{t("impl.diagram.caption")}</figcaption>
    </figure>
  );
}

/**
 * Naive word-wrapper: split a body string into <tspan> lines of at most `maxChars`,
 * stacked under the anchor (x, y). Keeps the SVG self-contained (no foreignObject) and
 * legible at the diagram's small scale.
 */
function wrapToTspans(text: string, x: number, y: number, maxChars: number) {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let current = "";
  for (const w of words) {
    const candidate = current ? `${current} ${w}` : w;
    if (candidate.length > maxChars && current) {
      lines.push(current);
      current = w;
    } else {
      current = candidate;
    }
  }
  if (current) lines.push(current);

  const lineHeight = 15;
  return lines.map((line, i) => (
    <tspan key={i} x={x} y={y + i * lineHeight}>
      {line}
    </tspan>
  ));
}

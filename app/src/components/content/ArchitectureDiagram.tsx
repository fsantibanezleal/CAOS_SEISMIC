import { useTranslation } from "react-i18next";

/**
 * Inline SVG of OUR chosen forecasting architecture (model-design.md §1–§2, §6).
 *
 * This is the architecture schematic for Methodology Tab 3 ("The version employed"): the
 * two-layer estimator stack that the product ships, drawn in the dark-technical palette and
 * theme-aware via the `.arch-*` classes in globals.css.
 *
 * Layout (left → right, the data/conditioning flow):
 *
 *   ┌ inputs (catalog spine + context covariates) ┐
 *   │  · full un-declustered catalog (triggering)  │ ── feeds ──▶ ┌ ETAS-class core (ships v0) ┐
 *   │  · declustered catalog (background μ)         │              │ space–time ETAS (reference)│
 *   │  · CNN spatial-context encoder (Slab2 /       │              │ smoothed-seismicity Poisson│
 *   │    faults / strain → context tensor)          │ ── gates ──▶ │   null  ·  R-J fallback     │
 *   └──────────────────────────────────────────────┘              └────────────┬───────────────┘
 *                              │ context tensor                                 │ λ(t,x,y | H_t)
 *                              ▼                                                 ▼
 *                    ┌ gated neural challenger ┐  ── must beat ETAS ──▶ ┌ exceedance + bounds ┐
 *                    │ context-conditioned TPP │     in CSEP, else      │ P(≥1 ≥ M*) for       │
 *                    │ (Hawkes inductive bias) │     stays behind flag  │ 1d/2d/7d · P10/50/90 │
 *                    └─────────────────────────┘                        └──────────────────────┘
 *
 * Rather than fight SVG auto-layout, the boxes are positioned on a fixed grid and connected by
 * straight/elbow connectors. Every label is translated (`method.employed.arch.*`); the SVG has
 * an accessible title + desc. The "gated" edge is dashed to signal the feature flag; the core
 * path is solid because the core is what actually ships in v0.
 */

interface BoxDef {
  key: string;
  titleKey: string;
  bodyKey: string;
  x: number;
  y: number;
  w: number;
  h: number;
  /** "core" = solid accent (ships v0); "input" = neutral; "gated" = dashed (flagged). */
  variant: "core" | "input" | "gated" | "output";
}

// viewBox is 760 × 560; boxes laid out on a deliberate grid.
const VB_W = 760;
const VB_H = 560;

const BOXES: BoxDef[] = [
  {
    key: "inputs",
    titleKey: "method.employed.arch.inputs.title",
    bodyKey: "method.employed.arch.inputs.body",
    x: 24,
    y: 24,
    w: 320,
    h: 150,
    variant: "input",
  },
  {
    key: "encoder",
    titleKey: "method.employed.arch.encoder.title",
    bodyKey: "method.employed.arch.encoder.body",
    x: 24,
    y: 206,
    w: 320,
    h: 130,
    variant: "input",
  },
  {
    key: "core",
    titleKey: "method.employed.arch.core.title",
    bodyKey: "method.employed.arch.core.body",
    x: 416,
    y: 24,
    w: 320,
    h: 150,
    variant: "core",
  },
  {
    key: "challenger",
    titleKey: "method.employed.arch.challenger.title",
    bodyKey: "method.employed.arch.challenger.body",
    x: 416,
    y: 206,
    w: 320,
    h: 130,
    variant: "gated",
  },
  {
    key: "output",
    titleKey: "method.employed.arch.output.title",
    bodyKey: "method.employed.arch.output.body",
    x: 220,
    y: 392,
    w: 320,
    h: 140,
    variant: "output",
  },
];

export function ArchitectureDiagram() {
  const { t } = useTranslation();

  const box = (k: string) => BOXES.find((b) => b.key === k)!;
  const inputs = box("inputs");
  const encoder = box("encoder");
  const core = box("core");
  const challenger = box("challenger");
  const output = box("output");

  const right = (b: BoxDef) => ({ x: b.x + b.w, y: b.y + b.h / 2 });
  const left = (b: BoxDef) => ({ x: b.x, y: b.y + b.h / 2 });
  const bottom = (b: BoxDef) => ({ x: b.x + b.w / 2, y: b.y + b.h });
  const top = (b: BoxDef) => ({ x: b.x + b.w / 2, y: b.y });

  return (
    <figure className="arch-figure">
      <svg
        className="arch-svg"
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        role="img"
        aria-labelledby="arch-title arch-desc"
        preserveAspectRatio="xMidYMid meet"
      >
        <title id="arch-title">{t("method.employed.arch.svgTitle")}</title>
        <desc id="arch-desc">{t("method.employed.arch.svgDesc")}</desc>

        <defs>
          <marker
            id="arch-arrow"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" className="arch-arrowhead" />
          </marker>
          <marker
            id="arch-arrow-dashed"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" className="arch-arrowhead gated" />
          </marker>
        </defs>

        {/* ── Connectors (drawn first, under the boxes) ───────────────────── */}
        {/* inputs → core (feeds the catalog spine) */}
        <g className="arch-connector">
          <line
            x1={right(inputs).x}
            y1={right(inputs).y}
            x2={left(core).x - 2}
            y2={left(core).y}
            className="arch-line"
            markerEnd="url(#arch-arrow)"
          />
          <text
            x={(right(inputs).x + left(core).x) / 2}
            y={right(inputs).y - 8}
            textAnchor="middle"
            className="arch-edge-label"
          >
            {t("method.employed.arch.edge.feeds")}
          </text>
        </g>

        {/* encoder → challenger (context tensor) */}
        <g className="arch-connector">
          <line
            x1={right(encoder).x}
            y1={right(encoder).y}
            x2={left(challenger).x - 2}
            y2={left(challenger).y}
            className="arch-line"
            markerEnd="url(#arch-arrow)"
          />
          <text
            x={(right(encoder).x + left(challenger).x) / 2}
            y={right(encoder).y - 8}
            textAnchor="middle"
            className="arch-edge-label"
          >
            {t("method.employed.arch.edge.context")}
          </text>
        </g>

        {/* core → challenger (gates: challenger inherits the Hawkes skeleton) — dashed */}
        <g className="arch-connector">
          <line
            x1={bottom(core).x}
            y1={bottom(core).y}
            x2={top(challenger).x}
            y2={top(challenger).y - 2}
            className="arch-line gated"
            markerEnd="url(#arch-arrow-dashed)"
          />
          <text
            x={bottom(core).x + 8}
            y={(bottom(core).y + top(challenger).y) / 2 + 4}
            textAnchor="start"
            className="arch-edge-label gated"
          >
            {t("method.employed.arch.edge.gates")}
          </text>
        </g>

        {/* core → output (the λ that ships) — solid */}
        <g className="arch-connector">
          <path
            d={`M ${bottom(core).x} ${bottom(core).y} V 364 H ${top(output).x} V ${top(output).y - 2}`}
            className="arch-line"
            fill="none"
            markerEnd="url(#arch-arrow)"
          />
          <text x={bottom(core).x + 8} y={188} textAnchor="start" className="arch-edge-label">
            {t("method.employed.arch.edge.lambda")}
          </text>
        </g>

        {/* challenger → output (only if it wins) — dashed */}
        <g className="arch-connector">
          <path
            d={`M ${bottom(challenger).x} ${bottom(challenger).y} V 372 H ${output.x + output.w - 60} V ${top(output).y - 2}`}
            className="arch-line gated"
            fill="none"
            markerEnd="url(#arch-arrow-dashed)"
          />
          <text
            x={bottom(challenger).x + 8}
            y={358}
            textAnchor="start"
            className="arch-edge-label gated"
          >
            {t("method.employed.arch.edge.ifWins")}
          </text>
        </g>

        {/* ── Boxes ───────────────────────────────────────────────────────── */}
        {BOXES.map((b) => (
          <g key={b.key} className={`arch-stage arch-${b.variant}`}>
            <rect
              x={b.x}
              y={b.y}
              width={b.w}
              height={b.h}
              rx={10}
              ry={10}
              className="arch-box"
            />
            <text x={b.x + 16} y={b.y + 26} className="arch-box-title">
              {t(b.titleKey)}
            </text>
            <WrappedBody text={t(b.bodyKey)} x={b.x + 16} y={b.y + 48} maxChars={42} />
          </g>
        ))}
      </svg>
      <figcaption className="arch-caption">{t("method.employed.arch.caption")}</figcaption>
    </figure>
  );
}

/**
 * Left-aligned multi-line text inside a box: split on the bullet separator " · " first
 * (each fragment starts a new line), then soft-wrap long fragments to `maxChars`.
 */
function WrappedBody({
  text,
  x,
  y,
  maxChars,
}: {
  text: string;
  x: number;
  y: number;
  maxChars: number;
}) {
  const fragments = text.split(" · ");
  const lines: { text: string; bullet: boolean }[] = [];
  for (const frag of fragments) {
    const wrapped = softWrap(frag, maxChars);
    wrapped.forEach((ln, i) => lines.push({ text: ln, bullet: fragments.length > 1 && i === 0 }));
  }
  const lineHeight = 16;
  return (
    <text className="arch-box-body">
      {lines.map((ln, i) => (
        <tspan key={i} x={x} y={y + i * lineHeight}>
          {ln.bullet ? "• " : ""}
          {ln.text}
        </tspan>
      ))}
    </text>
  );
}

function softWrap(text: string, maxChars: number): string[] {
  const words = text.split(/\s+/);
  const out: string[] = [];
  let cur = "";
  for (const w of words) {
    const cand = cur ? `${cur} ${w}` : w;
    if (cand.length > maxChars && cur) {
      out.push(cur);
      cur = w;
    } else {
      cur = cand;
    }
  }
  if (cur) out.push(cur);
  return out;
}

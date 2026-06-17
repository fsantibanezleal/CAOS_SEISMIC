import type { ReactNode } from "react";
import { Trans, useTranslation } from "react-i18next";

import { Callout } from "@/components/content/Callout";
import { Cite } from "@/components/content/Cite";
import { BlockEquation, Inline } from "@/components/content/Equation";
import { Figure } from "@/components/content/Figure";
import { SubTabs, type SubTabDef } from "@/components/content/SubTabs";
import type { CitationId } from "@/lib/citations";

/**
 * Methodology — Tab 1: "Theoretical approaches".
 *
 * This is the encyclopedic STATE-OF-THE-ART / BACKGROUND layer: the classical statistical and
 * physics-based models that form the field-standard baseline a forecaster must beat. It is NOT
 * the model the product ships (that is Tab 3 / Implementation). One vertical sub-tab per model,
 * each DEEP: prose theory, bulleted assumptions, the REAL governing equation(s) in KaTeX, the
 * modeling role, an inline theme-aware SVG analysis figure, and real DOI'd references.
 *
 * Mapped faithfully from research/02-classical-models and synthesis/methodology.md (Tab 1).
 * Nothing here is invented; every equation and citation is transcribed from the cited primary
 * source. Copy is i18n (`method.th.*`); equations and citation data are language-neutral.
 *
 * Models (sub-tabs): Gutenberg–Richter · Omori–Utsu · ETAS · Reasenberg–Jones · STEP · EEPAS ·
 * smoothed seismicity · BPT / renewal · rate-and-state + Coulomb stress.
 */

/* ── A small shared bibliography row under each sub-tab. ─────────────────────── */
function Refs({ ids }: { ids: CitationId[] }) {
  const { t } = useTranslation();
  return (
    <p className="th-refs">
      <span className="th-refs-label">{t("method.th.refsLabel")}</span>{" "}
      {ids.map((id, i) => (
        <span key={id}>
          {i > 0 ? " · " : null}
          <Cite id={id} paren={false} />
        </span>
      ))}
    </p>
  );
}

/* ── A theme-aware assumptions list. ────────────────────────────────────────── */
function Assumptions({ items, title }: { items: ReactNode[]; title: string }) {
  return (
    <div className="th-assume">
      <p className="th-assume-title">{title}</p>
      <ul className="th-assume-list">
        {items.map((it, i) => (
          <li key={i}>{it}</li>
        ))}
      </ul>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Inline SVG analysis figures — one per model. All read the CSS palette via the
   `.th-fig-*` classes in globals.css, so they follow light/dark automatically.
   Each is a schematic of the model's load-bearing behaviour, not a fitted plot.
   ═══════════════════════════════════════════════════════════════════════════ */

/** Log-linear frequency–magnitude line with slope −b above completeness Mc. */
function GrFigureSvg() {
  const { t } = useTranslation();
  // axes box
  const x0 = 48;
  const y0 = 24;
  const w = 300;
  const h = 150;
  const xr = x0 + w;
  const yb = y0 + h;
  // GR line: log10 N = a - bM. Draw a - b*M over magnitude axis.
  const mcX = x0 + 70; // completeness magnitude position
  return (
    <svg
      className="th-svg"
      viewBox="0 0 380 210"
      role="img"
      aria-label={t("method.th.gr.figCap")}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* axes */}
      <line className="th-axis" x1={x0} y1={y0} x2={x0} y2={yb} />
      <line className="th-axis" x1={x0} y1={yb} x2={xr} y2={yb} />
      <text className="th-axis-label" x={x0 - 8} y={y0 + 4} textAnchor="end">
        log₁₀N
      </text>
      <text className="th-axis-label" x={xr} y={yb + 16} textAnchor="end">
        M
      </text>
      {/* incomplete (rolled-off) region below Mc, dashed */}
      <path
        className="th-curve-faint"
        d={`M ${x0 + 8} ${yb - 12} Q ${x0 + 36} ${yb - 70} ${mcX} ${y0 + 36}`}
        fill="none"
        strokeDasharray="4 3"
      />
      {/* GR straight line above Mc */}
      <line className="th-curve" x1={mcX} y1={y0 + 36} x2={xr - 12} y2={yb - 12} />
      {/* Mc marker */}
      <line className="th-marker" x1={mcX} y1={y0} x2={mcX} y2={yb} strokeDasharray="3 3" />
      <text className="th-marker-label" x={mcX} y={y0 - 6} textAnchor="middle">
        Mc
      </text>
      {/* slope annotation */}
      <text className="th-note" x={x0 + 150} y={y0 + 58} textAnchor="start">
        slope = −b
      </text>
    </svg>
  );
}

/** Omori power-law decay n(t) = K/(t+c)^p — fast initial decay relaxing to background. */
function OmoriFigureSvg() {
  const { t } = useTranslation();
  const x0 = 40;
  const y0 = 20;
  const w = 310;
  const h = 150;
  const xr = x0 + w;
  const yb = y0 + h;
  // sample the curve n(t) ∝ 1/(t+c)
  const c = 0.6;
  const k = 1.0;
  const pts: string[] = [];
  for (let i = 0; i <= 40; i++) {
    const tt = (i / 40) * 6; // time axis 0..6
    const val = k / Math.pow(tt + c, 1.1);
    const px = x0 + (tt / 6) * w;
    const py = yb - Math.min(val / (k / Math.pow(c, 1.1)), 1) * h;
    pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
  }
  return (
    <svg
      className="th-svg"
      viewBox="0 0 380 200"
      role="img"
      aria-label={t("method.th.omori.figCap")}
      preserveAspectRatio="xMidYMid meet"
    >
      <line className="th-axis" x1={x0} y1={y0} x2={x0} y2={yb} />
      <line className="th-axis" x1={x0} y1={yb} x2={xr} y2={yb} />
      <text className="th-axis-label" x={x0 - 8} y={y0 + 4} textAnchor="end">
        n(t)
      </text>
      <text className="th-axis-label" x={xr} y={yb + 16} textAnchor="end">
        t
      </text>
      {/* background rate level */}
      <line className="th-marker" x1={x0} y1={yb - 18} x2={xr} y2={yb - 18} strokeDasharray="3 3" />
      <text className="th-note" x={xr - 4} y={yb - 24} textAnchor="end">
        background
      </text>
      <polyline className="th-curve" points={pts.join(" ")} fill="none" />
      <text className="th-note" x={x0 + 70} y={y0 + 36} textAnchor="start">
        ∝ (t + c)⁻ᵖ
      </text>
    </svg>
  );
}

/** ETAS branching tree: a mainshock seeding offspring, offspring seeding their own. */
function EtasFigureSvg() {
  const { t } = useTranslation();
  const node = (cx: number, cy: number, r: number, cls: string) => (
    <circle className={cls} cx={cx} cy={cy} r={r} />
  );
  return (
    <svg
      className="th-svg"
      viewBox="0 0 380 210"
      role="img"
      aria-label={t("method.th.etas.figCap")}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* background field */}
      <rect className="th-field" x={16} y={150} width={348} height={44} rx={6} />
      <text className="th-note" x={20} y={188} textAnchor="start">
        μ(x, y) background
      </text>
      {/* edges */}
      <g className="th-edges">
        <line x1={70} y1={40} x2={150} y2={92} />
        <line x1={70} y1={40} x2={210} y2={92} />
        <line x1={150} y1={92} x2={130} y2={140} />
        <line x1={150} y1={92} x2={195} y2={140} />
        <line x1={210} y1={92} x2={255} y2={140} />
        <line x1={210} y1={92} x2={300} y2={140} />
      </g>
      {/* mainshock */}
      {node(70, 40, 13, "th-node-main")}
      <text className="th-node-label" x={70} y={20} textAnchor="middle">
        mainshock
      </text>
      {/* first-generation offspring */}
      {node(150, 92, 9, "th-node-2")}
      {node(210, 92, 9, "th-node-2")}
      {/* second-generation offspring */}
      {node(130, 140, 6, "th-node-3")}
      {node(195, 140, 6, "th-node-3")}
      {node(255, 140, 6, "th-node-3")}
      {node(300, 140, 6, "th-node-3")}
      <text className="th-note" x={332} y={96} textAnchor="end">
        self-exciting offspring
      </text>
    </svg>
  );
}

/** Reasenberg–Jones: GR magnitude tail × Omori time decay → exceedance probability. */
function RjFigureSvg() {
  const { t } = useTranslation();
  return (
    <svg
      className="th-svg"
      viewBox="0 0 380 170"
      role="img"
      aria-label={t("method.th.rj.figCap")}
      preserveAspectRatio="xMidYMid meet"
    >
      <rect className="th-box" x={14} y={48} width={96} height={56} rx={8} />
      <text className="th-box-title" x={62} y={72} textAnchor="middle">
        GR magnitude
      </text>
      <text className="th-box-sub" x={62} y={90} textAnchor="middle">
        10^(b(Mm−M))
      </text>

      <text className="th-op" x={124} y={82} textAnchor="middle">
        ×
      </text>

      <rect className="th-box" x={138} y={48} width={96} height={56} rx={8} />
      <text className="th-box-title" x={186} y={72} textAnchor="middle">
        Omori decay
      </text>
      <text className="th-box-sub" x={186} y={90} textAnchor="middle">
        (t + c)⁻ᵖ
      </text>

      <line
        className="th-flow"
        x1={234}
        y1={76}
        x2={266}
        y2={76}
        markerEnd="url(#th-arrow)"
      />

      <rect className="th-box th-box-accent" x={270} y={48} width={96} height={56} rx={8} />
      <text className="th-box-title" x={318} y={72} textAnchor="middle">
        P(≥1)
      </text>
      <text className="th-box-sub" x={318} y={90} textAnchor="middle">
        1 − e^(−N)
      </text>

      <defs>
        <marker
          id="th-arrow"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="7"
          markerHeight="7"
          orient="auto-start-reverse"
        >
          <path className="th-arrowhead" d="M 0 0 L 10 5 L 0 10 z" />
        </marker>
      </defs>
      <text className="th-note" x={190} y={130} textAnchor="middle">
        rate λ(t, M) integrated over the horizon → expected count N
      </text>
    </svg>
  );
}

/** STEP: background grid + clustering component → blended gridded probability map. */
function StepFigureSvg() {
  const { t } = useTranslation();
  return (
    <svg
      className="th-svg"
      viewBox="0 0 380 180"
      role="img"
      aria-label={t("method.th.step.figCap")}
      preserveAspectRatio="xMidYMid meet"
    >
      <rect className="th-box" x={14} y={20} width={120} height={44} rx={8} />
      <text className="th-box-title" x={74} y={40} textAnchor="middle">
        background
      </text>
      <text className="th-box-sub" x={74} y={56} textAnchor="middle">
        time-independent
      </text>

      <rect className="th-box" x={14} y={92} width={120} height={44} rx={8} />
      <text className="th-box-title" x={74} y={112} textAnchor="middle">
        clustering (R–J)
      </text>
      <text className="th-box-sub" x={74} y={128} textAnchor="middle">
        generic · seq · gridded
      </text>

      <line className="th-flow" x1={134} y1={42} x2={178} y2={70} markerEnd="url(#th-arrow2)" />
      <line className="th-flow" x1={134} y1={114} x2={178} y2={86} markerEnd="url(#th-arrow2)" />

      {/* gridded map: 4x3 cells with graded fill */}
      <g className="th-grid-map">
        {Array.from({ length: 12 }).map((_, i) => {
          const col = i % 4;
          const row = Math.floor(i / 4);
          const op = 0.12 + ((col + row) / 5) * 0.7;
          return (
            <rect
              key={i}
              className="th-grid-cell"
              x={190 + col * 42}
              y={36 + row * 30}
              width={38}
              height={26}
              rx={3}
              style={{ fillOpacity: op }}
            />
          );
        })}
      </g>
      <text className="th-note" x={232} y={150} textAnchor="start">
        hourly shaking-probability map (MMI ≥ VI)
      </text>
      <defs>
        <marker
          id="th-arrow2"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="7"
          markerHeight="7"
          orient="auto-start-reverse"
        >
          <path className="th-arrowhead" d="M 0 0 L 10 5 L 0 10 z" />
        </marker>
      </defs>
    </svg>
  );
}

/** EEPAS: a precursor event projecting a magnitude / time / area scaling envelope. */
function EepasFigureSvg() {
  const { t } = useTranslation();
  return (
    <svg
      className="th-svg"
      viewBox="0 0 380 190"
      role="img"
      aria-label={t("method.th.eepas.figCap")}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* precursor */}
      {/* time axis */}
      <line className="th-axis" x1={40} y1={150} x2={356} y2={150} />
      <text className="th-axis-label" x={356} y={166} textAnchor="end">
        time →
      </text>
      <circle className="th-node-2" cx={70} cy={150} r={7} />
      <text className="th-node-label" x={70} y={172} textAnchor="middle">
        precursor Mp
      </text>
      {/* lognormal-in-time envelope → expected larger event */}
      <path
        className="th-field-stroke"
        d="M 150 150 C 200 60, 280 60, 320 150 Z"
        fill="none"
      />
      <text className="th-note" x={235} y={92} textAnchor="middle">
        expected Mm = a_M + b_M·Mp
      </text>
      <circle className="th-node-main" cx={235} cy={150} r={11} />
      <text className="th-node-label" x={235} y={138} textAnchor="middle">
        Mm
      </text>
      <text className="th-note" x={150} y={150 - 4} textAnchor="start">
        log₁₀T_P
      </text>
    </svg>
  );
}

/** Smoothed seismicity: scattered epicentres smoothed by an adaptive kernel into a density. */
function SmoothedFigureSvg() {
  const { t } = useTranslation();
  const seeds = [
    [90, 70],
    [120, 95],
    [105, 120],
    [150, 80],
    [140, 130],
    [175, 110],
  ];
  return (
    <svg
      className="th-svg"
      viewBox="0 0 380 190"
      role="img"
      aria-label={t("method.th.smoothed.figCap")}
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <radialGradient id="th-kernel" cx="50%" cy="50%" r="50%">
          <stop offset="0%" className="th-kernel-core" />
          <stop offset="100%" className="th-kernel-edge" />
        </radialGradient>
      </defs>
      {/* smoothed density blobs */}
      {seeds.map(([cx, cy], i) => (
        <circle key={`k${i}`} cx={cx} cy={cy} r={34} fill="url(#th-kernel)" />
      ))}
      {/* epicentres */}
      {seeds.map(([cx, cy], i) => (
        <circle key={`s${i}`} className="th-node-3" cx={cx} cy={cy} r={3.5} />
      ))}
      <text className="th-note" x={230} y={70} textAnchor="start">
        adaptive bandwidth d_i
      </text>
      <text className="th-note" x={230} y={92} textAnchor="start">
        = distance to n-th
      </text>
      <text className="th-note" x={230} y={114} textAnchor="start">
        nearest neighbour
      </text>
      <text className="th-note" x={70} y={172} textAnchor="start">
        declustered M ≥ 2 epicentres → time-independent μ(x, y)
      </text>
    </svg>
  );
}

/** BPT hazard: rises from ~0 after an event, peaks, then plateaus (vs flat Poisson). */
function BptFigureSvg() {
  const { t } = useTranslation();
  const x0 = 44;
  const y0 = 20;
  const w = 300;
  const h = 130;
  const xr = x0 + w;
  const yb = y0 + h;
  // BPT-like hazard: rise, peak, plateau
  const pts: string[] = [];
  for (let i = 0; i <= 50; i++) {
    const tt = i / 50;
    // smooth rise then plateau
    const val = 1 - Math.exp(-3.2 * tt);
    const px = x0 + tt * w;
    const py = yb - val * (h - 16);
    pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
  }
  return (
    <svg
      className="th-svg"
      viewBox="0 0 380 180"
      role="img"
      aria-label={t("method.th.bpt.figCap")}
      preserveAspectRatio="xMidYMid meet"
    >
      <line className="th-axis" x1={x0} y1={y0} x2={x0} y2={yb} />
      <line className="th-axis" x1={x0} y1={yb} x2={xr} y2={yb} />
      <text className="th-axis-label" x={x0 - 8} y={y0 + 4} textAnchor="end">
        hazard
      </text>
      <text className="th-axis-label" x={xr} y={yb + 16} textAnchor="end">
        time since last event
      </text>
      {/* Poisson constant hazard reference */}
      <line className="th-marker" x1={x0} y1={yb - 64} x2={xr} y2={yb - 64} strokeDasharray="4 3" />
      <text className="th-note" x={xr - 4} y={yb - 70} textAnchor="end">
        Poisson (constant)
      </text>
      {/* BPT curve */}
      <polyline className="th-curve" points={pts.join(" ")} fill="none" />
      <text className="th-note" x={x0 + 86} y={y0 + 30} textAnchor="start">
        BPT: rises → peaks → plateaus
      </text>
    </svg>
  );
}

/** Coulomb lobes + rate-and-state: ΔCFS promotes/suppresses, response decays Omori-like. */
function RsFigureSvg() {
  const { t } = useTranslation();
  return (
    <svg
      className="th-svg"
      viewBox="0 0 380 200"
      role="img"
      aria-label={t("method.th.rs.figCap")}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* fault line */}
      <line className="th-fault" x1={120} y1={30} x2={120} y2={160} />
      <text className="th-node-label" x={120} y={22} textAnchor="middle">
        source fault
      </text>
      {/* positive (promoting) lobes */}
      <ellipse className="th-lobe-pos" cx={78} cy={70} rx={34} ry={24} />
      <ellipse className="th-lobe-pos" cx={162} cy={120} rx={34} ry={24} />
      <text className="th-lobe-label pos" x={78} y={74} textAnchor="middle">
        +ΔCFS
      </text>
      <text className="th-lobe-label pos" x={162} y={124} textAnchor="middle">
        +ΔCFS
      </text>
      {/* negative (suppressing) lobes */}
      <ellipse className="th-lobe-neg" cx={78} cy={120} rx={34} ry={24} />
      <ellipse className="th-lobe-neg" cx={162} cy={70} rx={34} ry={24} />
      <text className="th-lobe-label neg" x={78} y={124} textAnchor="middle">
        −ΔCFS
      </text>
      <text className="th-lobe-label neg" x={162} y={74} textAnchor="middle">
        −ΔCFS
      </text>
      {/* rate-and-state response curve (Omori-like decay) */}
      <line className="th-axis" x1={232} y1={36} x2={232} y2={150} />
      <line className="th-axis" x1={232} y1={150} x2={356} y2={150} />
      <text className="th-axis-label" x={232} y={28} textAnchor="middle">
        R(t)/r
      </text>
      {(() => {
        const pts: string[] = [];
        for (let i = 0; i <= 40; i++) {
          const tt = (i / 40) * 5;
          const val = 1 + 4 * Math.exp(-tt * 1.3);
          const px = 232 + (tt / 5) * 120;
          const py = 150 - ((val - 1) / 4) * 100;
          pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
        }
        return <polyline className="th-curve" points={pts.join(" ")} fill="none" />;
      })()}
      <text className="th-note" x={294} y={70} textAnchor="middle">
        ∝ exp(ΔCFS/Aσ)
      </text>
      <text className="th-note" x={294} y={168} textAnchor="middle">
        Omori-like 1/t decay
      </text>
    </svg>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Per-model sub-tab bodies.
   ═══════════════════════════════════════════════════════════════════════════ */

function GutenbergRichter() {
  const { t } = useTranslation();
  return (
    <div className="th-model">
      <p>{t("method.th.gr.body")}</p>
      <BlockEquation
        math={String.raw`\log_{10} N(\geq M) = a - b\,M, \qquad
          f(M) = \beta\, e^{-\beta (M - M_c)}, \quad \beta = b \ln 10, \ M \geq M_c`}
        caption={t("method.th.gr.eqCap")}
      />
      <p>
        <Trans
          i18nKey="method.th.gr.estim"
          components={{ mc: <Inline math="M_c" />, b: <strong /> }}
        />
      </p>
      <BlockEquation
        math={String.raw`\hat b = \dfrac{\log_{10} e}{\bar m - \left(M_c - \tfrac{\Delta M}{2}\right)}`}
        caption={t("method.th.gr.eqCap2")}
      />
      <Assumptions
        title={t("method.th.assumptions")}
        items={[
          t("method.th.gr.a1"),
          t("method.th.gr.a2"),
          t("method.th.gr.a3"),
          t("method.th.gr.a4"),
        ]}
      />
      <Figure
        title={t("method.th.figureWord") + " 1"}
        caption={t("method.th.gr.figCap")}
      >
        <GrFigureSvg />
      </Figure>
      <p>
        <strong>{t("method.th.roleWord")}:</strong> {t("method.th.gr.role")}
      </p>
      <Refs ids={["gutenbergRichter1944", "aki1965", "tintiMulargia1987", "wiemerWyss2000", "woessnerWiemer2005"]} />
    </div>
  );
}

function OmoriUtsu() {
  const { t } = useTranslation();
  return (
    <div className="th-model">
      <p>{t("method.th.omori.body")}</p>
      <BlockEquation
        math={String.raw`n(t) = \dfrac{K}{(t + c)^{p}}, \qquad p \approx 1`}
        caption={t("method.th.omori.eqCap")}
      />
      <p>{t("method.th.omori.cumIntro")}</p>
      <BlockEquation
        math={String.raw`N(t_1, t_2) = \int_{t_1}^{t_2}\!\dfrac{K}{(t+c)^{p}}\,dt
          = \dfrac{K}{1-p}\left[(t_2 + c)^{1-p} - (t_1 + c)^{1-p}\right] \ (p \neq 1)`}
        caption={t("method.th.omori.eqCap2")}
      />
      <Assumptions
        title={t("method.th.assumptions")}
        items={[
          t("method.th.omori.a1"),
          t("method.th.omori.a2"),
          t("method.th.omori.a3"),
        ]}
      />
      <Callout tone="honest" title={t("method.th.omori.incompleteTitle")}>
        {t("method.th.omori.incomplete")}
      </Callout>
      <Figure title={t("method.th.figureWord") + " 2"} caption={t("method.th.omori.figCap")}>
        <OmoriFigureSvg />
      </Figure>
      <p>
        <strong>{t("method.th.roleWord")}:</strong> {t("method.th.omori.role")}
      </p>
      <Refs ids={["utsu1995", "ogata1983"]} />
    </div>
  );
}

function Etas() {
  const { t } = useTranslation();
  return (
    <div className="th-model">
      <p>
        <Trans i18nKey="method.th.etas.body" components={{ b: <strong /> }} />
      </p>
      <BlockEquation
        math={String.raw`\lambda(t, x, y \mid \mathcal{H}_t) = \mu(x,y)
          + \sum_{i:\,t_i < t} \underbrace{K\,e^{\alpha(M_i - M_0)}}_{\text{Utsu productivity}}
          \;\underbrace{\bigl(1 + \tfrac{t - t_i}{c}\bigr)^{-p}}_{\text{Omori kernel}}
          \;\underbrace{f(x - x_i,\, y - y_i \mid M_i)}_{\text{spatial kernel}}`}
        caption={t("method.th.etas.eqCap")}
      />
      <p>{t("method.th.etas.kernelIntro")}</p>
      <BlockEquation
        math={String.raw`f(x, y \mid M_i) = \dfrac{q-1}{\pi\,\zeta^2}
          \left(1 + \dfrac{r^2}{\zeta^2}\right)^{-q}, \quad
          \zeta = D\, e^{\gamma (M_i - M_0)}, \quad r^2 = (x-x_i)^2 + (y-y_i)^2`}
        caption={t("method.th.etas.eqCap2")}
      />
      <p>{t("method.th.etas.branchIntro")}</p>
      <BlockEquation
        math={String.raw`n = \int_{M_0}^{M_{\max}}\!\!\int_0^{\infty}
          K\,e^{\alpha(M-M_0)}\,g(\tau)\,\beta e^{-\beta(M-M_0)}\,d\tau\,dM`}
        caption={t("method.th.etas.eqCap3")}
      />
      <Callout tone="note" title={t("method.th.etas.gatesTitle")}>
        <ol className="ordered-steps">
          <li>
            <Trans
              i18nKey="method.th.etas.gate1"
              components={{
                m: <Inline math="\alpha < \beta" />,
                beta: <Inline math="\beta = b\,\ln 10" />,
              }}
            />
          </li>
          <li>
            <Trans
              i18nKey="method.th.etas.gate2"
              components={{ n: <Inline math="n < 1" />, nreject: <Inline math="n \geq 1" /> }}
            />
          </li>
        </ol>
      </Callout>
      <Assumptions
        title={t("method.th.assumptions")}
        items={[
          t("method.th.etas.a1"),
          t("method.th.etas.a2"),
          t("method.th.etas.a3"),
        ]}
      />
      <Figure title={t("method.th.figureWord") + " 3"} caption={t("method.th.etas.figCap")}>
        <EtasFigureSvg />
      </Figure>
      <p>{t("method.th.etas.fit")}</p>
      <p>
        <strong>{t("method.th.roleWord")}:</strong> {t("method.th.etas.role")}
      </p>
      <Refs ids={["ogata1988", "ogata1998", "zhuang2002"]} />
    </div>
  );
}

function ReasenbergJones() {
  const { t } = useTranslation();
  return (
    <div className="th-model">
      <p>{t("method.th.rj.body")}</p>
      <BlockEquation
        math={String.raw`\lambda(t, M) = \dfrac{10^{\,a + b\,(M_m - M)}}{(t + c)^{p}}`}
        caption={t("method.th.rj.eqCap")}
      />
      <p>{t("method.th.rj.probIntro")}</p>
      <BlockEquation
        math={String.raw`N(M; t_1, t_2) = \int_{t_1}^{t_2}\!\lambda(t, M)\,dt, \qquad
          P(M; t_1, t_2) = 1 - e^{-N(M; t_1, t_2)}`}
        caption={t("method.th.rj.eqCap2")}
      />
      <Assumptions
        title={t("method.th.assumptions")}
        items={[
          t("method.th.rj.a1"),
          t("method.th.rj.a2"),
          t("method.th.rj.a3"),
        ]}
      />
      <Figure title={t("method.th.figureWord") + " 4"} caption={t("method.th.rj.figCap")}>
        <RjFigureSvg />
      </Figure>
      <p>{t("method.th.rj.regimes")}</p>
      <p>
        <strong>{t("method.th.roleWord")}:</strong> {t("method.th.rj.role")}
      </p>
      <Refs ids={["reasenbergJones1989", "reasenbergJones1994", "page2016"]} />
    </div>
  );
}

function Step() {
  const { t } = useTranslation();
  return (
    <div className="th-model">
      <p>{t("method.th.step.body")}</p>
      <Assumptions
        title={t("method.th.assumptions")}
        items={[
          t("method.th.step.a1"),
          t("method.th.step.a2"),
          t("method.th.step.a3"),
        ]}
      />
      <Figure title={t("method.th.figureWord") + " 5"} caption={t("method.th.step.figCap")}>
        <StepFigureSvg />
      </Figure>
      <p>{t("method.th.step.structure")}</p>
      <p>
        <strong>{t("method.th.roleWord")}:</strong> {t("method.th.step.role")}
      </p>
      <Refs ids={["gerstenberger2005", "reasenbergJones1989"]} />
    </div>
  );
}

function Eepas() {
  const { t } = useTranslation();
  return (
    <div className="th-model">
      <p>{t("method.th.eepas.body")}</p>
      <BlockEquation
        math={String.raw`M_m = a_M + b_M\, M_p \ (b_M = 1), \quad
          \log_{10} T_P = a_T + b_T\, M_p, \quad
          \log_{10} A = a_A + b_A\, M_p`}
        caption={t("method.th.eepas.eqCap")}
      />
      <p>{t("method.th.eepas.densityIntro")}</p>
      <BlockEquation
        math={String.raw`\lambda(t, m, x, y) = \mu\,\lambda_0(m, x, y)
          + \sum_{t_i < t} w_i\, f(m \mid M_i)\, g(t - t_i \mid M_i)\, h(x, y \mid x_i, y_i, M_i)`}
        caption={t("method.th.eepas.eqCap2")}
      />
      <Assumptions
        title={t("method.th.assumptions")}
        items={[
          t("method.th.eepas.a1"),
          t("method.th.eepas.a2"),
          t("method.th.eepas.a3"),
        ]}
      />
      <Callout tone="honest" title={t("method.th.eepas.typosTitle")}>
        {t("method.th.eepas.typos")}
      </Callout>
      <Figure title={t("method.th.figureWord") + " 6"} caption={t("method.th.eepas.figCap")}>
        <EepasFigureSvg />
      </Figure>
      <p>
        <strong>{t("method.th.roleWord")}:</strong> {t("method.th.eepas.role")}
      </p>
      <Refs ids={["rhoadesEvison2004"]} />
    </div>
  );
}

function Smoothed() {
  const { t } = useTranslation();
  return (
    <div className="th-model">
      <p>{t("method.th.smoothed.body")}</p>
      <BlockEquation
        math={String.raw`\mu(x,y) = \sum_{i} K_{d_i}\!\big(\lVert (x,y) - (x_i,y_i)\rVert\big),
          \qquad K_{d}(r) = \dfrac{C(d)}{\big(r^2 + d^2\big)^{s}}`}
        caption={t("method.th.smoothed.eqCap")}
      />
      <Assumptions
        title={t("method.th.assumptions")}
        items={[
          t("method.th.smoothed.a1"),
          t("method.th.smoothed.a2"),
          t("method.th.smoothed.a3"),
        ]}
      />
      <Callout tone="note" title={t("method.th.smoothed.kernelNoteTitle")}>
        <Trans i18nKey="method.th.smoothed.kernelNote" components={{ s: <Inline math="s" /> }} />
      </Callout>
      <Figure title={t("method.th.figureWord") + " 7"} caption={t("method.th.smoothed.figCap")}>
        <SmoothedFigureSvg />
      </Figure>
      <p>
        <strong>{t("method.th.roleWord")}:</strong> {t("method.th.smoothed.role")}
      </p>
      <Refs ids={["helmstetter2007"]} />
    </div>
  );
}

function Bpt() {
  const { t } = useTranslation();
  return (
    <div className="th-model">
      <p>{t("method.th.bpt.body")}</p>
      <BlockEquation
        math={String.raw`f(t; \mu, \alpha) = \sqrt{\dfrac{\mu}{2\pi\,\alpha^2\, t^3}}
          \;\exp\!\left(-\dfrac{(t-\mu)^2}{2\,\mu\,\alpha^2\, t}\right), \qquad t > 0`}
        caption={t("method.th.bpt.eqCap")}
      />
      <p>{t("method.th.bpt.condIntro")}</p>
      <BlockEquation
        math={String.raw`P = \dfrac{\displaystyle\int_{T_e}^{T_e + \Delta T} f(t)\,dt}
          {\displaystyle\int_{T_e}^{\infty} f(t)\,dt}`}
        caption={t("method.th.bpt.eqCap2")}
      />
      <Assumptions
        title={t("method.th.assumptions")}
        items={[
          t("method.th.bpt.a1"),
          t("method.th.bpt.a2"),
          t("method.th.bpt.a3"),
        ]}
      />
      <Callout tone="honest" title={t("method.th.bpt.caveatTitle")}>
        {t("method.th.bpt.caveat")}
      </Callout>
      <Figure title={t("method.th.figureWord") + " 8"} caption={t("method.th.bpt.figCap")}>
        <BptFigureSvg />
      </Figure>
      <p>
        <strong>{t("method.th.roleWord")}:</strong> {t("method.th.bpt.role")}
      </p>
      <Refs ids={["matthews2002", "schwartzCoppersmith1984"]} />
    </div>
  );
}

function RateState() {
  const { t } = useTranslation();
  return (
    <div className="th-model">
      <p>{t("method.th.rs.body")}</p>
      <BlockEquation
        math={String.raw`\Delta\mathrm{CFS} = \Delta\tau - \mu'\,\Delta\sigma_n`}
        caption={t("method.th.rs.eqCap")}
      />
      <p>{t("method.th.rs.rsIntro")}</p>
      <BlockEquation
        math={String.raw`\dfrac{R}{r} = \exp\!\left(\dfrac{\Delta\mathrm{CFS}}{A\sigma}\right),
          \qquad d\gamma = \dfrac{1}{A\sigma_n}\big(dt - \gamma\, d\tau\big),
          \qquad t_a = \dfrac{A\sigma_n}{\dot\tau_r}`}
        caption={t("method.th.rs.eqCap2")}
      />
      <Assumptions
        title={t("method.th.assumptions")}
        items={[
          t("method.th.rs.a1"),
          t("method.th.rs.a2"),
          t("method.th.rs.a3"),
        ]}
      />
      <Figure title={t("method.th.figureWord") + " 9"} caption={t("method.th.rs.figCap")}>
        <RsFigureSvg />
      </Figure>
      <p>
        <Trans i18nKey="method.th.rs.bridge" components={{ b: <strong /> }} />
      </p>
      <p>
        <strong>{t("method.th.roleWord")}:</strong> {t("method.th.rs.role")}
      </p>
      <Refs ids={["kingSteinLin1994", "stein1999", "dieterich1994", "heimissonSegall2018"]} />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   The Tab-1 shell: intro + vertical SubTabs (one per classical model).
   ═══════════════════════════════════════════════════════════════════════════ */

export function MethodologyTheory() {
  const { t } = useTranslation();

  const tabs: SubTabDef[] = [
    { id: "gr", label: t("method.th.gr.tab"), content: <GutenbergRichter /> },
    { id: "omori", label: t("method.th.omori.tab"), content: <OmoriUtsu /> },
    { id: "etas", label: t("method.th.etas.tab"), content: <Etas /> },
    { id: "rj", label: t("method.th.rj.tab"), content: <ReasenbergJones /> },
    { id: "step", label: t("method.th.step.tab"), content: <Step /> },
    { id: "eepas", label: t("method.th.eepas.tab"), content: <Eepas /> },
    { id: "smoothed", label: t("method.th.smoothed.tab"), content: <Smoothed /> },
    { id: "bpt", label: t("method.th.bpt.tab"), content: <Bpt /> },
    { id: "rs", label: t("method.th.rs.tab"), content: <RateState /> },
  ];

  return (
    <div className="prose">
      <p className="muted">{t("method.th.intro")}</p>
      <Callout tone="strong" title={t("method.th.baselineTitle")}>
        <Trans i18nKey="method.th.baseline" components={{ b: <strong /> }} />
      </Callout>
      <SubTabs tabs={tabs} ariaLabel={t("method.th.subtabsAria")} orientation="vertical" />
    </div>
  );
}

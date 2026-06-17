import type { ReactNode } from "react";
import { Trans, useTranslation } from "react-i18next";
import {
  Activity,
  AlertTriangle,
  Binary,
  Brain,
  GitGraph,
  Layers,
  Network,
  Radar,
  Waves,
} from "lucide-react";

import { Callout } from "@/components/content/Callout";
import { Cite } from "@/components/content/Cite";
import { BlockEquation } from "@/components/content/Equation";
import { Figure } from "@/components/content/Figure";
import { SubTabs, type SubTabDef } from "@/components/content/SubTabs";

/**
 * Methodology — Tab 2: "Analytical / ML methods" (STATE OF THE ART, not our implementation).
 *
 * One DEEP sub-tab per method, mapping the deep research
 * (research/03-ml-approaches + synthesis/methodology.md §2): the temporal-point-process
 * conditional-intensity framework that unifies every model, the neural-TPP lineage
 * (RMTPP → Neural Hawkes → Self-Attentive / Transformer Hawkes), neural-TPP applied to
 * earthquakes (RECAST, FERN, and the decisive EarthquakeNPP benchmark), the CNN spatial
 * cautionary tale (DeVries 2018 refuted by Mignan & Broccardo 2019), GNN / RNN-LSTM, and the
 * hard line that detection (PhaseNet / EQTransformer / SeisLM) is NOT forecasting.
 *
 * Each sub-tab carries: theory prose, explicit assumptions, the KEY equation(s) in KaTeX, the
 * approach, an inline theme-aware SVG diagram, and real references with DOIs. Honest throughout
 * (the spatial-test gap, the AUC trap, the leakage lesson). All copy is i18n (`method.ml.*`);
 * nothing is invented, every equation and citation comes from the research.
 */

/** Section heading with a lucide glyph in the accent colour (matches the LDA-HSI look). */
function MethodHead({ icon, title }: { icon: ReactNode; title: ReactNode }) {
  return (
    <header className="method-head">
      <span className="method-head-icon" aria-hidden="true">
        {icon}
      </span>
      <h3>{title}</h3>
    </header>
  );
}

/** A compact "assumptions / limitations" list rendered from an i18n array. */
function AssumptionList({ titleKey, items }: { titleKey: string; items: string[] }) {
  const { t } = useTranslation();
  return (
    <div className="assumptions">
      <p className="assumptions-title">{t(titleKey)}</p>
      <ul className="method-assumption-list">
        {items.map((it) => (
          <li key={it}>{it}</li>
        ))}
      </ul>
    </div>
  );
}

/* ──────────────────────────────────────────────────────────────────────────────
   Inline SVG diagrams. All are theme-aware: fills/strokes read the CSS palette via
   the shared `.diagram-*` classes in globals.css, so they follow light/dark.
   ────────────────────────────────────────────────────────────────────────────── */

/** Marker defs shared by every diagram (one arrowhead). */
function ArrowDefs({ id }: { id: string }) {
  return (
    <defs>
      <marker
        id={id}
        viewBox="0 0 10 10"
        refX="8"
        refY="5"
        markerWidth="7"
        markerHeight="7"
        orient="auto-start-reverse"
      >
        <path d="M 0 0 L 10 5 L 0 10 z" className="diagram-arrowhead" />
      </marker>
    </defs>
  );
}

/**
 * The conditional-intensity timeline: events as ticks on a time axis, a λ*(t) curve that
 * jumps up at each event and decays (Omori-like) between them, illustrating self-excitation.
 */
function ConditionalIntensityDiagram() {
  const { t } = useTranslation();
  // Event times (viewBox x), decay curve sampled as a polyline.
  const x0 = 40;
  const x1 = 540;
  const baseY = 150;
  const topY = 30;
  const events = [90, 150, 250, 360, 460];
  // Build a self-exciting intensity: background + decaying jumps after each event.
  const pts: string[] = [];
  for (let x = x0; x <= x1; x += 4) {
    let lambda = 0.12; // background μ (fraction of full height)
    for (const e of events) {
      if (x >= e) lambda += 0.7 * Math.exp(-(x - e) / 38);
    }
    const y = baseY - Math.min(lambda, 1.05) * (baseY - topY);
    pts.push(`${x},${y.toFixed(1)}`);
  }
  return (
    <svg
      className="diagram-svg"
      viewBox="0 0 580 200"
      role="img"
      aria-label={t("method.ml.tpp.figAlt")}
      preserveAspectRatio="xMidYMid meet"
    >
      <ArrowDefs id="ci-arrow" />
      {/* axes */}
      <line x1={x0} y1={baseY} x2={x1 + 8} y2={baseY} className="diagram-axis" markerEnd="url(#ci-arrow)" />
      <line x1={x0} y1={baseY} x2={x0} y2={topY - 6} className="diagram-axis" markerEnd="url(#ci-arrow)" />
      <text x={x1} y={baseY + 18} className="diagram-axis-label" textAnchor="end">
        t
      </text>
      <text x={x0 - 8} y={topY - 2} className="diagram-axis-label" textAnchor="end">
        λ*(t)
      </text>
      {/* background level */}
      <line
        x1={x0}
        y1={baseY - 0.12 * (baseY - topY)}
        x2={x1}
        y2={baseY - 0.12 * (baseY - topY)}
        className="diagram-guide"
      />
      <text x={x1} y={baseY - 0.12 * (baseY - topY) - 4} className="diagram-tick" textAnchor="end">
        μ
      </text>
      {/* intensity curve */}
      <polyline points={pts.join(" ")} className="diagram-curve" />
      {/* events as ticks + history conditioning */}
      {events.map((e) => (
        <g key={e}>
          <line x1={e} y1={baseY} x2={e} y2={baseY + 12} className="diagram-event" />
          <circle cx={e} cy={baseY + 16} r={2.6} className="diagram-event-dot" />
        </g>
      ))}
      <text x={(events[0] ?? 0) + 6} y={baseY + 30} className="diagram-tick">
        {t("method.ml.tpp.figHist")}
      </text>
    </svg>
  );
}

/**
 * RMTPP: an RNN that consumes (time, mark) pairs and emits a hidden state h_j, from which a
 * log-linear intensity with an exponential decay term is read off.
 */
function RmtppDiagram() {
  const { t } = useTranslation();
  const cells = [
    { x: 60, label: "(t₁,m₁)" },
    { x: 210, label: "(t₂,m₂)" },
    { x: 360, label: "(t₃,m₃)" },
  ];
  const cy = 70;
  return (
    <svg
      className="diagram-svg"
      viewBox="0 0 540 180"
      role="img"
      aria-label={t("method.ml.rmtpp.figAlt")}
      preserveAspectRatio="xMidYMid meet"
    >
      <ArrowDefs id="rmtpp-arrow" />
      {cells.map((c, i) => (
        <g key={c.x}>
          {/* input token */}
          <rect x={c.x} y={cy - 22} width={96} height={44} rx={8} className="diagram-box" />
          <text x={c.x + 48} y={cy + 5} className="diagram-box-text" textAnchor="middle">
            {c.label}
          </text>
          {/* recurrent arrow to next cell */}
          {i < cells.length - 1 ? (
            <line
              x1={c.x + 96}
              y1={cy}
              x2={cells[i + 1]!.x}
              y2={cy}
              className="diagram-arrow"
              markerEnd="url(#rmtpp-arrow)"
            />
          ) : null}
          {/* state emission downward */}
          <line
            x1={c.x + 48}
            y1={cy + 22}
            x2={c.x + 48}
            y2={cy + 56}
            className="diagram-arrow"
            markerEnd="url(#rmtpp-arrow)"
          />
        </g>
      ))}
      <text x={cells[1]!.x + 48} y={cy - 34} className="diagram-tick" textAnchor="middle">
        {t("method.ml.rmtpp.figRnn")}
      </text>
      {/* intensity readout box */}
      <rect x={120} y={cy + 56} width={300} height={40} rx={8} className="diagram-box accent" />
      <text x={270} y={cy + 81} className="diagram-box-text accent" textAnchor="middle">
        λ*(t) = exp(vᵀhⱼ + w(t−tⱼ) + b)
      </text>
    </svg>
  );
}

/**
 * Neural Hawkes: a continuous-time LSTM whose cell state decays between events; intensity is a
 * softplus of the (decaying) hidden state, so past events can also *inhibit* future intensity.
 */
function NeuralHawkesDiagram() {
  const { t } = useTranslation();
  const x0 = 50;
  const x1 = 510;
  const baseY = 120;
  const topY = 30;
  const events = [120, 250, 360];
  const pts: string[] = [];
  for (let x = x0; x <= x1; x += 4) {
    // continuous-time decaying state: jumps up at events, decays toward a baseline that can be
    // pulled DOWN after the third event (inhibition) to show the NHP-only behaviour.
    let s = 0.3;
    if (x >= (events[0] ?? 0)) s += 0.55 * Math.exp(-(x - (events[0] ?? 0)) / 60);
    if (x >= (events[1] ?? 0)) s += 0.5 * Math.exp(-(x - (events[1] ?? 0)) / 60);
    if (x >= (events[2] ?? 0)) s -= 0.32 * Math.exp(-(x - (events[2] ?? 0)) / 70); // inhibition
    const y = baseY - Math.max(0.04, Math.min(s, 1.0)) * (baseY - topY);
    pts.push(`${x},${y.toFixed(1)}`);
  }
  return (
    <svg
      className="diagram-svg"
      viewBox="0 0 540 165"
      role="img"
      aria-label={t("method.ml.nhp.figAlt")}
      preserveAspectRatio="xMidYMid meet"
    >
      <ArrowDefs id="nhp-arrow" />
      <line x1={x0} y1={baseY} x2={x1 + 8} y2={baseY} className="diagram-axis" markerEnd="url(#nhp-arrow)" />
      <line x1={x0} y1={baseY} x2={x0} y2={topY - 6} className="diagram-axis" markerEnd="url(#nhp-arrow)" />
      <text x={x0 - 8} y={topY - 2} className="diagram-axis-label" textAnchor="end">
        λ*(t)
      </text>
      <polyline points={pts.join(" ")} className="diagram-curve" />
      {events.map((e, i) => (
        <line key={e} x1={e} y1={baseY} x2={e} y2={baseY + 11} className={i === 2 ? "diagram-event inhibit" : "diagram-event"} />
      ))}
      <text x={(events[2] ?? 0)} y={baseY + 26} className="diagram-tick inhibit" textAnchor="middle">
        {t("method.ml.nhp.figInhibit")}
      </text>
    </svg>
  );
}

/**
 * Transformer / Self-Attentive Hawkes: each event attends over all prior events (no RNN
 * memory decay); the attended representation parameterizes the continuous intensity.
 */
function AttentionHawkesDiagram() {
  const { t } = useTranslation();
  const xs = [80, 200, 320, 440];
  const yTok = 130;
  const yQuery = 50;
  return (
    <svg
      className="diagram-svg"
      viewBox="0 0 540 170"
      role="img"
      aria-label={t("method.ml.thp.figAlt")}
      preserveAspectRatio="xMidYMid meet"
    >
      <ArrowDefs id="thp-arrow" />
      {/* the query event (last) attends over all prior tokens */}
      <rect x={xs[3]! - 30} y={yQuery - 18} width={60} height={36} rx={8} className="diagram-box accent" />
      <text x={xs[3]!} y={yQuery + 5} className="diagram-box-text accent" textAnchor="middle">
        h(tᵢ)
      </text>
      {xs.map((x, i) => (
        <g key={x}>
          <rect x={x - 26} y={yTok - 16} width={52} height={32} rx={7} className="diagram-box" />
          <text x={x} y={yTok + 4} className="diagram-box-text" textAnchor="middle">
            e{i + 1}
          </text>
          {/* attention edge from each token to the query */}
          <line
            x1={x}
            y1={yTok - 16}
            x2={xs[3]!}
            y2={yQuery + 18}
            className="diagram-attn"
            markerEnd="url(#thp-arrow)"
          />
        </g>
      ))}
      <text x={270} y={yTok + 32} className="diagram-tick" textAnchor="middle">
        {t("method.ml.thp.figTokens")}
      </text>
      <text x={xs[3]!} y={yQuery - 26} className="diagram-tick" textAnchor="middle">
        {t("method.ml.thp.figQuery")}
      </text>
    </svg>
  );
}

/**
 * FERN / ETAS-generalizing encoder: a permutation-invariant (Deep-Sets) sum of per-event MLP
 * responses added to a background field — the ETAS skeleton with learned kernels.
 */
function FernDiagram() {
  const { t } = useTranslation();
  const events = [
    { x: 50, y: 36 },
    { x: 50, y: 78 },
    { x: 50, y: 120 },
  ];
  const mlpX = 180;
  const sumX = 350;
  const outX = 470;
  const midY = 78;
  return (
    <svg
      className="diagram-svg"
      viewBox="0 0 560 160"
      role="img"
      aria-label={t("method.ml.earthquakes.figAlt")}
      preserveAspectRatio="xMidYMid meet"
    >
      <ArrowDefs id="fern-arrow" />
      {events.map((e, i) => (
        <g key={i}>
          <rect x={e.x} y={e.y - 14} width={86} height={28} rx={7} className="diagram-box" />
          <text x={e.x + 43} y={e.y + 4} className="diagram-box-text" textAnchor="middle">
            (Δt,r,mᵢ)
          </text>
          <line x1={e.x + 86} y1={e.y} x2={mlpX} y2={e.y} className="diagram-arrow" markerEnd="url(#fern-arrow)" />
        </g>
      ))}
      {/* shared MLP */}
      <rect x={mlpX} y={20} width={70} height={116} rx={9} className="diagram-box accent" />
      <text x={mlpX + 35} y={midY + 4} className="diagram-box-text accent" textAnchor="middle">
        MLP
      </text>
      <text x={mlpX + 35} y={150} className="diagram-tick" textAnchor="middle">
        {t("method.ml.earthquakes.figShared")}
      </text>
      {/* sum node (permutation-invariant) */}
      <line x1={mlpX + 70} y1={midY} x2={sumX - 22} y2={midY} className="diagram-arrow" markerEnd="url(#fern-arrow)" />
      <circle cx={sumX} cy={midY} r={22} className="diagram-box" />
      <text x={sumX} y={midY + 7} className="diagram-sum" textAnchor="middle">
        Σ
      </text>
      <text x={sumX} y={midY + 44} className="diagram-tick" textAnchor="middle">
        + μ(x,y)
      </text>
      {/* output */}
      <line x1={sumX + 22} y1={midY} x2={outX - 4} y2={midY} className="diagram-arrow" markerEnd="url(#fern-arrow)" />
      <text x={outX + 30} y={midY + 5} className="diagram-box-text accent" textAnchor="middle">
        λ(x,y,t)
      </text>
    </svg>
  );
}

/**
 * The DeVries-vs-Mignan cautionary tale: a 12-input → 6×50 deep net (≈13.5k params, AUC 0.85)
 * beside a 2-parameter "one neuron" logistic regression that matches it.
 */
function DeepVsOneNeuronDiagram() {
  const { t } = useTranslation();
  return (
    <svg
      className="diagram-svg"
      viewBox="0 0 560 220"
      role="img"
      aria-label={t("method.ml.cnn.figAlt")}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* LEFT: deep net (over-parameterized) */}
      <text x={140} y={20} className="diagram-tick" textAnchor="middle">
        {t("method.ml.cnn.figDeep")}
      </text>
      {/* input column */}
      {[0, 1, 2, 3].map((i) => (
        <circle key={`in${i}`} cx={50} cy={45 + i * 34} r={6} className="diagram-node" />
      ))}
      {/* two hidden columns (sampled) */}
      {[0, 1, 2, 3, 4].map((i) => (
        <circle key={`h1${i}`} cx={130} cy={36 + i * 30} r={6} className="diagram-node" />
      ))}
      {[0, 1, 2, 3, 4].map((i) => (
        <circle key={`h2${i}`} cx={210} cy={36 + i * 30} r={6} className="diagram-node" />
      ))}
      <circle cx={272} cy={96} r={6} className="diagram-node accent" />
      {/* dense edges (sampled) */}
      {[0, 1, 2, 3].map((i) =>
        [0, 1, 2, 3, 4].map((j) => (
          <line
            key={`e1${i}${j}`}
            x1={56}
            y1={45 + i * 34}
            x2={124}
            y2={36 + j * 30}
            className="diagram-edge-faint"
          />
        )),
      )}
      {[0, 1, 2, 3, 4].map((i) =>
        [0, 1, 2, 3, 4].map((j) => (
          <line
            key={`e2${i}${j}`}
            x1={136}
            y1={36 + i * 30}
            x2={204}
            y2={36 + j * 30}
            className="diagram-edge-faint"
          />
        )),
      )}
      <text x={140} y={205} className="diagram-tick bad" textAnchor="middle">
        ≈13.5k params · AUC 0.85
      </text>

      {/* divider */}
      <line x1={320} y1={30} x2={320} y2={200} className="diagram-guide" />

      {/* RIGHT: one neuron */}
      <text x={440} y={20} className="diagram-tick" textAnchor="middle">
        {t("method.ml.cnn.figOne")}
      </text>
      <circle cx={390} cy={110} r={6} className="diagram-node" />
      <text x={390} y={134} className="diagram-tick" textAnchor="middle">
        x
      </text>
      <line x1={396} y1={110} x2={444} y2={110} className="diagram-arrow" markerEnd="url(#oneneuron-arrow)" />
      <ArrowDefs id="oneneuron-arrow" />
      <circle cx={460} cy={110} r={16} className="diagram-node accent" />
      <text x={460} y={114} className="diagram-sum" textAnchor="middle">
        σ
      </text>
      <text x={460} y={150} className="diagram-tick" textAnchor="middle">
        p = σ(wx+b)
      </text>
      <text x={440} y={205} className="diagram-tick good" textAnchor="middle">
        2 params · AUC ≈ 0.85
      </text>
    </svg>
  );
}

/**
 * Detection ≠ forecasting: a one-way pipeline from waveforms → ML detector → richer catalog →
 * the (separate) point-process forecaster, with a hard "not forecasting" stamp on the detector.
 */
function DetectionVsForecastDiagram() {
  const { t } = useTranslation();
  const y = 70;
  return (
    <svg
      className="diagram-svg"
      viewBox="0 0 580 150"
      role="img"
      aria-label={t("method.ml.detection.figAlt")}
      preserveAspectRatio="xMidYMid meet"
    >
      <ArrowDefs id="det-arrow" />
      {/* waveforms */}
      <rect x={20} y={y - 26} width={110} height={52} rx={9} className="diagram-box" />
      <text x={75} y={y + 5} className="diagram-box-text" textAnchor="middle">
        {t("method.ml.detection.figWave")}
      </text>
      <line x1={130} y1={y} x2={168} y2={y} className="diagram-arrow" markerEnd="url(#det-arrow)" />
      {/* detector (NOT forecasting) */}
      <rect x={170} y={y - 30} width={140} height={60} rx={9} className="diagram-box accent" />
      <text x={240} y={y - 2} className="diagram-box-text accent" textAnchor="middle">
        {t("method.ml.detection.figDetect")}
      </text>
      <text x={240} y={y + 16} className="diagram-tick bad" textAnchor="middle">
        {t("method.ml.detection.figNotForecast")}
      </text>
      <line x1={310} y1={y} x2={348} y2={y} className="diagram-arrow" markerEnd="url(#det-arrow)" />
      {/* richer catalog */}
      <rect x={350} y={y - 26} width={110} height={52} rx={9} className="diagram-box" />
      <text x={405} y={y + 5} className="diagram-box-text" textAnchor="middle">
        {t("method.ml.detection.figCatalog")}
      </text>
      <line x1={460} y1={y} x2={498} y2={y} className="diagram-arrow" markerEnd="url(#det-arrow)" />
      {/* forecaster */}
      <rect x={500} y={y - 26} width={70} height={52} rx={9} className="diagram-box good" />
      <text x={535} y={y + 1} className="diagram-box-text" textAnchor="middle">
        TPP /
      </text>
      <text x={535} y={y + 15} className="diagram-box-text" textAnchor="middle">
        ETAS
      </text>
    </svg>
  );
}

/* ──────────────────────────────────────────────────────────────────────────────
   Sub-tab content panels.
   ────────────────────────────────────────────────────────────────────────────── */

function PanelTpp() {
  const { t } = useTranslation();
  const assumptions = t("method.ml.tpp.assumptions", { returnObjects: true }) as string[];
  return (
    <div className="prose">
      <MethodHead icon={<Activity size={18} />} title={t("method.ml.tpp.title")} />
      <p>
        <Trans i18nKey="method.ml.tpp.body" components={{ b: <strong /> }} />
      </p>
      <BlockEquation
        math={String.raw`\lambda^{*}(t, x, y, m \mid \mathcal{H}_t) = \lim_{\Delta \to 0} \frac{\mathbb{E}\!\left[N\big([t, t+\Delta) \times dx\,dy\,dm\big) \mid \mathcal{H}_t\right]}{\Delta\,dx\,dy\,dm}`}
        caption={t("method.ml.tpp.capIntensity")}
      />
      <p>
        <Trans i18nKey="method.ml.tpp.likelihood" components={{ b: <strong /> }} />
      </p>
      <BlockEquation
        math={String.raw`\log \mathcal{L} = \sum_{i=1}^{n} \log \lambda^{*}(t_i) \;-\; \underbrace{\int_{0}^{T} \lambda^{*}(\tau)\, d\tau}_{\text{compensator / survival}}`}
        caption={t("method.ml.tpp.capLikelihood")}
      />
      <Figure
        title={t("method.ml.tpp.figTitle")}
        caption={t("method.ml.tpp.figCaption")}
        id="fig-tpp"
      >
        <ConditionalIntensityDiagram />
      </Figure>
      <AssumptionList titleKey="method.ml.tpp.assumptionsTitle" items={assumptions} />
      <Callout tone="strong" title={t("method.ml.tpp.calloutTitle")}>
        <Trans
          i18nKey="method.ml.tpp.callout"
          components={{ b: <strong />, cite: <Cite id="ogata1998" /> }}
        />
      </Callout>
      <p className="method-refs">
        <Trans
          i18nKey="method.ml.tpp.refs"
          components={{ c1: <Cite id="ogata1988" paren={false} />, c2: <Cite id="ogata1998" paren={false} /> }}
        />
      </p>
    </div>
  );
}

function PanelRmtpp() {
  const { t } = useTranslation();
  const assumptions = t("method.ml.rmtpp.assumptions", { returnObjects: true }) as string[];
  return (
    <div className="prose">
      <MethodHead icon={<Waves size={18} />} title={t("method.ml.rmtpp.title")} />
      <p>
        <Trans i18nKey="method.ml.rmtpp.body" components={{ b: <strong /> }} />
      </p>
      <BlockEquation
        math={String.raw`\lambda^{*}(t) = \exp\!\big(\mathbf{v}^{\top} \mathbf{h}_j + w\,(t - t_j) + b\big), \qquad t \in (t_j, t_{j+1}]`}
        caption={t("method.ml.rmtpp.capIntensity")}
      />
      <Figure title={t("method.ml.rmtpp.figTitle")} caption={t("method.ml.rmtpp.figCaption")} id="fig-rmtpp">
        <RmtppDiagram />
      </Figure>
      <AssumptionList titleKey="method.ml.rmtpp.assumptionsTitle" items={assumptions} />
      <p className="method-refs">
        <Trans i18nKey="method.ml.rmtpp.refs" components={{ c1: <Cite id="du2016" paren={false} /> }} />
      </p>
    </div>
  );
}

function PanelNeuralHawkes() {
  const { t } = useTranslation();
  const assumptions = t("method.ml.nhp.assumptions", { returnObjects: true }) as string[];
  return (
    <div className="prose">
      <MethodHead icon={<Brain size={18} />} title={t("method.ml.nhp.title")} />
      <p>
        <Trans i18nKey="method.ml.nhp.body" components={{ b: <strong /> }} />
      </p>
      <BlockEquation
        math={String.raw`\lambda^{*}(t) = \mathrm{softplus}\!\big(\mathbf{v}^{\top} \mathbf{h}(t)\big), \qquad \mathbf{h}(t)\ \text{decays continuously between events}`}
        caption={t("method.ml.nhp.capIntensity")}
      />
      <Figure title={t("method.ml.nhp.figTitle")} caption={t("method.ml.nhp.figCaption")} id="fig-nhp">
        <NeuralHawkesDiagram />
      </Figure>
      <AssumptionList titleKey="method.ml.nhp.assumptionsTitle" items={assumptions} />
      <p className="method-refs">
        <Trans i18nKey="method.ml.nhp.refs" components={{ c1: <Cite id="meiEisner2017" paren={false} /> }} />
      </p>
    </div>
  );
}

function PanelTransformerHawkes() {
  const { t } = useTranslation();
  const assumptions = t("method.ml.thp.assumptions", { returnObjects: true }) as string[];
  return (
    <div className="prose">
      <MethodHead icon={<Layers size={18} />} title={t("method.ml.thp.title")} />
      <p>
        <Trans i18nKey="method.ml.thp.body" components={{ b: <strong /> }} />
      </p>
      <BlockEquation
        math={String.raw`\lambda^{*}(t) = \mathrm{softplus}\!\Big( \alpha\, \tfrac{t - t_i}{t_i} + \mathbf{w}^{\top} \mathbf{h}(t_i) + b \Big), \quad t \in (t_i, t_{i+1}]`}
        caption={t("method.ml.thp.capIntensity")}
      />
      <Callout tone="honest" title={t("method.ml.thp.caveatTitle")}>
        <Trans i18nKey="method.ml.thp.caveat" components={{ b: <strong /> }} />
      </Callout>
      <Figure title={t("method.ml.thp.figTitle")} caption={t("method.ml.thp.figCaption")} id="fig-thp">
        <AttentionHawkesDiagram />
      </Figure>
      <AssumptionList titleKey="method.ml.thp.assumptionsTitle" items={assumptions} />
      <p className="method-refs">
        <Trans
          i18nKey="method.ml.thp.refs"
          components={{
            c1: <Cite id="zhang2020sahp" paren={false} />,
            c2: <Cite id="zuo2020thp" paren={false} />,
          }}
        />
      </p>
    </div>
  );
}

function PanelEarthquakes() {
  const { t } = useTranslation();
  const assumptions = t("method.ml.earthquakes.assumptions", { returnObjects: true }) as string[];
  return (
    <div className="prose">
      <MethodHead icon={<Network size={18} />} title={t("method.ml.earthquakes.title")} />
      <p>
        <Trans i18nKey="method.ml.earthquakes.body" components={{ b: <strong /> }} />
      </p>

      <h4 className="method-subhead">{t("method.ml.earthquakes.fernTitle")}</h4>
      <p>
        <Trans
          i18nKey="method.ml.earthquakes.fern"
          components={{ b: <strong />, cite: <Cite id="zlydenko2023" /> }}
        />
      </p>
      <BlockEquation
        math={String.raw`\lambda(x, y, t \mid \mathcal{H}_t) = \mu(x, y) + \sum_{i} \underbrace{T(t - t_i)\, S(x - x_i, y - y_i; M_i)}_{\text{fixed ETAS kernels} \;\to\; \text{learned MLPs}}`}
        caption={t("method.ml.earthquakes.capFern")}
      />
      <Figure
        title={t("method.ml.earthquakes.figTitle")}
        caption={t("method.ml.earthquakes.figCaption")}
        id="fig-fern"
      >
        <FernDiagram />
      </Figure>

      <h4 className="method-subhead">{t("method.ml.earthquakes.recastTitle")}</h4>
      <p>
        <Trans
          i18nKey="method.ml.earthquakes.recast"
          components={{ b: <strong />, cite: <Cite id="dascher2023" /> }}
        />
      </p>

      <h4 className="method-subhead">{t("method.ml.earthquakes.benchTitle")}</h4>
      <Callout tone="strong" title={t("method.ml.earthquakes.benchCalloutTitle")}>
        <Trans
          i18nKey="method.ml.earthquakes.bench"
          components={{ b: <strong />, cite: <Cite id="stockman2026" /> }}
        />
      </Callout>
      <Callout tone="honest" title={t("method.ml.earthquakes.leakageTitle")}>
        <Trans i18nKey="method.ml.earthquakes.leakage" components={{ b: <strong /> }} />
      </Callout>

      <AssumptionList titleKey="method.ml.earthquakes.assumptionsTitle" items={assumptions} />
      <p className="method-refs">
        <Trans
          i18nKey="method.ml.earthquakes.refs"
          components={{
            c1: <Cite id="zlydenko2023" paren={false} />,
            c2: <Cite id="dascher2023" paren={false} />,
            c3: <Cite id="stockman2026" paren={false} />,
            c4: <Cite id="schultz2026" paren={false} />,
          }}
        />
      </p>
    </div>
  );
}

function PanelCnn() {
  const { t } = useTranslation();
  const assumptions = t("method.ml.cnn.assumptions", { returnObjects: true }) as string[];
  return (
    <div className="prose">
      <MethodHead icon={<AlertTriangle size={18} />} title={t("method.ml.cnn.title")} />
      <p>
        <Trans
          i18nKey="method.ml.cnn.body"
          components={{ b: <strong />, cite: <Cite id="devries2018" /> }}
        />
      </p>
      <Callout tone="honest" title={t("method.ml.cnn.rebuttalTitle")}>
        <Trans
          i18nKey="method.ml.cnn.rebuttal"
          components={{ b: <strong />, cite: <Cite id="mignanBroccardo2019" /> }}
        />
      </Callout>
      <BlockEquation
        math={String.raw`p = \sigma(w\,x + b) = \frac{1}{1 + e^{-(w x + b)}} \qquad (\text{two free parameters } w, b)`}
        caption={t("method.ml.cnn.capOneNeuron")}
      />
      <Figure title={t("method.ml.cnn.figTitle")} caption={t("method.ml.cnn.figCaption")} id="fig-cnn">
        <DeepVsOneNeuronDiagram />
      </Figure>
      <AssumptionList titleKey="method.ml.cnn.assumptionsTitle" items={assumptions} />
      <Callout tone="strong" title={t("method.ml.cnn.aucTitle")}>
        <Trans i18nKey="method.ml.cnn.auc" components={{ b: <strong /> }} />
      </Callout>
      <p className="method-refs">
        <Trans
          i18nKey="method.ml.cnn.refs"
          components={{
            c1: <Cite id="devries2018" paren={false} />,
            c2: <Cite id="mignanBroccardo2019" paren={false} />,
            c3: <Cite id="devriesReply2019" paren={false} />,
          }}
        />
      </p>
    </div>
  );
}

function PanelGnnRnn() {
  const { t } = useTranslation();
  const assumptions = t("method.ml.gnn.assumptions", { returnObjects: true }) as string[];
  return (
    <div className="prose">
      <MethodHead icon={<GitGraph size={18} />} title={t("method.ml.gnn.title")} />

      <h4 className="method-subhead">{t("method.ml.gnn.rnnTitle")}</h4>
      <p>
        <Trans i18nKey="method.ml.gnn.rnn" components={{ b: <strong /> }} />
      </p>

      <h4 className="method-subhead">{t("method.ml.gnn.gnnTitle")}</h4>
      <p>
        <Trans i18nKey="method.ml.gnn.gnn" components={{ b: <strong /> }} />
      </p>
      <AssumptionList titleKey="method.ml.gnn.assumptionsTitle" items={assumptions} />
      <Callout tone="honest" title={t("method.ml.gnn.calloutTitle")}>
        <Trans i18nKey="method.ml.gnn.callout" components={{ b: <strong /> }} />
      </Callout>
    </div>
  );
}

function PanelDetection() {
  const { t } = useTranslation();
  const tools = t("method.ml.detection.tools", { returnObjects: true }) as {
    name: string;
    task: string;
  }[];
  return (
    <div className="prose">
      <MethodHead icon={<Radar size={18} />} title={t("method.ml.detection.title")} />
      <Callout tone="strong" title={t("method.ml.detection.lineTitle")}>
        <Trans i18nKey="method.ml.detection.line" components={{ b: <strong /> }} />
      </Callout>
      <p>
        <Trans i18nKey="method.ml.detection.body" components={{ b: <strong /> }} />
      </p>
      <Figure
        title={t("method.ml.detection.figTitle")}
        caption={t("method.ml.detection.figCaption")}
        id="fig-detection"
      >
        <DetectionVsForecastDiagram />
      </Figure>
      <table className="summary-table method-tools">
        <thead>
          <tr>
            <th>{t("method.ml.detection.colTool")}</th>
            <th>{t("method.ml.detection.colTask")}</th>
            <th>{t("method.ml.detection.colForecast")}</th>
          </tr>
        </thead>
        <tbody>
          {tools.map((tool) => (
            <tr key={tool.name}>
              <td className="mono">{tool.name}</td>
              <td>{tool.task}</td>
              <td className="method-tools-no">
                <Binary size={13} aria-hidden="true" /> {t("method.ml.detection.notForecast")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Callout tone="honest" title={t("method.ml.detection.seislmTitle")}>
        <Trans
          i18nKey="method.ml.detection.seislm"
          components={{ b: <strong />, cite: <Cite id="liu2024seislm" /> }}
        />
      </Callout>
      <p className="method-refs">
        <Trans
          i18nKey="method.ml.detection.refs"
          components={{
            c1: <Cite id="zhuBeroza2019" paren={false} />,
            c2: <Cite id="mousavi2020" paren={false} />,
            c3: <Cite id="woollam2022" paren={false} />,
            c4: <Cite id="liu2024seislm" paren={false} />,
          }}
        />
      </p>
    </div>
  );
}

export function MethodologyAnalytical() {
  const { t } = useTranslation();

  const subtabs: SubTabDef[] = [
    { id: "tpp", label: t("method.ml.tpp.tab"), content: <PanelTpp /> },
    { id: "rmtpp", label: t("method.ml.rmtpp.tab"), content: <PanelRmtpp /> },
    { id: "nhp", label: t("method.ml.nhp.tab"), content: <PanelNeuralHawkes /> },
    { id: "thp", label: t("method.ml.thp.tab"), content: <PanelTransformerHawkes /> },
    { id: "earthquakes", label: t("method.ml.earthquakes.tab"), content: <PanelEarthquakes /> },
    { id: "cnn", label: t("method.ml.cnn.tab"), content: <PanelCnn /> },
    { id: "gnn", label: t("method.ml.gnn.tab"), content: <PanelGnnRnn /> },
    { id: "detection", label: t("method.ml.detection.tab"), content: <PanelDetection /> },
  ];

  return (
    <div className="prose">
      <p className="muted">{t("method.ml.intro")}</p>

      <Callout tone="honest" title={t("method.ml.verdictTitle")}>
        <Trans
          i18nKey="method.ml.verdict"
          components={{ cite: <Cite id="stockman2026" />, b: <strong /> }}
        />
      </Callout>

      <SubTabs tabs={subtabs} ariaLabel={t("method.ml.subtabsAria")} orientation="vertical" />
    </div>
  );
}

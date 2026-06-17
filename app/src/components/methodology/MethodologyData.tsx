import { Trans, useTranslation } from "react-i18next";

import { Callout } from "@/components/content/Callout";
import { Cite } from "@/components/content/Cite";
import { BlockEquation, Inline } from "@/components/content/Equation";
import { Figure } from "@/components/content/Figure";
import { SubTabs, type SubTabDef } from "@/components/content/SubTabs";
import type { CitationId } from "@/lib/citations";

/**
 * Methodology — Tab 3: "Data & features".
 *
 * The EXPLICIT data layer Felipe asked for, split into two DEEP sub-tabs:
 *
 *  1. "Sources"           → the catalogs that drive the forecast (USGS ComCat, ISC / ISC-GEM,
 *                           GCMT, EMSC, regional networks) and the global enrichers that may add
 *                           context (Slab2, GEM faults, Bird plates, NGL GNSS, World Stress Map,
 *                           tides) — each with what it gives, its license, and its cadence.
 *  2. "Data types & features" → the catalog record, Mw homogenization, completeness Mc(x,y,t), the
 *                           catalog-derived features (recent rates, ETAS intensities, Zaliapin–
 *                           Ben-Zion η/T/R) and the context covariates as a real feature table.
 *
 * Mapped faithfully from synthesis/data-and-pipelines.md and the model-design feature list. Honest
 * throughout: the catalog is the foundation, every enricher is upside that must earn its place by
 * positive prospective information gain over a catalog-only ETAS baseline. No deterministic
 * prediction, no alarms, public-safe (only canonical sources, no machine paths, no private vault).
 *
 * Copy is i18n (`method.data.*`); citations and equations are language-neutral. Two inline,
 * theme-aware SVG figures reuse the shared `.th-*` palette classes (light/dark automatic).
 */

/* ── A small shared bibliography row, matching the theory tab. ───────────────── */
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

/* ═══════════════════════════════════════════════════════════════════════════
   Figure 1 — the catalog data-flow: raw multi-provider events → clean/dedupe →
   Mc + homogenize to Mw → declustering split → feature store. Reuses the `.th-box`
   / `.th-flow` / `.th-arrowhead` palette so it follows the theme automatically.
   ═══════════════════════════════════════════════════════════════════════════ */
function DataFlowSvg() {
  const { t } = useTranslation();
  const arrow = (x1: number, y1: number, x2: number, y2: number) => (
    <line className="th-flow" x1={x1} y1={y1} x2={x2} y2={y2} markerEnd="url(#data-arrow)" />
  );
  const box = (
    x: number,
    y: number,
    w: number,
    h: number,
    title: string,
    sub: string,
    accent = false,
  ) => (
    <g>
      <rect className={accent ? "th-box th-box-accent" : "th-box"} x={x} y={y} width={w} height={h} rx={8} />
      <text className="th-box-title" x={x + w / 2} y={y + 20} textAnchor="middle">
        {title}
      </text>
      <text className="th-box-sub" x={x + w / 2} y={y + 36} textAnchor="middle">
        {sub}
      </text>
    </g>
  );
  return (
    <svg
      className="th-svg"
      viewBox="0 0 560 280"
      role="img"
      aria-label={t("method.data.sources.flowAlt")}
      preserveAspectRatio="xMidYMid meet"
      style={{ maxWidth: 560 }}
    >
      <defs>
        <marker
          id="data-arrow"
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

      {/* Row 1 — multi-provider raw catalogs */}
      {box(14, 14, 122, 48, "ComCat", t("method.data.sources.flowSpine"))}
      {box(150, 14, 122, 48, "Regional FDSN", t("method.data.sources.flowRegional"))}
      {box(286, 14, 122, 48, "ISC-GEM / GCMT", t("method.data.sources.flowAnchor"))}
      {box(422, 14, 124, 48, "EMSC", t("method.data.sources.flowCross"))}

      {/* converge into clean/dedupe */}
      {arrow(75, 62, 230, 86)}
      {arrow(211, 62, 250, 86)}
      {arrow(347, 62, 290, 86)}
      {arrow(484, 62, 310, 86)}

      {/* Row 2 — clean + dedupe + keep magType */}
      {box(160, 88, 240, 48, t("method.data.sources.flowClean"), "magType " + t("method.data.sources.flowKept"))}
      {arrow(280, 136, 280, 158)}

      {/* Row 3 — Mc + homogenize to Mw */}
      {box(160, 160, 240, 48, t("method.data.sources.flowMcMw"), "Mc(x,y,t) · → Mw (TLS)", true)}

      {/* split into the dual catalogs */}
      {arrow(220, 208, 120, 232)}
      {arrow(340, 208, 440, 232)}

      {/* Row 4 — dual-catalog split */}
      {box(20, 234, 220, 40, t("method.data.sources.flowDecl"), "Gardner–Knopoff → μ(x,y)")}
      {box(320, 234, 226, 40, t("method.data.sources.flowFull"), "ZBZ η/T/R → " + t("method.data.sources.flowFeatures"))}
    </svg>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Figure 2 — the global context feature stack: the catalog foundation at the
   base, each enricher a thinner layer above it, ordered by expected lift, with
   the honest "upside, not foundation" reading. Reuses `.th-field` / `.th-box`.
   ═══════════════════════════════════════════════════════════════════════════ */
function ContextStackSvg() {
  const { t } = useTranslation();
  // bottom (widest, load-bearing) → top (thinnest, most speculative)
  const layers = [
    { label: t("method.data.sources.stackCatalog"), w: 360, accent: true },
    { label: "Slab2", w: 300, accent: false },
    { label: t("method.data.sources.stackFaults"), w: 250, accent: false },
    { label: t("method.data.sources.stackStrain"), w: 200, accent: false },
    { label: t("method.data.sources.stackStress"), w: 150, accent: false },
    { label: t("method.data.sources.stackTides"), w: 110, accent: false },
  ];
  const cx = 210;
  const h = 30;
  const gap = 6;
  const baseY = 14;
  return (
    <svg
      className="th-svg"
      viewBox="0 0 460 250"
      role="img"
      aria-label={t("method.data.sources.stackAlt")}
      preserveAspectRatio="xMidYMid meet"
    >
      {layers.map((layer, i) => {
        const y = baseY + i * (h + gap);
        const x = cx - layer.w / 2;
        return (
          <g key={layer.label}>
            <rect
              className={layer.accent ? "th-box th-box-accent" : "th-box"}
              x={x}
              y={y}
              width={layer.w}
              height={h}
              rx={6}
            />
            <text
              className={i === 0 ? "th-box-title" : "th-box-sub"}
              x={cx}
              y={y + h / 2 + 4}
              textAnchor="middle"
            >
              {layer.label}
            </text>
          </g>
        );
      })}
      {/* the "expected lift" axis on the right */}
      <text className="th-note" x={430} y={baseY + 18} textAnchor="end">
        {t("method.data.sources.stackUpside")}
      </text>
      <text className="th-note" x={430} y={baseY + layers.length * (h + gap) - 2} textAnchor="end">
        {t("method.data.sources.stackBase")}
      </text>
    </svg>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Sub-tab 1 — Sources.
   ═══════════════════════════════════════════════════════════════════════════ */

type SourceRow = {
  name: string;
  role: string;
  gives: string;
  license: string;
  cadence: string;
};

function SourcesTable({ titleKey, rowsKey }: { titleKey: string; rowsKey: string }) {
  const { t } = useTranslation();
  const rows = t(rowsKey, { returnObjects: true }) as SourceRow[];
  return (
    <div className="data-source-block">
      <h4 className="method-subhead">{t(titleKey)}</h4>
      <table className="summary-table data-source-table">
        <thead>
          <tr>
            <th>{t("method.data.sources.colName")}</th>
            <th>{t("method.data.sources.colGives")}</th>
            <th>{t("method.data.sources.colLicense")}</th>
            <th>{t("method.data.sources.colCadence")}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name}>
              <td>
                <span className="data-source-name">{r.name}</span>
                <span className="data-source-role">{r.role}</span>
              </td>
              <td>{r.gives}</td>
              <td>{r.license}</td>
              <td className="mono">{r.cadence}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PanelSources() {
  const { t } = useTranslation();
  return (
    <div className="prose">
      <p>
        <Trans i18nKey="method.data.sources.body" components={{ b: <strong /> }} />
      </p>

      <Callout tone="strong" title={t("method.data.sources.spineTitle")}>
        <Trans i18nKey="method.data.sources.spine" components={{ b: <strong /> }} />
      </Callout>

      <SourcesTable titleKey="method.data.sources.catalogsTitle" rowsKey="method.data.sources.catalogs" />

      <Figure title={t("method.th.figureWord") + " 1"} caption={t("method.data.sources.flowCap")}>
        <DataFlowSvg />
      </Figure>

      <SourcesTable titleKey="method.data.sources.regionalTitle" rowsKey="method.data.sources.regional" />

      <p>
        <Trans i18nKey="method.data.sources.enrichersBody" components={{ b: <strong /> }} />
      </p>
      <SourcesTable titleKey="method.data.sources.enrichersTitle" rowsKey="method.data.sources.enrichers" />

      <Figure title={t("method.th.figureWord") + " 2"} caption={t("method.data.sources.stackCap")}>
        <ContextStackSvg />
      </Figure>

      <Callout tone="honest" title={t("method.data.sources.licenseTitle")}>
        <Trans i18nKey="method.data.sources.license" components={{ b: <strong /> }} />
      </Callout>

      <Refs
        ids={[
          "gutenbergRichter1944",
          "woessnerWiemer2005",
          "helmstetter2007",
          "page2016",
        ]}
      />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Sub-tab 2 — Data types & features.
   ═══════════════════════════════════════════════════════════════════════════ */

type FeatureRow = {
  name: string;
  group: string;
  formula: string;
  meaning: string;
};

function FeatureTable() {
  const { t } = useTranslation();
  const rows = t("method.data.features.table", { returnObjects: true }) as FeatureRow[];
  return (
    <table className="summary-table data-feature-table">
      <thead>
        <tr>
          <th>{t("method.data.features.colGroup")}</th>
          <th>{t("method.data.features.colName")}</th>
          <th>{t("method.data.features.colFormula")}</th>
          <th>{t("method.data.features.colMeaning")}</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.name}>
            <td className="data-feature-group">{r.group}</td>
            <td>{r.name}</td>
            <td className="mono data-feature-formula">{r.formula}</td>
            <td>{r.meaning}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PanelFeatures() {
  const { t } = useTranslation();
  return (
    <div className="prose">
      <p>
        <Trans i18nKey="method.data.features.body" components={{ b: <strong /> }} />
      </p>

      {/* The catalog record */}
      <h4 className="method-subhead">{t("method.data.features.recordTitle")}</h4>
      <p>
        <Trans i18nKey="method.data.features.record" components={{ b: <strong /> }} />
      </p>

      {/* Mw homogenization */}
      <h4 className="method-subhead">{t("method.data.features.mwTitle")}</h4>
      <p>
        <Trans i18nKey="method.data.features.mw" components={{ b: <strong /> }} />
      </p>
      <BlockEquation
        math={String.raw`M_w = a + b\,m_{\text{native}} \quad (\text{total least squares, both axes in error}),
          \qquad \text{anchored on the ISC-GEM / GCMT overlap}`}
        caption={t("method.data.features.mwCap")}
      />

      {/* Completeness Mc */}
      <h4 className="method-subhead">{t("method.data.features.mcTitle")}</h4>
      <p>
        <Trans
          i18nKey="method.data.features.mc"
          components={{ b: <strong />, mc: <Inline math="M_c(x, y, t)" /> }}
        />
      </p>
      <Callout tone="honest" title={t("method.data.features.mcCaveatTitle")}>
        <Trans i18nKey="method.data.features.mcCaveat" components={{ b: <strong /> }} />
      </Callout>

      {/* Declustering as features, not labels */}
      <h4 className="method-subhead">{t("method.data.features.zbzTitle")}</h4>
      <p>
        <Trans i18nKey="method.data.features.zbz" components={{ b: <strong /> }} />
      </p>
      <BlockEquation
        math={String.raw`\eta_{ij} = t_{ij}\,(r_{ij})^{d_f}\,10^{-b\,m_i}, \qquad
          T_j = t_{ij}\,10^{-q b m_i}, \qquad
          R_j = (r_{ij})^{d_f}\,10^{-(1-q) b m_i}, \quad q \approx 0.5`}
        caption={t("method.data.features.zbzCap")}
      />
      <p>
        <Trans i18nKey="method.data.features.zbzBimodal" components={{ b: <strong /> }} />
      </p>

      {/* The real feature table */}
      <h4 className="method-subhead">{t("method.data.features.tableTitle")}</h4>
      <p className="muted">{t("method.data.features.tableIntro")}</p>
      <FeatureTable />

      {/* Context covariates */}
      <h4 className="method-subhead">{t("method.data.features.contextTitle")}</h4>
      <p>
        <Trans i18nKey="method.data.features.context" components={{ b: <strong /> }} />
      </p>

      {/* The hard leakage rule */}
      <Callout tone="strong" title={t("method.data.features.clockTitle")}>
        <Trans i18nKey="method.data.features.clock" components={{ b: <strong /> }} />
      </Callout>

      <Refs
        ids={[
          "wiemerWyss2000",
          "woessnerWiemer2005",
          "ogata1998",
          "zhuang2002",
          "helmstetter2007",
        ]}
      />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   The Tab-3 shell: intro + two horizontal sub-tabs.
   ═══════════════════════════════════════════════════════════════════════════ */

export function MethodologyData() {
  const { t } = useTranslation();

  const tabs: SubTabDef[] = [
    { id: "sources", label: t("method.data.sources.tab"), content: <PanelSources /> },
    { id: "features", label: t("method.data.features.tab"), content: <PanelFeatures /> },
  ];

  return (
    <div className="prose">
      <p className="muted">{t("method.data.intro")}</p>
      <Callout tone="strong" title={t("method.data.creedTitle")}>
        <Trans i18nKey="method.data.creed" components={{ b: <strong /> }} />
      </Callout>
      <SubTabs tabs={tabs} ariaLabel={t("method.data.subtabsAria")} />
    </div>
  );
}

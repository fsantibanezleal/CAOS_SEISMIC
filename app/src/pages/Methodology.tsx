import { useTranslation } from "react-i18next";

import { ReferenceList } from "@/components/content/Cite";
import { Tabs, type TabDef } from "@/components/content/Tabs";
import { MethodologyTheory } from "@/components/methodology/MethodologyTheory";
import { MethodologyAnalytical } from "@/components/methodology/MethodologyAnalytical";
import { MethodologyData } from "@/components/methodology/MethodologyData";
import { MethodologyEmployed } from "@/components/methodology/MethodologyEmployed";
import type { CitationId } from "@/lib/citations";

/**
 * Route 3 — Methodology (web-app-spec.md §4 / synthesis/methodology.md).
 *
 * A three-top-tab shell built on the `Tabs` primitive:
 *
 *  1. "Theoretical approaches"  → <MethodologyTheory/>     — the classical, citable equations.
 *  2. "Analytical / ML methods" → <MethodologyAnalytical/> — point processes + the honest
 *                                                            ML-vs-ETAS verdict.
 *  3. "Data & features"         → <MethodologyData/>       — the catalogs + enrichers (Sources)
 *                                                            and the catalog-derived + context
 *                                                            features the model ingests.
 *  4. "The version employed"    → <MethodologyEmployed/>   — the v0 ETAS-class model that ships.
 *
 * The three components carry the deep per-topic content (rendered with react-katex equations,
 * real citations, and the content primitives); this file only owns the tab shell, the page
 * header, and the consolidated reference list. Tab titles come from i18n `method.tabs3.*`.
 */

const METHOD_REFS: CitationId[] = [
  "gutenbergRichter1944",
  "aki1965",
  "tintiMulargia1987",
  "wiemerWyss2000",
  "woessnerWiemer2005",
  "utsu1995",
  "ogata1983",
  "ogata1988",
  "ogata1998",
  "zhuang2002",
  "reasenbergJones1989",
  "reasenbergJones1994",
  "page2016",
  "gerstenberger2005",
  "rhoadesEvison2004",
  "helmstetter2007",
  "matthews2002",
  "schwartzCoppersmith1984",
  "kingSteinLin1994",
  "stein1999",
  "dieterich1994",
  "heimissonSegall2018",
  "stockman2026",
  "dascher2023",
];

export default function Methodology() {
  const { t } = useTranslation();

  const tabs: TabDef[] = [
    { id: "theory", label: t("method.tabs3.theory"), content: <MethodologyTheory /> },
    { id: "analytical", label: t("method.tabs3.analytical"), content: <MethodologyAnalytical /> },
    { id: "data", label: t("method.tabs3.data"), content: <MethodologyData /> },
    { id: "employed", label: t("method.tabs3.employed"), content: <MethodologyEmployed /> },
  ];

  return (
    <article className="page-body prose">
      <header className="page-head">
        <h1>{t("method.title")}</h1>
        <p className="lede">{t("method.lede")}</p>
      </header>

      <Tabs tabs={tabs} ariaLabel={t("method.title")} />

      <ReferenceList ids={METHOD_REFS} heading={t("common.references")} />
    </article>
  );
}

import { Trans, useTranslation } from "react-i18next";

import { Callout } from "@/components/content/Callout";
import { Cite, ReferenceList } from "@/components/content/Cite";
import { Inline } from "@/components/content/Equation";
import { PipelineDiagram } from "@/components/content/PipelineDiagram";
import type { CitationId } from "@/lib/citations";

/**
 * Route 4 — Implementation (web-app-spec.md §5). The v0 model description + the inline SVG
 * flow diagram of the daily offline pipeline.
 *
 * Model (v0):
 *  - mandatory null: stationary smoothed-seismicity Poisson reference on a declustered catalog;
 *  - primary estimator + reference: ML space–time ETAS (Ogata 1998); R-J is a transparent
 *    fallback / sanity check;
 *  - dual-catalog rule (declustered → background; full → conditional/ETAS);
 *  - data hygiene (ordered): time-varying Mc → magnitude homogenization to Mw → declustering;
 *  - incompleteness-aware likelihood for the hours–days-after-a-large-event window (required,
 *    not optional);
 *  - gated ML challengers ship ONLY on positive, significant pseudo-prospective information
 *    gain over ETAS in CSEP tests; detection ≠ forecasting.
 *
 * Copy lives in i18n under `impl.*`; the diagram component carries its own translated labels.
 */

const IMPL_REFS: CitationId[] = [
  "ogata1998",
  "reasenbergJones1989",
  "page2016",
  "helmstetter2007",
  "wiemerWyss2000",
  "stockman2026",
  "dascher2023",
  "savran2022",
];

export default function Implementation() {
  const { t } = useTranslation();

  const hygieneSteps = t("impl.hygiene.steps", { returnObjects: true }) as string[];

  return (
    <article className="page-body prose">
      <header className="page-head">
        <h1>{t("impl.title")}</h1>
        <p className="lede">{t("impl.lede")}</p>
      </header>

      <section>
        <h2>{t("impl.model.title")}</h2>

        <div className="def-grid">
          <div className="def">
            <h3>{t("impl.model.null.title")}</h3>
            <p>{t("impl.model.null.body")}</p>
          </div>
          <div className="def">
            <h3>{t("impl.model.primary.title")}</h3>
            <p>
              <Trans
                i18nKey="impl.model.primary.body"
                components={{ cite: <Cite id="ogata1998" />, citeRJ: <Cite id="reasenbergJones1989" /> }}
              />
            </p>
          </div>
          <div className="def">
            <h3>{t("impl.model.background.title")}</h3>
            <p>
              <Trans i18nKey="impl.model.background.body" components={{ cite: <Cite id="helmstetter2007" /> }} />
            </p>
          </div>
          <div className="def">
            <h3>{t("impl.model.magnitude.title")}</h3>
            <p>
              <Trans
                i18nKey="impl.model.magnitude.body"
                components={{ mc: <Inline math="M_c" />, b: <Inline math="b" /> }}
              />
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2>{t("impl.dualCatalog.title")}</h2>
        <p>
          <Trans i18nKey="impl.dualCatalog.body" components={{ b: <strong /> }} />
        </p>
      </section>

      <section>
        <h2>{t("impl.hygiene.title")}</h2>
        <p className="muted">{t("impl.hygiene.intro")}</p>
        <ol className="ordered-steps">
          {hygieneSteps.map((step, i) => (
            <li key={i}>{step}</li>
          ))}
        </ol>
        <Callout tone="note" title={t("impl.hygiene.mcNoteTitle")}>
          <Trans i18nKey="impl.hygiene.mcNote" components={{ cite: <Cite id="wiemerWyss2000" /> }} />
        </Callout>
      </section>

      <section>
        <h2>{t("impl.incompleteness.title")}</h2>
        <Callout tone="honest">
          <Trans i18nKey="impl.incompleteness.body" components={{ mc: <Inline math="M_c(t)" />, b: <strong /> }} />
        </Callout>
      </section>

      <section>
        <h2>{t("impl.gated.title")}</h2>
        <p>
          <Trans
            i18nKey="impl.gated.body"
            components={{ cite: <Cite id="dascher2023" />, citeNPP: <Cite id="stockman2026" />, b: <strong /> }}
          />
        </p>
        <Callout tone="note">{t("impl.gated.detectionNote")}</Callout>
      </section>

      <section>
        <h2>{t("impl.pipeline.title")}</h2>
        <p className="muted">{t("impl.pipeline.intro")}</p>
        <PipelineDiagram />
      </section>

      <ReferenceList ids={IMPL_REFS} heading={t("common.references")} />
    </article>
  );
}

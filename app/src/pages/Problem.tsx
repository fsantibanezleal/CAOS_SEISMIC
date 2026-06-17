import { Trans, useTranslation } from "react-i18next";

import { Callout } from "@/components/content/Callout";
import { Cite, ReferenceList } from "@/components/content/Cite";
import type { CitationId } from "@/lib/citations";

/**
 * Route 2 — The problem. The honest epistemics of earthquake forecasting (web-app-spec.md §3):
 *  - the product creed, verbatim (the load-bearing message of the whole product);
 *  - deterministic prediction is effectively impossible (Geller et al. 1997), with the SOC
 *    grounding (Bak & Tang 1989) framed as a leading explanation, not settled physics;
 *  - the prediction-vs-forecast distinction (ICEF / Jordan et al. 2011) — a forecast is a
 *    probability strictly in (0, 1), never a binary call;
 *  - the honest absolute scale (short-term probabilities typically < 1% per day);
 *  - three teaching cases — Parkfield, Ridgecrest, L'Aquila — each with rendered citations;
 *  - what IS achievable: Operational Earthquake Forecasting (real, deployed, probabilistic).
 *
 * Every supporting point carries a canonical citation rendered on the page via <Cite/>; the
 * full reference block renders at the bottom. Copy lives in i18n under `problem.*`.
 */

/** The citations this page references, in render order, for the reference block. */
const PROBLEM_REFS: CitationId[] = [
  "geller1997",
  "bakTang1989",
  "icef2011",
  "bakunLindh1985",
  "bakun2005",
  "savran2020",
  "reasenbergJones1989",
  "page2016",
  "spassiani2023",
];

export default function Problem() {
  const { t } = useTranslation();

  return (
    <article className="page-body prose">
      <header className="page-head">
        <h1>{t("problem.title")}</h1>
        <p className="lede">{t("problem.lede")}</p>
      </header>

      {/* The creed, verbatim — the product's central honest message. */}
      <Callout tone="strong" title={t("problem.creedTitle")}>
        <span className="creed-verbatim">{t("disclaimer.creed")}</span>
      </Callout>

      <section>
        <h2>{t("problem.determinism.title")}</h2>
        <p>
          <Trans
            i18nKey="problem.determinism.body"
            components={{ cite: <Cite id="geller1997" /> }}
          />
        </p>
        <p>
          <Trans
            i18nKey="problem.determinism.soc"
            components={{ cite: <Cite id="bakTang1989" /> }}
          />
        </p>
      </section>

      <section>
        <h2>{t("problem.predVsForecast.title")}</h2>
        <p>
          <Trans
            i18nKey="problem.predVsForecast.body"
            components={{ cite: <Cite id="icef2011" />, b: <strong /> }}
          />
        </p>
        <Callout tone="note">{t("problem.predVsForecast.uiNote")}</Callout>
      </section>

      <section>
        <h2>{t("problem.scale.title")}</h2>
        <p>
          <Trans
            i18nKey="problem.scale.body"
            components={{ cite: <Cite id="icef2011" />, b: <strong /> }}
          />
        </p>
      </section>

      <section>
        <h2>{t("problem.cases.title")}</h2>
        <p className="muted">{t("problem.cases.intro")}</p>

        <div className="case-grid">
          <article className="card case">
            <h3>{t("problem.cases.parkfield.title")}</h3>
            <p>
              <Trans
                i18nKey="problem.cases.parkfield.body"
                components={{
                  cite1: <Cite id="bakunLindh1985" />,
                  cite2: <Cite id="bakun2005" />,
                }}
              />
            </p>
          </article>

          <article className="card case">
            <h3>{t("problem.cases.ridgecrest.title")}</h3>
            <p>
              <Trans
                i18nKey="problem.cases.ridgecrest.body"
                components={{ cite: <Cite id="savran2020" />, b: <strong /> }}
              />
            </p>
          </article>

          <article className="card case">
            <h3>{t("problem.cases.laquila.title")}</h3>
            <p>
              <Trans i18nKey="problem.cases.laquila.body" components={{ b: <strong /> }} />
            </p>
          </article>
        </div>
      </section>

      <section>
        <h2>{t("problem.achievable.title")}</h2>
        <p>
          <Trans
            i18nKey="problem.achievable.body"
            components={{
              citeRJ: <Cite id="reasenbergJones1989" />,
              citePage: <Cite id="page2016" />,
              citeINGV: <Cite id="spassiani2023" />,
              b: <strong />,
            }}
          />
        </p>
        <Callout tone="honest">{t("problem.achievable.honestNote")}</Callout>
      </section>

      <ReferenceList ids={PROBLEM_REFS} heading={t("common.references")} />
    </article>
  );
}

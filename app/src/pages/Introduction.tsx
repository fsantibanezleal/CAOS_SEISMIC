import { Trans, useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { Callout } from "@/components/content/Callout";

/**
 * Route 1 — Introduction. Above-the-fold framing (web-app-spec.md §2):
 *  - what the product IS (an independent, honest, calibrated research/education tool);
 *  - what it IS NOT (an authoritative civil-protection alarm; a predictor; a "safe" signal);
 *  - that it COMPLEMENTS official Operational Earthquake Forecasting (USGS, INGV, CSN,
 *    GeoNet, JMA), never competes with or replaces it;
 *  - that it emits CONDITIONAL PROBABILITIES (region × magnitude band × horizon), always
 *    shown next to the long-term baseline, with the live reliability diagram as the central
 *    credibility artifact.
 *
 * All copy comes from i18n keys under `intro.*` (EN source of truth + ES mirror). No
 * marketing tone, no deterministic-call language anywhere.
 */
export default function Introduction() {
  const { t } = useTranslation();

  const isList = t("intro.is.items", { returnObjects: true }) as string[];
  const isNotList = t("intro.isNot.items", { returnObjects: true }) as string[];

  return (
    <article className="page-body prose">
      <header className="page-head">
        <h1>{t("intro.title")}</h1>
        <p className="lede">{t("intro.lede")}</p>
      </header>

      <Callout tone="strong" title={t("intro.creedTitle")}>
        {t("disclaimer.creed")}
      </Callout>

      <div className="two-col">
        <section className="card">
          <h2>{t("intro.is.title")}</h2>
          <ul className="tick-list">
            {isList.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </section>

        <section className="card">
          <h2>{t("intro.isNot.title")}</h2>
          <ul className="cross-list">
            {isNotList.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </section>
      </div>

      <section>
        <h2>{t("intro.conditional.title")}</h2>
        <p>{t("intro.conditional.body")}</p>
        <p>
          <Trans
            i18nKey="intro.conditional.example"
            components={{ b: <strong />, code: <code /> }}
          />
        </p>
      </section>

      <section>
        <h2>{t("intro.complement.title")}</h2>
        <p>{t("intro.complement.body")}</p>
      </section>

      <section>
        <h2>{t("intro.credibility.title")}</h2>
        <p>{t("intro.credibility.body")}</p>
        <p>
          <Trans
            i18nKey="intro.credibility.cta"
            components={{
              problem: <Link to="/problem" />,
              backAnalysis: <Link to="/back-analysis" />,
              monitoring: <Link to="/monitoring" />,
            }}
          />
        </p>
      </section>
    </article>
  );
}

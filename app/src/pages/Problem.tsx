import { Trans, useTranslation } from "react-i18next";

import { Callout } from "@/components/content/Callout";
import { Cite, ReferenceList } from "@/components/content/Cite";
import { BlockEquation, Inline } from "@/components/content/Equation";
import { Figure } from "@/components/content/Figure";
import { PredictionVsForecastFigure } from "@/components/problem/PredictionVsForecastFigure";
import { MagnitudeFrequencyFigure } from "@/components/problem/MagnitudeFrequencyFigure";
import { SingleOutcomeFigure } from "@/components/problem/SingleOutcomeFigure";
import type { CitationId } from "@/lib/citations";

/**
 * Route 2 — The problem. The honest epistemics of earthquake forecasting
 * (research/01-problem-and-predictability + synthesis/methodology.md "Honest limits").
 *
 * This is the load-bearing honesty page of the whole product. It is intentionally dense and
 * equation/diagram heavy — a technical workbench, not plain text:
 *
 *  - the product creed, verbatim;
 *  - why deterministic *prediction* is effectively impossible (Geller et al. 1997), grounded in
 *    self-organized criticality / scale-invariance (Bak & Tang 1989) as the LEADING explanation,
 *    not settled physics;
 *  - the empirical scaling laws that DO hold — Gutenberg–Richter (with its log-linear plot) and
 *    the exceedance map P(>=1) = 1 - e^{-N} — and what they do / do not buy us;
 *  - the prediction-vs-forecast split (ICEF / Jordan et al. 2011): a forecast is a probability
 *    strictly in (0, 1), rendered as the contrast figure and a definition grid;
 *  - the honest absolute scale (< 1% per day) with the "always next to baseline" rule;
 *  - three teaching cases — Parkfield, Ridgecrest (the ~3% worked example, as a figure),
 *    L'Aquila — each with rendered citations;
 *  - what IS achievable: real, deployed Operational Earthquake Forecasting.
 *
 * Every supporting point carries a canonical citation via <Cite/>; the full reference block
 * renders at the bottom. Copy lives in i18n under `problem.*`; equations and citations are real
 * (no invented constants).
 */

/** The citations this page references, in render order, for the reference block. */
const PROBLEM_REFS: CitationId[] = [
  "geller1997",
  "bakTang1989",
  "icef2011",
  "ogata1998",
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

      {/* ── Why deterministic prediction is impossible ───────────────────── */}
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
            components={{ cite: <Cite id="bakTang1989" />, b: <strong /> }}
          />
        </p>

        <Figure
          id="fig-soc"
          title={t("problem.fig.mfd.figTitle")}
          caption={
            <Trans
              i18nKey="problem.fig.mfd.caption"
              components={{ cite: <Cite id="bakTang1989" paren={false} /> }}
            />
          }
        >
          <MagnitudeFrequencyFigure />
        </Figure>

        <p>
          <Trans
            i18nKey="problem.determinism.laws"
            components={{ b: <strong /> }}
          />
        </p>
        <BlockEquation
          id="eq-gr"
          math={String.raw`\log_{10} N(\geq M) = a - bM, \qquad b \approx 1`}
          caption={t("problem.eq.gr.caption")}
        />
        <p>
          <Trans
            i18nKey="problem.determinism.lawsBuy"
            components={{ b: <strong /> }}
          />
        </p>
      </section>

      {/* ── Prediction vs forecast (ICEF) ────────────────────────────────── */}
      <section>
        <h2>{t("problem.predVsForecast.title")}</h2>
        <p>
          <Trans
            i18nKey="problem.predVsForecast.body"
            components={{ cite: <Cite id="icef2011" />, b: <strong /> }}
          />
        </p>

        {/* The two ICEF definitions, verbatim-in-spirit, as a def grid. */}
        <div className="def-grid">
          <div className="def">
            <h3>{t("problem.predVsForecast.predDefTitle")}</h3>
            <p>{t("problem.predVsForecast.predDef")}</p>
          </div>
          <div className="def">
            <h3>{t("problem.predVsForecast.foreDefTitle")}</h3>
            <p>{t("problem.predVsForecast.foreDef")}</p>
          </div>
        </div>

        <Figure
          id="fig-pvf"
          title={t("problem.fig.pvf.figTitle")}
          caption={
            <Trans
              i18nKey="problem.fig.pvf.caption"
              components={{ cite: <Cite id="icef2011" paren={false} /> }}
            />
          }
        >
          <PredictionVsForecastFigure />
        </Figure>

        <Callout tone="note">{t("problem.predVsForecast.uiNote")}</Callout>
      </section>

      {/* ── The honest absolute scale ────────────────────────────────────── */}
      <section>
        <h2>{t("problem.scale.title")}</h2>
        <p>
          <Trans
            i18nKey="problem.scale.body"
            components={{ cite: <Cite id="icef2011" />, b: <strong /> }}
          />
        </p>
        <p>
          <Trans i18nKey="problem.scale.exceedance" components={{ b: <strong /> }} />
        </p>
        <BlockEquation
          id="eq-exceedance"
          math={String.raw`P(\geq 1 \text{ event} \geq M^{*}) = 1 - e^{-N}, \qquad N = \mathbb{E}[\text{count} \geq M^{*}]`}
          caption={t("problem.eq.exceedance.caption")}
        />
        <p>
          <Trans
            i18nKey="problem.scale.baselineRule"
            components={{ n: <Inline math="N" />, b: <strong /> }}
          />
        </p>
      </section>

      {/* ── Three teaching cases ─────────────────────────────────────────── */}
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
                  b: <strong />,
                }}
              />
            </p>
            <p className="case-lesson">{t("problem.cases.parkfield.lesson")}</p>
          </article>

          <article className="card case">
            <h3>{t("problem.cases.ridgecrest.title")}</h3>
            <p>
              <Trans
                i18nKey="problem.cases.ridgecrest.body"
                components={{ cite: <Cite id="savran2020" />, b: <strong /> }}
              />
            </p>
            <p className="case-lesson">{t("problem.cases.ridgecrest.lesson")}</p>
          </article>

          <article className="card case">
            <h3>{t("problem.cases.laquila.title")}</h3>
            <p>
              <Trans i18nKey="problem.cases.laquila.body" components={{ b: <strong /> }} />
            </p>
            <p className="case-lesson">{t("problem.cases.laquila.lesson")}</p>
          </article>
        </div>

        {/* The Ridgecrest ~3% worked example, as a figure. */}
        <Figure
          id="fig-single-outcome"
          title={t("problem.fig.soc.figTitle")}
          caption={
            <Trans
              i18nKey="problem.fig.soc.caption"
              components={{ cite: <Cite id="savran2020" paren={false} /> }}
            />
          }
        >
          <SingleOutcomeFigure />
        </Figure>
      </section>

      {/* ── What IS achievable: OEF ──────────────────────────────────────── */}
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

      {/* ── The do / never-do communication rule ─────────────────────────── */}
      <section>
        <h2>{t("problem.communication.title")}</h2>
        <p className="muted">{t("problem.communication.intro")}</p>
        <div className="two-col">
          <div className="card">
            <h3>{t("problem.communication.doTitle")}</h3>
            <ul className="tick-list">
              {(
                t("problem.communication.do", { returnObjects: true }) as string[]
              ).map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
          <div className="card">
            <h3>{t("problem.communication.neverTitle")}</h3>
            <ul className="cross-list">
              {(
                t("problem.communication.never", { returnObjects: true }) as string[]
              ).map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <ReferenceList ids={PROBLEM_REFS} heading={t("common.references")} />
    </article>
  );
}

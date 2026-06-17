import { Trans, useTranslation } from "react-i18next";

import { Callout } from "@/components/content/Callout";
import { Cite, ReferenceList } from "@/components/content/Cite";
import { BlockEquation, Inline } from "@/components/content/Equation";
import { Tabs, type TabDef } from "@/components/content/Tabs";
import type { CitationId } from "@/lib/citations";

/**
 * Route 3 — Methodology (web-app-spec.md §4). Two tabs:
 *
 *  1. "Classical theories" — the real, citable equations underpinning the field, each with
 *     its rendered LaTeX and reference: Gutenberg–Richter, Aki–Utsu b-value MLE, modified
 *     Omori–Utsu, Reasenberg–Jones (extended by Page et al. 2016), the space–time ETAS
 *     conditional intensity (Ogata 1998) with its two stationarity gates, smoothed
 *     seismicity (Helmstetter–Kagan–Jackson 2007), and the exceedance map the legend encodes.
 *
 *  2. "Analytical & ML" — the honest model-class verdict: under prospective CSEP-style
 *     testing, no neural point process has been shown to reliably beat a well-fit ETAS
 *     (EarthquakeNPP, Stockman et al. 2026), with the spatial-test caveat and the
 *     state-dependent, nats-not-bits framing of information gain.
 *
 * Equations are rendered with react-katex (KaTeX). Copy comes from i18n under `method.*`.
 */

const METHOD_REFS: CitationId[] = [
  "aki1965",
  "tintiMulargia1987",
  "utsu1995",
  "ogata1988",
  "ogata1998",
  "reasenbergJones1989",
  "page2016",
  "gerstenberger2005",
  "wiemerWyss2000",
  "woessnerWiemer2005",
  "helmstetter2007",
  "stockman2026",
  "dascher2023",
];

function ClassicalTab() {
  const { t } = useTranslation();
  return (
    <div className="prose">
      <p className="muted">{t("method.classical.intro")}</p>

      {/* Gutenberg–Richter */}
      <section>
        <h3>{t("method.gr.title")}</h3>
        <p>{t("method.gr.body")}</p>
        <BlockEquation
          math={String.raw`\log_{10} N(\geq M) = a - bM, \qquad b \approx 1`}
          caption={
            <>
              {t("method.gr.caption")} — <Cite id="aki1965" paren={false} />
            </>
          }
        />
      </section>

      {/* Aki–Utsu b-value MLE */}
      <section>
        <h3>{t("method.bvalue.title")}</h3>
        <p>
          <Trans
            i18nKey="method.bvalue.body"
            components={{
              mc: <Inline math="M_c" />,
              cite1: <Cite id="wiemerWyss2000" />,
              cite2: <Cite id="tintiMulargia1987" />,
            }}
          />
        </p>
        <BlockEquation
          math={String.raw`\hat b = \dfrac{\log_{10} e}{\bar m - (M_c - \Delta M / 2)}`}
          caption={t("method.bvalue.caption")}
        />
      </section>

      {/* Modified Omori–Utsu */}
      <section>
        <h3>{t("method.omori.title")}</h3>
        <p>{t("method.omori.body")}</p>
        <BlockEquation
          math={String.raw`n(t) = \dfrac{K}{(t + c)^{p}}, \qquad p \approx 1`}
          caption={
            <>
              {t("method.omori.caption")} — <Cite id="utsu1995" paren={false} />
            </>
          }
        />
      </section>

      {/* Reasenberg–Jones / Page */}
      <section>
        <h3>{t("method.rj.title")}</h3>
        <p>
          <Trans
            i18nKey="method.rj.body"
            components={{ cite1: <Cite id="reasenbergJones1989" />, cite2: <Cite id="page2016" /> }}
          />
        </p>
        <BlockEquation
          math={String.raw`\lambda(t, M) = 10^{\,a + b(M_m - M)}\,\dfrac{1}{(t + c)^{p}}, \qquad N = \int \lambda\,dt, \qquad P(\geq 1) = 1 - e^{-N}`}
          caption={t("method.rj.caption")}
        />
      </section>

      {/* Space–time ETAS */}
      <section>
        <h3>{t("method.etas.title")}</h3>
        <p>
          <Trans i18nKey="method.etas.body" components={{ cite: <Cite id="ogata1998" />, b: <strong /> }} />
        </p>
        <BlockEquation
          math={String.raw`\lambda(t, x, y \mid \mathcal{H}_t) = \mu(x,y) + \sum_{i:\,t_i < t} K\,e^{\alpha(M_i - M_0)}\,\bigl(1 + \tfrac{t - t_i}{c}\bigr)^{-p}\,f(x - x_i,\, y - y_i \mid M_i)`}
          caption={t("method.etas.caption")}
        />
        <Callout tone="note" title={t("method.etas.gatesTitle")}>
          <ol>
            <li>
              <Trans
                i18nKey="method.etas.gate1"
                components={{
                  m: <Inline math="\alpha < \beta" />,
                  beta: <Inline math="\beta = b\,\ln 10" />,
                  n: <Inline math="n" />,
                }}
              />
            </li>
            <li>
              <Trans
                i18nKey="method.etas.gate2"
                components={{ n: <Inline math="n < 1" />, nreject: <Inline math="n \geq 1" /> }}
              />
            </li>
          </ol>
        </Callout>
      </section>

      {/* Smoothed seismicity */}
      <section>
        <h3>{t("method.smoothed.title")}</h3>
        <p>
          <Trans i18nKey="method.smoothed.body" components={{ cite: <Cite id="helmstetter2007" /> }} />
        </p>
      </section>

      {/* Exceedance probability — what the legend encodes */}
      <section>
        <h3>{t("method.exceedance.title")}</h3>
        <p>{t("method.exceedance.body")}</p>
        <BlockEquation
          math={String.raw`P(\geq 1 \text{ event} \geq M^{*}) = 1 - e^{-N_{\geq M^{*}}}, \quad N_{\geq M^{*}} = \iint \lambda\,\Phi(M^{*})\,dx\,dy\,dt, \quad \Phi(M^{*}) = 10^{-b(M^{*} - M_c)}`}
          caption={t("method.exceedance.caption")}
        />
      </section>
    </div>
  );
}

function AnalyticalTab() {
  const { t } = useTranslation();
  return (
    <div className="prose">
      <p className="muted">{t("method.ml.intro")}</p>

      <Callout tone="honest" title={t("method.ml.verdictTitle")}>
        <Trans
          i18nKey="method.ml.verdict"
          components={{ cite: <Cite id="stockman2026" />, b: <strong /> }}
        />
      </Callout>

      <section>
        <h3>{t("method.ml.spatial.title")}</h3>
        <p>
          <Trans i18nKey="method.ml.spatial.body" components={{ b: <strong /> }} />
        </p>
      </section>

      <section>
        <h3>{t("method.ml.infogain.title")}</h3>
        <p>
          <Trans i18nKey="method.ml.infogain.body" components={{ b: <strong /> }} />
        </p>
      </section>

      <section>
        <h3>{t("method.ml.requirement.title")}</h3>
        <p>
          <Trans
            i18nKey="method.ml.requirement.body"
            components={{ cite: <Cite id="dascher2023" />, b: <strong /> }}
          />
        </p>
      </section>
    </div>
  );
}

export default function Methodology() {
  const { t } = useTranslation();

  const tabs: TabDef[] = [
    { id: "classical", label: t("method.tabs.classical"), content: <ClassicalTab /> },
    { id: "analytical", label: t("method.tabs.analytical"), content: <AnalyticalTab /> },
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

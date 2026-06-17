import { Trans, useTranslation } from "react-i18next";

import { ArchitectureDiagram } from "@/components/content/ArchitectureDiagram";
import { Callout } from "@/components/content/Callout";
import { Cite } from "@/components/content/Cite";
import { BlockEquation, Inline } from "@/components/content/Equation";

/**
 * Methodology — Tab 3: "The version employed" (model-design.md).
 *
 * This tab fixes WHAT WE ACTUALLY BUILD/SELECT, distinct from the field survey (Tab 1) and the
 * ML survey (Tab 2). It maps model-design.md faithfully:
 *
 *  · the chosen two-layer architecture — region-refit space–time ETAS as the core estimator AND
 *    the mandatory reference, the smoothed-seismicity Poisson null it must beat, the transparent
 *    Reasenberg–Jones fallback, and a gated context-conditioned neural TPP (Hawkes inductive
 *    bias) with a CNN spatial-context encoder — and WHY the CNN is an encoder, not a standalone;
 *  · the target definition — the full conditional magnitude distribution, exceedance at a
 *    region-appropriate M*, the 1d/2d/7d horizons, and a per-region Mmax that bounds the tail;
 *  · the data spine + the static geophysical context covariates the encoder ingests;
 *  · calibration (a release blocker) + real optimistic/expected/pessimistic bounds + cold-start
 *    flooring to the principled background;
 *  · the HONEST MODEL-CLASS VERDICT, rendered here as OUR verdict (not in the theory tabs): no
 *    NPP reliably beats ETAS in CSEP to date, the spatial-test gap, state-dependent info-gain in
 *    nats, and the gate-not-ban rule — the reason v0 ships ETAS-class only.
 *
 * Copy is in i18n under `method.employed.*` and the shared `impl.*` / `method.*` keys; equations
 * are KaTeX; the architecture SVG follows light/dark via the `.arch-*` palette classes. Nothing
 * here is invented and no internal vault / baseline is referenced.
 */
export function MethodologyEmployed() {
  const { t } = useTranslation();

  return (
    <div className="prose">
      <p className="muted">{t("method.employed.intro")}</p>

      {/* ── 1. The chosen architecture ─────────────────────────────────────── */}
      <section>
        <h2>{t("method.employed.arch.title")}</h2>
        <p>
          <Trans
            i18nKey="method.employed.arch.body"
            components={{
              cite: <Cite id="ogata1998" />,
              citeRJ: <Cite id="reasenbergJones1989" />,
              citeH: <Cite id="helmstetter2007" />,
              b: <strong />,
            }}
          />
        </p>

        <ArchitectureDiagram />

        <div className="def-grid">
          <div className="def">
            <h3>{t("method.employed.layer1.title")}</h3>
            <p>
              <Trans
                i18nKey="method.employed.layer1.body"
                components={{ cite: <Cite id="ogata1998" />, b: <strong /> }}
              />
            </p>
          </div>
          <div className="def">
            <h3>{t("method.employed.layer2.title")}</h3>
            <p>
              <Trans
                i18nKey="method.employed.layer2.body"
                components={{ cite: <Cite id="dascher2023" />, b: <strong /> }}
              />
            </p>
          </div>
        </div>

        <BlockEquation
          math={String.raw`\lambda(t, x, y \mid \mathcal{H}_t) = \mu(x,y) + \sum_{i:\,t_i < t} k(m_i)\,g(t - t_i)\,f(x - x_i,\, y - y_i \mid m_i)`}
          caption={t("method.etas.caption")}
        />

        <Callout tone="note" title={t("method.employed.cnnNote.title")}>
          <Trans i18nKey="method.employed.cnnNote.body" components={{ b: <strong /> }} />
        </Callout>
      </section>

      {/* ── 2. Target definition (the binding design decision) ─────────────── */}
      <section>
        <h2>{t("method.employed.target.title")}</h2>
        <p>
          <Trans
            i18nKey="method.employed.target.body"
            components={{ mstar: <Inline math="M^{*}" />, b: <strong /> }}
          />
        </p>
        <BlockEquation
          math={String.raw`P(\geq 1 \text{ event} \geq M^{*}) = 1 - e^{-N_{\geq M^{*}}}, \quad N_{\geq M^{*}} = \iint \lambda\,\Phi(M^{*})\,dx\,dy\,dt, \quad \Phi(M^{*}) = 10^{-b(M^{*} - M_c)}`}
          caption={t("method.employed.target.caption")}
        />
        <div className="def-grid">
          <div className="def">
            <h3>{t("method.employed.target.dist.title")}</h3>
            <p>{t("method.employed.target.dist.body")}</p>
          </div>
          <div className="def">
            <h3>{t("method.employed.target.horizon.title")}</h3>
            <p>{t("method.employed.target.horizon.body")}</p>
          </div>
          <div className="def">
            <h3>{t("method.employed.target.mmax.title")}</h3>
            <p>
              <Trans
                i18nKey="method.employed.target.mmax.body"
                components={{ mmax: <Inline math="M_{\max}" />, b: <strong /> }}
              />
            </p>
          </div>
        </div>
      </section>

      {/* ── 3. Data + context covariates ──────────────────────────────────── */}
      <section>
        <h2>{t("method.employed.data.title")}</h2>
        <p>
          <Trans i18nKey="method.employed.data.body" components={{ b: <strong /> }} />
        </p>
        <ul className="detail-list">
          {(t("method.employed.data.covariates", { returnObjects: true }) as string[]).map((c, i) => (
            <li key={i}>{c}</li>
          ))}
        </ul>
        <p className="muted">{t("method.employed.data.note")}</p>
      </section>

      {/* ── 4. Calibration + bounds ───────────────────────────────────────── */}
      <section>
        <h2>{t("method.employed.calibration.title")}</h2>
        <Callout tone="strong" title={t("method.employed.calibration.blockerTitle")}>
          <Trans
            i18nKey="method.employed.calibration.blockerBody"
            components={{ cite: <Cite id="schneider2022" />, b: <strong /> }}
          />
        </Callout>
        <div className="def-grid">
          <div className="def">
            <h3>{t("method.employed.bounds.title")}</h3>
            <p>
              <Trans
                i18nKey="method.employed.bounds.body"
                components={{ cite: <Cite id="kagan2017" />, b: <strong /> }}
              />
            </p>
          </div>
          <div className="def">
            <h3>{t("method.employed.coldstart.title")}</h3>
            <p>
              <Trans i18nKey="method.employed.coldstart.body" components={{ b: <strong /> }} />
            </p>
          </div>
        </div>
      </section>

      {/* ── 5. The honest model-class verdict (OURS — belongs in this tab) ──── */}
      <section>
        <h2>{t("method.employed.verdict.title")}</h2>
        <Callout tone="honest" title={t("method.employed.verdict.leadTitle")}>
          <Trans
            i18nKey="method.employed.verdict.lead"
            components={{ cite: <Cite id="stockman2026" />, b: <strong /> }}
          />
        </Callout>

        <h3>{t("method.employed.verdict.spatial.title")}</h3>
        <p>
          <Trans i18nKey="method.employed.verdict.spatial.body" components={{ b: <strong /> }} />
        </p>

        <h3>{t("method.employed.verdict.infogain.title")}</h3>
        <p>
          <Trans
            i18nKey="method.employed.verdict.infogain.body"
            components={{ b: <strong /> }}
          />
        </p>
        <BlockEquation
          math={String.raw`I_N(A, B) = \frac{1}{N}\sum_{i=1}^{N}\bigl(\ln \lambda_{A}(k_i) - \ln \lambda_{B}(k_i)\bigr) - \frac{\hat N_A - \hat N_B}{N}`}
          caption={t("method.employed.verdict.infogain.caption")}
        />

        <h3>{t("method.employed.verdict.rule.title")}</h3>
        <p>
          <Trans
            i18nKey="method.employed.verdict.rule.body"
            components={{ cite: <Cite id="dascher2023" />, citeS: <Cite id="serafini2025" />, b: <strong /> }}
          />
        </p>
        <Callout tone="note">{t("method.employed.verdict.detectionNote")}</Callout>
      </section>
    </div>
  );
}

<!-- markdownlint-disable MD013 MD033 -->
# Methodology

> **Honest framing (non-negotiable).** CAOS_SEISMIC produces **conditional probabilistic
> forecasts**, never deterministic predictions. Following the ICEF definition (Jordan et al.,
> 2011), a *prediction* is a deterministic statement that an event *will* or *will not* occur; a
> *forecast* gives a probability strictly between 0 and 1. Every number we publish is a probability
> in $(0, 1)$, scoped to a region, a magnitude band, and a horizon (1 d / 2 d / 7 d), reported with
> an uncertainty band and shown **alongside its long-term baseline**. Short-term probabilities of a
> large event "may vary over orders of magnitude but typically remain low in an absolute sense
> (< 1 % per day)" (Jordan et al., 2011). We never render an alarm, a countdown, a binary yes/no,
> or a "safe" state.

This page is the equation-rich explainer behind the forecast. It has three parts:

1. **Classical theories** — the analytical, physics-informed, and statistical models that define
   the field-standard baseline. Each is given with its governing equation, its parameters, and a
   peer-reviewed reference.
2. **Analytical / ML approaches** — temporal point processes, neural temporal point processes
   (NTPP), and the honest, evidence-grounded verdict on machine learning *vs.* ETAS.
3. **The CSEP evaluation backbone** — the credibility layer: consistency tests, comparative
   information gain, calibration, and the Molchan/ROC view.

**Conventions.** Magnitudes are homogenized to moment magnitude $M_w$ where possible; $M_c$ is the
magnitude of completeness; $b$ is the Gutenberg–Richter slope; $\beta = b\ln 10$; information gains
are reported in **nats** (natural-log units, the CSEP convention) — never bits.

---

## Part 1 — Classical theories

A well-tuned **ETAS** model is the de-facto short-term forecasting baseline that any candidate model
(including every neural model) must be benchmarked against, and beaten in prospective CSEP testing,
before it can claim forecasting skill. The classical stack below is what the system runs in
production for v0.

### 1.1 Gutenberg–Richter — the magnitude–frequency law

The frequency–magnitude distribution of earthquakes is, to first order, a power law. It supplies the
**magnitude term** of every forecast.

$$\log_{10} N(\ge M) = a - b\,M, \qquad M \ge M_c$$

Equivalently the magnitude density above completeness is exponential,

$$f(M) = \beta\, e^{-\beta (M - M_c)}, \qquad \beta = b\ln 10 .$$

The slope $b$ is typically near 1 but **is never hard-coded**. It is estimated by the Aki–Utsu
maximum-likelihood estimator with the Utsu / Tinti–Mulargia binning correction:

$$\hat b = \frac{\log_{10} e}{\bar m - \left(M_c - \tfrac{\Delta M}{2}\right)},$$

where $\bar m$ is the mean magnitude of events $\ge M_c$ and $\Delta M$ the magnitude bin width. The
estimator is **strongly biased if $M_c$ is mis-estimated**, so $M_c$ and $b$ are re-estimated on a
rolling space–time window and their uncertainty is propagated into the forecast.

- **Parameters:** $a$ (productivity / rate), $b$ (slope), $M_c$ (completeness), $\Delta M$ (bin).
- **References:** Aki (1965), *Bull. Earthq. Res. Inst.* 43, 237–239 (MLE);
  Tinti & Mulargia (1987), *BSSA* 77(6), 2125–2134 (binning correction);
  Wiemer & Wyss (2000), *BSSA* 90(4), 859–869 (doi:[10.1785/0119990114](https://doi.org/10.1785/0119990114)).

### 1.2 Omori–Utsu — aftershock decay

After a mainshock, the aftershock rate decays as a power law in time:

$$n(t) = \frac{K}{(t + c)^{p}}, \qquad p \approx 1 .$$

Its cumulative form (for $p \ne 1$) is

$$N(t_1, t_2) = \frac{K}{1 - p}\left[(t_2 + c)^{1-p} - (t_1 + c)^{1-p}\right].$$

- **Parameters:** $K$ (aftershock productivity), $c$ (time offset, hours–day), $p$ (decay exponent).
- **Reference:** Utsu, Ogata & Matsu'ura (1995), the modified Omori law; parameterization as used in
  Ogata (1988), *JASA* 83(401), 9–27
  (doi:[10.1080/01621459.1988.10478560](https://doi.org/10.1080/01621459.1988.10478560)).

> **Honest limit — short-term aftershock incompleteness.** Immediately after a large mainshock —
> exactly the highest-stakes, highest-traffic moment for a forecast — the catalog is grossly
> incomplete: $M_c$ spikes for hours to days while small events are buried in the coda. A naive fit
> then **underestimates productivity** precisely when a large aftershock is most likely. The
> real-time update therefore uses a time-dependent completeness $M_c(t)$ (incompleteness-aware
> ETAS) rather than a flat threshold.

### 1.3 ETAS — the Epidemic-Type Aftershock Sequence model (primary baseline)

ETAS is a self-exciting Hawkes point process: a stationary background plus the sum of all triggered
"offspring" of every past event. It stitches together the background rate, Utsu productivity, the
Omori–Utsu time kernel, and a spatial kernel into a single **conditional intensity**:

$$\lambda(t, x, y \mid \mathcal H_t) = \mu(x, y) + \sum_{i:\, t_i < t} K\, e^{\alpha (M_i - M_0)}\,
\Big(1 + \tfrac{t - t_i}{c}\Big)^{-p}\, f\!\left(x - x_i,\, y - y_i \mid M_i\right).$$

The Utsu productivity is $k(m) = K\, e^{\alpha (m - M_0)}$ and the Omori–Utsu kernel is
$g(t) = \tfrac{p-1}{c}\big(1 + t/c\big)^{-p}$. The Ogata (1998) inverse-power spatial kernel is

$$f(x, y \mid M_i) = \frac{q - 1}{\pi\,\zeta^2}\left(1 + \frac{r^2}{\zeta^2}\right)^{-q},
\qquad \zeta = D\, e^{\gamma (M_i - M_0)},\quad r^2 = (x-x_i)^2 + (y-y_i)^2 .$$

**Stability — two distinct gates.** Two logically separate conditions must hold:

1. **Finite branching:** the magnitude integral converges only if $\alpha < \beta$ (with
   $\beta = b\ln 10$). If $\alpha \ge \beta$ the productivity–magnitude integral diverges.
2. **Subcriticality / stationarity:** *given* $\alpha < \beta$, the branching ratio $n$ (expected
   direct offspring per event) must satisfy $n < 1$. $n \ge 1$ is supercritical (explosive) and
   signals a mis-fit; the fit is rejected.

- **Parameters:** $\mu(x,y)$ (background field), $K$, $\alpha$ (productivity scaling), $c$, $p$
  (Omori), $D$, $\gamma$, $q$ (spatial kernel), with $M_0$ a reference magnitude.
- **Fitting:** maximum-likelihood on the **full, un-declustered** catalog, with the background
  $\mu(x,y)$ recovered by stochastic declustering (Zhuang et al., 2002).
- **References:** Ogata (1988), *JASA* 83(401), 9–27 (temporal ETAS);
  Ogata (1998), *Ann. Inst. Statist. Math.* 50(2), 379–402, space–time ETAS
  (doi:[10.1023/A:1003403601725](https://doi.org/10.1023/A:1003403601725)).

The point-process log-likelihood maximized in the fit (reused as the L-test kernel, §E.2) is

$$\ln L = \sum_i \ln \lambda(t_i, x_i, y_i \mid \mathcal H_{t_i})
- \int_0^T\!\!\int_A \lambda(t, x, y \mid \mathcal H_t)\, dx\, dy\, dt .$$

### 1.4 Reasenberg–Jones — the transparent operational baseline

The most transparent "tomorrow's earthquakes" model. The rate of events of magnitude $\ge M$ after a
mainshock $M_m$ follows a Gutenberg–Richter magnitude term times a modified-Omori time decay:

$$\lambda(t, M) = \frac{10^{\,a + b(M_m - M)}}{(t + c)^{p}} .$$

The expected number over a forecast window is $N = \int \lambda\, dt$, and the probability of at
least one such event (non-homogeneous Poisson) is

$$P(\ge 1) = 1 - e^{-N} .$$

- **Parameters:** $a$ (sequence productivity), $b$, $c$, $p$.
- **Role:** transparent fallback and sanity-check alongside ETAS; the USGS Operational Aftershock
  Forecast (OAF) system runs both Reasenberg–Jones and ETAS.
- **References:** Reasenberg & Jones (1989), *Science* 243(4895), 1173–1176
  (doi:[10.1126/science.243.4895.1173](https://doi.org/10.1126/science.243.4895.1173));
  global tectonic-regime extension Page et al. (2016), *BSSA* 106(5), 2290–2301
  (doi:[10.1785/0120160073](https://doi.org/10.1785/0120160073)).

### 1.5 STEP — Short-Term Earthquake Probability

STEP is the production reference for the **product output shape**: it wraps Reasenberg–Jones
clustering plus a background term into gridded shaking-probability maps — a
"one-inference-per-(short interval)" probabilistic regional map, exactly the form of this product
(daily cadence in our case). The conditional rate at a cell is the background rate plus the summed
Reasenberg–Jones contribution of recent events.

- **Role:** template for the gridded daily probability map.
- **Reference:** Gerstenberger, Wiemer, Jones & Reasenberg (2005), *Nature* 435, 328–331
  (doi:[10.1038/nature03622](https://doi.org/10.1038/nature03622)).

### 1.6 EEPAS — Every Earthquake a Precursor According to Scale (medium term)

A medium-term (months–years) precursory-scaling model. Each earthquake of precursor magnitude $M_p$
contributes a rate density that is a product of three densities (magnitude, time, location),
governed by scaling relations:

$$M_m = a_M + b_M\, M_p, \qquad
\log_{10} T_P = a_T + b_T\, M_p, \qquad
\log_{10} A = a_A + b_A\, M_p ,$$

with the rate density $\propto$ (normal in magnitude) $\times$ (lognormal in time) $\times$
(bivariate-normal in location).

- **Honest caveat:** the published EEPAS density constants contain **known typos across papers**; if
  used, pin to a reference implementation (pyCSEP / floatCSEP) rather than transcribing constants.
  EEPAS sits outside the 1-week primary window — a feature/context source, not a short-term core
  model.
- **Reference:** Rhoades & Evison (2004), *Pure Appl. Geophys.* 161, 47–72
  (doi:[10.1007/s00024-003-2434-9](https://doi.org/10.1007/s00024-003-2434-9)).

### 1.7 Smoothed seismicity — the spatial background $\mu(x,y)$ and the mandatory null

A stationary, time-independent estimate of where earthquakes occur, obtained by smoothing a
**declustered** catalog with an adaptive kernel. The Helmstetter–Kagan–Jackson adaptive power-law
kernel sums per-event contributions with a bandwidth set by the distance to the $n$-th nearest
neighbour ($n \sim 6$, an optimized, region-specific hyperparameter):

$$\mu(x, y) = \sum_i K_{d_i}(r), \qquad K_d(r) = \frac{C(d)}{\left(r^2 + d^2\right)^{s}} .$$

> **Implementation note (do not hard-code the exponent).** The kernel exponent $s$ and normalization
> $C(d)$ vary across the Helmstetter–Kagan–Jackson family of papers (forms with $s = 1$ and
> $s = 3/2$ both appear depending on the specific kernel/normalization). $s$, $C(d)$, and the
> neighbour count are pinned to a reference implementation
> (Helmstetter–Kagan–Jackson 2007 or the pyCSEP/floatCSEP code), not assumed.

- **Role:** (a) the spatial background field $\mu(x,y)$ that feeds ETAS; (b) the **stationary
  Poisson reference** — the mandatory null any time-dependent model must beat.
- **Reference:** Helmstetter, Kagan & Jackson (2007), *SRL* 78(1), 78–86
  (doi:[10.1785/gssrl.78.1.78](https://doi.org/10.1785/gssrl.78.1.78)).

### 1.8 BPT / renewal — long-term time-dependent recurrence

Where paleoseismic data genuinely constrain a fault's mean recurrence interval, a Brownian Passage
Time (inverse-Gaussian) renewal model conditions the long-term background. The recurrence-time
density is

$$f(t; \mu, \alpha) = \sqrt{\frac{\mu}{2\pi\,\alpha^2 t^3}}\;
\exp\!\left(-\frac{(t - \mu)^2}{2\,\mu\,\alpha^2 t}\right),$$

whose hazard rises from zero, peaks, then plateaus.

- **Parameters:** $\mu$ (mean recurrence interval), $\alpha$ (aperiodicity / coefficient of
  variation).
- **Honest caveat:** with only a few observed cycles, $\alpha$ is poorly constrained and the gain
  over a plain Poisson background is marginal — we do not claim renewal skill where the data do not
  support it.
- **Reference:** Matthews, Ellsworth & Reasenberg (2002), *BSSA* 92, 2233–2250
  (doi:[10.1785/0120010267](https://doi.org/10.1785/0120010267)).

### 1.9 Rate-and-state friction + Coulomb stress — the mechanistic layer

Static stress transfer from a fault slip changes the Coulomb failure stress on neighbouring faults:

$$\Delta\mathrm{CFS} = \Delta\tau - \mu'\,\Delta\sigma_n ,$$

where $\Delta\tau$ is shear-stress change, $\Delta\sigma_n$ normal-stress change, and $\mu'$ the
effective friction. Dieterich's (1994) rate-and-state friction then predicts how the **seismicity
rate** responds to such a stress step, deriving an Omori-like $1/t$ decay *from first principles*.
The seismicity rate responds **exponentially** to a Coulomb stress step:

$$\frac{R}{r} = \exp\!\left(\frac{\Delta\mathrm{CFS}}{A\sigma}\right),
\qquad t_a = \frac{A\,\sigma_n}{\dot\tau_r} ,$$

with $t_a$ the characteristic aftershock-duration timescale.

- **Role:** mechanistic spatial priors (Coulomb lobes promote/suppress triggering) — optional,
  feature-flagged covariates, not a standalone forecaster.
- **References:** King, Stein & Lin (1994), *BSSA* 84, 935–953 (Coulomb);
  Dieterich (1994), *JGR* 99(B2), 2601–2618
  (doi:[10.1029/93JB02581](https://doi.org/10.1029/93JB02581), rate-and-state).

### 1.10 The exceedance probability the public sees

The single number rendered to a user combines the conditional intensity (from ETAS / R-J / STEP),
the Gutenberg–Richter magnitude tail, and the chosen horizon. The expected number of events above a
target magnitude $M^*$ in region $A$ over horizon $[0, T]$ is

$$N_{\ge M^*} = \int_0^T\!\!\int_A \lambda(t, x, y \mid \mathcal H_t)\,\Phi(M^*)\, dx\, dy\, dt,
\qquad \Phi(M^*) = 10^{-b(M^* - M_c)} ,$$

and the published probability is

$$P(\ge 1 \text{ event} \ge M^*) = 1 - e^{-N_{\ge M^*}} .$$

> The public formula $P = 1 - e^{-N}$ never changes. Only the **quality of $\lambda$** improves as
> the model improves. Quiet days correctly read near climatology; the number is always shown next to
> its long-term baseline so a user reads "X % vs Y % baseline," never an unanchored figure.

**Target shape.** A single fixed $M^*$ is brittle (a Chilean $M \ge 6$ is routine; a UK $M \ge 6$ is
unheard-of) and discards the Gutenberg–Richter information the model already computes. CAOS_SEISMIC
forecasts the **full conditional magnitude distribution** and derives the public scalar as an
exceedance probability at one or more **region-appropriate** thresholds (for Chile,
$M^* \in \{5.0, 6.0, 7.0\}$). The full distribution keeps the $b$-value information, enables the
M-test and CRPS, and lets the UI offer multiple thresholds. The maximum magnitude $M_{\max}$ that
bounds the exceedance integral is an explicit, documented per-region assumption (Chile: $M_{\max}=9.5$,
the 1960 Valdivia event) and sets the tail probability of the rare, high-impact events. See
[`model.md`](model.md) for the estimator detail.

---

## Part 2 — Analytical / ML model approaches

### 2.1 Temporal point processes — the unifying language

Every model above is a special case of a marked point process with conditional intensity
$\lambda(t, x, y \mid \mathcal H_t)$. The model is fit by maximizing the point-process log-likelihood
(§1.3) and scored by the same likelihood on held-out, time-causal data. ETAS is the parametric,
physics-informed member of this family; neural temporal point processes (NTPP) are the flexible,
learned members.

Foundational neural-TPP architectures (validated on **non-seismic** event streams — social, retail,
synthetic — their log-likelihood wins do **not** automatically transfer to seismicity):

- **RMTPP** — Du et al. (2016), KDD, exponential intensity
  (doi:[10.1145/2939672.2939875](https://doi.org/10.1145/2939672.2939875)).
- **Neural Hawkes Process** — Mei & Eisner (2017), NeurIPS, continuous-time LSTM with a softplus
  intensity that permits inhibition (arXiv:1612.09328).
- **Self-Attentive Hawkes** — Zhang et al. (2020), ICML (arXiv:1907.07561).
- **Transformer Hawkes Process** — Zuo et al. (2020), ICML (PMLR v119).

### 2.2 Neural TPP for earthquakes — what genuinely helps

Where learned models *can* add value over fixed-kernel ETAS, the gains come from two real ETAS gaps,
not from "deep-learning magic":

1. **Multivariate covariate ingestion** ETAS cannot easily absorb — sub-$M_c$ events, geodesy /
   InSAR, injection / pore-pressure data, multiple catalogs.
2. **Learned spatial anisotropy** — fault-aligned triggering structure recovered without explicit
   fault inputs.

Two honest exemplars:

- **RECAST** (Dascher-Cousineau et al., 2023, *GRL* 50, e2023GL103909,
  doi:[10.1029/2023GL103909](https://doi.org/10.1029/2023GL103909)) — a GRU-based encoder–decoder
  neural TPP. It **improves on temporal ETAS only when the training catalog is large**
  ($\gtrsim 10^4$ events); on smaller catalogs it merely *matches* ETAS.
- **FERN** (Zlydenko et al., 2023, *Sci. Rep.* 13,
  doi:[10.1038/s41598-023-38033-9](https://doi.org/10.1038/s41598-023-38033-9)) — an
  ETAS-generalizing encoder (MLPs replace fixed kernels). The **FERN+** variant (which ingests
  sub-$M_c$ events) reports a 4–12 % information-gain-per-earthquake improvement and learns
  fault-aligned anisotropy. **Crucial caveats stated by the authors themselves:** it is *not*
  CSEP-tested, provides *no* uncertainty quantification, and its test period ends *before* the 2011
  Tohoku $M_w$ 9.0. The gain came mostly from the two ETAS gaps above, not from network depth.

### 2.3 The honest ML-vs-ETAS verdict

**As of writing, no machine-learning model has been shown to reliably beat a well-fit ETAS for
short-term forecasting under fair, prospective CSEP-style testing.** The decisive evidence:

- **EarthquakeNPP** (Stockman, Lawson & Werner, *TMLR* 2026; arXiv:2410.08226) benchmarked five
  modern neural point processes (NSTPP, DeepSTPP, AutoSTPP, DSTPP, SMASH) on California 1971–2021
  with **strict chronological splits** and **CSEP consistency tests**. **None outperformed ETAS.**
  The authors conclude current NPP implementations are "not yet suitable for practical earthquake
  forecasting." On the ComCat dataset, ETAS passes the consistency tests at ~95.8 % (N-test),
  **92.0 % (spatial)**, 93.8 % (magnitude), 97.6 % (pseudo-likelihood), whereas the best NPP passes
  ~86–88 % on the number/pseudo-likelihood tests but only **~68.6 % on the spatial test** — exactly
  where forecasting value lives. The crucial methodological fix EarthquakeNPP made was to repair a
  **data-leakage flaw** in earlier neural-TPP-for-earthquakes work (non-chronological / alternating
  splits inflate metrics via triggering; excluding the Tohoku sequence makes the benchmark
  irrelevant). Once temporal splits and the big sequences are restored, the apparent neural
  advantage evaporates. This is *the* evaluation lesson.

> **Scope caveat.** This "NPPs do not beat ETAS" result is established on the California-only
> benchmark to date (1971–2021), not proven globally. It justifies shipping ETAS-class only for v0
> and gating any neural model behind a CSEP win — but it is stated as "on the benchmark to date,"
> not as an unconditional law. We do **not** claim ML can *never* add skill: some hybrid and neural
> models match or beat plain ETAS on information gain *in specific settings*. The honest statement
> is: *no pure ML / NPP has robustly beaten ETAS in prospective CSEP to date.*

### 2.4 The cautionary tale — over-parameterization and the wrong metric

The canonical warning is **DeVries et al. (2018)** (*Nature* 560, 632–634,
doi:[10.1038/s41586-018-0438-y](https://doi.org/10.1038/s41586-018-0438-y)): a deep net (6 hidden
layers, ~13,451 free parameters, 12 stress features) reported AUC 0.85 for aftershock spatial
pattern vs 0.58 for Coulomb stress. **Mignan & Broccardo (2019)** (*Nature* 575, E1–E3,
doi:[10.1038/s41586-019-1582-8](https://doi.org/10.1038/s41586-019-1582-8)) matched that 0.85 with a
**2-parameter logistic regression** ("one neuron") using a single feature — the sum of the absolute
values of the stress-change components. Root causes, each a guardrail for us:

- **Massive over-parameterization** vs only ~199 effective mainshocks → assume overfitting whenever
  parameters $\gg$ effective samples.
- **Per-cell "computer-vision" framing** inflated the apparent sample size (~131,000 correlated
  cells).
- **AUC is the wrong metric for rate forecasting** — it is invariant to monotone rescaling, hence
  blind to the calibration of the very probabilities a forecast publishes, and on a rare,
  per-cell-per-day task it largely measures *between-region rate differences* (it becomes a region
  classifier), not skill. **AUC / accuracy are banned as primary forecasting metrics.**

### 2.5 Detection is not forecasting

ML waveform models — PhaseNet, EQTransformer, PhaseNO, SeisBench, and the SeisLM foundation model
(arXiv:2410.15765) — are mature/production for phase-picking, detection, association, and
characterization. They build **better, more complete catalogs**, which helps both ETAS and neural
forecasters — the single biggest realizable near-term lever. But they **do not forecast**: SeisLM is
positioned for detection/characterization only, and its foreshock–aftershock task is *retrospective*
classification of existing waveforms relative to a *known* mainshock. We keep the
detection/forecasting line explicit in product copy so detection branding never implies prediction.

### 2.6 Roadmap rule

Ship calibrated ETAS-class as the defensible core. Treat any neural model as a **gated challenger**,
never the default: it reaches the public map only if it **beats ETAS in our own prospective CSEP
harness AND is calibrated**. The best-class challenger is a conditional spatio-temporal NTPP with a
Hawkes inductive bias (FERN spirit: keep the additive background + summed-triggering skeleton,
replace fixed kernels with small MLPs/attention) *and* explicit magnitude modelling (most NPPs lack
this — a real weakness EarthquakeNPP flags). Calibration (reliability / PIT) is a release blocker.

---

## Part 3 — The CSEP evaluation backbone

A forecast is only as honest as its prospective scoring. We adopt **CSEP** (the Collaboratory for
the Study of Earthquake Predictability) and its community-endorsed toolkit **pyCSEP** as the *only*
scoring framework — using standard tests, not bespoke metrics, is what makes the product defensible.
Each daily inference is **logged immutably at issue time, together with the exact catalog state it
was computed from**, then scored. The live **calibration / reliability diagram** ("when we said 5 %,
it happened ~5 % of the time") is the single most credibility-building artifact we ship. The full
back-analysis protocol is in [`evaluation.md`](evaluation.md); this section gives the equations.

### E.1 Two forecast representations

| Representation | What it is | Tests |
|---|---|---|
| **Gridded-rate** | Poisson expected count per space–magnitude cell | Poisson consistency tests |
| **Catalog-based** | ensemble of $\ge 10{,}000$ synthetic catalogs / day | empirical, non-Poisson tests |

Regional seismicity is **over-dispersed** relative to Poisson (variance $\gg$ mean, because of
clustering), so the Poisson grid tests **over-reject during aftershock sequences**. The
catalog-based tests (Savran et al., 2020, *BSSA* 110(4), 1799–1817,
doi:[10.1785/0120200026](https://doi.org/10.1785/0120200026)) relax the Poisson assumption and are
the **correct primary path** for a clustered daily forecast. We emit **both**, and the pessimistic
uncertainty bound is **wider than a naive Poisson interval** to honour the over-dispersion
(negative-binomial behaviour; cf. Kagan, 2017, *GJI* 211(1), 335–345,
doi:[10.1093/gji/ggx300](https://doi.org/10.1093/gji/ggx300)).

### E.2 Consistency tests (is one model calibrated?)

All built on the Poisson joint log-likelihood over space–magnitude bins:

$$L(\Omega \mid \Lambda) = \sum_{\text{bins } i}\Big[-\lambda_i + \omega_i \ln \lambda_i
- \ln(\omega_i!)\Big],$$

where $\lambda_i$ is the forecast expected count and $\omega_i$ the observed count in bin $i$.

- **N-test** (number). Poisson quantile scores
  $\delta_1 = 1 - F\big((N_{obs}-1)\mid N_{fore}\big)$ (small $\Rightarrow$ observed too **many** for
  the forecast → forecast too low) and $\delta_2 = F\big(N_{obs}\mid N_{fore}\big)$ (small
  $\Rightarrow$ too **few**), with $F$ the Poisson CDF and $N_{fore} = \sum_i \lambda_i$.
- **M-test** (magnitude / Gutenberg–Richter shape), quantile $\kappa$.
- **S-test** (spatial distribution), quantile $\zeta$.
- **L-test** (joint pseudo-likelihood), quantile $\gamma$; and the **CL-test** (conditional
  likelihood, conditioned on $N_{obs}$), preferred over the raw L-test because the L-test correlates
  strongly with the N-test. *(The L-test is not deprecated in current pyCSEP, but the conditional
  L-test and Poisson/negative-binomial number-test variants are preferred.)*

- **References:** Schorlemmer et al. (2007), *SRL* 78(1), 17–29 (origin of N/L/R tests,
  doi:[10.1785/gssrl.78.1.17](https://doi.org/10.1785/gssrl.78.1.17));
  Zechar, Gerstenberger & Rhoades (2010), *BSSA* 100(3), 1184–1195 (modern N/M/S definitions,
  doi:[10.1785/0120090192](https://doi.org/10.1785/0120090192)).

### E.3 Comparison tests (is model A *better* than model B?)

**Passing consistency tests is necessary but not sufficient.** Skill is established only by *winning
a comparison test* against a real baseline (smoothed-seismicity **and** ETAS) with a confidence
interval that excludes zero. The metric is **information gain per earthquake** (IGPE), in **nats**:

$$I_N(A, B) = \frac{1}{N}\sum_{i=1}^{N}\Big(\ln \lambda_{A}(k_i) - \ln \lambda_{B}(k_i)\Big)
- \frac{\hat N_A - \hat N_B}{N} ,$$

tested with the paired **T-test**

$$T = \frac{I_N(A,B)}{s / \sqrt{N}} \sim t_{N-1} ,$$

and corroborated by the non-parametric **W-test** (Wilcoxon signed-rank).

- **Reference:** Rhoades et al. (2011), *Acta Geophysica* 59(4), 728–747
  (doi:[10.2478/s11600-011-0013-5](https://doi.org/10.2478/s11600-011-0013-5)).

> **Honest framing of the magnitude of the gain.** ETAS-over-Poisson information gain is **strongly
> state-dependent**, not a fixed steady-state number. It is **positive and large during active
> aftershock sequences** (probability gains of up to orders of magnitude on peak days) and **near
> zero in quiet periods**, with a modest all-period average. For scale, *time-independent*
> model-vs-smoothed-seismicity contrasts in prospective California CSEP give IGPE of only about
> **−0.7 to +0.5 nats**. We therefore report the gain as state-dependent, **not** as a fabricated
> round figure, and always in nats — never bits.

### E.4 Proper scoring rules & the alarm/ROC view (communication aids, not substitutes)

On top of CSEP we report strictly proper scoring rules (Gneiting & Raftery, 2007, *JASA* 102(477),
359–378, doi:[10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)):

$$\text{LogS}(p, y) = -\ln p(y), \qquad
\text{BS} = \frac{1}{T}\sum_t (p_t - y_t)^2, \qquad
\text{CRPS} = \int \big(F(x) - \mathbf 1\{x \ge y\}\big)^2 dx .$$

The logarithmic score is the kernel of the CSEP L-test; the **Brier score** (originating with Brier,
1950) suits the bounded binary output; **CRPS** suits a full predictive distribution.

For an alarm-style/ROC view we use the **Molchan diagram** (miss rate $\nu = 1 - H$ vs alarm
fraction $\tau$) and the **Area Skill Score** (1 = perfect, 0.5 = random):

- **Reference:** Zechar & Jordan (2008), *GJI* 172(2), 715–724
  (doi:[10.1111/j.1365-246X.2007.03676.x](https://doi.org/10.1111/j.1365-246X.2007.03676.x)); ASS
  statistic Zechar & Jordan (2010), *PAGEOPH* 167, 893–906.

> Note on **ROC/AUC**: shown only as a communication aid, never as a primary skill metric — it is
> invariant to calibration and, on rare per-cell-per-day tasks, degenerates into a region classifier
> (the DeVries trap, §2.4).

### E.5 pyCSEP — the implementation

All tests above are implemented in **pyCSEP** (Savran et al., 2022, *SRL* 93(5), 2858–2870,
doi:[10.1785/0220220033](https://doi.org/10.1785/0220220033); docs at
[docs.cseptesting.org](https://docs.cseptesting.org)), which provides catalog access, both forecast
representations, and peer-reviewed implementations of the grid tests (number / magnitude / spatial /
likelihood / conditional_likelihood / paired_t / w) and catalog tests (number / spatial / magnitude
/ pseudolikelihood / calibration / resampled_magnitude / MLL). Using it means reviewers can dispute
our *model*, not our test code. *(There is no negative-binomial N-test in the core grid API;
over-dispersion is handled via the catalog-based number test.)*

A real operational template to sanity-check our scores: Serafini et al. (2025), *Scientific Data*
12, 1501 (doi:[10.1038/s41597-025-05766-3](https://doi.org/10.1038/s41597-025-05766-3)) — 25
automated $M \ge 3.95$ daily models, >50,000 daily next-day forecasts over California (2007–2018),
all scored with pyCSEP. The load-bearing finding: **no single model dominates the decade** (STEP
excels during aftershock sequences); ETAS is the consistent generalist.

---

## Honest limits (carry into product copy)

- **No deterministic prediction.** Whether a small rupture cascades into a great earthquake depends
  on unmeasurably fine details of the crust; deterministic prediction is effectively impossible
  (Geller et al., 1997, *Science* 275, 1616–1617). The physical picture is self-organized
  criticality (Bak & Tang, 1989, *JGR* 94(B11), 15635–15637) — a *leading explanatory framework, not
  settled physics*.
- **Absolute probabilities stay small.** Even during an active sequence, the absolute probability of
  a large event in the next day is usually well under a few percent. The *relative* gain over
  background can be large (1–3 orders of magnitude); the *absolute* number remains low. Always show
  it next to the baseline.
- **A single outcome neither validates nor invalidates a probabilistic forecast.** During the 2019
  Ridgecrest sequence, UCERF3-ETAS gave ~3 % (≈2.8 %) chance of a larger event in the first week;
  the $M$ 7.1 struck ~34 h later. A 3 % forecast is **not wrong** when the 3 % outcome occurs.
- **Operational systems are real and honest.** USGS OAF (Reasenberg–Jones → Page et al. 2016 global)
  and OEF-Italy (an ensemble of **three distinct models — ETAS + ETES + STEP**) run as scheduled
  services and publish calibrated probabilities with uncertainty. The OEF-Italy 10-year validation
  (Spassiani, Falcone, Murru & Marzocchi, 2023, *GJI* 234(3), 2501–2518) found it **broadly
  reliable, with a documented underestimation during the 2016–2017 Central Italy sequence** caused by
  post-mainshock catalog incompleteness — exactly the short-term incompleteness limit in §1.2.
- **The communication failure mode is the dangerous one.** The L'Aquila earthquake (2009, $M_w$
  ~6.1–6.3) and its trials are the field's cautionary tale: the harm was **false reassurance**, not a
  failure to predict. This product never over-reassures and never issues an alarm; it publishes
  calibrated conditional probabilities with explicit uncertainty, scored prospectively, as an
  independent research/education tool that **complements** — never replaces — official agencies.

---

## Reference index (canonical)

Aki 1965 · Bak & Tang 1989 (doi:10.1029/JB094iB11p15635) · Dascher-Cousineau et al. 2023
(doi:10.1029/2023GL103909) · DeVries et al. 2018 (doi:10.1038/s41586-018-0438-y) · Dieterich 1994
(doi:10.1029/93JB02581) · Geller et al. 1997 (doi:10.1126/science.275.5306.1616) · Gerstenberger
et al. 2005 (doi:10.1038/nature03622) · Gneiting & Raftery 2007 (doi:10.1198/016214506000001437) ·
Helmstetter, Kagan & Jackson 2007 (doi:10.1785/gssrl.78.1.78) · Jordan et al. 2011
(doi:10.4401/ag-5350) · Kagan 2017 (doi:10.1093/gji/ggx300) · King, Stein & Lin 1994 · Matthews,
Ellsworth & Reasenberg 2002 (doi:10.1785/0120010267) · Mignan & Broccardo 2019
(doi:10.1038/s41586-019-1582-8) · Ogata 1988 (doi:10.1080/01621459.1988.10478560) · Ogata 1998
(doi:10.1023/A:1003403601725) · Page et al. 2016 (doi:10.1785/0120160073) · Reasenberg & Jones 1989
(doi:10.1126/science.243.4895.1173) · Rhoades & Evison 2004 (doi:10.1007/s00024-003-2434-9) ·
Rhoades et al. 2011 (doi:10.2478/s11600-011-0013-5) · Savran et al. 2020 (doi:10.1785/0120200026) ·
Savran et al. 2022 (doi:10.1785/0220220033) · Schorlemmer et al. 2007 (doi:10.1785/gssrl.78.1.17) ·
Serafini et al. 2025 (doi:10.1038/s41597-025-05766-3) · Spassiani et al. 2023 (GJI 234(3),
2501–2518) · Stockman, Lawson & Werner 2026 (EarthquakeNPP, arXiv:2410.08226) · Tinti & Mulargia
1987 · Wiemer & Wyss 2000 (doi:10.1785/0119990114) · Woessner & Wiemer 2005 (doi:10.1785/0120040007)
· Zechar, Gerstenberger & Rhoades 2010 (doi:10.1785/0120090192) · Zechar & Jordan 2008
(doi:10.1111/j.1365-246X.2007.03676.x) · Zlydenko et al. 2023 (doi:10.1038/s41598-023-38033-9).

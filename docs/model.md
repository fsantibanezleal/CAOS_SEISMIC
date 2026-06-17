<!-- markdownlint-disable MD013 MD033 -->
# The Conditional Estimator

> **Honest framing (non-negotiable).** CAOS_SEISMIC is a *forecaster*, never a *predictor*. The
> model emits **bounded, calibrated conditional probabilities** scoped to a region × magnitude band
> × horizon, always shown next to a long-term baseline and scored CSEP-style. No deterministic call,
> no alarm, no countdown, no "safe" state. The equations behind the field are in
> [`methodology.md`](methodology.md); this page is the *implementation* of the estimator — what it
> targets, what feeds it, how it is calibrated, and the honest verdict on ETAS *vs.* machine
> learning.

This document is the design source of truth for the code under
[`src/caos_seismic/model/`](../src/caos_seismic/model/) and
[`src/caos_seismic/inference/`](../src/caos_seismic/inference/). Every claim here maps to a config
key in [`configs/`](../configs/) and a function in the package; the cross-references are explicit so
the prose and the code cannot drift.

---

## 1. Target definition (the binding decision)

A forecast is meaningless without a precisely-defined target. CAOS_SEISMIC fixes four axes —
space, magnitude, horizon, and the public scalar — and pins each to a config value.

### 1.1 The public scalar: an exceedance probability

The single number a user sees is the probability of **at least one** event above a target
magnitude $M^*$ in a cell over a horizon $[0, T]$. It is a non-homogeneous Poisson exceedance
probability:

$$P(\ge 1 \text{ event} \ge M^*) = 1 - e^{-N_{\ge M^*}}, \qquad
N_{\ge M^*} = \int_0^T\!\!\int_A \lambda(t, x, y \mid \mathcal H_t)\,\Phi(M^*)\, dx\, dy\, dt .$$

The magnitude term $\Phi(M^*)$ is the Gutenberg–Richter tail fraction
$\Phi(M^*) = 10^{-b(M^* - M_c)}$, *bounded* by the per-region maximum magnitude $M_{\max}$ so an
unbounded tail can never inflate the rare-event probability:

$$\Phi(M^*) = \frac{10^{-b(M^* - M_c)} - 10^{-b(M_{\max} - M_c)}}{1 - 10^{-b(M_{\max} - M_c)}},
\qquad M_c \le M^* \le M_{\max} .$$

> **The public formula $P = 1 - e^{-N}$ never changes.** Only the *quality of $\lambda$* improves as
> the model improves. Quiet days correctly read near climatology; the number is always rendered next
> to its long-term baseline, so a user reads "X % vs Y % baseline," never an unanchored figure.

In code this is exactly `poisson_p_at_least_one()` and `gr_exceedance_fraction()` in
[`model/_common.py`](../src/caos_seismic/model/_common.py) — the single shared implementation reused
by every forecaster, so there are not three subtly-different copies of the exceedance integral.

### 1.2 Forecast the *distribution*, threshold for display

A single fixed $M^*$ is brittle: a Chilean $M \ge 6$ is routine, a UK $M \ge 6$ is unheard-of, and a
hard threshold discards the Gutenberg–Richter information the model already computes. CAOS_SEISMIC
therefore forecasts the **full conditional magnitude distribution** (via the GR/$b$-value term) and
derives the public scalar as an exceedance probability at one or more **region-appropriate**
thresholds. For Chile, `configs/forecast.yaml` sets `magnitude_thresholds: [5.0, 6.0, 7.0]`. Keeping
the distribution (rather than a single binary) enables the CSEP **M-test**, the **CRPS**, and a UI
that offers multiple thresholds.

$M_{\max}$ is an explicit, documented per-region assumption that bounds the exceedance integral and
sets the tail probability of the rare, high-impact events. For Chile, `region.chile.yaml` sets
`m_max: 9.5` — the 1960 Valdivia event, the largest instrumentally recorded earthquake. Its
sensitivity is reported, not hidden.

### 1.3 Spatial cell: fit fine, render coarse

- **Fit & score on a fine grid** — regular $0.1° \times 0.1°$ space cells with $0.1$-magnitude bins
  (`configs/grid.yaml: fit.cell_deg = 0.1`, `fit.mag_bin = 0.1`), the CSEP California convention, so
  the **S-test** can resolve *where* and the **M-test** *what size*. This is the resolution at which
  ETAS is fit and CSEP-tested.
- **Render at region granularity** — the UI aggregates the fine grid into H3 hexbins
  (`grid.yaml: display.h3_resolution_world = 3`, `…_region = 5`) for display.
- **Couple cell size to data density** — events-per-cell-per-horizon must support fitting and
  scoring; in sparse cells the model borrows strength from the smoothed-seismicity background and
  regional priors (§6) rather than shrinking cells until they are empty.

### 1.4 Horizon

Three rolling horizons re-issued **daily**: **1 day, 2 days, 7 days**
(`configs/forecast.yaml: horizons_days = [1, 2, 7]`). The horizon selector is always visible in the
UI and visibly recolours the field. Honest caveat baked into the copy: the 7-day gain is near-zero
outside active sequences; quiet days correctly read near-climatology.

---

## 2. The estimator stack

Every model in the system implements one port — the `Forecaster` protocol in
[`contracts.py`](../src/caos_seismic/contracts.py): `fit(catalog, region, t_issue)` then
`expected_counts(region, cells, horizon_days, m_threshold, t_issue)`. That single seam lets the null,
the reference, and any future challenger be swapped, fit on the identical conditioning slice, and
scored against each other on identical bins.

### 2.1 The mandatory null — adaptive smoothed seismicity

A stationary, time-independent Poisson estimate of *where* earthquakes occur, obtained by smoothing
a **declustered** catalog with an adaptive power-law kernel (Helmstetter, Kagan & Jackson 2007,
*SRL* 78(1), 78–86, doi:[10.1785/gssrl.78.1.78](https://doi.org/10.1785/gssrl.78.1.78)):

$$\mu(x, y) = \sum_i K_{d_i}(r_i), \qquad K_d(r) = C(d)\,(r^2 + d^2)^{-s},$$

where the bandwidth $d_i$ is the great-circle distance to event $i$'s $n$-th nearest neighbour
(adaptive smoothing — dense regions sharpen, sparse regions broaden). This serves two roles: it is
(a) the spatial background field $\mu(x,y)$ that seeds ETAS and (b) the **stationary Poisson
reference** — the null any time-dependent model must beat in comparison testing.

> **Do not hard-code the exponent.** The kernel exponent $s$ and normalization $C(d)$ vary across the
> HKJ family (forms with $s = 1$ and $s = 3/2$ both appear). In the code these are *named reference
> profiles* (`KERNEL_PROFILES` in [`model/smoothed.py`](../src/caos_seismic/model/smoothed.py)),
> selected by config (`etas.yaml: background_model`), never assumed. $C(d)$ is derived analytically
> so each per-event kernel integrates to exactly one earthquake; the neighbour count $n$
> (`background_model.neighbors = 6`) is a region-tuned hyperparameter, not a universal constant.

### 2.2 The primary estimator and reference — space–time ETAS

The Epidemic-Type Aftershock Sequence model is a self-exciting Hawkes point process: a stationary
background plus the summed, decaying "offspring" of every past event. It is the de-facto operational
baseline and the v0 production model. Its conditional intensity (Ogata 1998, *Ann. Inst. Statist.
Math.* 50(2), 379–402, doi:[10.1023/A:1003403601725](https://doi.org/10.1023/A:1003403601725)) is

$$\lambda(t, x, y \mid \mathcal H_t) = \mu(x, y) + \sum_{i:\, t_i < t} K\, e^{\alpha (M_i - M_0)}\,
\Big(1 + \tfrac{t - t_i}{c}\Big)^{-p}\, f\!\left(x - x_i,\, y - y_i \mid M_i\right),$$

with Utsu productivity $k(m) = K\,e^{\alpha(m - M_0)}$, the Omori–Utsu time kernel
$g(t) = \tfrac{p-1}{c}\big(1 + t/c\big)^{-p}$, and the Ogata-1998 inverse-power spatial kernel

$$f(r \mid M_i) = \frac{q - 1}{\pi\,\zeta^2}\left(1 + \frac{r^2}{\zeta^2}\right)^{-q},
\qquad \zeta = D\, e^{\gamma (M_i - M_0)} .$$

The parameters $(K, \alpha, c, p, D, \gamma, q)$ are fit by maximum likelihood on the **full,
un-declustered** catalog (`etas.yaml: fit.method = mle`, `fit.background = stochastic_declustering`),
maximizing the point-process log-likelihood

$$\ln L = \sum_i \ln \lambda(t_i, x_i, y_i \mid \mathcal H_{t_i})
- \int_0^T\!\!\int_A \lambda(t, x, y \mid \mathcal H_t)\, dx\, dy\, dt ,$$

within the bounds in `etas.yaml: fit.bounds`.

**Two distinct stability gates** (kept logically separate in `etas.yaml: stability`):

1. **Finite branching** — `require_alpha_lt_beta`: the magnitude integral converges only if
   $\alpha < \beta$ with $\beta = b\ln 10$. If $\alpha \ge \beta$ the productivity–magnitude integral
   diverges.
2. **Subcriticality / stationarity** — `reject_supercritical`: *given* $\alpha < \beta$, the
   branching ratio $n$ (expected direct offspring per event) must satisfy $n < 1$. A fit with
   $n \ge 1$ is supercritical (explosive), signals a mis-fit, and is rejected.

### 2.3 The transparent fallback — Reasenberg–Jones

The most transparent "tomorrow's earthquakes" model and the human-auditable cross-check. The rate of
aftershocks $\ge M$ following a mainshock $M_m$ is a Gutenberg–Richter magnitude term times a
modified-Omori decay (Reasenberg & Jones 1989, *Science* 243(4895), 1173–1176,
doi:[10.1126/science.243.4895.1173](https://doi.org/10.1126/science.243.4895.1173)):

$$\lambda(t, M) = \frac{10^{\,a + b(M_m - M)}}{(t + c)^{p}}, \qquad
N = \int \lambda\, dt, \qquad P(\ge 1) = 1 - e^{-N} .$$

In [`model/reasenberg_jones.py`](../src/caos_seismic/model/reasenberg_jones.py) this conditions on
the single largest triggering event before `t_issue`, distributes the regional total over cells by a
Wells–Coppersmith rupture-length spatial kernel, and **flags its constants as
California-derived** — the methodology's hard rule *"do not reuse California parameters for Chile"* is
encoded in `params_used` and surfaced in the manifest. R-J is a sanity check, not the primary spatial
forecaster: where ETAS is the production estimator, R-J answers "does ETAS roughly agree with the
textbook Omori extrapolation?".

---

## 3. The stronger model, honestly delivered

The "stronger model" is delivered in two layers, and the second is **gated**, not default.

**Layer 1 — region-refit space–time ETAS with the full hygiene pipeline** the simplistic baselines
omit: per-region rolling $M_c$, Mw homogenization, dual-catalog declustering, propagated parameter
uncertainty, and full CSEP testability on a fine grid. This alone is *materially stronger by design*
than any hand-binned Hawkes / coarse-rectangle approach, because it is likelihood-fit, calibrated,
and S-testable.

**Layer 2 — a gated neural challenger** (feature-flagged, never the default): a conditional
spatio-temporal Neural Point Process with a **Hawkes inductive bias** — keep the additive background
+ summed-triggering skeleton, replace the fixed kernels with small MLPs/attention, and **model
magnitude explicitly** (a real gap in most NPPs). The challenger reaches the public map **only if**
it beats ETAS in *our* prospective CSEP harness (positive information gain, T-test CI excluding zero)
**and** is calibrated. Otherwise it stays behind the flag.

---

## 4. The honest ETAS-vs-ML verdict

**As of writing, no machine-learning model has been shown to reliably beat a well-fit ETAS for
short-term forecasting under fair, prospective, CSEP-style testing.** This is a *requirement* we
gate against, not a claim that ML can never add skill.

The decisive evidence is the **EarthquakeNPP** benchmark (Stockman, Lawson & Werner, *TMLR* 2026,
arXiv:[2410.08226](https://arxiv.org/abs/2410.08226)): five modern neural point processes (NSTPP,
DeepSTPP, AutoSTPP, DSTPP, SMASH) on California 1971–2021 with **strict chronological splits** and
**CSEP consistency tests** — **none outperformed ETAS**. On the ComCat dataset, ETAS passes the
consistency tests at roughly 95.8 % (N-test), **92.0 % (spatial)**, 93.8 % (magnitude), and 97.6 %
(pseudo-likelihood); the best NPP reaches ~86–88 % on the number / pseudo-likelihood tests but only
**~68.6 % on the spatial test** — exactly the dimension that matters for a map. The crucial
methodological fix was repairing a **data-leakage flaw** in earlier neural-TPP-for-earthquakes work
(non-chronological splits inflate metrics via triggering; excluding the 2011 Tohoku sequence makes
the benchmark irrelevant). Once temporal splits and the big sequences are restored, the apparent
neural advantage evaporates. This is *the* evaluation lesson.

> **Scope caveat.** This "NPPs do not beat ETAS" result is established on the California benchmark to
> date (1971–2021), not proven globally. It justifies shipping ETAS-class only for v0 and gating any
> neural model behind a CSEP win — stated as *"on the benchmark to date,"* not as an unconditional
> law. Some hybrid/neural models match or beat plain ETAS on information gain *in specific settings*.

**Where learned value genuinely comes from.** The honest exemplars — RECAST (Dascher-Cousineau et
al. 2023, *GRL* 50, e2023GL103909,
doi:[10.1029/2023GL103909](https://doi.org/10.1029/2023GL103909)) and FERN (Zlydenko et al. 2023,
*Sci. Rep.* 13, doi:[10.1038/s41598-023-38033-9](https://doi.org/10.1038/s41598-023-38033-9)) — gain
from two *ETAS gaps*, not from network depth: (1) **multivariate covariate ingestion** ETAS cannot
easily absorb (sub-$M_c$ events, geodesy, multiple catalogs), and (2) **learned spatial anisotropy**.
RECAST improves on temporal ETAS only when the training catalog is large ($\gtrsim 10^4$ events) and
merely matches on smaller ones. FERN+'s reported 4–12 % information-gain improvement came mostly from
ingesting sub-$M_c$ events and learning fault-aligned anisotropy — and the authors' own caveats are
release-blockers for us: it was *not* CSEP-tested, gave *no* uncertainty quantification, and its test
period ended *before* Tohoku $M_w$ 9.0.

**The cautionary tale.** DeVries et al. (2018, *Nature* 560, 632–634,
doi:[10.1038/s41586-018-0438-y](https://doi.org/10.1038/s41586-018-0438-y)) reported AUC 0.85 for
aftershock spatial pattern from a deep net (~13,451 free parameters); Mignan & Broccardo (2019,
*Nature* 575, E1–E3, doi:[10.1038/s41586-019-1582-8](https://doi.org/10.1038/s41586-019-1582-8))
matched it with a **2-parameter logistic regression** on a single feature. The guardrails we keep
from it: assume overfitting whenever parameters $\gg$ effective samples; never inflate sample size
with correlated per-cell framing; and **AUC / accuracy are banned as primary forecasting metrics** —
AUC is invariant to monotone rescaling, hence blind to the calibration of the very probabilities a
forecast publishes.

**Detection is not forecasting.** ML waveform models (PhaseNet, EQTransformer, SeisBench, the SeisLM
foundation model, arXiv:[2410.15765](https://arxiv.org/abs/2410.15765)) are mature for phase-picking,
detection, and characterization. They build **better, more complete catalogs** — which helps both
ETAS and any neural forecaster, the single biggest realizable near-term lever — but they **do not
forecast**. This line stays explicit in product copy so detection branding never implies prediction.

---

## 5. Calibration and uncertainty (release blockers)

### 5.1 Calibration

The public probability is **recalibrated** (isotonic regression; `forecast.yaml: calibration.method =
isotonic`) and validated with a **reliability diagram per horizon** ("when we said 5 %, it happened
~5 % of the time"). Calibration is a **release blocker** (`calibration.release_blocker = true`) — an
uncalibrated probability does not ship. The number is always rendered next to the climatological /
Poisson baseline so a user reads "X % vs Y % baseline." The reliability diagram is computed by
`reliability_diagram()` in [`eval/csep.py`](../src/caos_seismic/eval/csep.py) and emitted in the
artifact's `calibration.reliability` field exactly as
`CalibrationSummary.reliability = [[forecast_prob, observed_freq, n], …]`.

### 5.2 Uncertainty bounds must be *real*

The UI ships an **optimistic (P10) · expected (median) · pessimistic (P90)** triad
(`forecast.yaml: bounds.quantiles = [0.10, 0.50, 0.90]`) — the empirically best uncertainty design
(Schneider et al. 2022, *NHESS* 22, 1499–1518,
doi:[10.5194/nhess-22-1499-2022](https://doi.org/10.5194/nhess-22-1499-2022)). The bounds are a
genuine epistemic+aleatory decomposition, sourced from:

1. **ETAS parameter uncertainty** — MLE covariance / bootstrap (or a Bayesian posterior).
2. **$M_c$ / $b$-value estimation uncertainty** propagated through the exceedance integral. The
   Shi & Bolt (1982) $\sigma_b$ from `aki_utsu_b_value()` in
   [`catalog/completeness.py`](../src/caos_seismic/catalog/completeness.py) is a first-class input
   here, not a decoration.
3. **Structural / model-selection uncertainty** (ETAS variant choice).
4. **Over-dispersion** — regional seismicity is over-dispersed relative to Poisson (variance $\gg$
   mean, because of clustering), so the pessimistic bound is **wider than a naive Poisson quantile**
   (`bounds.overdispersion = negative_binomial`; cf. Kagan 2017, *GJI* 211(1), 335–345,
   doi:[10.1093/gji/ggx300](https://doi.org/10.1093/gji/ggx300)). A pessimistic bound that is just a
   Poisson quantile systematically under-warns at the tail and is **not acceptable**.

These bounds land in every artifact cell as `lo` / `hi` around the expected `p`
(see [`contracts.py: CellForecast`](../src/caos_seismic/contracts.py)).

---

## 6. Cold-start / low-seismicity regions

Most of any map's area-and-time is quiet, so cold-start is the *dominant* regime and must be honest
there.

- **Floor to a principled background, not an arbitrary constant.** Where recent seismicity is
  sparse/zero, the conditional rate floors to the long-term smoothed-seismicity Poisson background
  $\mu(x,y)$ (§2.1) — never a hard-coded per-day floor.
- **Borrow strength spatially.** Hierarchical / empirical-Bayes pooling and regionalized priors let
  cells with few events inherit a sensible prior from their tectonic neighbourhood rather than
  producing noisy or fake rates.
- **Three honest UI states, visually distinct:** (a) *low but nonzero, poorly constrained* — wide
  bounds, near-baseline expected value; (b) *genuinely quiescent* — near-baseline with tight bounds;
  (c) *no data / out-of-coverage* — an explicit hatch/mask (the artifact's `coverage_mask`). **Blank
  must never read as "safe."**

Because the reliability diagram is dominated by these quiet cells, calibration is validated
*specifically* in the cold-start regime, not only during active sequences.

---

## 7. Short-term aftershock incompleteness (the highest-stakes window)

Immediately after a large mainshock — exactly when the forecast matters most and is most consumed —
the catalog is grossly incomplete: $M_c$ spikes for hours-to-days while small events are buried in
the coda. A naive ETAS fit then **under-forecasts productivity** precisely when a large aftershock is
most likely. The daily job therefore uses an **incompleteness-aware likelihood** — a time-dependent
$M_c(t)$ post-mainshock (`completeness.yaml: short_term_incompleteness.method =
time_dependent_mc`, triggered above `trigger_magnitude`) rather than a flat threshold. This is a
decided method, not an open question; under-forecasting at this moment is both a credibility and an
(indirect) communication failure. OEF-Italy's documented under-estimation of the 2016–2017 Central
Italy sequence (Spassiani et al. 2023, *GJI* 234(3), 2501–2518) is the field's worked example of
exactly this failure mode.

---

## 8. Why this beats a simplistic baseline, by design

| Failure mode of a simplistic baseline | What CAOS_SEISMIC does |
|---|---|
| No $M_c$, flat thresholds, network-bias contamination | Rolling per-region $M_c(x,y,t)$ (MAXC+GFT/EMR), incompleteness-aware post-mainshock |
| No declustering decision | Explicit dual-catalog rule (ZBZ/GK background, full catalog for triggering) |
| Mixed mb/Ms/Mw | Mw homogenization anchored on ISC-GEM / GCMT |
| Binary ROC-AUC, no null, calibration-blind | CSEP N/M/S/L/CL consistency + IGPE T/W comparison vs Poisson **and** ETAS; reliability diagram; AUC banned |
| Coarse rectangles (un-S-testable) | Fine $0.1°$ grid (S-testable); region-level only for display |
| Hand-set constants, no CIs | MLE fit, propagated parameter + structural + over-dispersion uncertainty |
| Arbitrary per-day floor | Principled smoothed-seismicity background + hierarchical pooling for cold-start |
| Soft feature/label leak | Forecast-clock causal cutoff (features $<t$, label $[t, t{+}H)$) + immutable input snapshot |

Information gain over a Poisson reference is **state-dependent** (reported in **nats**, the CSEP
unit, never bits): positive and large during active aftershock sequences, near zero in quiet
periods, with a modest all-period average. For scale, time-independent model-vs-smoothed-seismicity
contrasts in prospective California CSEP give IGPE of only about $-0.7$ to $+0.5$ nats. We report the
gain as state-dependent, never as a fabricated round figure.

---

## Reference index

Dascher-Cousineau et al. 2023 (doi:10.1029/2023GL103909) · DeVries et al. 2018
(doi:10.1038/s41586-018-0438-y) · Geller et al. 1997 (doi:10.1126/science.275.5306.1616) ·
Helmstetter, Kagan & Jackson 2007 (doi:10.1785/gssrl.78.1.78) · Jordan et al. 2011
(doi:10.4401/ag-5350) · Kagan 2017 (doi:10.1093/gji/ggx300) · Mignan & Broccardo 2019
(doi:10.1038/s41586-019-1582-8) · Ogata 1998 (doi:10.1023/A:1003403601725) · Reasenberg & Jones 1989
(doi:10.1126/science.243.4895.1173) · Schneider et al. 2022 (doi:10.5194/nhess-22-1499-2022) ·
Shi & Bolt 1982 (BSSA 72(5), 1677–1687) · Spassiani et al. 2023 (GJI 234(3), 2501–2518) ·
Stockman, Lawson & Werner 2026 (EarthquakeNPP, arXiv:2410.08226) · Wells & Coppersmith 1994
(BSSA 84(4), 974–1002) · Zlydenko et al. 2023 (doi:10.1038/s41598-023-38033-9).

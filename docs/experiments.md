# Experiment register — CAOS_SEISMIC

A living, append-only record of every substantive experiment and modelling change: the **motivation**
(what problem / hypothesis), the **change**, the **result** (with measured numbers), and the
**justification** for the decision taken. The point is that no change to the model or pipeline is silent
— each is traceable to evidence. Dates are absolute (UTC). This is the technical source; the public
narrative is mirrored to the wiki (`Methodology-History`, `Changelog-and-Progress`).

> Honest-framing note carried into every entry: these are **conditional forecasts, never predictions**.
> A change is only "an improvement" if it shows **prospective** (leakage-free, forward-in-time) skill in
> the CSEP back-analysis — retrospective fit gains are recorded but never published as skill.

**Entry format.** `E<n> — <title> (<date>)` · Motivation · Change · Result · Decision/justification · Refs.
Metrics: IGPE = information gain per earthquake (nats, vs the stated baseline); N/M/S/L = CSEP consistency
tests; Mc = magnitude of completeness; b = Gutenberg–Richter b-value.

---

## E1 — Global catalog enrichment via literature Mw homogenization (2026-06-16)

**Motivation.** The raw USGS ComCat pull, homogenized to moment magnitude (Mw), collapsed from **228,008
→ 16,540** events (−93%). Root cause: Mw conversion required a per-network station-pair regression
(ISC-GEM / GCMT reference) that was not fetched, so ~80% of `mb`/`Ms`-only events were dropped instead of
converted. Standing directive: *never lose records*.

**Change.** `data/clean.py`: added `LITERATURE_DEFAULTS` — Scordilis (2006) global `mb→Mw` and `Ms→Mw`
orthogonal-regression conversions used as a **fallback** when no station-pair regression is available
(`use_literature_defaults=True`, `TLSFit.source` records provenance: `scordilis2006` vs a fitted line).

**Result.** **16,540 → 97,230 events (6×)**, 1990–2026, M≥4.5 global. Caveat: the global b-value inflates
(Scordilis compresses the saturated-`mb` tail), later measured at b≈1.59 on the naive global fit and
b≈1.337 once the regime-tiled fit estimates b per tile.

**Decision/justification.** Enrich rather than discard — a 6× larger conditioning catalog is worth a
documented, bounded magnitude-conversion bias. The b inflation is flagged as **open** (→ E-pending: GCMT
data-driven conversion). Provenance is recorded per event so the bias is auditable, not hidden.

**Refs.** Scordilis (2006), *J. Seismol.* 10, 225–236.

---

## E2 — Pipeline made tractable: five O(N²) / architecture fixes (2026-06-17)

**Motivation.** `train` and `infer` on the 97k global catalog hung — multiple all-pairs hot-spots, plus a
hidden architecture error.

**Change.**
1. **ETAS MLE** — neighbour cutoffs (`max_parent_dist_km`, `max_parent_days`) + precomputed vectorized
   parent→child pairs, so each likelihood evaluation is one `np.bincount` (no Python per-event loop).
2. **`expected_counts`** — rewritten: the Omori temporal window integral is now **closed-form**
   (`G(age+H) − G(age)`, the Omori-Utsu CDF; exact, replaces the 24-step midpoint quadrature) and the
   cell↔parent spatial pairs come from a single unit-sphere `cKDTree.sparse_distance_matrix` with a
   vectorized per-pair density + scatter-add.
3. **Smoothed-seismicity** adaptive bandwidth + null inference → `cKDTree` k-nearest (was all-pairs).
4. **Gardner–Knopoff declustering** → `cKDTree` `query_ball_point` (was all-pairs sweep).
5. **TiledForecaster** memoizes the cell→tile routing across the repeated per-(horizon, threshold) calls.
6. Global fit grid carved coarsely worldwide + refined only around M≥6 events: **948k → ~120k** scoreable
   cells.
7. **`infer` switched from a monolithic global ETAS to the regime-tiled primary** — the actual reason
   inference hung. A single ETAS over the worldwide 10⁵-event catalog is both O(N²) AND physically wrong
   (subduction ≠ stable interior); it also disagreed with the model `train` reports in the manifest.

**Result.** Tiled `expected_counts` over the 119k-cell global field **79.4 s → 12.1 s (6.5×)**, identical
result (sum 64.07; the small delta is the exact integral vs the old quadrature). Full pipeline runs
end-to-end (~2 min). 59 unit tests green throughout.

**Decision/justification.** Correctness-preserving accelerations (KD-tree, closed-form integral) plus the
architectural alignment that `infer` must serve the SAME regime-tiled model `train` fits.

**Refs.** Ogata (1998); Helmstetter, Kagan & Jackson (2007) for the adaptive background.

---

## E3 — First trained global model + live forecast (2026-06-17)

**Result.** Regime-tiled ETAS: **195 ETAS tiles** fitted (265 tiles fall back to their smoothed null),
b=1.337, Mc=5.35 on 96,981 conditioning events. Single-window self-consistency: **IGPE 4.65 nats vs the
Poisson null** (the conditional model adds real skill over the stationary baseline). N-test **fails**
(forecast under-counts the 248-event holdout) — calibration pending, reported honestly. First artifact:
31,344 world H3 cells (res 3) + 13 country views (res 5), P(M≥5, 7 d) up to 0.185 in the most active
cells. **LIVE** at `seismic.fasl-work.com` (1.18 MB gzipped artifact, git-as-data → GitHub Pages).

**Decision/justification.** Ship the honest ETAS field with the N-test failure surfaced, not hidden. The
self-consistency IGPE is a single-window diagnostic; the authoritative skill measure is the prospective
back-analysis (E5), not this number.

---

## E4 — Back-analysis efficiency: fit-cadence + recondition + coarse scoring grid (2026-06-17)

**Motivation.** The multi-country pseudo-prospective back-analysis re-fit the ENTIRE per-tile MLE at every
issue date — exhaustive but ~4–5 h for 60 days × 7 views. Two distinct costs were conflated.

**Change.**
1. **Fit on a cadence, recondition daily.** The ETAS parameters and the long-term smoothed background are
   stable over a refit cadence (publish.yaml `train_cadence.full_refit = weekly`); only the triggering
   conditioning (which events are parents, their ages) moves day to day. Added
   `ETASForecaster.recondition` / `TiledForecaster.recondition` (refresh parents, hold parameters +
   background) and a `refit_every_days` (default 7) in `run_back_analysis`. Leakage-safe (recondition
   admits only events < t_issue).
2. **Coarsen the SCORING grid to 0.5°** (`DEFAULT_SCORING_CELL_DEG`). M≥5 events are spatially sparse, so a
   0.1° grid is almost entirely empty cells — the CSEP S/L tests are under-powered AND ~25× more cells are
   inferred every issue day (the dominant cost once the MLE is cadenced).

**Result.**
- Recondition validated **correct**: forecast field agrees with full-refit-every-day to **0.05% median /
  ~0.5% typical** deviation (the baseline result is preserved, just computed efficiently).
- 7-day × 7-view multi-country run: **34 min → 2.6 min (13×)** with both levers. Full 60-day baseline
  now ~30–40 min (was ~4–5 h).

**Decision/justification.** A measurement-preserving efficiency change that ALSO mirrors how the live
system runs (fit weekly, condition daily). The coarser scoring grid is methodologically **sounder** for
sparse large events (better-powered CSEP cells), not a quality compromise. The published daily FORECAST
still uses the fine grid; only multi-day skill SCORING coarsens. Key insight recorded: for the
multi-country back-analysis the bottleneck is the per-day INFERENCE, not the MLE — so the grid (not the
fit cadence) is the dominant lever for small/dense views.

---

## E5 — Multi-country prospective baseline: the high-vs-low-seismicity bias (2026-06-17)

**Motivation.** The thesis question: does the model's skill generalise, or does it only look good because
it over-fits high-seismicity margins? Measure IGPE-vs-null per country across high- and low-seismicity
classes (leakage-free, 60-day pseudo-prospective).

**Result (IGPE vs the Poisson null, 60-day window, 1-day horizon).**

| View | Class | IGPE vs null (1d) | IGPE vs null (7d) | N-test pass |
|---|---|---:|---:|---:|
| JP Japan | high | **+0.072** | +0.062 | 0.95 |
| CL Chile | high | +0.0015 | +0.0024 | 1.00 |
| US-CA California | high | ~0 | +0.0016 | 0.99 |
| NZ New Zealand | high | ~0 | −0.001 | 0.98 |
| US-CE Central US | low | 0.0 | 0.0 | 1.00 |
| EU-W W. Europe | low | 0.0 | 0.0 | — |
| AU-E E. Australia | low | 0.0 | 0.0 | — |

(Global whole-Earth view: pending the running job at time of writing.)

**Interpretation / justification.** Exactly the honest, expected pattern: ETAS adds skill over the
stationary null **where there is triggering to exploit** (Japan, the densest margin), and collapses to the
null where there isn't (low-seismicity interiors). This **is** the high/low bias, quantified — the
adversarial question has a measured answer rather than an assertion. Low-seismicity 0.0 is not a failure;
it is the correct statement that a self-exciting model has nothing to add without aftershock sequences.

---

## E6 — Neural challenger trained on GPU (2026-06-17)

**Motivation.** The thesis headline is the information gain of a **context-conditioned** model over
catalog-only ETAS. Train the neural TPP (`model/context_tpp.py`: a CNN context encoder feeding a
Hawkes-structured conditional intensity — a RECAST-class ETAS-residual design, not a raw transformer).

**Change/result.** Trained on the global catalog on the RTX 4070 Laptop GPU (CUDA, torch 2.6+cu124),
checkpoint persisted (`data/weights/context_tpp_global_*.pt`, outside git). The ETAS-comparison GATE
(IGPE neural-vs-ETAS on a held-out tail) was running at time of writing. Context channel is currently
**seismicity-only** (the geodetic/stress/tidal enrichers are a later data wave), so the honest
expectation is a near-zero gate until real context covariates are wired — surfaced via
`context_channel_active`, never faked positive.

**Decision/justification.** Use the GPU for the one genuinely GPU-bound workload (the neural challenger);
the classical ETAS/tiled/smoothed pipeline stays on CPU by nature. Skill is established ONLY by the
prospective back-analysis, never by this in-loop gate alone.

**Refs.** Dascher-Cousineau et al. (2023) RECAST; the EarthquakeNPP benchmark (no NPP has shown robust
prospective gain over ETAS as of 2024) — see the pending deep-research synthesis.

---

## E7 — Continuous heatmap field mode (web, visual only) (2026-06-17)

Added a "Style" toggle on the Monitoring map: a continuous GPU-KDE heatmap (`@deck.gl/aggregation-layers`)
alongside the discrete H3 hexbins. Same per-cell values, same perceptually-uniform colormap — purely a
different rendering, never a new value or a danger ramp. Hexbins remain the default.

---

## E8 — Ensemble forecaster: weighted linear opinion pool (2026-06-17)

**Motivation.** The most reliable evidence-backed lever to beat any single short-term seismicity model is
to **combine** them — the CSEP consensus across collaboratories is that a well-weighted ensemble matches
or beats the best single component, because the components fail in different places (ETAS over-relies on
recent triggering; smoothed-seismicity carries the stationary background; Reasenberg–Jones anchors the
aftershock decay).

**Change.** New `model/ensemble.py`: `EnsembleForecaster` combines the per-cell expected counts of
already-fitted components as a linear opinion pool `λ_ens = Σ_k w_k λ_k` (weights normalized over the
components that evaluate). `build_default_ensemble` wraps a fitted model family (tiled ETAS + smoothed
null + R-J). Equal weights for now; score-weighted stacking deferred. 5 unit tests; 64 total green.

**Result.** Implementation + unit tests only — **prospective skill not yet measured**. It will be scored
by the same back-analysis harness (a new component alongside ETAS and the null) and only then gets a
measured IGPE; recorded here so the change is traceable even before its result lands.

**Decision/justification.** Linear (not log-linear) pooling is the CSEP-standard rate combination: it is
conservative (the ensemble rate never collapses because one component does) and, with the mandatory
smoothed null as a member, never reads below the long-term Poisson floor. The ensemble is a forecaster
like any other — it earns a place in the public field ONLY if it shows prospective IGPE in the
back-analysis, never by assertion.

**Refs.** Marzocchi, Zechar & Jordan (2012), *BSSA* 102; Rhoades et al. (2014, 2018) hybrid CSEP models.

---

## E9 — Multi-model benchmark: the naive ensemble underperforms ETAS (2026-06-17)

**Motivation.** Score every model (ETAS, the ensemble, Reasenberg–Jones, the null) head-to-head — the
"benchmark all models" deliverable — and test whether the E8 ensemble actually beats ETAS.

**Change.** A single-window, leakage-free **IGPE-vs-the-Poisson-null** benchmark over 8 views (countries +
global): fit the model family at a 30-day-back cutoff, score each model on the held-out tail
(`results/benchmark.json`).

**Result (IGPE vs null, nats; the held-out window).**

| View | n_obs | ETAS | ensemble | R-J |
|---|---:|---:|---:|---:|
| global | 248 | **+3.712** | +2.795 | −0.025 |
| CL | 11 | **+0.036** | −0.061 | −0.182 |
| JP | 5 | −0.081 | −0.013 | −0.438 |
| NZ | 2 | −0.007 | −0.012 | −0.038 |

(Absolute values are window-specific — a 30-day-horizon single window, distinct from the daily-averaged
back-analysis E5; the RELATIVE ranking is the finding.) ETAS beats the null where there is signal; the
**equal-weight ensemble (ETAS + null + R-J) is consistently WORSE than ETAS alone** (global +2.80 < +3.71;
CL −0.06 < +0.036); R-J ≈ the null or worse.

**Decision/justification.** The naive ensemble is **not** an improvement — averaging in the null and the
weak R-J **dilutes** the ETAS triggering signal that *is* the skill. This exactly confirms the cited
evidence ([improvement-evidence](improvement-evidence.md) F1: ensembles help only when **score-weighted**
with **strong** members). So we do **not** ship the equal-weight ensemble; the real lever is score-weighted
stacking of ETAS *variants* (next). Recording the negative result is the point — it is honest evidence,
and it redirects effort away from a dead end.

---

## E10 — Geodetic covariate (GNSS strain) wired into the neural challenger (2026-06-17)

**Motivation.** The product's thesis is "how much does global context contribute over catalog-only ETAS?"
Until now the neural challenger's context channel was **seismicity-only**, so its gate was ≈ 0 by
construction (E6). The cited evidence ([improvement-evidence](improvement-evidence.md) F2) says **GNSS
strain rate adds GLOBAL skill** (GEAR1; Strader 2018) but **not regional** skill (Bayona 2022). Turn the
channel ON with the real covariate and measure the prospective gain.

**Change.**
- Fixed the NGL MIDAS enricher (`data/enrichers/gnss.py`): HTTPS (the HTTP:80 endpoint times out) + a
  correct fixed-column lat/lon parse (the old value-range heuristic mis-read the E/N velocity pair as
  lon/lat → 0 stations). Now 20,168 stations; strain rate **high on active margins** (Japan 95,
  California 154, Chile 64 nstrain/yr) and **low in stable interiors** (Central US 3.6, W Europe 5.6).
- New `data/covariate_provider.py` (`make_strain_provider`): a `CovariateFieldProvider` that grids the
  strain into the CNN's `CovariateField` (`gnss_strain_rate` channel) at a coarse 1° global pitch.
- Retraining `context_tpp` with the channel **active** and gating its IGPE vs ETAS (the gate itself was
  fixed in E6-followup to use the coarse grid + tiled-ETAS reference, so it is now tractable).

**Result (2026-06-18) — INCONCLUSIVE on the geodetic question, due to a neural calibration bug.**
The strain channel was filled and the neural retrained (NLL 243), but the gate is catastrophically
negative: `igpe_vs_etas_nats = -485`. The cause is NOT the covariate — it is the neural's **absolute-rate
calibration**: its `expected_counts` forecasts **122,148** events over the 30-day global holdout when
**248** occurred (ETAS forecasts **173.7**, ≈ correct). So the neural over-predicts the rate ~**490×**, and
the IGPE's rate-normalization term `(N̂_chal − N̂_etas)/N ≈ +492` dominates the score. Diagnosed: the
softplus `mu_head` (background) is ~490× too high; the training compensator (Monte-Carlo
`integrated_intensity`) does not constrain the forecast-grid integral to the true rate, so the NLL never
penalises the over-prediction enough.

**Update (shape-only gate added) — the geodetic context IS a positive contribution.** A SHAPE-only gate
(renormalize the challenger field to the ETAS total before scoring, isolating the spatial/temporal skill
from the absolute-rate bug) gives **`igpe_vs_etas_shape_nats = +0.0533`** (raw, calibration-dominated:
−391.8; neural forecasts 98,932 vs 248 observed; ETAS 173.7). So **net of the calibration bug, the
strain-conditioned neural places M≥5 events BETTER than ETAS by +0.053 nats/eq** — the same order as the
global-vs-null IGPE (0.0835) and Japan (0.072). This is the thesis's positive signal: **the global
geodetic context improves the forecast SHAPE**. Honest caveats: single-window measurement, the neural's
absolute calibration is still broken (N-test fails) so it is not production-usable as-is, and the literature
sets the prior at a *modest* global gain (consistent with +0.053).

**Root cause of the calibration bug (diagnosed).** The training compensator (`_compensator_term`)
approximates the background spatial integral `∫mu dA` by `mu.mean()` at **event locations** — that
constrains mu only where events occur, leaving the `mu_head` free to inflate mu in event-free cells, so the
forecast-grid integral in `expected_counts` over-estimates (~490×). The fix is a calibration that ties the
absolute level to the training rate.

**Calibration FIXED (2026-06-18) — the calibrated context-neural beats ETAS on the in-loop gate.** Added a
fit-time `rate_cal` scalar (anchors the integrated 1-day forecast rate to the training rate ≥ Mc), computed
on the **same multi-resolution grid the gate uses** (the first attempt over-corrected — 98,932 → 3.25 —
because it was fit on a uniform 1° grid whose inferred per-cell area differs from the gate's grid). Result:

| | neural | ETAS | observed |
|---|---:|---:|---:|
| forecast count | **218.9** | 173.7 | 248 |
| N-test | **PASS** (q=0.028) | — | — |

`igpe_vs_etas_nats = +0.0748` (calibrated, raw), `igpe_vs_etas_shape_nats = +0.026`, **`gate_passed = True`**.
So the **GNSS-strain-conditioned neural TPP, once calibrated, beats ETAS by +0.075 nats/eq on the in-loop
gate and passes the N-test** — a calibrated positive result, and the strain context is a real channel
(`gnss_strain_rate` is active; the other geophysical channels remain honestly zero-filled).

**Decision/justification — measured honesty (critical, it IS the product ethos).** This is a **single-window**
in-loop gate, NOT the authoritative pseudo-prospective back-analysis. The literature is unambiguous that
single-window / retrospective wins do **not** generalise — that is precisely why no NPP has beaten ETAS
prospectively, and why the reference repos' wins evaporate under rigorous testing. So this is **not** a
claim that "we beat ETAS"; it is an **encouraging, calibrated signal that the geodetic context helps, which
now needs pseudo-prospective validation** (running the context-neural as the primary across many issue
dates) before it can be called a real prospective result. The blocker for that validation is that the
neural re-trains per fit (~74 min) — a cadenced/reconditioned neural-training path is the engineering
prerequisite (a new pending experiment). The strain enricher + provider + the rate_cal calibration are
correct and reusable.

---

## E11 — Pseudo-prospective validation of the neural, dual-horizon (2026-06-18)

**Motivation.** E10's +0.075 nats was a *single 30-day window* in-loop gate. The literature (and the
EarthquakeNPP benchmark) is unambiguous that single-window wins do not generalise. E11 is the
authoritative test: does the geodetic-context neural beat ETAS **prospectively**, across many leakage-free
windows, and **at which horizon**?

**Design.** Fit the strain-conditioned neural **once** at the earliest cutoff, then **recondition** it
forward across 8 weekly leakage-free windows (the cadenced path — no retrain per window), scoring the
IGPE of the neural over the tiled ETAS at **two horizons: 7 days (the product horizon) and 30 days (the
gate horizon)**. The two-horizon design directly tests the hypothesis that the geodetic advantage is
**horizon-dependent** (helps where the background/context dominates, i.e. longer horizons; loses where
ETAS triggering dominates, i.e. short horizons).

**Change that made it tractable (justification — Felipe's efficiency principle).** The neural's
`expected_counts` was a per-cell Python loop over the 119,717-cell global grid × 12 time steps (~1.4M net
forward passes, **~51 min per scoring window** — the full 8-window dual-horizon run would have been ~7 h).
It was **vectorized**: the productivity `kappa` and spatial scale `zeta` depend only on the parent events
and the temporal kernel `g` only on the step, so they are computed once and the cell loop collapses to a
chunked great-circle distance matrix. Pinned numerically identical to the old quadrature by
`tests/test_context_tpp_vectorization.py` (max relative error 6.7e-8, float32 round-off). Per window
dropped from ~51 min to seconds.

**Result (DECISIVE, horizon-dependent).**
- **7-day product horizon:** mean IGPE vs ETAS = **−0.053** over 8 windows, only **4/8 positive**, very
  noisy (per-window {−0.16, −0.08, +0.14, +0.09, −0.09, +0.23, +0.23, **−0.80**}; the last is a sequence
  window the neural badly missed). **The neural does NOT beat ETAS at the operational horizon.**
- **30-day horizon:** mean IGPE vs ETAS = **+0.078** over 4 windows, **4/4 positive** ({+0.052, +0.019,
  +0.103, +0.137}). **E10's single-window +0.075 gate win GENERALISES at 30 days.**

**Interpretation (the key finding).** The geodetic context is a **real lever, but only at the longer,
background-dominated horizon** — not at the 1–7 day product horizon where ETAS triggering dominates. The
decisive mechanistic detail: the calibrated neural's total forecast is **constant across windows** (18.1 at
7 d, 77.3 at 30 d, ratio ≈ 30/7) ⇒ it deploys a time-flat **geodetic background**, not a triggering model.
That is *why* it loses at 7 d (can't track sequences) and wins at 30 d (better background dominates the
integral). This does not retract E10; it **bounds where that signal is real**, and it directly motivates a
**14–30 day outlook product** (geodetic background) + the `mu = mu_neural` hybrid at 7 d gated to
no-regression (`wip/horizon-aware-and-hybrid-design.md`; frontier-research rank 2).

---

## E12 — Score-weighted ETAS-variant stacking ensemble (2026-06-18)

**Motivation.** The deep-research evidence base (improvement-evidence.md, F1) is that **score-weighted
ETAS-family ensembles are the *only* proven prospective lever over a single well-fit ETAS** — and only by
a *small* margin (+0.016 ± 0.028 IGPE; the CI already straddles zero, Herrmann & Marzocchi 2023). E8-E9
recorded that the *naive equal-weight* pool of heterogeneous models {ETAS, smoothed-null, R-J}
**underperforms** ETAS (it dilutes the triggering signal). E12 builds the construction the evidence
actually supports.

**Design (adversarial panel synthesis, `wip/e12-stacked-ensemble-design.json`).** A convex
**log-score-optimal stacking** of **3 ETAS-FAMILY variants** — V0 base tiled ETAS (the pinned anchor),
V1 short-memory (fast Omori: `max_parent_days` 730→120, `max_parent_dist_km` 500→250, `p∈[1.0,1.4]`), V2
long-memory (late-aftershock tail: `max_parent_days` 730→1825, `p∈[0.9,1.1]`). A **single global** weight
vector is learned from the rolling strictly-past Poisson log-score (== the IGPE numerator) and **shrunk
hard toward the V0 vertex** (L2/Dirichlet). The data budget is the binding constraint (~248 global M≥5 per
30-day window): with K=3 we fit only 2 free weights — no per-regime weights, no model zoo.

**The anti-dilution guarantee (the E8-E9 fix, structural):** every member is a tiled-ETAS triggering
model; the smoothed null enters **only** as each member's `mu(x,y)` background + the downstream cold-start
floor, **never** as a weighted component (`build_etas_stack_ensemble` raises otherwise). Weights are
**learned**, so a useless member is driven to its floor; the anchor is pinned and the optimum is provably
≥ base in-sample and collapses to exactly base on sparse/quiet holdouts. Honest expectation: a small
(+0.01..+0.02) and **possibly non-significant** gain.

**Pre-registered ship / no-ship rule (fixed BEFORE the run, so the decision cannot move after seeing the
number).** E12 **ships** (replaces base tiled ETAS as primary) iff ALL hold: (a) mean paired IGPE(E12 vs
base tiled ETAS) > 0 on the GLOBAL view AND in ≥2 high-seismicity regional views; (b) the Rhoades-2011
paired-T 95% CI excludes zero on the positive side at the **7-day** horizon; (c) the W-test corroborates
(p<0.05, positive median); (d) per-window IGPE positive in ≥60% of weekly windows and the sign does not
flip on the most-recent quintile; (e) E12 loses in no low-seismicity view by more than the noise floor
(mean IGPE ≥ −0.005); (f) a negative-control shuffled-label fit does NOT score as well as the real fit;
(g) E12 passes N/M/S/L consistency no worse than base. **If any of (a)–(g) fails, E12 is recorded as a
dead-end** with the measured IGPE — "no demonstrated prospective gain over single tiled ETAS" is itself a
valid result, exactly as E9 was recorded.

**Status.** Core IMPLEMENTED + unit-tested (8 tests). Single-window global gate gave +0.0087 (10/12
windows), beating the shuffled-label negative-control's max.

**Multi-region gate (the pre-registered decision) — NO-SHIP.** The leakage-free 14-window 7-day
back-analysis scored the single global weight vector on the GLOBAL view AND each high-seismicity view:

| view | mean IGPE vs base | windows + | n events | Rhoades-T CI |
|---|---|---|---|---|
| **GLOBAL** | **+0.0111** | 13/14 | 751 | **excludes zero (+)** |
| Chile | −0.0001 | 3/10 | 19 | crosses zero |
| Japan | −0.0008 | 8/13 | 54 | crosses zero |
| California | +0.0077 | 3/3 | **3 (noise)** | crosses zero |
| New Zealand | −0.0008 | 0/3 | 5 | crosses zero |

The pre-registered rule requires GLOBAL **and ≥2** high-seismicity views positive; only **1/4** is
(California, on 3 events — noise). **E12 does NOT ship.** The honest reading: the global gain is *real and
significant* (the long-memory variant V2 captures late-aftershock structure the base 730-day cutoff
misses, earning ~+0.011 pooled over 751 events) but it lives in the **diffuse global field, not the canonical
active margins** — at the regional level (few M≥5 events) it washes out. Recorded as a dead-end, exactly as
E9 was: "no robust prospective generalisation across regions." The reusable machinery (the stacking solver,
the `igpe_vs_base` channel, the negative-control + multi-region gate harness) is the lasting value.

**E12-adaptive (temporal time-decay of the holdout) — also NO-SHIP, and a lesson in scepticism.** Extended
the solver with per-window weights (`window_weights`) and re-ran with an exponential decay (τ=3 windows) so
the weights track the current sequence regime (the OEF-Italy lever). A *naive* gate reads "SHIP" (3/4
regions now have a positive mean), but scrutiny refutes it: (1) adaptive **reduces** the global gain
(+0.0048 vs +0.0111 static), failing the pre-registered "must beat base **and** static-E12" criterion; and
(2) the regional "improvements" are **noise** — CL +1e-5, JP +6e-4, California +3e-3 on **3 events**, NZ
still negative; with 3-54 events over 14 windows the IGPE noise floor is ~±0.05 and **no regional CI
excludes zero**. The 3/4-positive is a multiple-comparison artifact of sign-flipping noise. **Meta-finding:
the regional 7-day M≥5 gate is underpowered** — regional generalisation cannot be established either way at
this event density; the global view (751 events) is the only adequately-powered test, and there the static
stack already gave the honest answer. Temporal adaptivity does not rescue it.

---

## Research-2 — Frontier paths, adversarially validated: the honest 7-day ceiling (2026-06-18)

**Motivation (Felipe's directive).** Our 1–7 day results are honest but modest (E11 neural fails at 7 d,
E12 stacking gains only +0.0087). So: a NEW deep research into 2024–2026 frontier paths, with **every
hypothesis adversarially refuted** before it can enter the menu — the discipline that keeps us from
chasing retrospective hype. (38 agents; 6 frontier search angles → 10 hypotheses → 3 skeptics each → rank.)

**Result — all 10 hypotheses were killed (0 survived ≥2/3 refutation).** The convergent verdict:
**base tiled ETAS is at/near the practical ceiling for mean-rate 1–7 d IGPE over global M≥5.** No
neural/foundation TPP has beaten ETAS prospectively; the only proven lever (score-weighted ETAS ensembles)
caps at +0.016 ± 0.028 (CI crosses 0); our own E11/E12 confirm it. **Anyone promising a large 7-day gain is
selling retrospective/hype.** Killed paths included: time-varying b in the GR tail (OAF holds b fixed,
corrects the rate side; STAI is sub-Mc), regime-gated NPP routing (misreads EarthquakeNPP), EEPAS
multiplicative strain (long-term/retrospective), GCMT-anisotropic kernel (retrospective rupture
reconstruction; 90° plane ambiguity), MAGNET magnitude head (Mc 1.6–2.5, location handed in), STEP stack
member (same principles → no orthogonality), quadtree multi-resolution (raises S-test power, not IGPE).

**The redirection (the value of the null result).** "Near the ceiling on mean-rate IGPE" ≠ "nothing to
fix." Our **actually-measured binding failure** at 7 d is NOT spatial shape (which every killed path
attacked) — it is **count over-dispersion**: the benchmark global view has ETAS `n_forecast = 64.2` vs
`n_observed = 248` with the **Poisson N-test at quantile 0**, because the gridded-Poisson likelihood
assumes `Var[N] = E[N]` and over-rejects clustered (branching) sequence counts (Werner 2010; Kagan 2017;
Savran 2020). The adversarially-survived menu (`wip/frontier-paths-2026-06-18.json`):

1. **Over-dispersion-honest scoring (E13)** — negative-binomial / catalog-based N-test. Fixes the measured
   consistency failure; leakage-free; publishable even at zero IGPE movement. **Strongest evidence.**
2. **Horizon-aware deployment** — ship the geodetic-neural background as a NEW 14–30 d outlook (where E11
   measured a robust +0.05…+0.10) + hybrid `mu = mu_neural` at 7 d gated to no-regression.
3. **Temporally-adaptive E12 stack** — dynamic weights reweighted during sequences (the OEF-Italy lever);
   small headroom (+0.005…+0.015), possibly non-significant.
4. **Permanent stratification + negative-control guard** — institutionalize the over-fitting check (the
   shuffled-label control that exposed half of E12's headline as a non-spatial artifact).

## E13 — Over-dispersion-honest N-test: negative-binomial (2026-06-18)

**Motivation.** The forecast *product* already emits over-dispersed (negative-binomial / Gamma-mixture)
count bounds (`inference.daily`, `nb_r = 4`), but the *scoring* still used the **Poisson** N-test — so a
vigorous-but-plausible sequence is read as a gross miscalibration (the benchmark N-test fails at quantile
0). That mismatch is the binding consistency failure Research-2 surfaced.

**Change.** `csep.n_test_negbinom` scores the observed total against a negative-binomial null with mean
`N_fore` and variance `N_fore(1 + N_fore/r)`, `r` = the same over-dispersion the product uses; `r → ∞`
recovers the Poisson test exactly. 4 tests: Poisson limit; accepts moderate over-dispersion the Poisson
test wrongly rejects; **STILL fails the extreme 64-vs-248 case** (a 3.9× *rate* under-forecast is a real
bias, not dispersion — a dispersion fix must never whitewash it); degenerate-`r` fallback.

**Result.** The NB N-test corrects the *evaluation* (IGPE unchanged) for typical over-dispersed sequences,
and is now reported **alongside** the Poisson N-test in the back-analysis (`n_test_nb_pass_rate`).

**E13b — the catalog-based (branching-simulation) layer + the secondary-cascade finding.** Implemented
`model/simulate.py`: an ETAS branching-process forward simulator (pooled across sims, vectorized per
generation, counts only — locations are irrelevant to the total) that produces the **honest** in-window
`M≥mc` count distribution, plus a `catalog_based_n_test` against it (5 unit tests). Applied to the real
global tiled ETAS (per-tile params, interior parents → no halo double-counting; decomposition reproduces
`N_fore` exactly, ratio 1.00) on a 30-day global window:

| quantity | value |
|---|---|
| `N_fore` (frozen-intensity forecast) | 64.3 (background 60.1 + first-gen triggering 4.2) |
| simulated mean (full cascade) | **82.1** (+28% — the within-window secondary triggering the frozen forecast omits) |
| Fano (Var/mean, over-dispersion) | 1.8 |
| observed `M≥mc` | 96 |
| Poisson N-test | q=0.0001 → **FAIL** |
| catalog-based N-test | q=0.13 → **PASS** (sim p05=63, median=82, p95=103) |

**Two honest findings.** (1) The apparent under-forecast (64 vs 96) on this window is **over-dispersion +
secondary cascade, NOT a rate bias** — the catalog-based test passes where the Poisson test (which assumes
`Var=mean`) over-rejects. (2) A real, previously-unquantified effect: the frozen-intensity forecast
**systematically under-counts by ~28%** because it omits within-window secondary triggering — the realised
mean (82) exceeds `N_fore` (64). The catalog-based N-test is the right consistency tool; whether to also
publish the cascade-corrected expected count is a follow-up. (One 30-day global window; the method
generalises — full multi-window back-analysis integration is the next step.)

---

## Pending experiments (evidence-ranked menu — to be slotted in as run)

Ranked by expected prospective payoff, **grounded in the cited evidence base
([improvement-evidence.md](improvement-evidence.md), deep-research 2026-06-17)**: score-weighted
ETAS-family ensembles are the only proven prospective lever (and only by a *small* margin, +0.016 IGPE);
NO neural / foundation TPP beats ETAS prospectively; geodetic covariates help globally but can *hurt*
regionally. Each item gets its own E-entry with measured results when executed.

1. **Ensemble** — **IMPLEMENTED (E8) and BENCHMARKED (E9): the naive equal-weight version UNDERPERFORMS
   ETAS** (it dilutes the triggering signal). The remaining lever, **score-weighted stacking of ETAS
   *variants***, is now **E12 — core IMPLEMENTED + unit-tested** (the convex log-score-optimal solver with
   anchor shrinkage + the structural ETAS-family guard); the leakage-free multi-region back-analysis with
   the `igpe_vs_base_tiled` channel is the remaining run, against the pre-registered ship rule.
2. **Context covariates (the thesis experiment)** — **DONE for GNSS strain (E10): the calibrated
   context-neural beats ETAS on the in-loop gate (+0.075 nats, N-test passes).** **Pseudo-prospective
   validation RUNNING (E11)** via the new neural `recondition` (fit once, recondition forward across weekly
   windows) — the authoritative test of whether the single-window win generalises. The remaining covariates
   (Coulomb stress, fault proximity, slab, tides) are zero-filled and would be wired the same way.
3. **GCMT data-driven Mw conversion** — **DEFERRED, low priority.** The motivating problem (inflated b) is
   largely resolved: the naive global fit's b=1.586 (Scordilis tail-compression) becomes **b=1.337 under the
   per-tile fit** actually used in production. GCMT would sharpen the Mw homogenization further (a
   data-quality refinement), but the production b is already reasonable.
4. **Global event-event coupling channel (E11-bis — Felipe's teleseismic question)** — the current model
   conditions LOCALLY (~500 km ETAS cutoff). A channel of recent worldwide seismicity (or dynamic-stress)
   would test whether "Japan→Chile"-type coupling adds skill. Candidate, but the evidence is that remote
   triggering of large events has weak/unestablished **prospective** value (Parsons & Velasco 2011) — to be
   measured honestly, with ≈ 0 the likely (and still valid) outcome.
5. **Anisotropic / finite-fault aftershock kernel** — replace the isotropic spatial kernel for large
   events; ETAS variants that beat vanilla ETAS prospectively use fault-aware spatial decay.
6. **Score-weighted ETAS-variant ensemble** (from item 1) + **foundation-model** pre-training
   (`CAOS_RES_Foundation_Models`) — the longer-horizon levers.

> The authoritative, cited evidence base for this menu is the deep-research synthesis launched 2026-06-17;
> its findings will be folded into this register and the wiki `Models-*` pages.

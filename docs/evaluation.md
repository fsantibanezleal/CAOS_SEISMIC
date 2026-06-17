<!-- markdownlint-disable MD013 MD033 -->
# Evaluation

> **A forecast is only as honest as its prospective scoring.** Skill is established **only** by
> winning *comparison* tests against real baselines. Passing *consistency* (calibration) tests is
> necessary but **not sufficient**. The only scoring framework is **CSEP / pyCSEP** — no bespoke
> metrics. This page is the concrete, leakage-free back-analysis protocol; the governing equations
> are in [`methodology.md`](methodology.md) §Part 3.

CAOS_SEISMIC adopts **CSEP** (the Collaboratory for the Study of Earthquake Predictability) and its
community-endorsed toolkit **pyCSEP** (Savran et al. 2022, *SRL* 93(5), 2858–2870,
doi:[10.1785/0220220033](https://doi.org/10.1785/0220220033)). Using standard tests — not invented
metrics — is what makes the product defensible: reviewers can dispute the *model*, not the test code.
The scoring primitives live in [`eval/csep.py`](../src/caos_seismic/eval/csep.py); pyCSEP is the
authoritative path when installed, with dependency-free numpy fallbacks for the closed-form subset
(N-test, Poisson log-likelihood, information gain in nats, Brier, reliability bins) that carry an
explicit `pycsep_used: False` flag so a consumer never mistakes a fallback for the authoritative
result.

---

## 1. Two forecast representations (we emit both)

| Representation | What it is | Tests |
|---|---|---|
| **Gridded-rate** | Poisson expected count per space (×magnitude) cell | Poisson consistency tests; comparable to the published CSEP California daily benchmark |
| **Catalog-based** | ensemble of $\ge 10{,}000$ synthetic catalogs/day | empirical, non-Poisson tests |

Regional seismicity is **over-dispersed** relative to Poisson (variance $\gg$ mean, because of
clustering), so the Poisson grid tests **over-reject during aftershock sequences**. The catalog-based
tests (Savran et al. 2020, *BSSA* 110(4), 1799–1817,
doi:[10.1785/0120200026](https://doi.org/10.1785/0120200026)) relax the Poisson assumption and are
the **correct primary path** for a clustered daily forecast. Both representations are emitted
(`forecast.yaml: ensemble.emit_gridded_rate` and `emit_catalog_based`); a Poisson-grid-test failure
during a sequence is always reported **paired with** its catalog-based result. The pessimistic bound
is **wider than a naive Poisson interval** to honour the over-dispersion (negative-binomial behaviour;
cf. Kagan 2017, *GJI* 211(1), 335–345, doi:[10.1093/gji/ggx300](https://doi.org/10.1093/gji/ggx300)).

---

## 2. Pseudo-prospective is the primary mode

True prospective testing (depositing forecasts with a testing centre) is the gold standard and the
live tail; retrospective out-of-sample is weak; in-sample is a fit diagnostic only. The primary
back-analysis mode is **pseudo-prospective**, driven by a *forecast clock* (taxonomy per Mizrahi et
al. 2024, *Rev. Geophys.* 62, doi:[10.1029/2023RG000823](https://doi.org/10.1029/2023RG000823)).

**The forecast-clock driver is structural, not disciplinary.** At each daily issue time $t$, the model
is handed **only** the catalog slice $(-\infty, t)$ (`conditioning_slice` in
[`inference/clock.py`](../src/caos_seismic/inference/clock.py)), the forecast is sealed, then the clock
advances. The conditioning window $(-\infty, t)$ is half-open on the left and the target window
$[t, t+H)$ is half-open on the right, so back-to-back daily horizons tile the timeline without
double-counting a boundary event. `assert_no_leakage()` is a cheap always-on invariant called right
before fitting — defence in depth on top of the slice. The same code drives the live product and the
back-analysis, so they run *identical* logic. The back-analysis is invoked by
`caos-seismic backanalysis --start … --end …` (see [`cli.py`](../src/caos_seismic/cli.py)).

**Five leakage failure modes engineered against:**

1. **Temporal leakage** — solved by the forecast clock (features $<t$, label $[t, t+H)$).
2. **Catalog-revision leakage** — score against the catalog *as it would have been known at issue
   time*; snapshot the exact input catalog state for every daily issue (`snapshot_id()` in
   [`inference/provenance.py`](../src/caos_seismic/inference/provenance.py)) so a forecast is
   byte-reproducible. Where reconstructing historical real-time states is infeasible, the optimistic
   bias is documented, not hidden.
3. **$M_c$-inconsistency leakage** — $M_c$ estimated on the training window only; one uniform
   $M_{\min}$ at/above the worst regional $M_c$ (`completeness.yaml: target.m_min`), applied
   identically to the model and **every** baseline.
4. **Region/parameter snooping** — pre-register region polygons, horizons, thresholds, grid, and
   declustering; freeze hyperparameters across the test split.
5. **Multiple-testing inflation** — report **every** region × horizon cell *including failures*; never
   select the best cells post hoc.

A learning split of roughly **60–70 %** and a pseudo-prospective testing split of **30–40 %** with
frozen hyperparameters; pre-register the protocol before scoring.

---

## 3. Regions and periods

Run the back-analysis across **≥4 tectonically diverse regions** to prevent cherry-picking:

| Region | Authoritative catalog | Why included |
|---|---|---|
| Chile / N. Chile | CSN (via EarthScope/IRIS) | v0 target; subduction megathrust; strong region-specific ETAS |
| California | ANSS/ComCat, SCEDC/NCEDC | direct comparison to 25 published CSEP next-day models |
| Japan | JMA (internal use only; not redistributable) | dense network, very low $M_c$, abundant sequences |
| New Zealand | GeoNet | mixed subduction/crustal; established CSEP history |
| Italy (optional) | INGV | OEF-Italy operational analogue |

California specifically anchors against the gold-standard operational template: Serafini et al. 2025
(*Sci. Data* 12, 1501, doi:[10.1038/s41597-025-05766-3](https://doi.org/10.1038/s41597-025-05766-3))
— 25 automated $M \ge 3.95$ daily models, >50,000 next-day forecasts over California (Aug 2007–Aug
2018), all scored in pyCSEP. The load-bearing finding: **no single model dominates the decade** (STEP
excels during aftershock sequences); ETAS is the consistent generalist.

**Region-specific parameters.** Do **not** reuse California generics for Chilean subduction; source /
refit region-appropriate ETAS / Reasenberg–Jones parameters. Subduction megathrusts violate
isotropic-kernel and point-source assumptions for great events — note where a finite-fault/anisotropic
kernel is needed.

---

## 4. The target, declustering, and what "skill" means

The scored target is the **non-declustered** catalog — we deliberately forecast clustering. The
dual-catalog rule applies to *inputs* (declustered for the background, full for triggering; see
[`data-and-pipelines.md`](data-and-pipelines.md) §2), **not** to the target.

**The trap, stated plainly.** Because the target is mostly aftershocks, a trivial "aftershocks follow
mainshocks" model passes *consistency* tests, and a naive skill number can be an artifact of
clustering that *both* the model and ETAS capture. Therefore:

- **Consistency tests calibrate one model; they never establish skill.**
- **Skill = winning comparison tests against a real ETAS baseline** *and* a smoothed-seismicity
  baseline. The information-gain-over-ETAS measures genuine added skill precisely because both models
  already reproduce Omori clustering — so a positive, significant gain is not "I predicted
  aftershocks," it is "I predicted them better than the standard self-exciting model."

---

## 5. Exact tests and metrics

### 5.1 Consistency tests (calibration of one model)

All built on the Poisson joint log-likelihood over space (×magnitude) bins,
$L(\Omega \mid \Lambda) = \sum_i [-\lambda_i + \omega_i \ln \lambda_i - \ln(\omega_i!)]$
(`poisson_joint_log_likelihood()`):

- **N-test** (number; Poisson tails). With $N_{\text{fore}} = \sum_i \lambda_i$ and $F$ the Poisson
  CDF,

  $$\delta_1 = 1 - F\big(N_{\text{obs}} - 1 \mid N_{\text{fore}}\big), \qquad
  \delta_2 = F\big(N_{\text{obs}} \mid N_{\text{fore}}\big).$$

  Small $\delta_1 \Rightarrow$ observed too **many** for the forecast (forecast too low); small
  $\delta_2 \Rightarrow$ too **few**. Closed-form, always available (`n_test_poisson()`); two-sided
  rejection at $\min(\delta_1, \delta_2) < \alpha/2$.
- **M-test** (magnitude / GR shape, quantile $\kappa$), **S-test** (spatial, quantile $\zeta$),
  **L-test** (joint pseudo-likelihood, quantile $\gamma$), and the preferred **CL-test** (conditional
  likelihood, conditioned on $N_{\text{obs}}$ — preferred over the raw L-test because the L-test
  correlates strongly with the N-test; standardized definitions from Zechar, Gerstenberger & Rhoades
  2010, *BSSA* 100(3), 1184–1195, doi:[10.1785/0120090192](https://doi.org/10.1785/0120090192)). These
  are simulation-based: `consistency_tests()` routes them through pyCSEP's `poisson_evaluations`
  (`number_test` / `magnitude_test` / `spatial_test` / `likelihood_test` /
  `conditional_likelihood_test`). The code **refuses to fake** a simulation-based quantile — without
  pyCSEP they return `passed=None` with an actionable note rather than a fabricated number.

The catalog-based equivalents (number / spatial / magnitude / pseudo-likelihood) are the
over-dispersion-honest primary path; the L-test is **not** deprecated in current pyCSEP, but the
conditional L-test and the Poisson/negative-binomial number-test variants are preferred. (There is no
negative-binomial N-test in the core grid API; over-dispersion is handled via the catalog-based
number test.)

### 5.2 Comparison tests (where skill lives)

**Passing consistency tests is necessary but not sufficient.** Skill is established only by *winning a
comparison test* against a real baseline with a confidence interval that excludes zero. The metric is
**information gain per earthquake (IGPE)**, in **nats** (`information_gain_per_earthquake()`):

$$I_N(A, B) = \frac{1}{N}\sum_{i=1}^{N}\Big(\ln \lambda_{A}(k_i) - \ln \lambda_{B}(k_i)\Big)
- \frac{\hat N_A - \hat N_B}{N},$$

tested with the paired **T-test** $T = I_N(A,B) / (s/\sqrt{N}) \sim t_{N-1}$
(`comparison_tests()` → `_paired_t_test_igpe()`) and corroborated by the non-parametric **W-test**
(Wilcoxon signed-rank, `_w_test_igpe()`); Rhoades et al. 2011, *Acta Geophysica* 59(4), 728–747,
doi:[10.2478/s11600-011-0013-5](https://doi.org/10.2478/s11600-011-0013-5). The per-earthquake paired
differences are produced by expanding each bin by its observed count, so the T/W tests see the true
per-event series.

**Mandatory baselines** (the model must beat **both**): a **smoothed-seismicity** model and an
**ETAS** model. `comparison_tests()` is called once per baseline; `skill_demonstrated` is flagged
**only** when `igpe > 0` and the T-test CI excludes zero on the positive side, corroborated by the
W-test. Both results land in the artifact as `calibration.info_gain_vs_poisson_nats` and
`calibration.info_gain_vs_etas_nats` (see [`contracts.py: CalibrationSummary`](../src/caos_seismic/contracts.py)).

> **Honest framing of the gain.** ETAS-over-Poisson information gain is **strongly state-dependent**,
> not a fixed steady-state number: positive and large during active aftershock sequences (probability
> gains of up to orders of magnitude on peak days), near zero in quiet periods, with a modest
> all-period average. For scale, *time-independent* model-vs-smoothed-seismicity contrasts in
> prospective California CSEP give IGPE of only about $-0.7$ to $+0.5$ nats. We report the gain as
> state-dependent — never as a fabricated round figure, and always in **nats**, never bits.

### 5.3 Proper scoring rules and the alarm/ROC view (communication aids, not substitutes)

On top of CSEP we report strictly proper scoring rules (Gneiting & Raftery 2007, *JASA* 102(477),
359–378, doi:[10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)):

$$\text{LogS}(p, y) = -\ln p(y), \qquad
\text{BS} = \frac{1}{T}\sum_t (p_t - y_t)^2, \qquad
\text{CRPS} = \int \big(F(x) - \mathbf 1\{x \ge y\}\big)^2 dx .$$

The logarithmic score is the kernel of the L-test; the **Brier score** (`brier_score()`; origin Brier
1950) suits the bounded binary exceedance output; **CRPS** suits the full predictive distribution. For
an alarm-style view we use the **Molchan diagram** and **Area Skill Score** (Zechar & Jordan 2008,
*GJI* 172(2), 715–724, doi:[10.1111/j.1365-246X.2007.03676.x](https://doi.org/10.1111/j.1365-246X.2007.03676.x)).

> **ROC/AUC is shown only as a communication aid, never as a primary skill metric** — it is invariant
> to calibration and, on rare per-cell-per-day tasks, degenerates into a region classifier (the
> DeVries trap; see [`methodology.md`](methodology.md) §2.4). `eval/csep.py` deliberately does **not**
> implement AUC as a skill metric.

### 5.4 Calibration

A **reliability diagram per horizon per region** is the headline credibility artifact
(`reliability_diagram()` → `CalibrationSummary.reliability`). Calibration is validated **specifically
in the cold-start / quiet regime**, which dominates the diagram because most cells are quiet — not
only during active sequences — and the pyCSEP `calibration` test is reported alongside.

---

## 6. The full report table (report every cell, including failures)

For **each region × horizon (1d/2d/7d)** cell, report:

| Block | Contents |
|---|---|
| Consistency | N ($\delta_1, \delta_2$), M ($\kappa$), S ($\zeta$), L/CL ($\gamma$) — gridded **and** catalog-based |
| Comparison | IGPE (nats) vs smoothed-seismicity **and** vs ETAS, with T-test CI + W-test p |
| Calibration | reliability diagram + pyCSEP calibration test |
| Communication | Molchan / Area Skill Score |
| Scoring rules | Brier, Log score, CRPS |

**Report all cells including failures.** A region × horizon where the model fails to beat ETAS is
published as such — selective reporting is itself the selection-bias trap CSEP exists to prevent.

---

## 7. Provenance and reproducibility (existential)

Honest pseudo-prospective scoring is impossible without input-state snapshots — otherwise today's
model is scored against a retroactively-improved catalog (optimistic leakage). Each daily forecast
therefore persists, immutably and versioned: the **forecast** (gridded + catalog-based) with its
**issue timestamp**; the **exact input catalog snapshot** (events, magnitudes); and the **$M_c$ grid**,
declustering choice, model version, and all parameter values. This makes any past forecast
**byte-reproducible** and auditable months later — the property the entire value proposition depends
on (`build_manifest()` / `write_manifest()` in
[`inference/provenance.py`](../src/caos_seismic/inference/provenance.py)).

---

## 8. How results feed the web app

- **CSEP calibration badge** (green/amber/red) for **model quality only** — N/S/L/CL + reliability
  diagram. The traffic-light triad is reserved **exclusively** for model quality and **never** for the
  forecast field itself (green = within CSEP consistency, amber = borderline, red = rejected/
  under-tested — the only place red appears).
- **Reliability diagram per horizon** — the public "when we said X %, it happened ~X %" artifact,
  refreshed as the live record grows.
- **Expected-count-vs-baseline bars** and an **expected-vs-observed time series** in the no-map summary
  view (zero map-bundle cost, accessible / no-WebGL).
- **Per-region × per-horizon results**, including the cells where the model does **not** beat ETAS —
  published honestly.
- **Worked teaching example:** the 2019 Ridgecrest case (a ~3 % first-week forecast was *not wrong*
  when the ~3 % outcome occurred) to teach that single outcomes neither validate nor invalidate a
  probabilistic forecast (the UCERF3-ETAS pseudo-prospective evaluation, Savran et al. 2020).
- A prominent **last-run / staleness banner** and **coverage mask** so blank never reads as "safe," and
  the explicit statement that Poisson grid tests over-reject during sequences (paired with the
  catalog-based result).

---

## 9. Governance posture (public-launch gate)

Technical honesty (calibration, real uncertainty bands) is necessary but **not sufficient** for a
public live-number product. The L'Aquila earthquake (2009) and its trials are the field's cautionary
tale: the harm was **false reassurance**, not a failure to predict. CAOS_SEISMIC positions strictly as
an **independent, honest, calibrated research/education tool that complements** official OEF
(USGS/INGV/CSN), never as an authoritative civil-protection alarm. A small-but-elevated number is
framed to trigger **neither panic nor false reassurance** — always shown vs. the long-term baseline,
with explicit uncertainty, never a binary call. The inevitable "you said 2 % and it happened / you
said elevated and nothing happened" press cycle is met by **leading with the live reliability
record**, which is exactly how the field defines a good forecast.

---

## Reference index

Gneiting & Raftery 2007 (doi:10.1198/016214506000001437) · Kagan 2017 (doi:10.1093/gji/ggx300) ·
Mizrahi et al. 2024 (doi:10.1029/2023RG000823) · Rhoades et al. 2011 (doi:10.2478/s11600-011-0013-5)
· Savran et al. 2020 (doi:10.1785/0120200026) · Savran et al. 2022 (doi:10.1785/0220220033) ·
Schorlemmer et al. 2007 (doi:10.1785/gssrl.78.1.17) · Serafini et al. 2025
(doi:10.1038/s41597-025-05766-3) · Werner et al. 2011 (doi:10.1785/0120090340) · Zechar, Gerstenberger
& Rhoades 2010 (doi:10.1785/0120090192) · Zechar & Jordan 2008 (doi:10.1111/j.1365-246X.2007.03676.x).

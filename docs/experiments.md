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

## Pending experiments (evidence-ranked menu — to be slotted in as run)

Ranked by expected prospective payoff (informed by the SOTA: ensembles are the most reliable lever; no
single neural model has beaten ETAS prospectively as of 2024–2026). Each will get its own E-entry with
measured results when executed.

1. **Ensemble** (highest expected gain) — **IMPLEMENTED (E8)**; next step is to SCORE it in the
   back-analysis (a component alongside ETAS + null) to get a prospective IGPE, then add score-weighted
   stacking (weights from the rolling log-score) and EEPAS/STEP components.
2. **GCMT data-driven Mw conversion** — replace the Scordilis literature default with fitted station-pair
   regressions; expected to correct the inflated b and sharpen the magnitude tail (→ closes E1's caveat).
3. **Anisotropic / finite-fault aftershock kernel** — replace the isotropic spatial kernel for large
   events; ETAS variants that beat vanilla ETAS prospectively use fault-aware spatial decay.
4. **Context covariates** (the thesis experiment) — wire GNSS strain rate, Coulomb stress, fault
   proximity, slab geometry, tidal stress into `context_tpp`; re-train; measure whether the
   context-conditioned model shows **prospective** IGPE over ETAS (most such claims fail prospectively —
   measuring it honestly is the contribution).
5. **Foundation-model direction** (frontier) — pre-train on global seismicity, transfer per region
   (see `CAOS_RES_Foundation_Models`); document current limitations.

> The authoritative, cited evidence base for this menu is the deep-research synthesis launched 2026-06-17;
> its findings will be folded into this register and the wiki `Models-*` pages.

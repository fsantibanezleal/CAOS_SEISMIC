# H6 — Quadtree multi-resolution grid: adversarial validation (2026-06-18)

**Hypothesis.** Re-gridding our global M>=5 product onto a catalog-derived quadtree multi-resolution
grid (fine where dense, coarse where sparse) will RAISE the measured 7-day IGPE and S-test power over
the current uniform grid, without changing the ETAS model, by removing many-empty-cells dilution.

**Verdict: survives = FALSE.** The prospective/method evidence is real, but it is misread by the
hypothesis. Quadtree raises S-TEST POWER (testability), not INFORMATION GAIN; and the authors of the
exact cited work say IG comparison must use the *highest* resolution, i.e. the opposite of coarsening.

## What is genuinely true (evidence is real, not hype)

- Khawaja, Schorlemmer, Hainzl, Iturrieta, Savran, Bayona, Werner (2023, *BSSA* 113(1):333-347,
  doi:10.1785/0120220028) "Multi-Resolution Grids in Earthquake Forecasting: The Quadtree Approach" —
  real, peer-reviewed, integrated into pyCSEP. https://doi.org/10.1785/0120220028
- Khawaja, Hainzl, Schorlemmer et al. (2023, *GJI* 233(3):2053, doi:10.1093/gji/ggad030) "Statistical
  power of spatial earthquake forecast tests" — the S-test power paper. The "~8 earthquakes for max
  power vs >32,000 on the uniform high-res grid" figure is real.
  https://academic.oup.com/gji/article/233/3/2053/7000831
- It IS new for us: our grid is uniform (configs/grid.yaml world 1.0 deg / tile 0.5 deg fit), we have
  never used adaptive quadtree gridding. Essentially free to try.

## Why it does NOT support the hypothesis (the refutation)

1. **Power != skill — the cited paper says the opposite of H6 for IG.** The GJI power paper's own
   conclusion: *"pair-wise comparative testing of earthquake forecast models based on information gain
   (T-test) should be conducted using the highest available spatial resolution."* The quadtree boosts
   the **S-test's power to REJECT a spatially uninformative forecast** (test sensitivity), NOT the
   model's information gain. Higher S-test power means the test can more easily detect a *bad* spatial
   forecast — it can just as easily make a mediocre forecast FAIL. It is not a skill lever.

2. **IG systematically INCREASES with resolution; coarsening cannot generically raise it.** The
   literature consensus (and the quadtree authors) is that information gain of a forecast rises with
   the spatial resolution at which it is scored, because finer cells resolve where events actually go.
   Aggregating a fixed forecast onto a COARSER quadtree throws spatial information away. Any apparent
   IGPE rise from coarsening is a representation artifact (probability-mass concentration / changed
   normalisation), and our own back-analysis already exposes such non-spatial artifacts via the
   shuffled-label negative control (the E12 stacking finding). So "removing empty-cell dilution" is not
   a free IG gain — it is a different (lower-resolution) scoring of the SAME model.

3. **IGPE is per-EARTHQUAKE and largely grid-invariant for our headline.** Our IGPE (eval/csep.py,
   `information_gain_per_earthquake`) sums log-rate differences over the N observed events, normalised
   by N. Re-binning ETAS and the smoothed null onto the same quadtree shifts BOTH numerator terms
   together; the per-event log-rate *difference* (ETAS vs null) is what survives, and it is set by the
   model, not the grid. Coarsening mostly averages that difference DOWN, it does not manufacture it.
   The empty ocean cells the hypothesis blames contribute ~0 to IGPE already (no observed events, and
   the log-rate difference there is ~0), so they are not "diluting" the per-earthquake metric.

4. **Scope mismatch — validated regime is long-term, not 1-7 day.** Both papers validate on
   **time-independent GLOBAL M>=5.95, multi-year (1-yr scaled to 5-6 yr)** forecasts (GEAR1/WHEEL
   class). The "8 earthquakes" power figure is for that magnitude/horizon. Our product is M>=5,
   1-7 DAY, triggering-dominated. The operational short-term gold standard (CSEP-Italy weekly,
   Herrmann & Marzocchi 2023) uses a **UNIFORM 0.1 deg grid** — the adaptive-grid community has not
   moved short-term daily/weekly evaluation to quadtree. No prospective short-term evidence exists that
   quadtree improves a triggering forecast's measured skill.

5. **Does not beat our current stack.** Base ETAS already captures triggering; the geodetic-neural
   already failed at 7d; the proven lever is the modest score-weighted ETAS-family stack (E12,
   +0.0087 nats, half of it a non-spatial artifact). H6 changes the SCORING REPRESENTATION, not the
   generative model. It cannot add genuine forecast skill the model does not have; at best it changes
   the number we report, and per (1)-(2) it tends to change it DOWN for IG while changing it UP for
   S-test power. A higher S-test pass/power on a coarser grid is not a comparison-test win over ETAS,
   which is where our methodology (correctly) locates skill.

## Where a quadtree IS legitimately useful (not the hypothesis)

- As an EVALUATION DIAGNOSTIC: report the S-test at a quadtree resolution to gain detection power with
  our small M>=5 7-day event counts, ALONGSIDE (never replacing) the highest-resolution IG/T-test.
  This is exactly the intended pyCSEP use. It would strengthen our N/M/S/L *consistency* reporting, not
  raise the headline IGPE. Worth a small, clearly-labelled diagnostic add — but it is NOT a model
  improvement and must not be presented as raising skill.

## Bottom line

The evidence is real but the hypothesis inverts it: quadtree buys S-test POWER (testability), and the
cited authors explicitly say IG comparison should use the FINEST resolution. Coarsening cannot
generically raise a per-earthquake information gain that the model does not already produce, and the
validated regime is multi-year M>=5.95, not 1-7 day M>=5 triggering. survives = FALSE.

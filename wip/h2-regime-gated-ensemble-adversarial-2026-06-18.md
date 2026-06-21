# H2 (regime/phase-gated ensemble) — adversarial verdict

**Date:** 2026-06-18. **Verdict: DOES NOT SURVIVE.**

## Hypothesis
A phase-GATED ensemble routes background-dominated 7-day windows to our geodetic-neural ContextTPP and
active-aftershock windows to tiled ETAS, claiming to beat both base ETAS and our flat convex stack at 1-7d.

## What is genuinely TRUE (steel-man)
- The phase-decomposition claim is REAL and verified in EarthquakeNPP (arXiv:2410.08226, v3 Mar 2026):
  "model performance is strongest during 'background' periods... ETAS models the background with a
  **constant rate**, while the NPPs improve upon this by capturing the non-stationary nature of the
  background data" and "relative performance to ETAS is poorest during large earthquake sequences."
- This is genuine pseudo-prospective California evidence (24h CSEP sims), not pure hype. Claim (1) passes.

## Why it FAILS for OUR system (the refutations)

1. **The gap the NPP exploits is ALREADY CLOSED in our stack.** EarthquakeNPP's background advantage is
   measured against a **constant-rate** ETAS background. Our tiled ETAS does NOT use a constant background —
   it uses a per-tile **spatially-varying smoothed-seismicity** `mu(x,y)` as the ETAS background (tiled.py
   lines 13, 260-261, 322; that same smoothed null is also the mandatory Poisson floor). The non-stationary
   background skill the NPP buys you over constant-rate ETAS is the skill our smoothed background already has.
   The hypothesis imports an advantage measured against a baseline we do not run.

2. **Our NPP's "background" is geodetic-context-conditioned but TIME-FLAT — and it already lost head-to-head.**
   Our own E11 measurement: the neural forecast is constant across windows (18.0 at 7d, 77.3 at 30d, ratio
   ~30/7), i.e. it learned a static geodetic background, not triggering. At 7d it was mixed/weak
   (-0.16, -0.08, +0.14...). A gate that routes "background windows" to this model is routing them to a
   static field that, on our M>=5 global catalog, did not robustly beat ETAS at 7d even in the windows that
   ARE background-dominated. The phase split helps an NPP that has temporal-background skill over CONSTANT
   ETAS; ours competes against a smoothed-background ETAS and shows no robust 7d edge to gate toward.

3. **The EarthquakeNPP advantage is TEMPORAL and on DENSE SMALL-M catalogs; our product is SPATIAL on M>=5.**
   Their advantage is in temporal log-likelihood on Mc 0.6-3.0 California catalogs with thousands of events
   feeding the non-stationary-background estimate. "ETAS consistently outperforms ALL NPPs in spatial
   log-likelihood." Our product is a spatial field of M>=5 counts per cell, globally — exactly the axis
   (spatial) and regime (sparse large-M) where NPPs are weakest, and the improvement they report is
   "marginal" even on their favorable temporal axis.

4. **The gate ADDS estimation variance with almost no signal to gate on.** At M>=5 global, 7-day windows
   carry ~15 expected events (e12 n_fc_base ~15). A per-cell background-vs-active classifier (time-since-last
   -large-parent / recent triggering intensity) must be fit/validated prospectively per cell with this thin
   signal. The flat stack's HONEST 7d gain is only +0.0087 nats and ~half is a non-spatial artifact the
   negative control already exposes (shuffled max +0.0075). A noisy gate spends exactly this margin on
   routing variance. The literature prior on stacking is +0.016 +- 0.028 (CI crosses 0); a gate is strictly
   more parameters chasing a sub-noise effect.

5. **SCEC-2025 (Stockman/Werner, SCEC pub 14827) cuts AGAINST the framing.** Its abstract: generative NPPs
   "consistently underperform"; LL-NPPs show only "marginal temporal information gains over USGS"; and the
   seismologically-informed gains are "particularly for **immediate aftershocks**" — the ACTIVE regime, the
   opposite of routing the neural to background windows. No NPP beats ETAS operationally. The CLAIMED
   evidence that the gate is "the untested lever they leave open" is an inference we are adding, not a
   prospective result they report.

## Bottom line
(1) prospective phase-split evidence: REAL. (2) applies to global M>=5 7d: NO — it's temporal LL on dense
small-M California, vs a constant-rate ETAS we don't use, on the spatial axis where NPPs are worst.
(3) plausibly beats our current stack: NO — the background gap is already closed by our smoothed `mu(x,y)`,
our neural already failed at 7d, and the gate burns the sub-noise stacking margin on routing variance.
Default survives=false stands.

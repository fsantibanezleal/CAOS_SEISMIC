# Horizon-aware forecasting + the ETAS-triggering / geodetic-neural-background hybrid

**Status:** design (2026-06-18). Grounded in our OWN prospective measurement (E11), pending the full E11
means + the frontier research's adversarial verdict before implementation.

## The empirical fact that drives this (E11)

The pseudo-prospective dual-horizon validation shows the calibrated GNSS-strain neural is **horizon-
dependent** vs base tiled ETAS:

| window | 7-day IGPE | 30-day IGPE |
|---|---|---|
| w0 2026-04-22 | −0.162 | +0.052 |
| w1 2026-04-29 | −0.076 | +0.019 |
| w2 2026-05-06 | +0.145 | +0.103 |
| w3 2026-05-13 | +0.087 | … |

**The decisive detail:** the neural's total forecast is **constant across windows** — `neural_fc = 18.0`
at 7 d and `77.3` at 30 d in *every* window, and `77.3 / 18.0 ≈ 4.3 ≈ 30/7`. The forecast is **time-flat
and scales linearly with the horizon** ⇒ the neural's *triggering* contribution is negligible; what it
actually learned and deploys is a **geodetic-context-conditioned BACKGROUND rate**, not a triggering model.

That single fact explains everything:
- **Short horizon (1–7 d):** dominated by aftershock *triggering*. ETAS's Omori/productivity response to
  recent events wins; the neural's static background cannot track a vigorous sequence → it loses (w0/w1).
- **Long horizon (~30 d):** the background term dominates the integral; the neural's *better, geodetically
  informed background* wins → +0.05…+0.10 consistently.

## Two designs (the simple one Felipe endorsed, and the principled one the data points to)

### A. Horizon-switching ensemble (simple; the endorsed idea)
Pick the prospectively-best model per horizon: ETAS (or the E12 stack) for 1–7 d; the geodetic-neural
background for the ~30 d outlook. A per-horizon weight `w(H)` learned from the rolling prospective
log-score (reuse `fit_weights_from_history`, but indexed by horizon). Honest, trivial to ship, and it
turns the horizon-dependence into a feature. Naturally extends the product to a **30-day outlook tab**
where the geodetic context genuinely adds skill.

### B. ETAS-triggering + neural-background HYBRID (principled; what the constancy fact implies)
Since the neural's value IS its background and ETAS's value IS its triggering, the clean model is **one**
forecaster whose conditional intensity is:

```
lambda(x, t) = mu_neural(x | geodetic context)   +   sum_i  ETAS_triggering(x, t | event_i)
```

i.e. replace ETAS's smoothed-seismicity background `mu(x,y)` with the geodetic-context-neural's learned
background, and keep ETAS's analytic Omori/productivity/spatial triggering. Properties:
- **Short horizon:** triggering dominates ⇒ behaves like ETAS (no short-horizon regression — fixes the
  −0.16 failure mode).
- **Long horizon:** the better geodetic background dominates ⇒ inherits the neural's +0.05…+0.10.
- Single model, no switch, no double-counting (the neural background *replaces* the smoothed null as `mu`;
  the smoothed null remains the cold-start floor). This is exactly the **hybrid / neural-augmented ETAS**
  family the frontier research is evaluating (RECAST/deep-ETAS class) — but motivated here by our own data,
  not the literature.

**B subsumes A**: the hybrid automatically weights background vs triggering correctly at every horizon, so
it is the horizon-switching idea taken to its limit. A is the fast win; B is the real model.

## Disciplined gate before building (Felipe's own rule)
1. Wait for the **full E11 means** (8 windows) to confirm the crossover horizon and that 30 d is robustly
   positive (not just w0–w3).
2. Wait for the **frontier research adversarial verdict** on hybrid-ETAS / horizon-switching — if the
   skeptics refute it, record why and reconsider.
3. Confirm the neural-background quality directly: does `mu_neural` beat the smoothed null as a *background*
   (long-horizon, declustered) in a leakage-free score? If yes, B is well-founded.

## Next experiment (when the gate clears)
- **E13a (cheap):** horizon-indexed weights — score ETAS vs neural at H ∈ {1,2,7,14,30} prospectively,
  fit `w(H)`, report the per-horizon ensemble IGPE. Ships the 30-day outlook.
- **E13b (the model):** implement the hybrid `mu = mu_neural` inside `ETASForecaster`/`TiledForecaster`
  (a `background=` injection point already exists — the neural becomes the background provider), refit,
  run the leakage-free back-analysis at 1–7 d, gate against base tiled ETAS with the E12 pre-registered
  rule. Honest expectation: matches ETAS at 1–7 d (no regression) and gains at the longer outlook.

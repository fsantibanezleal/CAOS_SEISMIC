# H8 adversarial validation — STEP as a different-family stack member (2026-06-18)

## Hypothesis
Adding a STEP-style instantaneous-Omori clustering model (a different FAMILY) as a stack
member — instead of more tiled-ETAS variants — yields a larger/more stable 7-day stacking gain
than our all-ETAS convex pool, because stacking only extracts gain from members that fail in
different places and our V1-V4 are >0.9 correlated. Cites OEF-Italy (ETAS-LM + ETES + STEP) as
the canonical operational template.

## Verdict: survives = FALSE (refuted)

## Why the cited evidence refutes the hypothesis
H8's own canonical citation is Herrmann & Marzocchi 2023, GJI 234:73 ("Maximizing the
forecasting skill of an ensemble model", the OEF-Italy ensemble) — the SAME paper we already
cite as the +0.016 ± 0.028 anchor. In that paper:
- STEP (STEP_LG) is the WEAKEST member: CumIGPE = -0.861 ± 0.056 vs the SMA reference;
  ETES_FCM = -0.191; ETAS_LM (best) = +0.061 ± 0.027.
- The paper states STEP "often plays a minor or no role" and "degrades performance during
  major sequences" — i.e. it is driven toward ZERO weight, it does not supply the orthogonality.
- The re-weighted ensemble's edge over the BEST single model (ETAS_LM) is the same
  +0.016 ± 0.028 whose CI crosses zero. The gain is robustness ("never worse than best"),
  carried by ETAS_LM — not by STEP family diversity.
- Independent corroboration: comparative-eval Italy paper (arXiv 2405.10712, 2024) ranks the
  STEP/LG model LAST of 5; redundant in the ensemble (M>=4, Italy, 7-day daily, 2005-2020).

So the operational template H8 invokes shows the OPPOSITE of its claim: STEP's residuals are
"orthogonal" because STEP is WORSE, not because it fails where ETAS loses log-score in a way
that helps. A reliably-worse, occasionally-anti-correlated member earns ~0 convex weight.

## Prospective track record (STEP vs ETAS, 1-day)
- CSEP Italy daily + California: ETAS / K3 OUTPERFORM STEP at 1-day forecasts.
- The one "STEP ~ ETAS" result (Ward 2025, California, super-thinned/Voronoi residuals) is a
  residual-diagnostic comparison, "very comparable" (not a log-score/IGPE win), and regional
  California — not global M>=5 log-score.

## Scale/horizon mismatch with our product
OEF-Italy = M>=4 regional. Ours = GLOBAL M>=5 (Mc=5.35). STEP's R&J productivity is tuned on
dense regional aftershock clouds whose distinguishing events sit mostly BELOW our M>=5 threshold,
collapsing STEP toward the same triggering signal base ETAS already captures — REDUCING the very
orthogonality H8 needs, not increasing it.

## Collinearity premise
Correct that our V1-V4 are ~0.9+ correlated (our own E12 risk register flags this; the measured
+0.0087, ~half a non-spatial artifact, is consistent with a near-flat log-score in the weights).
But the proposed remedy is wrong: family diversity is NECESSARY, not SUFFICIENT. The member must
be competitively CALIBRATED at the scored log-score AND fail where ETAS loses. Every prospective
dataset shows STEP is not competitive with ETAS-LM at log-score, so it gets ~0 weight and cannot
push us past the +0.016 ± 0.028 ceiling (CI crosses 0) we already sit at with +0.0087.

## Honest counter-consideration (does not rescue it)
STEP is a genuinely maintained operational model with a real distinct ingredient (real-time
secondary aftershocks + R&J spatial anisotropy), so this is not pure hype, and building it would
be real family diversity. But every prospective IGPE/log-score comparison found ranks STEP below
ETAS-LM, which is exactly the condition under which a convex stack assigns it near-zero weight.

## Sources
- Herrmann & Marzocchi 2023, GJI 234:73 — academic.oup.com/gji/article/234/1/73/6994524
- Marzocchi et al. 2023, GJI, OEF-Italy 10-yr validation — dx.doi.org/10.1093/gji/ggad256
- Comparative evaluation (Italy) — arxiv.org/html/2405.10712v1
- EarthquakeNPP (Stockman et al. 2024) — arxiv.org/abs/2410.08226
- Ward 2025, Environmetrics — onlinelibrary.wiley.com/doi/10.1002/env.70014

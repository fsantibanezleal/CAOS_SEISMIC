# Improvement evidence base — how (and whether) to beat ETAS, with prospective evidence

A cited, adversarially-verified synthesis (deep-research run 2026-06-17; 5 search angles, 23 primary
sources fetched, 107 claims extracted → 25 verified → 9 high-confidence findings) of what the literature
**actually** shows about improving a short-term (1–7 day) probabilistic seismicity forecaster beyond a
well-fit space-time ETAS, as of 2024–2026. It is the evidence base that **justifies the
[experiment register](experiments.md) pending menu** — every proposed change points back here.

> Headline, uncomfortable, and load-bearing: **as of 2026, beating a well-fit ETAS prospectively is
> hard, and most published "wins" do not survive rigorous prospective testing.** This is not a reason to
> stop — it is the reason our product's contribution is *honest prospective measurement*, not a claim.

## The nine findings (all 3-0 adversarial votes unless noted)

1. **Score-weighted / BMA ensembles of ETAS-family models are the only approach with demonstrated
   prospective gain over single ETAS — and the gain is small.** OEF-Italy runs an operational
   score-model-averaging ensemble of ETAS + ETES + STEP (good expected-vs-observed agreement after 10 yr).
   Herrmann & Marzocchi (2023, *GJI* 234:73) improve the SMA ensemble by +0.078±0.023 IGPE but only
   **+0.016±0.028 over the best single member (ETAS_LM)** — "modest", not significant; a second fitting
   scheme gives +0.002±0.019. Framework: Marzocchi, Zechar & Jordan (2012, *BSSA* 102:2574) — Bayes
   factors + posterior / inverse-log-likelihood weights, applied to the RELM California prospective test.

2. **Geodetic covariates help GLOBALLY, not regionally.** GEAR1 (Strader et al. 2018) — smoothed
   seismicity + geodetic strain — is more informative than either alone at global scale. But at regional
   scale, geodetic-hybrid models (HKJ-SHEN) scored **negative information (~−0.68) vs the plain HKJ
   smoothed-seismicity baseline** over 2011–2020 (Bayona et al. 2022, *GJI* 229:1736). "Ensembles have not
   always performed much better than the best individual model" (Mizrahi et al. 2024, *Rev. Geophys.*).

3. **No neural point process beats ETAS in the EarthquakeNPP benchmark.** Stockman/Lapins et al. (TMLR
   2026, arXiv:2410.08226): **none of five NPPs outperform ETAS** on log-likelihood or generative metrics;
   "current NPP implementations are not yet suitable for practical earthquake forecasting." Prior NPP
   "wins" were inflated by **data leakage** (earthquake triggering across alternating train/test splits),
   **omission of the largest sequence** (2011 Tohoku M9 in the Japan benchmark), and **no comparison to
   state-of-the-art seismology baselines**.

4. **The 2023 neural wins (RECAST, FERN) are retrospective-only and data-hungry.** RECAST
   (Dascher-Cousineau et al. 2023, *GRL*, GRU NTPP) beats temporal ETAS in S. California **only when
   training >10⁴ events**, on retrospective splits / synthetic ETAS data — never under prospective/CSEP
   testing. FERN (Zlydenko et al. 2023, *Sci. Rep.*) is only "slightly better" than an **isotropic-kernel
   (weak-baseline) ETAS** on a Japan split, and **loses to ETAS in some regions**.

5. **Physics (Coulomb rate-state) is only comparable to ETAS, not better** — even with secondary
   triggering, variable slip and optimized parameters — and shares ETAS's structural weakness (it does not
   encode the next event's magnitude). The field's caution is grounded: DeVries et al. (2018, *Nature*)
   deep-learning aftershock model was shown by Mignan & Broccardo (2019, *Nature*) to be **no more
   informative than a two-parameter ("one-neuron") logistic regression**.

6. **No single model dominates all horizons → horizon-specific (or horizon-weighted) ensembling is a real
   lever.** Pseudo-prospective Italy (train 1990–2011, test 2012–2023, 27 Mw≥5 targets): **ETAS has the
   highest SHORT-term skill** (IGPA=1.35 at 1 month) while EEPAS dominates long-term (IGPA=0.72 at 6 yr,
   where ETAS falls to second-lowest). **For a 1–7 day global model this confirms ETAS-family triggering as
   the correct short-horizon backbone.**

7. **Only prospective evaluation is rigorous; temporally-stratified testing is essential.** Models can
   beat a baseline on validation yet **degrade on the most recent held-out quintile** (Koehler et al.
   2025) — overfitting to early seismicity. Information-gain-per-earthquake (IGPE) vs a reference is the
   right metric. **This validates our pseudo-prospective forecast-clock back-analysis + IGPE methodology.**

8. **The gold-standard prospective archive contains no ML.** The decade-long fully-prospective CSEP
   California archive (Serafini et al. 2025, *Nature Sci. Data*; 25 next-day models, >50k daily forecasts
   2008–2018) is almost entirely **ETAS variants + ETAS/STEP ensembles, with no neural/ML models** — so no
   direct prospective ML-vs-ETAS comparison exists; the ML-vs-ETAS evidence is all retrospective/pseudo.

9. **Foundation-model / flexible-neural frontiers are temporal-only, retrospective, no prospective gain
   (medium confidence).** NMRP (Zhan, Zhuang & Wu 2026, *Earth's Future*) reinterprets NPPs as neural
   modulated renewal processes, evaluated on EarthquakeNPP with **no physical covariates**;
   multimodal/foundation capability is proposed future work. The claim "NMRP beats ETAS" was **explicitly
   refuted (0-3)** — they only match/slightly trail.

## What this means for OUR model — the evidence-ranked, revised plan

| # | Lever | Evidence verdict | Action for us |
|---|---|---|---|
| 1 | **Score-weighted ensemble of ETAS-family** | Only proven prospective gain; **small** (+0.016 IGPE) | Keep the `EnsembleForecaster` (E8) but evolve it: **(a)** weight by the rolling prospective log-score, not equal; **(b)** make members ETAS *variants* (regimes, kernels), not ETAS+null+RJ; **(c)** add **horizon-specific weights** (F6). Score it in the back-analysis and report the honest (likely modest) IGPE. |
| 2 | **Horizon-specific weighting** | ETAS best short-term (F6) | Our 1–7 d window keeps ETAS-family as the backbone; let the ensemble down-weight long-horizon-only members. |
| 3 | **Geodetic strain covariate (global)** | Global yes, regional **no** (F2) | Wire GNSS strain into `context_tpp` and measure **global** IGPE; do **not** expect (and honestly report the absence of) regional gain. |
| 4 | **Neural / foundation TPP** | No prospective win anywhere (F3, F4, F9) | Treat as R&D only; keep the honest gate; pursue only in data-rich regimes (>10⁴ events) and never publish a retrospective "win". Our neural is documented as ≈ETAS until real covariates. |
| 5 | **Physics (Coulomb) covariate** | Comparable, not better (F5) | Low priority as a skill lever; useful as an interpretable feature, not a silver bullet. |
| — | **Evaluation rigor** | Prospective-only; temporal stratification (F7, F8) | Already our design. Keep IGPE + leakage-free clock + report failures; add temporally-stratified (recent-quintile) reporting. |

**Bottom line.** The evidence says the realistic, honest gain over our ETAS baseline comes from a
**score-weighted, horizon-aware ensemble of ETAS-family models** (small but real) plus **global** geodetic
context — not from a neural model (none beats ETAS prospectively) and not from regional covariate hybrids
(they can hurt). The product's distinctive value is doing the **prospective, multi-region, honest
measurement** that the field's gold-standard archive (all-ETAS, no-ML) does not provide for ML.

## Primary sources

- Herrmann & Marzocchi (2023). *GJI* 234(1):73. https://academic.oup.com/gji/article/234/1/73/6994524
- Marzocchi, Zechar & Jordan (2012). *BSSA* 102(6):2574. (BMA / Bayes-factor ensemble framework.)
- Marzocchi et al. (2023). *GJI* 234(3):2501. https://academic.oup.com/gji/article/234/3/2501/7207398 (OEF-Italy 10-yr.)
- Mizrahi, Schorlemmer et al. (2024). *Rev. Geophys.* 2023RG000823. https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023RG000823
- Bayona et al. (2022). *GJI* 229:1736 (prospective geodetic-hybrid, negative info scores).
- Strader et al. (2018). *SRL* (GEAR1 global seismicity+strain).
- Stockman, Lapins et al. (2026). *TMLR*. arXiv:2410.08226 (EarthquakeNPP — no NPP beats ETAS).
- Dascher-Cousineau et al. (2023). *GRL* (RECAST). Zlydenko et al. (2023). *Sci. Rep.* (FERN).
- DeVries et al. (2018). *Nature*; Mignan & Broccardo (2019). *Nature* (the one-neuron rebuttal).
- Serafini et al. (2025). *Nature Sci. Data* (decade-long prospective CSEP California archive).
- Zhan, Zhuang & Wu (2026). *Earth's Future* (NMRP; "beats ETAS" refuted).

*Full machine-readable synthesis (claims, votes, all sources) archived from the deep-research run on
2026-06-17. This document is the canonical justification; the wiki `Models-*` pages mirror the public
narrative.*

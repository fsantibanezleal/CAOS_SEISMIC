"""E13 — ETAS branching-process count simulation (the over-dispersion-honest catalog-based N-test).

Pins the properties that make the simulated count distribution a correct null:

1. No background + no parents ⇒ zero counts.
2. Background only (no triggering, K=0) ⇒ counts ~ Poisson(bg) ⇒ Fano ≈ 1 and mean ≈ bg (the simulation
   reduces to the deterministic forecast when there is nothing to cascade).
3. Triggering present ⇒ Fano > 1 (over-dispersed) AND the simulated mean EXCEEDS the frozen-intensity
   first-generation forecast (the secondary within-window cascade the forecast omits).
4. The catalog-based N-test accepts an observed total inside the simulated spread and rejects one far
   outside it (a genuine rate miss is NOT whitewashed).
"""

from __future__ import annotations

import numpy as np

from caos_seismic.model.simulate import catalog_based_n_test, simulate_window_counts

B = 1.0
BETA = B * np.log(10.0)


def _sim(**kw):
    base = dict(
        K=0.0, alpha=1.0, c=0.01, p=1.1, m0=3.5, mc=5.0, beta=BETA,
        bg_expected=0.0, parent_ages=np.array([]), parent_mags=np.array([]),
        horizon_days=7.0, n_sims=4000, rng=np.random.default_rng(0),
    )
    base.update(kw)
    return simulate_window_counts(**base)


def test_empty_gives_zero_counts():
    r = _sim()
    assert r.counts.sum() == 0
    assert r.simulated_mean == 0.0


def test_background_only_is_poisson():
    """K=0 ⇒ no triggering ⇒ counts ~ Poisson(bg): mean ≈ bg, Fano ≈ 1, forecast == simulated mean."""
    r = _sim(K=0.0, bg_expected=30.0, n_sims=8000, rng=np.random.default_rng(1))
    assert abs(r.simulated_mean - 30.0) < 1.5
    assert abs(r.fano - 1.0) < 0.15  # Poisson
    assert abs(r.forecast_first_gen - 30.0) < 1e-9


def test_triggering_is_overdispersed_and_exceeds_frozen_forecast():
    """With real parents + productivity, the cascade makes Var/mean > 1 and lifts the mean above the
    frozen-intensity first-generation forecast."""
    rng = np.random.default_rng(2)
    # a handful of recent moderate-large parents
    ages = np.array([0.5, 1.0, 2.0, 5.0])
    mags = np.array([6.5, 6.0, 6.8, 6.2])
    r = simulate_window_counts(
        K=0.25, alpha=1.0, c=0.02, p=1.15, m0=3.5, mc=5.0, beta=BETA,
        bg_expected=5.0, parent_ages=ages, parent_mags=mags,
        horizon_days=7.0, n_sims=8000, rng=rng,
    )
    assert r.fano > 1.2, f"expected over-dispersion, got Fano={r.fano:.2f}"
    # secondary cascade lifts the realised mean above the first-generation forecast
    assert r.simulated_mean > r.forecast_first_gen + 1e-6


def test_catalog_based_n_test_accepts_inside_rejects_outside():
    rng = np.random.default_rng(3)
    r = simulate_window_counts(
        K=0.2, alpha=1.0, c=0.02, p=1.15, m0=3.5, mc=5.0, beta=BETA,
        bg_expected=20.0, parent_ages=np.array([0.3, 1.0]), parent_mags=np.array([6.5, 6.0]),
        horizon_days=7.0, n_sims=8000, rng=rng,
    )
    inside = catalog_based_n_test(r.counts, int(round(r.sim_median if hasattr(r, "sim_median") else np.median(r.counts))))
    assert inside["passed"] is True
    # an observed total far above the 95th percentile is correctly rejected (genuine rate miss)
    extreme = int(r.counts.max() + 10 * (np.std(r.counts) + 1))
    outside = catalog_based_n_test(r.counts, extreme)
    assert outside["passed"] is False
    assert outside["delta1"] < 0.025  # "too many observed" tail


def test_n_test_quantiles_are_well_formed():
    r = _sim(K=0.15, bg_expected=15.0, parent_ages=np.array([1.0]), parent_mags=np.array([6.0]),
             n_sims=6000, rng=np.random.default_rng(4))
    res = catalog_based_n_test(r.counts, 15)
    assert 0.0 <= res["quantile"] <= 0.5
    assert res["sim_p05"] <= res["sim_median"] <= res["sim_p95"]

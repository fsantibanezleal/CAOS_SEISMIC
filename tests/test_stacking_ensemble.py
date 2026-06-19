"""E12 — score-weighted ETAS-variant stacking ensemble.

Pins the three structural guarantees that make the stack a real, fail-safe lever and not a repeat of the
E8-E9 equal-weight dilution dead-end:

1. **Plumbing** — ``TiledForecaster`` forwards the per-tile ETAS kernel overrides, so V1/V2 (short/long
   memory) are genuinely different triggering models.
2. **Stacking solver** — convex log-score-optimal weights: collapses to the anchor on a sparse/quiet
   holdout, recovers a clear signal when one member forecasts better, always returns a valid simplex
   point, and is never worse than the anchor in objective (the no-worse-than-base property).
3. **Structural guard** — ``build_etas_stack_ensemble`` refuses any non-ETAS-family weighted member (the
   smoothed null / R-J), which is exactly what diluted E8-E9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caos_seismic.contracts import BBox, Region
from caos_seismic.model.ensemble import (
    EnsembleForecaster,
    _solve_stacking_weights,
    build_etas_stack_ensemble,
)
from caos_seismic.model.smoothed import SmoothedSeismicityForecaster
from caos_seismic.model.tiled import TiledForecaster


def _region() -> Region:
    return Region(
        id="chile",
        name_en="Chile (test)",
        name_es="Chile (test)",
        bbox=BBox(lat_min=-37.0, lat_max=-21.0, lon_min=-75.0, lon_max=-68.0),
        m_max=9.0,
    )


def _two_cluster_catalog(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    times = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 700, n)), unit="D")
    lat = np.concatenate([rng.normal(-33.0, 0.6, n // 2), rng.normal(-25.0, 0.6, n - n // 2)])
    lon = np.concatenate([rng.normal(-72.0, 0.6, n // 2), rng.normal(-70.0, 0.6, n - n // 2)])
    mw = 3.5 + rng.exponential(0.45, n)
    return (
        pd.DataFrame(
            {
                "event_id": [f"e{i}" for i in range(n)],
                "time": times,
                "latitude": lat,
                "longitude": lon,
                "depth_km": rng.uniform(10.0, 60.0, n),
                "mag": mw,
                "mag_type": "Mw",
                "mw": mw,
                "source": "synthetic",
            }
        )
        .sort_values("time")
        .reset_index(drop=True)
    )


# ── 1. plumbing ───────────────────────────────────────────────────────────────


def test_tiled_forwards_etas_kernel_overrides():
    """A TiledForecaster(etas_max_parent_days=120, ...) yields per-tile ETAS with those cutoffs/bounds."""
    cat = _two_cluster_catalog()
    tf = TiledForecaster(
        tile_deg=10.0, mc=3.5, b_value=1.0,
        etas_max_parent_days=120.0, etas_max_parent_dist_km=250.0, etas_bounds={"p": (1.0, 1.4)},
    )
    tf.fit(cat, _region(), pd.Timestamp("2025-06-01", tz="UTC"))
    etas_tiles = [t for t in tf._tiles if getattr(t, "is_etas", False)]
    assert etas_tiles, "expected at least one ETAS tile in the two-cluster catalog"
    m = etas_tiles[0].model
    assert m.max_parent_days == 120.0
    assert m.max_parent_dist_km == 250.0
    assert m.bounds["p"] == (1.0, 1.4)
    # a base TiledForecaster keeps the ETAS defaults (no forwarding)
    base = TiledForecaster(tile_deg=10.0, mc=3.5, b_value=1.0)
    base.fit(cat, _region(), pd.Timestamp("2025-06-01", tz="UTC"))
    bm = next(t.model for t in base._tiles if getattr(t, "is_etas", False))
    assert bm.max_parent_days == 730.0  # ETAS default, untouched


# ── 2. stacking solver ─────────────────────────────────────────────────────────


def _synthetic_holdout(K, n_cells, signal_component=None, n_events=200, seed=0):
    """One-window holdout: each component a (n_cells,K) rate matrix + an omega vector.

    If signal_component is set, that component's rate field matches where the events actually are
    (so a good solver moves weight to it); component 0 is a diffuse uniform anchor.
    """
    rng = np.random.default_rng(seed)
    true_field = rng.random(n_cells) ** 3  # spiky truth
    true_field *= n_events / true_field.sum()
    omega = rng.poisson(true_field).astype(float)
    L = np.zeros((n_cells, K))
    L[:, 0] = omega.sum() / n_cells  # anchor: flat (knows the total, not the shape)
    for k in range(1, K):
        L[:, k] = omega.sum() / n_cells  # default flat
    if signal_component is not None:
        # the signal member knows the shape (a noised version of the truth, same total)
        sig = true_field * rng.uniform(0.8, 1.2, n_cells)
        L[:, signal_component] = sig * (omega.sum() / sig.sum())
    return [(L, omega)]


def test_solver_collapses_to_anchor_when_sparse():
    """< n_min events ⇒ weights are exactly e_0 (trust the base, don't fit 2 weights on a handful)."""
    holdout = _synthetic_holdout(K=3, n_cells=400, signal_component=1, n_events=5, seed=1)
    w = _solve_stacking_weights(holdout, 3, rho=10.0, w_min=0.0, n_min=8, anchor_index=0)
    assert np.allclose(w, [1.0, 0.0, 0.0])


def test_solver_recovers_signal():
    """When member 1 forecasts the event SHAPE and the anchor is flat, weight moves to member 1."""
    holdout = _synthetic_holdout(K=2, n_cells=600, signal_component=1, n_events=300, seed=2)
    w = _solve_stacking_weights(holdout, 2, rho=1.0, w_min=0.0, n_min=8, anchor_index=0)
    assert abs(w.sum() - 1.0) < 1e-9
    assert w[1] > 0.3, f"expected weight to move to the skilful member, got {w}"


def test_solver_valid_simplex_with_collinear_members():
    """Two identical members ⇒ a valid simplex point, no NaN, no crash."""
    rng = np.random.default_rng(3)
    omega = rng.poisson(0.3, 500).astype(float)
    L = np.tile((omega.sum() / 500.0), (500, 2))  # identical columns
    w = _solve_stacking_weights([(L, omega)], 2, rho=5.0, w_min=0.0, n_min=8, anchor_index=0)
    assert np.all(np.isfinite(w))
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.all(w >= -1e-12)


def test_solver_never_worse_than_anchor():
    """J(w*) >= J(e_0): the returned objective is no worse than putting all mass on the base."""
    holdout = _synthetic_holdout(K=3, n_cells=500, signal_component=2, n_events=250, seed=4)
    K = 3
    w = _solve_stacking_weights(holdout, K, rho=2.0, w_min=0.0, n_min=8, anchor_index=0)
    L = holdout[0][0]
    om = holdout[0][1]
    e0 = np.array([1.0, 0.0, 0.0])
    eps = np.finfo(float).tiny

    def J(wv):
        lam = np.clip(L @ wv, eps, None)
        return float(np.sum(om * np.log(lam) - lam)) - 0.5 * 2.0 * float(np.sum((wv - e0) ** 2))

    assert J(w) >= J(e0) - 1e-6


def test_fit_weights_from_history_sets_and_records():
    """The forecaster method sets component weights and records the stacking provenance."""
    # three dummy ETAS-family stand-ins (only the .name matters for the guard; weights come from holdout)
    class _Dummy:
        def __init__(self, name):
            self.name = name

    ens = EnsembleForecaster(
        components=[("base", _Dummy("tiled_etas"), 1.0),
                    ("v1", _Dummy("tiled_etas"), 1.0),
                    ("v2", _Dummy("tiled_etas"), 1.0)],
    )
    holdout = _synthetic_holdout(K=3, n_cells=500, signal_component=1, n_events=250, seed=5)
    ens.fit_weights_from_history(holdout, rho=1.0, n_min=8)
    ws = [w for _, _, w in ens.components]
    assert abs(sum(ws) - 1.0) < 1e-9
    assert "stacking" in ens.params_used
    assert ens.params_used["stacking"]["n_holdout_windows"] == 1


# ── 3. structural guard (the E8-E9 fix) ────────────────────────────────────────


def test_build_etas_stack_rejects_non_etas_member():
    """The smoothed null must NOT be admissible as a weighted stacking member."""
    base = TiledForecaster(tile_deg=10.0)
    null = SmoothedSeismicityForecaster()
    with pytest.raises(ValueError, match="ETAS-family"):
        build_etas_stack_ensemble(base, [("smoothed_null", null)])


def test_build_etas_stack_accepts_tiled_variants():
    """A base + two tiled-ETAS variants build a stack with the base as the anchor (index 0)."""
    base = TiledForecaster(tile_deg=10.0)
    v1 = TiledForecaster(tile_deg=10.0, etas_max_parent_days=120.0)
    v2 = TiledForecaster(tile_deg=10.0, etas_max_parent_days=1825.0)
    ens = build_etas_stack_ensemble(base, [("v1_short", v1), ("v2_long", v2)])
    assert ens.name == "etas_stack_ensemble"
    assert [n for n, _, _ in ens.components] == ["etas_tiled_base", "v1_short", "v2_long"]

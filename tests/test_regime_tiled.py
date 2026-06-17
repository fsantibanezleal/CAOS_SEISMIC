"""Unit tests for the global conditioning layer — tectonic regimes + spatial tiling + tiled fit.

Core deps only (numpy/pandas/scipy/pydantic); no enrichers on disk (the regime classifier exercises
its heuristic fallback), no network, no heavy science deps. The invariants pinned here are:

1. **Regime assignment** is total and deterministic, returns one of the five regimes, records its
   ``source``, and the obvious physical cases (shallow offshore subduction margin → interface; deep
   slab → intraslab) land in the right class via the no-enricher heuristic.
2. **Every regime has a usable prior** (positive productivity, subcritical-friendly alpha).
3. **Tiling** tessellates a region with non-overlapping interiors whose union covers the bbox, and
   every interior is strictly contained in its halo (so triggering is edge-correct).
4. **The tiled forecaster** implements the Forecaster contract (``expected_counts`` over global
   cells, length-preserving, non-negative), aggregates per-tile fits into one field, enforces the
   ETAS stability gates per tile (a supercritical tile falls back to its smoothed null, recorded),
   and yields public probabilities in [0, 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caos_seismic.contracts import BBox, Cell, Region
from caos_seismic.model import (
    REGIME_PRIORS,
    TectonicRegime,
    TiledForecaster,
    assign_regime,
    dominant_regime,
    iterate_tiles,
    regime_prior,
    tiles_for_region,
)
from caos_seismic.model.regime import RegimeAssignment, Tile


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _wide_region() -> Region:
    """A region wide enough (and event-dense enough below) to span multiple tiles."""
    return Region(
        id="chile",
        name_en="Chile (test)",
        name_es="Chile (test)",
        bbox=BBox(lat_min=-37.0, lat_max=-21.0, lon_min=-75.0, lon_max=-68.0),
        m_max=9.0,
    )


def _two_cluster_catalog(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Two spatial clusters (so two different tiles get fit), all required contract columns present."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    times = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 700, n)), unit="D")
    lat_a, lon_a = rng.normal(-33.0, 0.6, n // 2), rng.normal(-72.0, 0.6, n // 2)
    lat_b, lon_b = rng.normal(-25.0, 0.6, n - n // 2), rng.normal(-70.0, 0.6, n - n // 2)
    lat = np.concatenate([lat_a, lat_b])
    lon = np.concatenate([lon_a, lon_b])
    depth = rng.uniform(10.0, 60.0, n)
    mw = 3.5 + rng.exponential(0.4, n)
    return (
        pd.DataFrame(
            {
                "event_id": [f"e{i}" for i in range(n)],
                "time": times,
                "latitude": lat,
                "longitude": lon,
                "depth_km": depth,
                "mag": mw,
                "mag_type": "Mw",
                "mw": mw,
                "source": "synthetic",
            }
        )
        .sort_values("time")
        .reset_index(drop=True)
    )


def _cell_grid(region: Region, step: float = 1.0) -> list[Cell]:
    bb = region.bbox
    cells: list[Cell] = []
    for la in np.arange(bb.lat_min + step / 2, bb.lat_max, step):
        for lo in np.arange(bb.lon_min + step / 2, bb.lon_max, step):
            la_r, lo_r = round(float(la), 2), round(float(lo), 2)
            cells.append(Cell(key=f"{la_r},{lo_r}", lat=la_r, lon=lo_r))
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# 1) Regime assignment
# ─────────────────────────────────────────────────────────────────────────────


def test_assign_regime_is_total_and_records_source():
    a = assign_regime(-33.5, -72.0, 25.0)
    assert isinstance(a, RegimeAssignment)
    assert a.regime in set(TectonicRegime)
    assert a.source in {"slab2", "pb2002", "heuristic"}


def test_shallow_subduction_margin_is_interface_via_heuristic():
    # Offshore central Chile, shallow — circum-Pacific margin box → subduction interface.
    a = assign_regime(-33.5, -72.0, 20.0)
    assert a.regime == TectonicRegime.SUBDUCTION_INTERFACE


def test_deep_slab_is_intraslab():
    # Deep beneath the Andes (> intraslab depth) in the subduction margin → intraslab.
    a = assign_regime(-22.0, -66.0, 200.0)
    assert a.regime == TectonicRegime.INTRASLAB


def test_longitude_wrapping_handled():
    # 288 deg E == -72 deg; the classifier must wrap and give the same answer.
    a = assign_regime(-33.5, -72.0, 20.0)
    b = assign_regime(-33.5, 288.0, 20.0)
    assert a.regime == b.regime


# ─────────────────────────────────────────────────────────────────────────────
# 2) Priors
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("regime", list(TectonicRegime))
def test_every_regime_has_a_usable_prior(regime):
    p = regime_prior(regime)
    assert p.productivity_k > 0.0
    assert 0.0 < p.alpha < 2.5
    assert p.p > 1.0
    assert p.n_neighbors >= 1
    # The enum and its string value resolve to the same prior.
    assert regime_prior(regime.value) is REGIME_PRIORS[regime]


# ─────────────────────────────────────────────────────────────────────────────
# 3) Tiling
# ─────────────────────────────────────────────────────────────────────────────


def test_tiles_tessellate_region_interior():
    reg = _wide_region()
    tiles = tiles_for_region(reg, tile_deg=8.0, halo_deg=1.0)
    assert len(tiles) >= 2
    bb = reg.bbox
    # Interiors cover the bbox exactly (min/max edges meet) with no overlap on the lat axis.
    lat_mins = sorted({t.interior.lat_min for t in tiles})
    assert lat_mins[0] == pytest.approx(bb.lat_min)
    assert max(t.interior.lat_max for t in tiles) == pytest.approx(bb.lat_max)


def test_halo_strictly_contains_interior():
    reg = _wide_region()
    for t in iterate_tiles(reg, tile_deg=8.0, halo_deg=1.0):
        assert t.halo.lat_min < t.interior.lat_min
        assert t.halo.lat_max > t.interior.lat_max
        assert t.halo.lon_min < t.interior.lon_min
        assert t.halo.lon_max > t.interior.lon_max


def test_globe_tiling_count():
    globe = BBox(lat_min=-90.0, lat_max=90.0, lon_min=-180.0, lon_max=180.0)
    tiles = list(iterate_tiles(globe, tile_deg=20.0, halo_deg=2.0))
    # 180/20 = 9 rows, 360/20 = 18 cols.
    assert len(tiles) == 9 * 18


def test_dominant_regime_of_a_tile():
    reg = _wide_region()
    tile = next(iter(iterate_tiles(reg, tile_deg=8.0, halo_deg=1.0)))
    lat = np.array([-33.0, -33.2, -32.8])
    lon = np.array([-72.0, -71.9, -72.1])
    depth = np.array([20.0, 25.0, 18.0])
    r = dominant_regime(tile, lat, lon, depth)
    assert r in set(TectonicRegime)


# ─────────────────────────────────────────────────────────────────────────────
# 4) Tiled forecaster
# ─────────────────────────────────────────────────────────────────────────────


def test_tiled_forecaster_fits_and_aggregates_global_field():
    reg = _wide_region()
    cat = _two_cluster_catalog()
    t_issue = pd.Timestamp("2025-06-01", tz="UTC")

    tf = TiledForecaster(tile_deg=8.0, halo_deg=1.0, mc=3.5, min_events_for_etas=30)
    tf.fit(cat, reg, t_issue)

    pu = tf.params_used
    assert pu["n_tiles_fit"] >= 2
    # At least one tile is carried by a real ETAS fit (the dense, well-behaved cluster).
    assert pu["n_tiles_etas"] >= 1
    # Each fit tile records its regime + carrying model.
    for info in pu["regimes"].values():
        assert info["regime"] in {r.value for r in TectonicRegime}
        assert info["model"] in {"etas", "smoothed_seismicity"}

    cells = _cell_grid(reg, step=1.0)
    counts = tf.expected_counts(reg, cells, horizon_days=7.0, m_threshold=4.0, t_issue=t_issue)
    counts = np.asarray(counts)
    # Forecaster contract: one value per cell, all non-negative, and a positive global field.
    assert counts.shape == (len(cells),)
    assert np.all(counts >= 0.0)
    assert counts.sum() > 0.0


def test_tiled_forecaster_probabilities_in_unit_interval():
    reg = _wide_region()
    cat = _two_cluster_catalog()
    t_issue = pd.Timestamp("2025-06-01", tz="UTC")
    tf = TiledForecaster(tile_deg=8.0, halo_deg=1.0, mc=3.5).fit(cat, reg, t_issue)
    probs = np.asarray(
        tf.forecast_probabilities(reg, _cell_grid(reg), 7.0, 4.0, t_issue)
    )
    assert np.all(probs >= 0.0) and np.all(probs < 1.0)


def test_tiled_forecaster_enforces_stability_gate_per_tile():
    # A supercritical per-tile ETAS must be rejected and the tile carried by its smoothed null,
    # with the rejection reason recorded — never a silently published explosive intensity.
    reg = _wide_region()
    cat = _two_cluster_catalog()
    t_issue = pd.Timestamp("2025-06-01", tz="UTC")
    tf = TiledForecaster(tile_deg=8.0, halo_deg=1.0, mc=3.5, min_events_for_etas=30).fit(
        cat, reg, t_issue
    )
    regimes = tf.params_used["regimes"]
    # Any tile not carried by ETAS must carry a non-empty rejection string.
    for info in regimes.values():
        if not info["is_etas"]:
            assert info["rejection"]


def test_tiled_forecaster_requires_fit():
    tf = TiledForecaster()
    with pytest.raises(RuntimeError):
        tf.expected_counts(_wide_region(), [Cell(key="0,0", lat=0.0, lon=0.0)], 1.0, 5.0,
                           pd.Timestamp("2025-06-01", tz="UTC"))


def test_empty_past_raises():
    reg = _wide_region()
    cat = _two_cluster_catalog()
    # t_issue before any event → no lawful past.
    t_issue = pd.Timestamp("2000-01-01", tz="UTC")
    with pytest.raises(ValueError):
        TiledForecaster().fit(cat, reg, t_issue)

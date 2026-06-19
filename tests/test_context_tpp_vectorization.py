"""The vectorized neural triggering field must reproduce the per-cell quadrature exactly.

`ContextTPPForecaster.expected_counts` was a 119k-cell × 12-step Python loop calling the net per cell
(~1.4 M forward passes, ~50 min on the global grid). It is now a single net pass over the parents plus a
chunked distance matrix (`_triggering_field`). This test pins the two to float32 round-off so the speed-up
can never silently change a forecast — the regression guard for the optimization in docs/experiments.md E11.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caos_seismic.contracts import BBox, Cell, Region
from caos_seismic.model.context_tpp import ContextTPPConfig, ContextTPPForecaster

torch = pytest.importorskip("torch")


def _two_cluster_catalog(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    times = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 700, n)), unit="D")
    lat = np.concatenate([rng.normal(-33.0, 0.6, n // 2), rng.normal(-25.0, 0.6, n - n // 2)])
    lon = np.concatenate([rng.normal(-72.0, 0.6, n // 2), rng.normal(-70.0, 0.6, n - n // 2)])
    mw = 3.5 + rng.exponential(0.4, n)
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


@pytest.fixture(scope="module")
def fitted():
    reg = Region(
        id="chile",
        name_en="Chile (test)",
        name_es="Chile (test)",
        bbox=BBox(lat_min=-37.0, lat_max=-21.0, lon_min=-75.0, lon_max=-68.0),
        m_max=9.0,
    )
    cfg = ContextTPPConfig(epochs=2, temporal_mc_samples=32, patch_radius=2, context_dim=8)
    m = ContextTPPForecaster(config=cfg, mc=3.5, b_value=1.0, checkpoint_dir=None, device="cpu")
    m.fit(_two_cluster_catalog(), reg, pd.Timestamp("2025-06-01", tz="UTC"))
    return reg, m


def _cells():
    return [
        Cell(key=f"{la:.3f},{lo:.3f}", lat=float(la), lon=float(lo))
        for la in np.linspace(-36, -22, 7)
        for lo in np.linspace(-74, -69, 7)
    ]


def test_triggering_field_matches_per_cell_quadrature(fitted):
    """`_triggering_field` == looping `_triggering_intensity` over cells × steps, to float32 round-off."""
    _, m = fitted
    cells = _cells()
    lats = np.array([c.lat for c in cells])
    lons = np.array([c.lon for c in cells])
    steps = 12
    edges = np.linspace(0.0, 7.0, steps + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    dts = np.diff(edges)

    m._net.eval()
    with torch.no_grad():
        fast = m._triggering_field(m._net, lats, lons, mids, dts)
        ref = np.array(
            [
                sum(
                    m._triggering_intensity(m._net, float(lats[i]), float(lons[i]), float(s)) * float(w)
                    for s, w in zip(mids, dts)
                )
                for i in range(len(cells))
            ]
        )

    rel = np.abs(fast - ref) / np.maximum(np.abs(ref), 1e-9)
    assert rel.max() < 1e-5, f"max relative error {rel.max():.2e} exceeds float32 round-off"
    assert ref.max() > 0.0  # the kernels are actually exercised (non-trivial triggering)


def test_expected_counts_is_finite_and_nonnegative(fitted):
    """The vectorized forecast field stays a valid expected-count field (finite, >= 0)."""
    reg, m = fitted
    ec = np.asarray(m.expected_counts(reg, _cells(), 7.0, 5.0, pd.Timestamp("2025-06-01", tz="UTC")), float)
    assert np.isfinite(ec).all()
    assert (ec >= 0.0).all()
    assert ec.sum() > 0.0

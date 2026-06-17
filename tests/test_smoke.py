"""Pure unit smoke tests — NO network, core deps only (numpy/pandas/scipy/pydantic/h3).

These are the fast invariants every build must keep green. They deliberately avoid the data fetch
(no ComCat call), heavy science deps (no obspy/pycsep), and any committed output: each test builds
its own tiny in-memory fixture. Three properties are pinned:

1. **Catalog schema contract** — :func:`caos_seismic.contracts.validate_catalog` accepts a frame with
   the required columns and rejects one missing any of them.
2. **The exceedance formula never changes** — ``P(>=1 event) = 1 - e^{-N}`` (methodology §1.10), as
   implemented by :func:`caos_seismic.model._common.poisson_p_at_least_one`, including its boundary
   behaviour (N=0 -> 0, large N -> ~1, negative N clipped to 0).
3. **ForecastArtifact round-trip** — a tiny artifact built against the contract serializes to the
   compact on-disk form (H3 aggregate + quantize + gzip) and loads back with its structure and
   (de-quantized) rates intact, via :mod:`caos_seismic.inference.artifact`.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from caos_seismic.contracts import (
    CATALOG_COLUMNS,
    CalibrationSummary,
    ForecastArtifact,
    Region,
    Staleness,
    validate_catalog,
)
from caos_seismic.contracts import BBox
from caos_seismic.model._common import poisson_p_at_least_one


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures (tiny, in-memory — no IO, no network)
# ─────────────────────────────────────────────────────────────────────────────


def _tiny_catalog() -> pd.DataFrame:
    """A 3-event catalog with every required contract column populated."""
    times = pd.to_datetime(
        ["2025-01-01T00:00:00Z", "2025-01-02T06:00:00Z", "2025-01-03T12:00:00Z"], utc=True
    )
    return pd.DataFrame(
        {
            "event_id": ["a1", "a2", "a3"],
            "time": times,
            "latitude": [-22.0, -21.5, -23.1],
            "longitude": [-70.5, -70.2, -70.9],
            "depth_km": [30.0, 25.0, 40.0],
            "mag": [4.2, 5.1, 4.8],
            "mag_type": ["mb", "Mww", "ml"],
            "mw": [4.2, 5.1, 4.8],
            "source": ["usgs_comcat", "usgs_comcat", "usgs_comcat"],
        }
    )


def _tiny_region() -> Region:
    return Region(
        id="chile",
        name_en="Chile (test)",
        name_es="Chile (test)",
        bbox=BBox(lat_min=-24.0, lat_max=-20.0, lon_min=-71.5, lon_max=-69.5),
        m_max=9.5,
        attribution=["test"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1) Catalog schema contract
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_catalog_accepts_full_schema():
    df = _tiny_catalog()
    out = validate_catalog(df)
    # Returns the same frame unchanged, and every required column is present.
    assert out is df
    assert set(CATALOG_COLUMNS).issubset(out.columns)


@pytest.mark.parametrize("missing", sorted(CATALOG_COLUMNS))
def test_validate_catalog_rejects_missing_column(missing):
    df = _tiny_catalog().drop(columns=[missing])
    with pytest.raises(ValueError) as exc:
        validate_catalog(df)
    assert missing in str(exc.value)


# ─────────────────────────────────────────────────────────────────────────────
# 2) The exceedance formula P = 1 - e^{-N}
# ─────────────────────────────────────────────────────────────────────────────


def test_exceedance_zero_rate_is_zero():
    assert poisson_p_at_least_one(0.0) == 0.0


def test_exceedance_negative_rate_clipped_to_zero():
    # A negative expected count is non-physical; the formula clips N to >= 0 (P = 0), never NaN/<0.
    assert poisson_p_at_least_one(-3.0) == 0.0


def test_exceedance_large_rate_approaches_one():
    p = poisson_p_at_least_one(50.0)
    assert 0.0 < p <= 1.0
    assert p > 1.0 - 1e-9


@pytest.mark.parametrize("n", [0.01, 0.1, 0.5, 1.0, 2.5, 5.0])
def test_exceedance_matches_closed_form(n):
    # Exactly 1 - e^{-N}, the formula that "NEVER changes" (methodology §1.10).
    assert poisson_p_at_least_one(n) == pytest.approx(1.0 - math.exp(-n), rel=0, abs=1e-12)


def test_exceedance_monotone_increasing():
    rates = [0.0, 0.05, 0.2, 0.8, 1.5, 3.0]
    ps = [poisson_p_at_least_one(n) for n in rates]
    assert all(b >= a for a, b in zip(ps, ps[1:]))


# ─────────────────────────────────────────────────────────────────────────────
# 3) ForecastArtifact round-trip (build -> write -> read)
# ─────────────────────────────────────────────────────────────────────────────


def _tiny_artifact() -> ForecastArtifact:
    """A minimal but schema-valid artifact: 2 fine cells, 1 horizon, 1 threshold."""
    region = _tiny_region()
    issued_at = "2025-01-04T00:00:00Z"
    # forecast[cell][str(horizon)][str(M*)] -> {p, lo, hi, rate, baseline}
    forecast = {
        "-22.05,-70.55": {
            "1": {"5.0": {"p": 0.012, "lo": 0.004, "hi": 0.030, "rate": 0.0121, "baseline": 0.002}}
        },
        "-21.55,-70.25": {
            "1": {"5.0": {"p": 0.008, "lo": 0.003, "hi": 0.021, "rate": 0.0080, "baseline": 0.001}}
        },
    }
    return ForecastArtifact(
        issued_at=issued_at,
        region=region,
        horizons_days=[1],
        magnitude_thresholds=[5.0],
        m_max=region.m_max,
        grid={"type": "h3", "resolution": 5},
        forecast=forecast,
        calibration=CalibrationSummary(
            reliability=[[0.01, 0.011, 100]], csep={"isotonic_fitted": False}
        ),
        coverage_mask=[],
        provenance={"code_git_sha": None, "model": {"name": "smoothed_seismicity"}},
        staleness=Staleness(
            generated=issued_at, next_run="2025-01-05T00:00:00Z", ok=True
        ),
    )


def test_forecast_artifact_round_trip(tmp_path: Path):
    from caos_seismic.inference.artifact import load_artifact, write_artifact

    artifact = _tiny_artifact()
    paths = write_artifact(artifact, results_dir=tmp_path)

    # A gzipped artifact file + an index were written to the throwaway dir (never committed results/).
    assert paths["artifact"].exists()
    assert paths["artifact"].suffix == ".gz"
    assert paths["index"].name == "index.json"

    reloaded = load_artifact(paths["artifact"])

    # Top-level contract fields survive the round-trip.
    assert reloaded.product == "CAOS_SEISMIC"
    assert reloaded.issued_at == artifact.issued_at
    assert reloaded.region.id == artifact.region.id
    assert reloaded.horizons_days == [1]
    assert reloaded.magnitude_thresholds == [5.0]
    assert reloaded.m_max == artifact.m_max
    assert reloaded.staleness.ok is True

    # The forecast nest survives (aggregated to >=1 H3 cell), with the {p,lo,hi,rate,baseline} keys
    # and de-quantized float rates back in place.
    assert len(reloaded.forecast) >= 1
    cell_key = next(iter(reloaded.forecast))
    row = reloaded.forecast[cell_key]["1"]["5.0"]
    for key in ("p", "lo", "hi", "rate", "baseline"):
        assert key in row
        assert isinstance(row[key], float)
    # Probabilities stay in [0, 1] and ordered after aggregation.
    assert 0.0 <= row["lo"] <= row["p"] <= row["hi"] <= 1.0
    # The quantized rate de-quantizes to a small positive float (within an order of magnitude of input).
    assert row["rate"] > 0.0


def test_artifact_serialize_is_compact_and_self_describing():
    from caos_seismic.inference.artifact import serialize_artifact

    payload = serialize_artifact(_tiny_artifact(), grid_cfg={"display": {"h3_resolution_region": 5}})
    # The compact payload embeds the rate legend (so a loader de-quantizes without external state)
    # and a compaction block, and quantizes the rate to an integer code `q` (not a float `rate`).
    assert payload["rate_legend"]["kind"] == "log_uint16"
    assert "compaction" in payload
    any_cell = next(iter(payload["forecast"].values()))
    row = any_cell["1"]["5.0"]
    assert "q" in row and isinstance(row["q"], int)
    assert "rate" not in row  # the float rate is replaced by the quantized code on disk

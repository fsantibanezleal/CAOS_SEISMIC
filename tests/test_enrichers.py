"""Unit tests for the global geophysical enrichers — NO network, core deps only.

These pin the enricher *contract* without downloading anything or importing heavy geospatial deps:

1. **Registry integrity** — every enricher in :data:`caos_seismic.data.enrichers.ENRICHERS` exposes
   ``download`` / ``features_at`` / ``FEATURE_NAMES`` and the registry's flattened feature names are
   unique.
2. **Lazy-import discipline** — importing the enrichers subpackage and the feature bridge does NOT
   pull any heavy geospatial dependency into ``sys.modules`` (the hard package rule).
3. **Tides analytic path** — the semidiurnal phase + fortnightly Mf envelope are returned without
   pygtide (analytic), in their valid ranges, and the ΔCFS columns degrade to ``None``.
4. **Plates ASCII parser** — a tiny synthetic ``PB2002_steps.dat`` yields the expected
   distance / convergent-type / relative-velocity covariates.
5. **`build_context_features` join** — the grid → context-matrix machinery assembles ``key/lat/lon``
   + each enricher's columns, accepts both ``list[Cell]`` and a lat/lon DataFrame, and tolerates an
   enricher raising (records NaN, never corrupts the matrix). A lightweight stub enricher keeps the
   test off the heavy deps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from caos_seismic.contracts import Cell


# ─────────────────────────────────────────────────────────────────────────────
# 1) Registry integrity
# ─────────────────────────────────────────────────────────────────────────────


def test_registry_contract_and_unique_feature_names():
    from caos_seismic.data.enrichers import ALL_FEATURE_NAMES, ENRICHERS, feature_names_for

    assert set(ENRICHERS) == {"slab2", "faults", "plates", "gnss", "stress", "tides"}
    for name, mod in ENRICHERS.items():
        assert callable(getattr(mod, "download", None)), f"{name} missing download()"
        assert callable(getattr(mod, "features_at", None)), f"{name} missing features_at()"
        assert getattr(mod, "FEATURE_NAMES", None), f"{name} missing FEATURE_NAMES"

    # No two enrichers emit a column with the same name (the matrix schema must be unambiguous).
    assert len(ALL_FEATURE_NAMES) == len(set(ALL_FEATURE_NAMES))
    assert feature_names_for("slab2") == list(ENRICHERS["slab2"].FEATURE_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Lazy-import discipline (no heavy deps at import time)
# ─────────────────────────────────────────────────────────────────────────────


def test_no_heavy_deps_imported_at_module_load():
    # Run in a CLEAN subprocess: the package's lazy-import discipline is a property of importing the
    # package, not of the current pytest session's sys.modules — which an earlier test that fits a
    # neural model (importing torch) would otherwise pollute, making this guard order-dependent.
    import subprocess
    import sys as _sys

    heavy = ["geopandas", "shapely", "xarray", "netCDF4", "pygtide", "obspy", "torch"]
    code = (
        "import sys\n"
        "import caos_seismic.catalog.features\n"
        "import caos_seismic.data.enrichers\n"
        f"heavy = {heavy!r}\n"
        "leaked = [m for m in heavy if m in sys.modules]\n"
        "print(','.join(leaked))\n"
        "sys.exit(1 if leaked else 0)\n"
    )
    proc = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"heavy deps imported at module top level: {proc.stdout.strip()}"


# ─────────────────────────────────────────────────────────────────────────────
# 3) Tides analytic path (pygtide-free)
# ─────────────────────────────────────────────────────────────────────────────


def test_tides_analytic_phase_and_envelope_without_pygtide():
    from caos_seismic.data.enrichers import tides

    f = tides.phase_features_at(-33.0, -71.0, t_issue="2026-06-16T00:00:00Z")
    assert -1.0 <= f["tidal_phase_sin"] <= 1.0
    assert -1.0 <= f["tidal_phase_cos"] <= 1.0
    assert 0.0 <= f["tidal_mf_envelope"] <= 1.0
    # ΔCFS/stressing-rate need pygtide -> None in the analytic-only helper.
    assert f["tidal_dCFS_kpa"] is None
    assert f["tidal_stress_rate_kpa_per_hr"] is None


def test_tides_features_at_degrades_without_pygtide():
    pytest.importorskip  # noqa: B018  (documents intent: this path runs whether or not pygtide exists)
    from caos_seismic.data.enrichers import tides

    if "pygtide" in sys.modules or _module_available("pygtide"):
        pytest.skip("pygtide installed — graceful-degradation path not exercised")
    f = tides.features_at(-33.0, -71.0, t_issue="2026-06-16T00:00:00Z")
    assert f["tidal_phase_sin"] is not None and f["tidal_mf_envelope"] is not None
    assert f["tidal_dCFS_kpa"] is None  # degraded, not raised


def test_tides_strict_raises_without_pygtide():
    from caos_seismic.data.enrichers import tides

    if _module_available("pygtide"):
        pytest.skip("pygtide installed — strict ImportError path not exercised")
    with pytest.raises(ImportError) as exc:
        tides.TidesEnricher().features_at(-33.0, -71.0, t_issue="2026-06-16T00:00:00Z", strict=True)
    assert "pygtide" in str(exc.value)


# ─────────────────────────────────────────────────────────────────────────────
# 4) Plates ASCII parser (core deps only)
# ─────────────────────────────────────────────────────────────────────────────


def test_plates_parses_steps_and_extracts_features(tmp_path: Path):
    from caos_seismic.data.enrichers import plates

    (tmp_path / "PB2002_steps.dat").write_text(
        "-72.0 -34.0 0.0 65.5 90.0 SUB\n"
        "-71.5 -33.5 0.0 64.0 92.0 SUB\n"
        "  10.0  45.0 0.0  5.0  3.0 CCB\n",
        encoding="utf-8",
    )
    enr = plates.PlatesEnricher(dest=tmp_path)
    f = enr.features_at(-33.7, -71.7)
    assert f["plate_is_convergent"] == 1.0
    assert f["plate_boundary_type_code"] == float(plates.BOUNDARY_TYPE_CODES["SUB"])
    assert f["plate_boundary_dist_km"] < 100.0
    assert f["plate_rel_velocity_mm_yr"] is not None and f["plate_rel_velocity_mm_yr"] > 0


def test_plates_missing_cache_raises_actionable(tmp_path: Path):
    from caos_seismic.data.enrichers import plates

    with pytest.raises(FileNotFoundError) as exc:
        plates.PlatesEnricher(dest=tmp_path).features_at(0.0, 0.0)
    assert "download" in str(exc.value)


# ─────────────────────────────────────────────────────────────────────────────
# 5) build_context_features join machinery (stub enricher → no heavy deps)
# ─────────────────────────────────────────────────────────────────────────────


class _StubEnricher:
    """A minimal in-process enricher used to test the join without touching real datasets."""

    FEATURE_NAMES = ("stub_a", "stub_b")

    @staticmethod
    def download(**_):  # pragma: no cover - not exercised here
        raise NotImplementedError

    @staticmethod
    def features_at(lat, lon, **_):
        return {"stub_a": lat + lon, "stub_b": None}


class _BoomEnricher:
    """An enricher that always raises — to prove a failure becomes NaN, never a corrupt matrix."""

    FEATURE_NAMES = ("boom_x",)

    @staticmethod
    def download(**_):  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def features_at(lat, lon, **_):
        raise RuntimeError("boom")


@pytest.fixture()
def _stub_registry(monkeypatch):
    import caos_seismic.data.enrichers as enr_pkg

    stub = {"stub": _StubEnricher, "boom": _BoomEnricher}
    monkeypatch.setattr(enr_pkg, "ENRICHERS", stub, raising=True)
    return stub


def test_build_context_features_with_cell_list(_stub_registry):
    from caos_seismic.catalog.features import build_context_features

    cells = [Cell(key="a", lat=-33.0, lon=-71.0), Cell(key="b", lat=-34.0, lon=-72.0)]
    m = build_context_features(cells, enrichers=["stub"], t_issue="2026-06-16T00:00:00Z")
    assert list(m.columns) == ["key", "lat", "lon", "stub_a", "stub_b"]
    assert m.shape == (2, 5)
    assert m.loc[0, "stub_a"] == pytest.approx(-104.0)  # lat+lon
    assert pd.isna(m.loc[0, "stub_b"])  # None → NaN


def test_build_context_features_with_dataframe_grid(_stub_registry):
    from caos_seismic.catalog.features import build_context_features

    grid = pd.DataFrame({"latitude": [-33.0], "longitude": [-71.0]})
    m = build_context_features(grid, enrichers=["stub"])
    assert m.loc[0, "key"] == "-33.0,-71.0"
    assert m.loc[0, "stub_a"] == pytest.approx(-104.0)


def test_build_context_features_tolerates_enricher_failure(_stub_registry):
    from caos_seismic.catalog.features import build_context_features

    cells = [Cell(key="a", lat=1.0, lon=2.0)]
    m = build_context_features(cells, enrichers=["stub", "boom"], t_issue="2026-06-16T00:00:00Z")
    # The failing enricher's column exists but is NaN; the good enricher is intact.
    assert "boom_x" in m.columns
    assert pd.isna(m.loc[0, "boom_x"])
    assert m.loc[0, "stub_a"] == pytest.approx(3.0)


def test_build_context_features_rejects_unknown_enricher(_stub_registry):
    from caos_seismic.catalog.features import build_context_features

    with pytest.raises(ValueError) as exc:
        build_context_features([Cell(key="a", lat=0.0, lon=0.0)], enrichers=["nope"])
    assert "nope" in str(exc.value)


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None

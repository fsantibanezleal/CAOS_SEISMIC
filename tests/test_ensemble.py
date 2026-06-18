"""Unit tests for the linear-opinion-pool EnsembleForecaster (pure combination logic — no fitting)."""

from __future__ import annotations

import math

import pandas as pd

from caos_seismic.contracts import Cell
from caos_seismic.model.ensemble import EnsembleForecaster, build_default_ensemble


class _StubForecaster:
    """A forecaster whose expected_counts is a fixed per-cell constant (one value per cell)."""

    def __init__(self, name: str, per_cell: float, fitted: bool = True):
        self.name = name
        self.version = "0.0.0"
        self._val = float(per_cell)
        self.params = {"stub": True} if fitted else {}

    def expected_counts(self, region, cells, horizon_days, m_threshold, t_issue):
        return [self._val] * len(cells)


def _cells(n: int = 4):
    return [Cell(key=f"c{i}", lat=float(i), lon=float(-i)) for i in range(n)]


def _t():
    return pd.Timestamp("2026-01-01T00:00:00Z")


def test_linear_pool_is_the_weighted_average_of_component_rates():
    cells = _cells()
    ens = EnsembleForecaster(
        components=[("a", _StubForecaster("a", 0.4), 1.0), ("b", _StubForecaster("b", 0.8), 3.0)]
    )
    out = ens.expected_counts(None, cells, 1.0, 5.0, _t())
    # weights normalize to (0.25, 0.75): 0.25*0.4 + 0.75*0.8 = 0.7 per cell.
    assert len(out) == 4
    assert all(math.isclose(v, 0.7, rel_tol=1e-9) for v in out)


def test_equal_weights_by_default_and_probabilities_use_exceedance_formula():
    cells = _cells(2)
    ens = EnsembleForecaster(
        components=[("a", _StubForecaster("a", 0.2), 1.0), ("b", _StubForecaster("b", 0.6), 1.0)]
    )
    lam = ens.expected_counts(None, cells, 1.0, 5.0, _t())[0]
    assert math.isclose(lam, 0.4, rel_tol=1e-9)  # (0.2 + 0.6) / 2
    p = ens.forecast_probabilities(None, cells, 1.0, 5.0, _t())[0]
    assert math.isclose(p, 1.0 - math.exp(-0.4), rel_tol=1e-9)


def test_zero_weight_component_is_recorded_but_contributes_nothing():
    cells = _cells(1)
    ens = EnsembleForecaster(
        components=[("keep", _StubForecaster("keep", 1.0), 1.0), ("mute", _StubForecaster("mute", 99.0), 0.0)]
    )
    assert ens.expected_counts(None, cells, 1.0, 5.0, _t()) == [1.0]
    assert ens.params_used["n_components"] == 2  # the muted component is still auditable


def test_failed_component_is_dropped_and_pool_renormalizes():
    class _Broken(_StubForecaster):
        def expected_counts(self, *a, **k):
            raise RuntimeError("cannot evaluate")

    cells = _cells(1)
    ens = EnsembleForecaster(
        components=[("ok", _StubForecaster("ok", 0.5), 1.0), ("broken", _Broken("broken", 0.0), 1.0)]
    )
    # The broken component drops out; the pool renormalizes over the survivor -> its own value.
    assert ens.expected_counts(None, cells, 1.0, 5.0, _t()) == [0.5]


def test_build_default_ensemble_from_a_model_family_dedupes_primary_and_etas():
    primary = _StubForecaster("etas", 0.3)
    ens = build_default_ensemble(
        {"primary": primary, "etas": primary, "smoothed": _StubForecaster("smoothed", 0.1),
         "reasenberg_jones": _StubForecaster("rj", 0.2)}
    )
    # primary IS etas (same object) -> deduped; smoothed + rj distinct -> 3 components total.
    assert ens.params_used["n_components"] == 3
    names = {c["name"] for c in ens.params_used["components"]}
    assert names == {"etas_tiled", "smoothed_null", "reasenberg_jones"}

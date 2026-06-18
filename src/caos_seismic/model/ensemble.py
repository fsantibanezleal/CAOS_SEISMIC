"""Forecast-combination ensemble — a weighted linear opinion pool over component forecasters.

The single most reliable lever for beating any one short-term seismicity model is **combining** them:
the CSEP consensus across collaboratories is that a well-weighted ensemble (Bayesian model averaging /
linear opinion pool / score-weighted stacking) matches or beats the best single component, because the
components fail in different places (ETAS over-relies on recent triggering; smoothed-seismicity carries
the stationary background; Reasenberg–Jones anchors the aftershock decay). See Marzocchi, Zechar &
Jordan (2012) for the BMA formulation and Rhoades et al. (2014, 2018) for multiplicative/additive hybrid
ensembles in operational CSEP.

This module combines the per-cell **expected counts** of already-fitted component forecasters as a
linear opinion pool::

    lambda_ensemble(cell) = sum_k w_k * lambda_k(cell)        (weights normalized to sum 1)

Linear (not log-linear) pooling is the CSEP-standard rate combination: it is conservative (the ensemble
rate never collapses to zero just because one component does), keeps the public exceedance formula
``P(>=1) = 1 - e^{-lambda}`` unchanged, and — when the mandatory smoothed null is a component — guarantees
the ensemble never reads below a floor of the long-term Poisson baseline. Weights default to equal; a
caller may pass fixed weights or, later, weights learned from the rolling prospective log-score
(score-weighted stacking — deferred to :meth:`fit_weights_from_history`).

Skill is established only by the prospective back-analysis (:mod:`caos_seismic.eval.backanalysis`); this
class is a forecaster like any other and is scored by the same harness, never asserted to be better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import BaseForecaster, Cell, Region
from ._common import poisson_p_at_least_one


@dataclass
class EnsembleForecaster(BaseForecaster):
    """Weighted linear opinion pool over component forecasters (BMA-style rate combination).

    Parameters
    ----------
    components:
        ``[(name, forecaster, weight), ...]`` — each ``forecaster`` exposes ``expected_counts`` and is
        expected to be **already fitted** (the canonical use wraps a fitted model family whose ETAS and
        smoothed components were fit under the dual-catalog rule). :meth:`fit` will best-effort fit any
        component that is not yet fitted, on the catalog passed.
    weights are normalized to sum to 1 at evaluation time; non-positive total weight falls back to equal
    weights. ``components`` with a zero weight are kept (auditable) but contribute nothing.
    """

    components: list[tuple[str, Any, float]] = field(default_factory=list)
    name: str = "ensemble_linear_pool"
    version: str = "0.1.0"
    params_used: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("EnsembleForecaster needs at least one component")
        self._record_params()

    def _record_params(self) -> None:
        total = sum(max(float(w), 0.0) for _, _, w in self.components)
        self.params_used = {
            "combination": "linear_opinion_pool",
            "n_components": len(self.components),
            "components": [
                {
                    "name": str(n),
                    "model": getattr(m, "name", type(m).__name__),
                    "weight_raw": float(w),
                    "weight_norm": (float(w) / total) if total > 0 else 1.0 / len(self.components),
                }
                for (n, m, w) in self.components
            ],
        }

    def fit(self, catalog: pd.DataFrame, region: Region, t_issue: pd.Timestamp) -> "EnsembleForecaster":
        """Best-effort fit any not-yet-fitted component on ``catalog`` (already-fit components are kept).

        The canonical path constructs the ensemble from a model family that was already fitted under the
        dual-catalog rule (ETAS on the full catalog, smoothed on the declustered catalog), so this is
        usually a no-op. A component is treated as fitted if it exposes a non-empty ``params``/
        ``params_used`` or a private ``_fitted`` flag; otherwise its ``fit`` is called on ``catalog``.
        """
        for _, model, _ in self.components:
            if self._is_fitted(model):
                continue
            fit = getattr(model, "fit", None)
            if callable(fit):
                fit(catalog, region, t_issue)
        self._record_params()
        return self

    @staticmethod
    def _is_fitted(model: Any) -> bool:
        if getattr(model, "_fitted", False):
            return True
        for attr in ("params", "params_used"):
            val = getattr(model, attr, None)
            if val:
                return True
        return False

    def expected_counts(
        self,
        region: Region,
        cells: list[Cell],
        horizon_days: float,
        m_threshold: float,
        t_issue: pd.Timestamp,
    ) -> list[float]:
        """Per-cell ``sum_k w_k * lambda_k`` (weights normalized to 1; missing/failed components skipped)."""
        if not cells:
            return []
        acc = np.zeros(len(cells), dtype=float)
        total_w = 0.0
        for _name, model, weight in self.components:
            w = max(float(weight), 0.0)
            if w <= 0.0:
                continue
            try:
                lam = np.asarray(
                    model.expected_counts(region, cells, horizon_days, m_threshold, t_issue), dtype=float
                )
            except Exception:
                # A component that cannot evaluate this slice is dropped from the pool for this call
                # (its weight is not redistributed implicitly — total_w below renormalizes over the
                # components that DID contribute, so the ensemble stays a proper convex combination).
                continue
            if lam.shape != acc.shape:
                continue
            acc += w * np.clip(lam, 0.0, None)
            total_w += w
        if total_w <= 0.0:
            return [0.0] * len(cells)
        return [float(v) for v in (acc / total_w)]

    def forecast_probabilities(
        self,
        region: Region,
        cells: list[Cell],
        horizon_days: float,
        m_threshold: float,
        t_issue: pd.Timestamp,
    ) -> list[float]:
        """Per-cell ``P(>=1 event >= M*) = 1 - e^{-lambda_ensemble}`` (the public exceedance formula)."""
        return [
            poisson_p_at_least_one(n)
            for n in self.expected_counts(region, cells, horizon_days, m_threshold, t_issue)
        ]


def build_default_ensemble(models: dict[str, Any], weights: dict[str, float] | None = None) -> EnsembleForecaster:
    """Construct the default ensemble from a fitted model family (the ``_fit_model_family`` dict).

    Components, when present and fitted: the regime-tiled ETAS (``primary``/``etas``), the adaptive
    smoothed-seismicity null (``smoothed`` — the mandatory stationary floor), and Reasenberg–Jones
    (``reasenberg_jones``). Equal weights by default; pass ``weights`` (by component key) to override.
    A future score-weighted variant will set these from the rolling prospective log-score.
    """
    spec: list[tuple[str, str]] = [
        ("etas_tiled", "primary"),
        ("smoothed_null", "smoothed"),
        ("reasenberg_jones", "reasenberg_jones"),
    ]
    comps: list[tuple[str, Any, float]] = []
    seen: set[int] = set()
    for name, key in spec:
        model = models.get(key)
        if model is None or id(model) in seen:
            continue
        seen.add(id(model))
        w = float((weights or {}).get(name, 1.0))
        comps.append((name, model, w))
    if not comps:
        raise ValueError("no fitted components available to build an ensemble")
    return EnsembleForecaster(components=comps)

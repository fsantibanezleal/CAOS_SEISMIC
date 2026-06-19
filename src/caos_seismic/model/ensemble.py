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

    # ── score-weighted stacking (E12) ─────────────────────────────────────────
    def fit_weights_from_history(
        self,
        holdout: list[tuple[np.ndarray, np.ndarray]],
        *,
        rho: float = 100.0,
        w_min: float = 0.0,
        n_min: int = 8,
        anchor_index: int = 0,
    ) -> "EnsembleForecaster":
        """Set the pool weights by **log-score-optimal convex stacking** on a strictly-past holdout.

        This is the only literature-proven prospective lever over a single well-fit ETAS (score-weighted
        ETAS-family ensembles; Marzocchi-Zechar-Jordan 2012 BMA, Serafini et al. 2022, Bayona et al. 2021).
        It is implemented to be **fail-safe ON THE FIT HOLDOUT**: the anchor (the base tiled ETAS,
        ``anchor_index``) is the shrinkage target ``e_0``, so the fitted weights never score worse than
        ``e_0`` *on the holdout objective they are fit to*, and on a sparse/quiet holdout collapse to
        ``w = e_0`` (E12 == base). **This is NOT a guarantee on the separately-scored out-of-sample
        window** — that forecast is genuinely prospective and its IGPE vs base may be positive OR negative;
        only the pre-registered paired T/W test over many windows decides whether E12 has real skill.

        Parameters
        ----------
        holdout:
            ``[(L_j, omega_j), ...]`` — one entry per past leakage-free window. ``L_j`` is the
            ``(n_cells, K)`` matrix of each component's expected counts for that window and ``omega_j``
            the ``(n_cells,)`` observed counts. The caller MUST build these from windows whose target
            period ends strictly before the forecast issue time (leakage is the caller's contract).
        rho:
            L2 (Dirichlet-style) shrinkage strength toward ``e_0``. Larger ⇒ harder pull to the base
            vertex. Conservative default; tuned only on a development period disjoint from the evaluation.
        w_min:
            Per-weight floor (keeps every member auditable; 0 lets a useless member be pruned to zero).
        n_min:
            Minimum total observed events in the holdout to fit weights at all. Below it, return ``e_0``
            (cold start: trust the base model rather than 2 free weights on a handful of events).
        anchor_index:
            Index of the base/anchor component (the ``e_0`` vertex). The builder puts the base first.

        The objective maximized is exactly the held-out Poisson joint log-score of the pooled rate
        ``lambda(w) = L w`` (the kernel of CSEP's L-test / the IGPE numerator), minus the shrinkage:
        ``J(w) = sum_j sum_c [omega log(L w) - (L w)] - (rho/2)||w - e_0||^2`` over the simplex.
        """
        K = len(self.components)
        w = _solve_stacking_weights(holdout, K, rho=rho, w_min=w_min, n_min=n_min, anchor_index=anchor_index)
        self.components = [(n, m, float(w[k])) for k, (n, m, _) in enumerate(self.components)]
        self._record_params()
        n_obs = int(sum(float(np.asarray(om).sum()) for _, om in holdout)) if holdout else 0
        self.params_used["stacking"] = {
            "method": "logscore_optimal_convex_stacking",
            "rho": float(rho),
            "w_min": float(w_min),
            "n_min": int(n_min),
            "anchor_index": int(anchor_index),
            "n_holdout_windows": len(holdout),
            "n_holdout_events": n_obs,
            "weights": [float(x) for x in w],
            "collapsed_to_anchor": bool(np.argmax(w) == anchor_index and w[anchor_index] > 0.999),
        }
        return self


def _solve_stacking_weights(
    holdout: list[tuple[np.ndarray, np.ndarray]],
    K: int,
    *,
    rho: float,
    w_min: float,
    n_min: int,
    anchor_index: int,
) -> np.ndarray:
    """Convex log-score-optimal stacking weights over the simplex (the E12 solver).

    Maximizes the held-out Poisson joint log-score of the pooled rate ``lambda = L w`` with an L2 pull
    toward the anchor vertex ``e_0``: ``J(w) = sum [omega·log(L w) - (L w)] - (rho/2)||w - e_0||^2``.
    The Poisson log-score is concave in ``lambda`` and ``lambda`` is linear in ``w``, so ``J`` is concave
    over the simplex — a small SLSQP solve (K is 2-4). Fail-safe ON THE FIT HOLDOUT: any solver failure,
    an empty/sparse holdout (< ``n_min`` events), or a solution that does not beat the anchor on the
    holdout objective returns ``e_0`` exactly. This bounds the *fitted weights* by the anchor on the data
    they are fit to; it does NOT bound the out-of-sample IGPE of the resulting forecast (that is what the
    prospective back-analysis measures).
    """
    e0 = np.zeros(K, dtype=float)
    e0[anchor_index] = 1.0
    if not holdout or K <= 1:
        return e0
    try:
        L = np.vstack([np.asarray(Lj, dtype=float).reshape(-1, K) for Lj, _ in holdout])
        om = np.concatenate([np.asarray(omj, dtype=float).reshape(-1) for _, omj in holdout])
    except Exception:
        return e0
    if L.shape[0] != om.shape[0] or float(om.sum()) < float(n_min):
        return e0
    eps = np.finfo(float).tiny
    Lc = np.clip(L, 0.0, None)

    def neg_J(w: np.ndarray) -> float:
        lam = np.clip(Lc @ w, eps, None)
        ll = float(np.sum(om * np.log(lam) - lam))
        pen = 0.5 * float(rho) * float(np.sum((w - e0) ** 2))
        return -(ll - pen)

    def neg_grad(w: np.ndarray) -> np.ndarray:
        lam = np.clip(Lc @ w, eps, None)
        g = Lc.T @ (om / lam - 1.0)
        return -(g - float(rho) * (w - e0))

    try:
        from scipy.optimize import minimize

        cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: np.ones(K)},)
        bnds = [(float(w_min), 1.0)] * K
        res = minimize(
            neg_J, e0.copy(), jac=neg_grad, method="SLSQP",
            bounds=bnds, constraints=cons, options={"maxiter": 300, "ftol": 1e-10},
        )
        w = np.asarray(res.x, dtype=float) if getattr(res, "success", False) else e0.copy()
    except Exception:
        w = e0.copy()

    w = np.clip(w, max(float(w_min), 0.0), None)
    s = float(w.sum())
    w = w / s if s > 0 else e0.copy()
    # Fail-safe: never return a pool that scores worse than the anchor on the holdout objective.
    if not np.all(np.isfinite(w)) or neg_J(w) > neg_J(e0) + 1e-9:
        return e0
    return w


def _is_tiled_etas_family(model: Any) -> bool:
    """True iff ``model`` is a tiled-ETAS-family forecaster (the only admissible stacking member).

    Duck-typed first (``name`` starts with ``tiled``) to avoid a hard import at module load, with an
    ``isinstance`` backstop. The smoothed null and Reasenberg-Jones deliberately fail this check.
    """
    if str(getattr(model, "name", "")).startswith("tiled"):
        return True
    try:
        from .tiled import TiledForecaster

        return isinstance(model, TiledForecaster)
    except Exception:
        return False


def build_etas_stack_ensemble(
    base: Any, variants: list[tuple[str, Any]] | dict[str, Any] | None = None
) -> EnsembleForecaster:
    """Build the E12 score-weighted ETAS-variant stack: base tiled ETAS (anchor, index 0) + variants.

    **Structural guard against re-creating the E8-E9 dilution dead-end:** every weighted component MUST
    be a tiled-ETAS-family forecaster. The smoothed-seismicity null and Reasenberg-Jones are NOT weighted
    members here — the null enters only as each member's ``mu(x,y)`` background and the downstream
    cold-start floor. Raises if a non-ETAS-family model is passed.

    Components start at equal weight; call :meth:`EnsembleForecaster.fit_weights_from_history` to set the
    score-weighted stacking weights (anchor index 0). Variants accepts ``[(name, model), ...]`` or a dict.
    """
    items: list[tuple[str, Any]] = []
    if isinstance(variants, dict):
        items = list(variants.items())
    elif variants:
        items = list(variants)
    spec: list[tuple[str, Any]] = [("etas_tiled_base", base), *items]
    comps: list[tuple[str, Any, float]] = []
    for name, model in spec:
        if model is None:
            continue
        if not _is_tiled_etas_family(model):
            raise ValueError(
                f"build_etas_stack_ensemble: component '{name}' "
                f"({getattr(model, 'name', type(model).__name__)}) is not a tiled-ETAS-family forecaster. "
                "The score-weighted stack admits ETAS-family variants only; the smoothed null / "
                "Reasenberg-Jones must never be weighted members (the E8-E9 dilution dead-end)."
            )
        comps.append((name, model, 1.0))
    if len(comps) < 1:
        raise ValueError("build_etas_stack_ensemble needs a base tiled-ETAS forecaster")
    return EnsembleForecaster(components=comps, name="etas_stack_ensemble")


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

"""Space-time ETAS (Ogata 1998) — the primary conditional estimator *and* the reference to beat.

The Epidemic-Type Aftershock Sequence model is the physics-free, self-exciting Hawkes point process
that is the de-facto operational baseline for short-horizon seismicity forecasting. Any candidate
forecaster (neural or otherwise) must demonstrate positive, significant information gain over a
well-fit ETAS in prospective CSEP-style testing, or it adds no forecasting skill (model-design §1.1).

Conditional intensity (model-design §1.2; configs/etas.yaml)::

    lambda(t,x,y | H_t) = mu(x,y) + sum_{i: t_i < t} k(m_i) g(t - t_i) f(r | m_i)

with the canonical Ogata-1998 *separable* kernels:

  * Utsu productivity:        k(m) = K e^{alpha (m - M0)}
  * Omori-Utsu temporal decay: g(t) = (p - 1) / c * (1 + t/c)^{-p}            [days, integrates to 1]
  * Inverse-power spatial:     f(r | m) = (q - 1) / (pi zeta^2) (1 + r^2/zeta^2)^{-q},
                               zeta(m) = D e^{gamma (m - M0)}                  [integrates to 1 over plane]

The background term ``mu(x, y)`` is supplied by the adaptive smoothed-seismicity field
(:class:`~caos_seismic.model.smoothed.SmoothedSeismicityForecaster`) fit on a *declustered* catalog —
the dual-catalog rule (model-design §5). The triggering sum uses the *full, un-declustered* catalog,
because aftershock/foreshock triggering *is* the predictable signal.

The magnitude distribution is independent of history (separability) and follows Gutenberg-Richter,
so the expected count above a display threshold ``M*`` is the integrated intensity times the GR tail
fraction, and the public probability is the non-homogeneous Poisson exceedance
``P(>=1) = 1 - e^{-N}`` (model-design §3.2). This formula never changes; only the quality of the rate
feeding ``N`` improves.

**Stability — two distinct gates (kept separate, both enforced after the fit):**

1. ``alpha < beta`` with ``beta = b ln 10`` is required for the productivity x magnitude integral to
   converge (finite branching ratio ``n``).
2. Given that, ``n < 1`` is the subcritical / stationary condition. A fit with ``n >= 1`` is
   supercritical (explosive) and is rejected as a mis-fit.

References
----------
Ogata, Y. (1988), *JASA* 83(401), 9-27, doi:10.1080/01621459.1988.10478560.
Ogata, Y. (1998), *Ann. Inst. Statist. Math.* 50(2), 379-402, doi:10.1023/A:1003403601725.
Zhuang, J., Ogata, Y. & Vere-Jones, D. (2002), *JASA* 97(458), 369-380 (stochastic declustering /
    branching-ratio formulation), doi:10.1198/016214502760046925.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contracts import BaseForecaster, Cell, Region, validate_catalog
from ._common import (
    DEG2KM,
    EARTH_RADIUS_KM,
    LN10,
    bvalue_aki_utsu,
    gr_exceedance_fraction,
    haversine_km,
    poisson_p_at_least_one,
    sample_truncated_gr,
)
from .smoothed import SmoothedSeismicityForecaster

#: Order of the seven fitted parameters in the optimizer vector.
PARAM_NAMES: tuple[str, ...] = ("K", "alpha", "c", "p", "D", "gamma", "q")

#: Default optimizer bounds (configs/etas.yaml ``fit.bounds``). Distances ``D`` in degrees, times
#: ``c`` in days. Used when a caller does not override them. ``q > 1`` keeps the spatial integral
#: finite; ``p > 1`` keeps the temporal kernel normalizable as a proper density.
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "K": (1.0e-6, 10.0),
    "alpha": (0.1, 2.5),
    "c": (1.0e-4, 1.0),
    "p": (0.8, 1.5),
    "D": (1.0e-4, 1.0),
    "gamma": (0.1, 2.0),
    "q": (1.1, 3.0),
}


class ETASStabilityError(ValueError):
    """Raised when a fitted ETAS violates a stability gate (alpha >= beta, or branching ratio n >= 1).

    Carries the offending quantities so the caller / manifest can record *why* the fit was rejected
    rather than silently publishing a supercritical (explosive) intensity.
    """

    def __init__(self, message: str, *, alpha: float, beta: float, branching_ratio: float) -> None:
        super().__init__(message)
        self.alpha = alpha
        self.beta = beta
        self.branching_ratio = branching_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Separable Ogata-1998 kernels (each normalized to integrate to 1 over its domain)
# ─────────────────────────────────────────────────────────────────────────────


def utsu_productivity(m: np.ndarray | float, K: float, alpha: float, m0: float) -> np.ndarray:
    """Utsu productivity ``k(m) = K e^{alpha (m - M0)}`` — expected direct aftershocks of an event ``m``.

    This is the *number* of first-generation offspring an event of magnitude ``m`` triggers; the
    temporal and spatial kernels below distribute that number over time and space as densities.
    """
    return K * np.exp(alpha * (np.asarray(m, dtype=float) - m0))


def omori_utsu_density(t: np.ndarray | float, c: float, p: float) -> np.ndarray:
    """Omori-Utsu temporal *density* ``g(t) = (p-1)/c (1 + t/c)^{-p}`` (days), integrating to 1 on ``[0, inf)``.

    Normalized form (requires ``p > 1``): :math:`\\int_0^\\infty g(t)\\,dt = 1`, so multiplying by the
    productivity ``k(m)`` gives a kernel whose time-integral is exactly ``k(m)`` expected offspring.
    """
    t = np.asarray(t, dtype=float)
    return (p - 1.0) / c * np.power(1.0 + t / c, -p)


def omori_utsu_cumulative(t: np.ndarray | float, c: float, p: float) -> np.ndarray:
    """Cumulative Omori-Utsu mass ``G(t) = 1 - (1 + t/c)^{-(p-1)}`` on ``[0, t]`` (fraction of offspring).

    For ``p > 1`` this is the integral of :func:`omori_utsu_density` and lies in ``[0, 1)``; it gives
    the expected fraction of an event's aftershocks that have occurred within ``t`` days. Used to
    integrate the triggering contribution over a forecast window in closed form (no quadrature).
    """
    t = np.asarray(t, dtype=float)
    return 1.0 - np.power(1.0 + t / c, -(p - 1.0))


def spatial_scale(m: np.ndarray | float, D: float, gamma: float, m0: float) -> np.ndarray:
    """Magnitude-dependent spatial scale ``zeta(m) = D e^{gamma (m - M0)}`` (degrees).

    Larger events spread their aftershocks over a wider area; ``D`` sets the scale at the reference
    magnitude ``M0`` and ``gamma`` controls how fast the aftershock zone grows with magnitude.
    """
    return D * np.exp(gamma * (np.asarray(m, dtype=float) - m0))


def spatial_density(
    r_deg: np.ndarray | float, m: np.ndarray | float, D: float, gamma: float, q: float, m0: float
) -> np.ndarray:
    """Inverse-power spatial *density* ``f(r | m) = (q-1)/(pi zeta^2) (1 + r^2/zeta^2)^{-q}``.

    ``r_deg`` is the epicentral distance in degrees (small-region planar approximation, consistent
    with the 0.1° fit grid). Normalized so :math:`\\int_0^\\infty 2\\pi r f(r|m)\\,dr = 1` for ``q > 1``,
    i.e. the kernel distributes exactly one event's worth of probability mass over the plane.
    """
    zeta = spatial_scale(m, D, gamma, m0)
    r2 = np.asarray(r_deg, dtype=float) ** 2
    return (q - 1.0) / (np.pi * zeta**2) * np.power(1.0 + r2 / zeta**2, -q)


def _unit_xyz(lat: np.ndarray | float, lon: np.ndarray | float) -> np.ndarray:
    """Map geographic ``(lat, lon)`` in degrees to 3-D unit-sphere Cartesian coords for a KD-tree.

    The Euclidean distance between two such unit vectors is the *chord*; the great-circle arc is
    ``2 * arcsin(chord / 2)`` (radians). Same convention as the smoothed-seismicity KD-tree, so the
    ETAS triggering cutoff and the background field agree geometrically.
    """
    latr = np.radians(np.asarray(lat, dtype=float))
    lonr = np.radians(np.asarray(lon, dtype=float))
    coslat = np.cos(latr)
    return np.column_stack([coslat * np.cos(lonr), coslat * np.sin(lonr), np.sin(latr)])


# ─────────────────────────────────────────────────────────────────────────────
# The forecaster
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ETASForecaster(BaseForecaster):
    """Space-time ETAS (Ogata 1998) conditional forecaster — the production estimator and reference.

    Fit by maximum likelihood on the **full un-declustered** catalog slice before ``t_issue``; the
    background ``mu(x, y)`` is delegated to an adaptive smoothed-seismicity field fit on the
    *declustered* catalog (dual-catalog rule, model-design §5). After every fit, both stability gates
    are enforced (:class:`ETASStabilityError` on violation). Forecasting uses the simulation-free
    closed-form expected count by default; :meth:`simulate` provides the seeded synthetic-catalog
    Monte-Carlo path used for over-dispersion-honest bounds.

    Parameters
    ----------
    m0:
        Reference magnitude ``M0`` for the productivity / spatial-scale exponents (configs/etas.yaml
        ``m0``); must be at/above the target ``m_min``.
    mc, b_value:
        Magnitude of completeness and Gutenberg-Richter ``b``. If ``None`` they are estimated on the
        fit catalog (``b`` via the binning-corrected Aki-Utsu MLE; ``mc`` defaults to the catalog
        minimum ``mw`` as a conservative proxy — the real pipeline passes the rolling per-region Mc).
    bounds:
        Optimizer box constraints per parameter (defaults to :data:`DEFAULT_BOUNDS`).
    require_alpha_lt_beta, reject_supercritical:
        The two independent stability gates (configs/etas.yaml ``stability``). Both default on.
    background:
        An already-constructed :class:`SmoothedSeismicityForecaster` for ``mu(x, y)``. If ``None`` one
        is built with default hyperparameters and fit on the (declustered) catalog passed to
        :meth:`fit`. Pass an explicitly declustered-catalog-fit instance to honor the dual-catalog
        rule precisely.
    integration_steps:
        Number of sub-steps used for the time integral of the productivity term inside the forecast
        window (the spatial/magnitude parts are closed-form; only the residual coupling between the
        window edge and each parent's age needs a light quadrature).
    regime, regime_prior:
        Optional tectonic-regime tag (:class:`~caos_seismic.model.regime.TectonicRegime`) and its
        :class:`~caos_seismic.model.regime.RegimePrior`. When this ETAS is fit on one tile of the
        global tiled forecaster, the regime prior **seeds the MLE start** (regime-appropriate ``K``,
        ``alpha``, ``c``, ``p``) so a thin tile is pulled toward its regime's worldwide behaviour
        rather than a generic point (empirical-Bayes "borrow strength spatially", model-design §8).
        The prior is only a *start*; the MLE and both stability gates are unchanged. ``None`` keeps
        the original generic data-informed start (full backward compatibility).
    """

    name: str = "etas"
    version: str = "0.1.0"

    m0: float = 3.5
    mc: float | None = None
    b_value: float | None = None
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BOUNDS))
    require_alpha_lt_beta: bool = True
    reject_supercritical: bool = True
    background: SmoothedSeismicityForecaster | None = None
    integration_steps: int = 24
    #: Triggering-sum neighbour cutoffs — the O(N^2) → O(N·k) accelerator that makes the global,
    #: multi-decade, 10^5-event MLE tractable. A parent older than ``max_parent_days`` or farther than
    #: ``max_parent_dist_km`` from a child has, by construction, a negligible Omori/spatial kernel value
    #: (both kernels have decayed to ~0 there), so it is dropped from that child's triggering sum. The
    #: approximation error is bounded by the kernel tails *at* the cutoffs and is recorded in the model
    #: manifest. Defaults: 2 years of triggering memory, 500 km of aftershock-zone reach.
    max_parent_days: float = 730.0
    max_parent_dist_km: float = 500.0
    #: MLE optimizer budget. ``n_restarts`` extra random restarts beyond the single data-informed
    #: (regime-prior-seeded) start guard against shallow local optima; ``0`` keeps just the informed
    #: start, which is the cheap default the global tiled fit uses across its hundreds of tiles. A
    #: focused single-region fit can raise it for robustness.
    n_restarts: int = 1
    maxiter: int = 200
    regime: str | None = None
    regime_prior: object | None = None  # caos_seismic.model.regime.RegimePrior (avoids an import cycle)

    # ── Fitted state ─────────────────────────────────────────────────────────
    params: dict[str, float] = field(default_factory=dict, repr=False)
    _b: float = field(default=0.0, repr=False)
    _mc: float = field(default=0.0, repr=False)
    _beta: float = field(default=0.0, repr=False)
    _branching_ratio: float = field(default=0.0, repr=False)
    _t_issue: pd.Timestamp | None = field(default=None, repr=False)
    _region: Region | None = field(default=None, repr=False)
    # Parent catalog (events before t_issue), arrays for fast vectorized intensity.
    _ev_t: np.ndarray | None = field(default=None, repr=False)  # days before t_issue (>= 0)
    _ev_lat: np.ndarray | None = field(default=None, repr=False)
    _ev_lon: np.ndarray | None = field(default=None, repr=False)
    _ev_m: np.ndarray | None = field(default=None, repr=False)
    # Precomputed neighbour pairs (parent→child, within the cutoffs) + per-event background, built ONCE
    # at fit time so every MLE evaluation is an O(pairs) vectorized numpy sum with NO per-event Python
    # loop — the second half of the O(N^2)→O(N·k) acceleration (the cutoffs bound k; this removes the
    # Python overhead). ``_pair_child[p]`` is the child index of pair ``p``; the other arrays carry that
    # pair's parent magnitude, age gap (days) and epicentral distance (deg).
    _mu_ev: np.ndarray | None = field(default=None, repr=False)
    _pair_child: np.ndarray | None = field(default=None, repr=False)
    _pair_parent_m: np.ndarray | None = field(default=None, repr=False)
    _pair_dt: np.ndarray | None = field(default=None, repr=False)
    _pair_r_deg: np.ndarray | None = field(default=None, repr=False)
    _loglik: float = field(default=float("nan"), repr=False)
    params_used: dict = field(default_factory=dict, repr=False)

    # ── Forecaster.fit ───────────────────────────────────────────────────────
    def fit(self, catalog: pd.DataFrame, region: Region, t_issue: pd.Timestamp) -> "ETASForecaster":
        """MLE-fit the seven ETAS parameters on the full un-declustered catalog before ``t_issue``.

        Steps:

        1. Slice the catalog to ``time < t_issue`` (leakage backstop on top of the forecast clock).
        2. Estimate ``Mc`` / ``b`` (hence ``beta = b ln 10``) unless fixed by the caller.
        3. Build the smoothed-seismicity background ``mu(x, y)`` if one was not supplied.
        4. Maximize the space-time point-process log-likelihood by L-BFGS-B over the bounded box.
        5. Enforce the two stability gates; raise :class:`ETASStabilityError` if either fails.

        Returns ``self``. Raises ``ValueError`` for an empty/degenerate fit window and
        :class:`ETASStabilityError` for a supercritical / non-convergent fit.
        """
        validate_catalog(catalog)
        df = catalog.loc[catalog["time"] < t_issue].copy()
        if df.empty:
            raise ValueError("no events strictly before t_issue to fit ETAS")
        df = df.sort_values("time")

        self._t_issue = pd.Timestamp(t_issue)
        self._region = region

        # Completeness + b-value (estimated, never hard-coded to 1.0).
        self._mc = float(self.mc) if self.mc is not None else float(df["mw"].min())
        complete = df.loc[df["mw"] >= self._mc - 1e-9]
        if len(complete) < 2:
            raise ValueError(f"need >= 2 events at/above Mc={self._mc} to fit ETAS")
        if self.b_value is not None:
            self._b = float(self.b_value)
        else:
            self._b, _ = bvalue_aki_utsu(complete["mw"].to_numpy(), self._mc, delta_m=0.1)
        self._beta = self._b * LN10

        # Parent-event arrays (ages in days before t_issue, >= 0).
        t_days = (self._t_issue - complete["time"]).dt.total_seconds().to_numpy() / 86400.0
        self._ev_t = np.clip(t_days, 0.0, None)
        self._ev_lat = complete["latitude"].to_numpy(dtype=float)
        self._ev_lon = complete["longitude"].to_numpy(dtype=float)
        self._ev_m = complete["mw"].to_numpy(dtype=float)
        train_days = float(max(self._ev_t.max(), 1.0))  # observation window length T

        # Background mu(x, y): use the supplied (ideally declustered-catalog-fit) field, else build one.
        if self.background is None:
            self.background = SmoothedSeismicityForecaster(
                b_value=self._b, mc=self._mc
            ).fit(catalog, region, t_issue)
        elif self.background._ev_lat is None:  # not yet fit
            self.background.fit(catalog, region, t_issue)

        # Precompute the bounded parent→child neighbour pairs + per-event background ONCE, so each of
        # the thousands of MLE likelihood evaluations is a vectorized O(pairs) sum (no per-event loop).
        self._precompute_pairs()

        # MLE over the bounded parameter box.
        best = self._maximize_likelihood(train_days)
        self.params = best
        self._loglik = -self._negative_loglik(self._vector(best), train_days)

        # Stability gates (kept separate, both checked).
        self._branching_ratio = self._compute_branching_ratio(best)
        self._enforce_stability(best)

        self.params_used = {
            **best,
            "m0": self.m0,
            "mc": self._mc,
            "b": self._b,
            "beta": self._beta,
            "branching_ratio": self._branching_ratio,
            "loglik": self._loglik,
            "n_parents": int(self._ev_t.size),
            "train_days": train_days,
            "background": self.background.name,
            "regime": self.regime,
            "gates": {
                "require_alpha_lt_beta": self.require_alpha_lt_beta,
                "reject_supercritical": self.reject_supercritical,
            },
        }
        return self

    def recondition(self, catalog: pd.DataFrame, t_issue: pd.Timestamp) -> "ETASForecaster":
        """Advance the conditioning to a new issue time WITHOUT re-running the MLE.

        The seven ETAS parameters (and ``Mc``/``b``) are physical and stable over a refit cadence
        (configs/publish.yaml ``train_cadence.full_refit``); day-to-day only the *conditioning* changes —
        which events are parents and their ages. This re-slices the lawful past and refreshes just the
        parent arrays (``_ev_*``) against the new ``t_issue``, keeping the fitted ``params``/``Mc``/``b``
        AND the smoothed background (the long-term declustered rate barely moves within a cadence
        window). It lets the back-analysis / daily clock advance in O(N) per day instead of paying the
        full L-BFGS-B MLE again. Requires a prior :meth:`fit`.

        Leakage-safe: like :meth:`fit`, only events strictly before ``t_issue`` are admitted, so a
        reconditioned forecast never sees its own scoring window.
        """
        if not self.params or self._ev_t is None:
            raise RuntimeError("recondition() requires a prior fit()")
        validate_catalog(catalog)
        self._t_issue = pd.Timestamp(t_issue)
        df = catalog.loc[catalog["time"] < self._t_issue]
        complete = df.loc[df["mw"] >= self._mc - 1e-9].sort_values("time")
        if complete.empty:
            # No lawful parents at/above Mc: the field collapses to the (held) background. Keep params,
            # empty the triggering parents so conditional_intensity returns just mu.
            self._ev_t = np.empty(0, dtype=float)
            self._ev_lat = np.empty(0, dtype=float)
            self._ev_lon = np.empty(0, dtype=float)
            self._ev_m = np.empty(0, dtype=float)
            return self
        t_days = (self._t_issue - complete["time"]).dt.total_seconds().to_numpy() / 86400.0
        self._ev_t = np.clip(t_days, 0.0, None)
        self._ev_lat = complete["latitude"].to_numpy(dtype=float)
        self._ev_lon = complete["longitude"].to_numpy(dtype=float)
        self._ev_m = complete["mw"].to_numpy(dtype=float)
        return self

    # ── Likelihood machinery ──────────────────────────────────────────────────
    def _vector(self, params: dict[str, float]) -> np.ndarray:
        return np.array([params[k] for k in PARAM_NAMES], dtype=float)

    def _unpack(self, x: np.ndarray) -> dict[str, float]:
        return {k: float(v) for k, v in zip(PARAM_NAMES, x)}

    def _background_density_deg2(self, lat: float, lon: float) -> float:
        """Background rate ``mu(x, y)`` in events / day / **deg^2** at a point.

        The smoothed-seismicity field reports ``mu`` in events / day / km^2, whereas the ETAS spatial
        kernel :func:`spatial_density` is normalized per **deg^2** (its scale ``zeta`` is in degrees).
        Adding the two requires a common areal unit, so the per-km^2 background is converted to per-deg^2
        using the local area element ``km^2 per deg^2 = DEG2KM^2 * cos(lat)``. Doing this keeps the
        background and triggering terms on the same footing (the bug that otherwise lets the MLE inflate
        ``K`` because a per-km^2 background is ~10^4x too small next to a per-deg^2 trigger).
        """
        km2_per_deg2 = (DEG2KM**2) * np.cos(np.radians(lat))
        return self.background.background_rate_density(lat, lon) * km2_per_deg2

    def _precompute_pairs(self) -> None:
        """Build the bounded parent→child neighbour pairs + per-event background ONCE for the MLE.

        For each event ``j`` (a potential child) gather the strictly-older events within
        ``max_parent_days`` (a contiguous tail of the time-descending ``age`` array, found in O(log N)
        by ``searchsorted``) AND within ``max_parent_dist_km``. All surviving (parent, child) pairs are
        flattened into the ``_pair_*`` arrays and the per-event smoothed background cached in
        ``_mu_ev``. The likelihood then evaluates as one vectorized ``np.bincount`` over these pairs, so
        the O(N·k) neighbour search runs **once** instead of on every one of the optimizer's thousands
        of steps — the change that turns a multi-decade global fit from intractable into seconds/tile.
        """
        n = self._ev_t.size
        age = self._ev_t
        neg_age = -age  # ascending, for the O(log N) time-window search
        t_cut = float(self.max_parent_days)
        r_cut = float(self.max_parent_dist_km)

        self._mu_ev = np.array(
            [self._background_density_deg2(self._ev_lat[j], self._ev_lon[j]) for j in range(n)],
            dtype=float,
        )

        child_parts: list[np.ndarray] = []
        pm_parts: list[np.ndarray] = []
        dt_parts: list[np.ndarray] = []
        rdeg_parts: list[np.ndarray] = []
        for j in range(n):
            i_lo = int(np.searchsorted(neg_age, -(age[j] + t_cut), side="left"))
            if j <= i_lo:
                continue
            sl = slice(i_lo, j)
            dt = age[sl] - age[j]
            r_km = haversine_km(
                self._ev_lat[j], self._ev_lon[j], self._ev_lat[sl], self._ev_lon[sl]
            )
            keep = (dt > 1e-12) & (r_km <= r_cut)
            if not np.any(keep):
                continue
            cnt = int(np.count_nonzero(keep))
            child_parts.append(np.full(cnt, j, dtype=np.intp))
            pm_parts.append(self._ev_m[sl][keep])
            dt_parts.append(dt[keep])
            rdeg_parts.append(r_km[keep] / DEG2KM)

        if child_parts:
            self._pair_child = np.concatenate(child_parts)
            self._pair_parent_m = np.concatenate(pm_parts)
            self._pair_dt = np.concatenate(dt_parts)
            self._pair_r_deg = np.concatenate(rdeg_parts)
        else:
            self._pair_child = np.array([], dtype=np.intp)
            self._pair_parent_m = np.array([], dtype=float)
            self._pair_dt = np.array([], dtype=float)
            self._pair_r_deg = np.array([], dtype=float)

    def _conditional_intensity_at_events(self, p: dict[str, float]) -> np.ndarray:
        """Background + triggering intensity evaluated at each observed event time/place (the sum term).

        For event ``j`` only earlier events ``i < j`` (strictly older, larger age) contribute to the
        triggering sum — the Hawkes causality constraint. ``mu`` is the smoothed background rate at the
        event location. Returns an array of ``lambda(t_j, x_j, y_j)`` values (events / day / deg^2-ish
        intensity in the planar approximation), used in the first (log) term of the log-likelihood.
        """
        K, alpha, c, pp, D, gamma, q = self._vector(p)
        # Vectorized over the precomputed parent→child pairs (built once by :meth:`_precompute_pairs`,
        # within the temporal + spatial cutoffs). Each pair's triggering contribution
        # ``k(m_parent) g(dt) f(r | m_parent)`` is summed onto its child via ``np.bincount`` — an
        # O(pairs) numpy reduction with no per-event Python loop. ``mu`` (the smoothed background) is
        # fixed during the fit, so it is added per event from the precomputed ``_mu_ev``.
        lam = self._mu_ev.copy()
        if self._pair_child is not None and self._pair_child.size:
            k_m = utsu_productivity(self._pair_parent_m, K, alpha, self.m0)
            g_t = omori_utsu_density(self._pair_dt, c, pp)
            f_r = spatial_density(self._pair_r_deg, self._pair_parent_m, D, gamma, q, self.m0)
            lam += np.bincount(self._pair_child, weights=k_m * g_t * f_r, minlength=lam.size)
        return lam

    def _integrated_intensity(self, p: dict[str, float], train_days: float) -> float:
        """``∫_0^T ∫_A lambda dx dy dt`` — the compensator term of the log-likelihood.

        With separable, individually-normalized kernels the triggering integral collapses in closed
        form: each parent contributes ``k(m_i) * G(T - t_i)`` expected offspring over the window
        (its spatial kernel integrates to 1 over the plane, its temporal kernel to ``G`` over the
        elapsed window). The background contributes ``mu_total * T`` where ``mu_total`` is the
        region-integrated background rate (events / day), recovered from the smoothed field's own
        normalization (its spatial integral equals the declustered event rate per day).
        """
        K, alpha, c, pp, D, gamma, q = self._vector(p)
        # Background mass over the window: the smoothed field integrates to its declustered rate/day.
        bg_total = float(self.background._rate_per_day) * train_days
        # Triggering mass: each parent's productivity times the temporal CDF over its in-window age.
        # A parent of age a (days before t_issue) was active for (train_days - a) ... T of the window;
        # but the compensator integrates the *future* contribution within [0, T]. For an event that
        # occurred at forward time s_i = train_days - age_i within the window, its offspring mass over
        # the remaining window is k(m_i) * G(T - s_i) = k(m_i) * G(age_i). So age_i is exactly the
        # elapsed time available for triggering inside the observation window.
        k_m = utsu_productivity(self._ev_m, K, alpha, self.m0)
        g_cdf = omori_utsu_cumulative(self._ev_t, c, pp)
        trig_total = float(np.sum(k_m * g_cdf))
        return bg_total + trig_total

    def _negative_loglik(self, x: np.ndarray, train_days: float) -> float:
        """Negative space-time point-process log-likelihood ``-(sum ln lambda_i - ∫∫∫ lambda)``.

        Minimized by the optimizer. Guards against non-positive intensities (returns a large finite
        penalty) so L-BFGS-B never sees ``log(<=0)``.
        """
        p = self._unpack(x)
        lam = self._conditional_intensity_at_events(p)
        if np.any(~np.isfinite(lam)) or np.any(lam <= 0):
            return 1e12
        ll = float(np.sum(np.log(lam))) - self._integrated_intensity(p, train_days)
        if not np.isfinite(ll):
            return 1e12
        return -ll

    def _maximize_likelihood(self, train_days: float) -> dict[str, float]:
        """Bounded L-BFGS-B MLE from a data-informed start, with a coarse multi-start for robustness.

        SciPy is imported lazily so the package stays importable with numpy alone; a clear error is
        raised if SciPy is unavailable. The starting point puts ``alpha`` safely below ``beta`` to seed
        a subcritical region, and the optimizer is restarted from a few perturbations to avoid shallow
        local optima in the (mildly multimodal) ETAS likelihood.
        """
        try:
            from scipy.optimize import minimize
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without scipy
            raise ModuleNotFoundError(
                "ETAS MLE requires SciPy (scipy.optimize). Install the 'scipy' dependency."
            ) from exc

        bnds = [self.bounds[k] for k in PARAM_NAMES]

        def clamp(name: str, value: float) -> float:
            lo, hi = self.bounds[name]
            return float(min(max(value, lo), hi))

        # Data-informed start: alpha just under beta (subcritical), Omori p slightly > 1, modest K.
        # A regime prior (when supplied by the tiled forecaster) shifts the start toward the regime's
        # worldwide behaviour, keeping alpha safely subcritical; otherwise the generic start is used.
        rp = self.regime_prior
        if rp is not None:
            k_start = clamp("K", float(getattr(rp, "productivity_k", 0.1)))
            alpha0 = clamp("alpha", min(float(getattr(rp, "alpha", 0.8)), 0.9 * self._beta))
            c_start = clamp("c", float(getattr(rp, "c", 0.01)))
            p_start = clamp("p", float(getattr(rp, "p", 1.1)))
        else:
            k_start = clamp("K", 0.1)
            alpha0 = clamp("alpha", min(0.8, 0.9 * self._beta))
            c_start = clamp("c", 0.01)
            p_start = clamp("p", 1.1)
        x0 = np.array(
            [
                k_start,
                alpha0,
                c_start,
                p_start,
                clamp("D", 0.05),
                clamp("gamma", 0.5),
                clamp("q", 1.5),
            ],
            dtype=float,
        )

        starts = [x0]
        rng = np.random.default_rng(0)
        for _ in range(max(int(self.n_restarts), 0)):
            perturb = x0 * (1.0 + 0.25 * rng.standard_normal(x0.size))
            perturb = np.array([clamp(k, v) for k, v in zip(PARAM_NAMES, perturb)])
            starts.append(perturb)

        best_x, best_f = None, np.inf
        for s in starts:
            res = minimize(
                self._negative_loglik,
                s,
                args=(train_days,),
                method="L-BFGS-B",
                bounds=bnds,
                options={"maxiter": int(self.maxiter), "ftol": 1e-7},
            )
            if res.fun < best_f:
                best_f, best_x = float(res.fun), res.x
        if best_x is None:  # pragma: no cover - minimize always returns something
            best_x = x0
        return self._unpack(best_x)

    def _compute_branching_ratio(self, p: dict[str, float]) -> float:
        """Branching ratio ``n`` = expected direct offspring per event, integrated over the GR law.

        With ``k(m) = K e^{alpha(m - M0)}`` and the bounded GR density ``f(m) = beta e^{-beta(m-Mc)} /
        (1 - e^{-beta(m_max-Mc)})`` on ``[Mc, m_max]``, and the temporal/spatial kernels each
        integrating to 1::

            n = K e^{alpha(Mc - M0)} * beta / (beta - alpha)
                * (1 - e^{-(beta - alpha)(m_max - Mc)}) / (1 - e^{-beta (m_max - Mc)})

        valid for ``alpha < beta`` (finite-branching gate 1). When ``alpha >= beta`` the integral
        diverges (unbounded ``m_max``) or is dominated by the largest events; we return ``+inf`` to
        force rejection by gate 1. ``m_max`` comes from the region (bounded GR), which keeps ``n``
        finite and physical even near ``alpha ~ beta``.

        References: Zhuang et al. (2002); Helmstetter & Sornette (2003), *JGR* 108(B10), 2457.
        """
        K, alpha = p["K"], p["alpha"]
        beta, mc = self._beta, self._mc
        m_max = float(self._region.m_max) if self._region is not None else mc + 4.0
        base = K * np.exp(alpha * (mc - self.m0))
        span = m_max - mc
        denom_norm = 1.0 - np.exp(-beta * span)  # GR normalization on [Mc, m_max]
        if denom_norm <= 0:
            return float("inf")
        if abs(beta - alpha) < 1e-9:
            # limit: integral of e^{(alpha-beta)(m-Mc)} over the span -> span itself.
            integral = beta * span
        else:
            integral = beta / (beta - alpha) * (1.0 - np.exp(-(beta - alpha) * span))
        n = float(base * integral / denom_norm)
        return n if np.isfinite(n) and n >= 0 else float("inf")

    def _enforce_stability(self, p: dict[str, float]) -> None:
        """Apply the two independent stability gates; raise :class:`ETASStabilityError` on violation."""
        alpha = p["alpha"]
        if self.require_alpha_lt_beta and not (alpha < self._beta):
            raise ETASStabilityError(
                f"ETAS finite-branching gate failed: alpha={alpha:.4f} >= beta={self._beta:.4f} "
                f"(beta = b ln10, b={self._b:.3f}); productivity x magnitude integral diverges.",
                alpha=alpha,
                beta=self._beta,
                branching_ratio=self._branching_ratio,
            )
        if self.reject_supercritical and not (self._branching_ratio < 1.0):
            raise ETASStabilityError(
                f"ETAS supercritical: branching ratio n={self._branching_ratio:.4f} >= 1 "
                "(explosive cascade — rejected as a mis-fit).",
                alpha=alpha,
                beta=self._beta,
                branching_ratio=self._branching_ratio,
            )

    # ── Conditional intensity at an arbitrary forecast point ───────────────────
    def conditional_intensity(
        self, t_days_ahead: float, lat: float, lon: float, *, parents_only_before_issue: bool = True
    ) -> float:
        """Conditional intensity ``lambda(t, x, y | H_t)`` at ``t_issue + t_days_ahead`` and ``(lat, lon)``.

        Background plus the triggering sum over all parent events (the catalog before ``t_issue``).
        ``t_days_ahead`` is measured forward from ``t_issue``; each parent's elapsed time is its
        pre-issue age plus ``t_days_ahead``. Returns the intensity in events / day / deg^2 (planar
        approximation). This is the quantity integrated by :meth:`expected_counts`.
        """
        self._require_fit()
        K, alpha, c, pp, D, gamma, q = self._vector(self.params)
        mu = self._background_density_deg2(lat, lon)  # events / day / deg^2 (matches the spatial kernel)
        # Same neighbour cutoff as the fit: only parents within the spatial reach and the temporal
        # memory contribute a non-negligible kernel value, so the per-point triggering sum is bounded.
        r_km = haversine_km(lat, lon, self._ev_lat, self._ev_lon)
        keep = (r_km <= float(self.max_parent_dist_km)) & (self._ev_t <= float(self.max_parent_days))
        if not np.any(keep):
            return float(mu)
        dt = self._ev_t[keep] + float(t_days_ahead)  # elapsed days since each parent at the forecast time
        r_deg = r_km[keep] / DEG2KM
        k_m = utsu_productivity(self._ev_m[keep], K, alpha, self.m0)
        g_t = omori_utsu_density(dt, c, pp)
        f_r = spatial_density(r_deg, self._ev_m[keep], D, gamma, q, self.m0)
        return float(mu + np.sum(k_m * g_t * f_r))

    # ── Forecaster.expected_counts ─────────────────────────────────────────────
    def expected_counts(
        self,
        region: Region,
        cells: list[Cell],
        horizon_days: float,
        m_threshold: float,
        t_issue: pd.Timestamp,
    ) -> list[float]:
        """Expected count ``N_{>=M*}`` per cell over ``[t_issue, t_issue + horizon_days)``.

        Implements the methodology integral (model-design §3.2)::

            N_{>=M*} = ∫∫ lambda(t,x,y) dx dt * 10^{-b (M* - Mc)}            (bounded GR tail)

        Per cell we integrate the conditional intensity over the horizon (light time quadrature; the
        background part is time-flat) times the cell area, then multiply by the Gutenberg-Richter
        exceedance fraction :func:`gr_exceedance_fraction` (bounded by the region ``m_max``). The
        public probability is then ``1 - e^{-N}`` (see :meth:`forecast_probabilities`).
        """
        self._require_fit()
        if not cells:
            return []
        cell_area_deg2 = self._cell_area_deg2(cells)
        mag_frac = gr_exceedance_fraction(m_threshold, self._b, self._mc, region.m_max)
        if mag_frac <= 0.0:
            return [0.0] * len(cells)

        K, alpha, c, pp, D, gamma, q = self._vector(self.params)
        H = float(horizon_days)
        clat = np.fromiter((cell.lat for cell in cells), dtype=float, count=len(cells))
        clon = np.fromiter((cell.lon for cell in cells), dtype=float, count=len(cells))

        # Background mu(x, y) is time-flat over the window, so its window integral is mu * H. The
        # smoothed field is queried per cell (k-nearest KD-tree); the tiled forecaster routes only a
        # bounded subset of cells into each per-tile call, so this stays cheap.
        mu = np.array(
            [self._background_density_deg2(float(la), float(lo)) for la, lo in zip(clat, clon)],
            dtype=float,
        )
        integral = mu * H  # events / deg^2 over [0, H) from the background

        # Triggering term — fully vectorized, with the SAME neighbour cutoff as the fit. The window
        # integral of the Omori kernel is closed-form (no quadrature, and exact):
        #   ∫_0^H g(age_j + s) ds = G(age_j + H) - G(age_j)      with G the Omori-Utsu CDF,
        # so a parent j of (pre-issue) age ``age_j`` contributes a window-integrated productivity
        # ``kt_j = k(m_j) * [G(age_j + H) - G(age_j)]`` spread spatially by ``f(r_ij | m_j)``. Only
        # parents within the temporal memory carry mass; the spatial cutoff is applied pairwise below.
        ages = self._ev_t
        in_mem = ages <= float(self.max_parent_days)
        if np.any(in_mem):
            pm = self._ev_m[in_mem]
            page = ages[in_mem]
            kt = utsu_productivity(pm, K, alpha, self.m0) * (
                omori_utsu_cumulative(page + H, c, pp) - omori_utsu_cumulative(page, c, pp)
            )
            trig = self._triggering_field(
                clat, clon, self._ev_lat[in_mem], self._ev_lon[in_mem], pm, kt, D, gamma, q
            )
            integral = integral + trig

        counts = integral * cell_area_deg2 * mag_frac
        return [float(v) if v > 0.0 else 0.0 for v in counts]

    def _triggering_field(
        self,
        clat: np.ndarray,
        clon: np.ndarray,
        plat: np.ndarray,
        plon: np.ndarray,
        pm: np.ndarray,
        kt: np.ndarray,
        D: float,
        gamma: float,
        q: float,
    ) -> np.ndarray:
        """Window-integrated triggering rate at every cell — vectorized over cell↔parent pairs.

        Returns ``Σ_j kt_j · f(r_ij | m_j)`` per cell ``i`` (events / day / deg^2 already integrated
        over the window through ``kt_j``), where the sum runs over parent events within
        ``max_parent_dist_km`` of the cell — the same spatial cutoff used by the fit. Pairs are found
        with a unit-sphere :class:`scipy.spatial.cKDTree` and a single ``sparse_distance_matrix`` call
        (chord radius), so there is no Python-level per-cell or per-parent loop; the per-pair spatial
        density and the scatter-add are both vectorized. Falls back to a bounded per-cell numpy sweep
        if SciPy is unavailable (cell sets routed per tile are small, so the fallback is still usable).
        """
        out = np.zeros(clat.size, dtype=float)
        if clat.size == 0 or plat.size == 0:
            return out
        chord_cut = 2.0 * np.sin(min(float(self.max_parent_dist_km) / EARTH_RADIUS_KM, np.pi) / 2.0)
        try:
            from scipy.spatial import cKDTree
        except ModuleNotFoundError:  # pragma: no cover - exercised only without scipy
            for i in range(clat.size):
                r_km = haversine_km(float(clat[i]), float(clon[i]), plat, plon)
                keep = r_km <= float(self.max_parent_dist_km)
                if np.any(keep):
                    r_deg = r_km[keep] / DEG2KM
                    f_r = spatial_density(r_deg, pm[keep], D, gamma, q, self.m0)
                    out[i] = float(np.sum(kt[keep] * f_r))
            return out

        coo = (
            cKDTree(_unit_xyz(clat, clon))
            .sparse_distance_matrix(cKDTree(_unit_xyz(plat, plon)), chord_cut, output_type="coo_matrix")
        )
        ci, pj, chord = coo.row, coo.col, coo.data
        if ci.size == 0:
            return out
        # chord (unit-sphere Euclidean) → great-circle arc → degrees, matching haversine + DEG2KM.
        r_deg = (2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)) * EARTH_RADIUS_KM) / DEG2KM
        f_r = spatial_density(r_deg, pm[pj], D, gamma, q, self.m0)
        np.add.at(out, ci, kt[pj] * f_r)
        return out

    def forecast_probabilities(
        self,
        region: Region,
        cells: list[Cell],
        horizon_days: float,
        m_threshold: float,
        t_issue: pd.Timestamp,
    ) -> list[float]:
        """Convenience: per-cell ``P(>=1 event >= M*) = 1 - e^{-N}`` (the public exceedance formula)."""
        return [
            poisson_p_at_least_one(n)
            for n in self.expected_counts(region, cells, horizon_days, m_threshold, t_issue)
        ]

    # ── Seeded synthetic-catalog simulation (Monte-Carlo path for over-dispersion bounds) ──
    def simulate(
        self,
        region: Region,
        horizon_days: float,
        *,
        seed: int,
        m_min: float | None = None,
        max_generations: int = 100,
    ) -> pd.DataFrame:
        """Simulate ONE synthetic next-window catalog over ``[t_issue, t_issue + horizon_days)``.

        Branching-process (thinning) simulation seeded by ``seed`` for byte-reproducibility:

        * **Background** events are drawn as a homogeneous Poisson process in time with the
          region-integrated background rate, placed spatially by sampling the smoothed field's parent
          events (each background event inherits a fitted-bandwidth Gaussian jitter around a randomly
          chosen historical epicenter — a fast, normalization-consistent surrogate for the kernel).
        * **Aftershocks** of every existing parent (historical *and* newly simulated) are generated
          generation-by-generation: a parent of magnitude ``m`` spawns ``Poisson(k(m) * remaining
          Omori mass in window)`` offspring, each with an Omori-distributed time, an inverse-power
          spatial offset (``zeta(m)`` scale), and a bounded-GR magnitude.

        The loop terminates when no generation produces offspring inside the window (guaranteed in
        finite time because the fit is subcritical, ``n < 1``) or after ``max_generations``.

        Returns a DataFrame with the catalog columns (``time, latitude, longitude, mw, mag, mag_type,
        source, event_id``) for the synthetic events only. Drawing many of these with different seeds
        and binning per cell yields the over-dispersion-honest catalog-based forecast and the
        P10/median/P90 bounds (model-design §7.2, §9 step 4).
        """
        self._require_fit()
        rng = np.random.default_rng(seed)
        K, alpha, c, pp, D, gamma, q = self._vector(self.params)
        m_max = float(region.m_max)
        mc = self._mc
        m_min = float(m_min) if m_min is not None else mc
        T = float(horizon_days)

        rows: list[dict] = []

        # 1) Background events: homogeneous Poisson in time at the region-integrated rate.
        bg_rate_per_day = float(self.background._rate_per_day)
        n_bg = int(rng.poisson(bg_rate_per_day * T))
        if n_bg > 0 and self.background._ev_lat is not None and self.background._ev_lat.size > 0:
            parent_idx = rng.integers(0, self.background._ev_lat.size, size=n_bg)
            for k in range(n_bg):
                pi = int(parent_idx[k])
                d_km = float(self.background._ev_d[pi])  # adaptive bandwidth (km) as jitter scale
                jitter_deg = (d_km / DEG2KM) * rng.standard_normal(2)
                lat = float(self.background._ev_lat[pi] + jitter_deg[0])
                lon = float(self.background._ev_lon[pi] + jitter_deg[1])
                t = float(rng.random() * T)
                m = float(sample_truncated_gr(rng, 1, self._b, mc, m_max)[0])
                rows.append(self._mk_event(t, lat, lon, m, "sim_background"))

        # 2) Branching cascade. Seed the queue with (a) historical parents (their in-window offspring)
        #    and (b) the freshly simulated background events.
        # Historical parents trigger offspring over the *remaining* window; their pre-issue age shifts
        # the Omori clock so a long-decayed mainshock contributes little.
        Generation = list[tuple[float, float, float, float]]  # (t_in_window, lat, lon, m)
        queue: Generation = [
            (-float(age), float(la), float(lo), float(mm))
            for age, la, lo, mm in zip(self._ev_t, self._ev_lat, self._ev_lon, self._ev_m)
        ]
        queue += [(r["_t"], r["latitude"], r["longitude"], r["mw"]) for r in rows]

        for _gen in range(max_generations):
            next_queue: Generation = []
            for (t_parent, la, lo, mm) in queue:
                # Expected offspring inside the window: productivity * Omori mass from max(t_parent, 0)
                # to T, measured from the parent's own origin (t_parent may be negative = pre-issue).
                k_m = float(utsu_productivity(mm, K, alpha, self.m0))
                t0 = max(t_parent, 0.0)
                # elapsed-time bounds for the parent's Omori clock within the window:
                a_lo = t0 - t_parent  # age at window start for this parent (>= 0)
                a_hi = T - t_parent    # age at window end
                if a_hi <= a_lo:
                    continue
                mass = float(
                    omori_utsu_cumulative(a_hi, c, pp) - omori_utsu_cumulative(a_lo, c, pp)
                )
                lam = k_m * max(mass, 0.0)
                n_off = int(rng.poisson(lam)) if lam > 0 else 0
                for _ in range(n_off):
                    # Omori-distributed age within [a_lo, a_hi] via inverse-CDF on the truncated kernel.
                    u = rng.random()
                    g_lo = float(omori_utsu_cumulative(a_lo, c, pp))
                    g_hi = float(omori_utsu_cumulative(a_hi, c, pp))
                    g = g_lo + u * (g_hi - g_lo)
                    age = c * ((1.0 - g) ** (-1.0 / (pp - 1.0)) - 1.0)
                    t_child = t_parent + age
                    if t_child < 0.0 or t_child >= T:
                        continue
                    # Inverse-power spatial offset at scale zeta(mm); isotropic direction.
                    zeta_deg = float(spatial_scale(mm, D, gamma, self.m0))
                    uu = rng.random()
                    r_deg = zeta_deg * np.sqrt((1.0 - uu) ** (-1.0 / (q - 1.0)) - 1.0)
                    theta = 2.0 * np.pi * rng.random()
                    lat_c = la + r_deg * np.sin(theta)
                    lon_c = lo + r_deg * np.cos(theta) / max(np.cos(np.radians(la)), 1e-6)
                    m_c = float(sample_truncated_gr(rng, 1, self._b, mc, m_max)[0])
                    rows.append(self._mk_event(float(t_child), float(lat_c), float(lon_c), m_c, "sim_triggered"))
                    next_queue.append((float(t_child), float(lat_c), float(lon_c), m_c))
            if not next_queue:
                break
            queue = next_queue

        if not rows:
            return self._empty_catalog()
        df = pd.DataFrame(rows)
        df = df.loc[df["mw"] >= m_min - 1e-9].copy()
        # Convert relative window time to absolute UTC timestamps.
        df["time"] = self._t_issue + pd.to_timedelta(df["_t"], unit="D")
        df = df.drop(columns=["_t"]).sort_values("time").reset_index(drop=True)
        return df

    # ── internals ──────────────────────────────────────────────────────────────
    def _mk_event(self, t_days: float, lat: float, lon: float, m: float, source: str) -> dict:
        """Build one synthetic catalog row (relative time kept in ``_t`` until absolute conversion)."""
        return {
            "event_id": "",  # filled by caller / dedup if needed
            "_t": float(t_days),
            "latitude": float(lat),
            "longitude": float(lon),
            "depth_km": np.nan,
            "mag": float(m),
            "mag_type": "Mw",
            "mw": float(m),
            "source": source,
        }

    def _empty_catalog(self) -> pd.DataFrame:
        cols = ["event_id", "time", "latitude", "longitude", "depth_km", "mag", "mag_type", "mw", "source"]
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    def _cell_area_deg2(self, cells: list[Cell]) -> float:
        """Area of one fit cell in deg^2 (matches the planar deg-based spatial kernel normalization).

        The spatial density :func:`spatial_density` is normalized per unit area in *degrees* (its
        ``zeta`` is in degrees), so the intensity it produces is per deg^2 and the cell area used to
        turn an intensity into a count must also be in deg^2. We infer the regular pitch from the
        cell latitudes (the 0.1° fit grid) and correct longitude by ``cos(lat)``.
        """
        if len(cells) >= 2:
            lats = np.array([c.lat for c in cells])
            pitch = np.median(np.diff(np.unique(np.round(lats, 4))))
            if not np.isfinite(pitch) or pitch <= 0:
                pitch = 0.1
        else:
            pitch = 0.1
        mean_lat = float(np.mean([c.lat for c in cells])) if cells else 0.0
        return float(pitch * pitch * np.cos(np.radians(mean_lat)))

    def _require_fit(self) -> None:
        if self._ev_t is None or not self.params:
            raise RuntimeError("ETASForecaster.fit() must be called before use")

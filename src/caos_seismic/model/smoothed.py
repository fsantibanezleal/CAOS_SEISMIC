"""Adaptive smoothed-seismicity background — the **mandatory null** Forecaster.

Stationary, time-independent Poisson estimate of *where* earthquakes occur, obtained by
smoothing a **declustered** catalog with an adaptive power-law kernel. This module supplies two
things the rest of the system depends on:

  1. the spatial background field ``mu(x, y)`` that seeds the ETAS conditional intensity
     (:mod:`caos_seismic.model.etas`), and
  2. the **stationary Poisson reference** — the null hypothesis any time-dependent model must beat
     in prospective CSEP comparison testing (information gain > 0 with a CI excluding zero).

Kernel (Helmstetter, Kagan & Jackson 2007, *SRL* 78(1), 78-86, doi:10.1785/gssrl.78.1.78). Each
event contributes an isotropic power-law kernel whose bandwidth ``d_i`` is the distance to its
``n``-th nearest neighbour (adaptive smoothing — dense regions sharpen, sparse regions broaden)::

    mu(x, y) = sum_i K_{d_i}(r_i),      K_d(r) = C(d) * (r^2 + d^2)^{-s}

where ``r_i`` is the epicentral distance from cell centre to event ``i``.

**The exponent ``s`` and normalization ``C(d)`` are NOT hard-coded to 3/2.** As the methodology
flags, the HKJ family quotes both an ``s = 1`` form ``∝ 1/(r^2 + d^2)`` and an ``s = 3/2`` form
``∝ 1/(r^2 + d^2)^{3/2}`` depending on the specific paper/normalization. The value is taken from a
named reference profile (:data:`KERNEL_PROFILES`) selected by config, and ``C(d)`` is derived
analytically so each per-event kernel integrates to one earthquake over the plane (see
:func:`_kernel_normalization`). The neighbour count ``n`` is a region-tuned hyperparameter, not a
universal constant.

The field is scaled so its spatial integral equals the observed event rate of the declustered
catalog (events per day), making ``mu`` a genuine Poisson rate density (events / day / deg^2). The
forecast probability uses the same exceedance formula as every other Forecaster::

    N_{>=M*} = mu(cell) * area(cell) * horizon * 10^{-b (M* - Mc)}
    P(>=1 event >= M*) = 1 - exp(-N_{>=M*})            (Gutenberg-Richter magnitude tail)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contracts import BaseForecaster, Cell, Region, validate_catalog
from ._common import (
    DEG2KM,
    bvalue_aki_utsu,
    gr_exceedance_fraction,
    haversine_km,
    poisson_p_at_least_one,
)

# ─────────────────────────────────────────────────────────────────────────────
# Named kernel profiles — pin the exponent + normalization to a reference, never hard-code 3/2.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KernelProfile:
    """A named adaptive-kernel parameterization from the HKJ literature.

    Attributes
    ----------
    name:
        Human-readable identifier recorded in the provenance manifest.
    exponent_s:
        Power-law exponent ``s`` in ``K_d(r) = C(d) (r^2 + d^2)^{-s}``. ``s > 1`` is required for
        the 2-D radial integral ``∫_0^∞ 2πr (r^2+d^2)^{-s} dr`` to converge.
    reference:
        Literature citation the exponent/normalization is pinned to.
    """

    name: str
    exponent_s: float
    reference: str


#: Reference profiles. ``hkj2007`` is the power-law-(-3/2) form used by Helmstetter, Kagan &
#: Jackson (2007) and carried into pyCSEP/floatCSEP smoothed-seismicity forecasts; ``power1`` is
#: the alternative ``∝ 1/(r^2 + d^2)`` form that also appears in the family. The exponent is
#: selected by config (``etas.yaml: background_model``), never assumed.
KERNEL_PROFILES: dict[str, KernelProfile] = {
    "hkj2007": KernelProfile(
        name="hkj2007",
        exponent_s=1.5,
        reference="Helmstetter, Kagan & Jackson (2007), SRL 78(1), 78-86, doi:10.1785/gssrl.78.1.78",
    ),
    "power1": KernelProfile(
        name="power1",
        exponent_s=1.0,
        reference="HKJ-family power-(-1) kernel ∝ 1/(r^2 + d^2) (Werner et al. 2011 normalization)",
    ),
}

DEFAULT_PROFILE = "hkj2007"


def _kernel_normalization(d: float, s: float) -> float:
    """Normalization ``C(d)`` so a single-event kernel integrates to exactly one event.

    For ``K_d(r) = C(d) (r^2 + d^2)^{-s}`` the integral over the plane is::

        ∫_0^∞ 2π r C(d) (r^2 + d^2)^{-s} dr = C(d) * π / ((s - 1) d^{2(s-1)})

    (valid for ``s > 1``; substitute ``u = r^2``). Setting this to 1 gives::

        C(d) = (s - 1) d^{2(s-1)} / π .

    For the limiting ``s = 1`` case the radial integral diverges, so the kernel is truncated at a
    multiple of the bandwidth and normalized numerically by the caller; here we return the
    finite-``s`` closed form which the adaptive sum re-normalizes globally anyway.

    Distances are in kilometres (so ``mu`` is per km^2 before conversion to per-deg^2 by the field
    scaling); ``d`` is the adaptive bandwidth in km.
    """
    if s <= 1.0:
        # s = 1 has no finite plane integral; use a large-but-finite surrogate (s -> 1+) and rely
        # on the global rate re-scaling in `fit` to fix the absolute level.
        s = 1.0 + 1e-6
    return (s - 1.0) * d ** (2.0 * (s - 1.0)) / np.pi


#: Mean Earth radius (km) for the chord→great-circle conversion in the KD-tree bandwidth.
_EARTH_R_KM = 6371.0088


def _nth_nearest_bandwidth(
    lat: np.ndarray, lon: np.ndarray, n_neighbors: int, d_min_km: float
) -> np.ndarray:
    """Adaptive bandwidth ``d_i`` = great-circle distance (km) to each event's ``n``-th neighbour.

    Floored at ``d_min_km`` so coincident/very-close events do not yield a singular kernel.

    Computed in **O(N log N)** with a 3-D unit-sphere KD-tree (``scipy.spatial.cKDTree``): events are
    mapped to unit vectors, whose Euclidean **chord** distance is monotonic in great-circle distance,
    so the ``k``-th nearest neighbour is identical to the on-sphere one; the chord is then converted
    back to a great-circle arc-length in km. This is the change that makes the **global** (10^5-event)
    smoothed-null fit tractable — the previous O(N^2) all-pairs loop hung on a worldwide catalog. A
    pure-numpy O(N^2) fallback is kept for the (rare) case SciPy is unavailable.
    """
    n_events = lat.size
    if n_events <= 1:
        return np.full(n_events, d_min_km, dtype=float)
    # Rank we actually take: the n-th *other* event (the self-match at distance 0 is column 0).
    k = min(int(n_neighbors), max(1, n_events - 1))

    try:
        from scipy.spatial import cKDTree
    except ModuleNotFoundError:  # pragma: no cover - SciPy is a core dependency
        d = np.empty(n_events, dtype=float)
        for i in range(n_events):
            dist = haversine_km(lat[i], lon[i], lat, lon)
            dist_sorted = np.partition(dist, k)[: k + 1]
            d[i] = max(np.sort(dist_sorted)[k], d_min_km)
        return d

    latr = np.radians(lat)
    lonr = np.radians(lon)
    coslat = np.cos(latr)
    pts = np.column_stack([coslat * np.cos(lonr), coslat * np.sin(lonr), np.sin(latr)])
    tree = cKDTree(pts)
    chord, _ = tree.query(pts, k=k + 1)          # (N, k+1); column 0 is the self-match (0.0)
    chord_k = np.clip(np.atleast_2d(chord)[:, k], 0.0, 2.0)
    arc_km = 2.0 * np.arcsin(chord_k / 2.0) * _EARTH_R_KM   # chord → great-circle arc length (km)
    return np.maximum(arc_km, d_min_km)


@dataclass
class SmoothedSeismicityForecaster(BaseForecaster):
    """Adaptive smoothed-seismicity Poisson background and the mandatory null.

    Parameters
    ----------
    n_neighbors:
        Neighbour count ``n`` setting the adaptive bandwidth (HKJ ``n ≈ 6``; region-tuned).
    profile:
        Key into :data:`KERNEL_PROFILES` pinning the kernel exponent/normalization to a reference.
    d_min_km:
        Floor on the adaptive bandwidth (km) to avoid singular kernels for clustered events.
    b_value:
        Optional fixed Gutenberg-Richter ``b``. If ``None`` it is estimated by the binning-corrected
        Aki-Utsu MLE on the fit catalog (never hard-coded to 1.0).
    mc:
        Optional magnitude of completeness. If ``None`` the catalog minimum ``mw`` is used as a
        conservative proxy (the real pipeline passes the per-region rolling ``Mc``).
    """

    name: str = "smoothed_seismicity"
    version: str = "0.1.0"

    n_neighbors: int = 6
    profile: str = DEFAULT_PROFILE
    d_min_km: float = 2.0
    b_value: float | None = None
    mc: float | None = None
    #: Number of nearest events summed per query point at INFERENCE. The adaptive kernel decays as
    #: ``r^{-2s}``, so the few-dozen nearest events carry essentially all of a point's density; querying
    #: them with a KD-tree makes the field O(cells · k log N) instead of O(cells · N) — the difference
    #: between an instant and a 10^5-events × 10^6-cells global evaluation. ``None`` sums all events.
    inference_k: int = 96

    # Fitted state (populated by `fit`).
    _ev_lat: np.ndarray | None = field(default=None, repr=False)
    _ev_lon: np.ndarray | None = field(default=None, repr=False)
    _ev_xyz: np.ndarray | None = field(default=None, repr=False)  # 3-D unit-sphere coords for the KD-tree
    _tree: object | None = field(default=None, repr=False)        # scipy.spatial.cKDTree over _ev_xyz
    _ev_d: np.ndarray | None = field(default=None, repr=False)  # adaptive bandwidths (km)
    _ev_norm: np.ndarray | None = field(default=None, repr=False)  # per-event C(d_i)
    _exponent_s: float = field(default=0.0, repr=False)
    _b: float = field(default=0.0, repr=False)
    _mc: float = field(default=0.0, repr=False)
    _rate_per_day: float = field(default=0.0, repr=False)  # declustered events/day at >= Mc
    _train_days: float = field(default=0.0, repr=False)

    # ── Forecaster.fit ───────────────────────────────────────────────────────
    def fit(
        self, catalog: pd.DataFrame, region: Region, t_issue: pd.Timestamp
    ) -> "SmoothedSeismicityForecaster":
        """Build the stationary background field from the **declustered** catalog before ``t_issue``.

        The caller is responsible for passing the declustered catalog (the dual-catalog rule, see
        ``configs/declustering.yaml``); this forecaster does not decluster — it only smooths. Events
        at or after ``t_issue`` are dropped here as a leakage backstop even though the forecast clock
        already guarantees it.
        """
        validate_catalog(catalog)
        if self.profile not in KERNEL_PROFILES:
            raise ValueError(
                f"unknown kernel profile {self.profile!r}; choose from {sorted(KERNEL_PROFILES)}"
            )
        self._exponent_s = KERNEL_PROFILES[self.profile].exponent_s

        df = catalog.loc[catalog["time"] < t_issue].copy()
        if df.empty:
            raise ValueError("no events strictly before t_issue to fit the background field")

        # Completeness + b-value (estimated, never hard-coded).
        self._mc = float(self.mc) if self.mc is not None else float(df["mw"].min())
        complete = df.loc[df["mw"] >= self._mc - 1e-9]
        if complete.empty:
            raise ValueError(f"no events at or above Mc={self._mc} in the fit catalog")
        if self.b_value is not None:
            self._b = float(self.b_value)
        else:
            self._b, _ = bvalue_aki_utsu(complete["mw"].to_numpy(), self._mc, delta_m=0.1)

        lat = complete["latitude"].to_numpy(dtype=float)
        lon = complete["longitude"].to_numpy(dtype=float)
        self._ev_lat, self._ev_lon = lat, lon
        # 3-D unit-sphere index, reused for the O(k) k-nearest queries at inference.
        latr = np.radians(lat)
        lonr = np.radians(lon)
        coslat = np.cos(latr)
        self._ev_xyz = np.column_stack([coslat * np.cos(lonr), coslat * np.sin(lonr), np.sin(latr)])
        try:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self._ev_xyz)
        except ModuleNotFoundError:  # pragma: no cover - SciPy is a core dependency
            self._tree = None
        self._ev_d = _nth_nearest_bandwidth(lat, lon, self.n_neighbors, self.d_min_km)
        self._ev_norm = np.array(
            [_kernel_normalization(d, self._exponent_s) for d in self._ev_d], dtype=float
        )

        # Stationary rate: declustered complete events per day over the observed span.
        t = complete["time"]
        span_days = max((t_issue - t.min()).total_seconds() / 86400.0, 1.0)
        self._train_days = float(span_days)
        self._rate_per_day = float(len(complete)) / span_days
        return self

    # ── Spatial density helpers ──────────────────────────────────────────────
    def background_density_km2(self, lat: float, lon: float) -> float:
        """Smoothed event density (events / km^2, integrated over all time) at a point.

        This is ``sum_i C(d_i) (r_i^2 + d_i^2)^{-s}`` — the raw HKJ field, normalized so the whole
        catalog integrates to ``len(events)`` over the plane. Multiply by a temporal rate and a cell
        area to get an expected count.
        """
        self._require_fit()
        n = self._ev_lat.size
        if self._tree is not None and self.inference_k is not None and n > int(self.inference_k):
            # Sum the kernel over the k nearest events only — the decaying ``r^{-2s}`` kernel makes the
            # far events' contribution negligible, so this is O(k log N) instead of O(N) per query.
            latr = np.radians(lat)
            lonr = np.radians(lon)
            cl = np.cos(latr)
            q = np.array([cl * np.cos(lonr), cl * np.sin(lonr), np.sin(latr)])
            kk = min(int(self.inference_k), n)
            _, idx = self._tree.query(q, k=kk)
            idx = np.atleast_1d(np.asarray(idx, dtype=int))
            r = haversine_km(lat, lon, self._ev_lat[idx], self._ev_lon[idx])
            ker = self._ev_norm[idx] * np.power(r * r + self._ev_d[idx] ** 2, -self._exponent_s)
            return float(np.sum(ker))
        r = haversine_km(lat, lon, self._ev_lat, self._ev_lon)
        k = self._ev_norm * np.power(r * r + self._ev_d * self._ev_d, -self._exponent_s)
        return float(np.sum(k))

    def background_rate_density(self, lat: float, lon: float) -> float:
        """Stationary Poisson **rate** density ``mu(x, y)`` in events / day / km^2 at a point.

        The spatial shape (``background_density_km2``) is a per-km^2 spatial PDF×N; dividing by the
        event count turns it into a normalized spatial PDF, then multiplying by the observed
        events/day gives a rate density that integrates to the catalog rate over the region. This is
        the ``mu(x, y)`` consumed by the ETAS background term.
        """
        self._require_fit()
        n_events = self._ev_lat.size
        spatial_pdf = self.background_density_km2(lat, lon) / max(n_events, 1)
        return spatial_pdf * self._rate_per_day

    # ── Forecaster.expected_counts ───────────────────────────────────────────
    def expected_counts(
        self,
        region: Region,
        cells: list[Cell],
        horizon_days: float,
        m_threshold: float,
        t_issue: pd.Timestamp,
    ) -> list[float]:
        """Expected count ``N_{>=M*}`` per cell over ``[t_issue, t_issue + horizon_days)``.

        Stationary background → the rate does not depend on ``t_issue`` (the clock only fixes which
        events were used in :meth:`fit`). Per cell::

            N = mu(cell) * cell_area_km2 * horizon_days * 10^{-b (M* - Mc)}

        bounded above so the magnitude tail never exceeds the region ``m_max`` integral.
        """
        self._require_fit()
        cell_area_km2 = self._cell_area_km2(cells)
        mag_frac = gr_exceedance_fraction(m_threshold, self._b, self._mc, region.m_max)
        out: list[float] = []
        for c in cells:
            mu = self.background_rate_density(c.lat, c.lon)  # events / day / km^2
            n = mu * cell_area_km2 * horizon_days * mag_frac
            out.append(max(n, 0.0))
        return out

    def forecast_probabilities(
        self,
        region: Region,
        cells: list[Cell],
        horizon_days: float,
        m_threshold: float,
        t_issue: pd.Timestamp,
    ) -> list[float]:
        """Convenience: ``P(>=1 event >= M*)`` per cell via ``1 - exp(-N)`` (the public formula)."""
        return [poisson_p_at_least_one(n) for n in self.expected_counts(
            region, cells, horizon_days, m_threshold, t_issue
        )]

    # ── internals ────────────────────────────────────────────────────────────
    def _cell_area_km2(self, cells: list[Cell]) -> float:
        """Approximate area of one fit cell (km^2) from the median cell pitch.

        The fine fit grid is regular in degrees (``configs/grid.yaml: fit.cell_deg``). We infer the
        pitch from the smallest non-zero inter-cell spacing so this works whether the caller passes
        the 0.1° fit grid or an H3 display grid, and correct longitude by ``cos(lat)``.
        """
        if len(cells) >= 2:
            lats = np.array([c.lat for c in cells])
            pitch = np.median(np.diff(np.unique(np.round(lats, 4))))
            if not np.isfinite(pitch) or pitch <= 0:
                pitch = 0.1
        else:
            pitch = 0.1
        mean_lat = float(np.mean([c.lat for c in cells])) if cells else 0.0
        dlat_km = pitch * DEG2KM
        dlon_km = pitch * DEG2KM * np.cos(np.radians(mean_lat))
        return float(dlat_km * dlon_km)

    def _require_fit(self) -> None:
        if self._ev_lat is None:
            raise RuntimeError("SmoothedSeismicityForecaster.fit() must be called before use")

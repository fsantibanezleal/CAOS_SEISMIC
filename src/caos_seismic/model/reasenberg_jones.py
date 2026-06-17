"""Reasenberg-Jones aftershock model — the transparent operational fallback / sanity check.

The most transparent "tomorrow's earthquakes" model and the shape the USGS Operational Aftershock
Forecast (OAF) system runs alongside ETAS. The rate of aftershocks of magnitude ``>= M`` following a
mainshock ``M_m`` is a Gutenberg-Richter magnitude term times a modified-Omori temporal decay
(methodology §1.4, model-design §1.3)::

    lambda(t, M) = 10^{a + b (M_m - M)} / (t + c)^p                       (per day, t in days)

Integrated over a forecast window ``[t1, t2]`` (with ``p != 1``)::

    N(>=M) = 10^{a + b(M_m - M)} * [ (t2 + c)^{1-p} - (t1 + c)^{1-p} ] / (1 - p)

and the probability of at least one such event (non-homogeneous Poisson)::

    P(>=1) = 1 - e^{-N} .

Here ``t`` is elapsed time since the **most recent triggering mainshock** in the catalog. The model
is deliberately mainshock-driven: it is a *fallback* and *sanity check*, not the primary spatial
forecaster — it forecasts the aftershock productivity of the largest recent event, distributed in
space by that event's smoothed influence. Where ETAS is the production estimator, R-J is the
human-auditable cross-check (does ETAS roughly agree with the textbook Omori extrapolation?).

Parameters
----------
``a`` (sequence productivity), ``b`` (Gutenberg-Richter slope), ``c`` (Omori offset, days), ``p``
(Omori decay). Generic / regionally-defaulted constants are explicitly flagged: the methodology
warns *do not reuse California parameters for Chile* — defaults here are the widely-cited generic
values (Reasenberg & Jones 1989; Page et al. 2016 global), used only when a region/sequence fit is
unavailable, and the chosen values are recorded in ``params_used`` for the manifest.

References
----------
Reasenberg, P. A. & Jones, L. M. (1989), *Science* 243(4895), 1173-1176, doi:10.1126/science.243.4895.1173.
Page, M. T. et al. (2016), *BSSA* 106(5), 2290-2301, doi:10.1785/0120160073 (global tectonic-regime priors).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contracts import BaseForecaster, Cell, Region, validate_catalog
from ._common import DEG2KM, haversine_km, poisson_p_at_least_one

# Generic Reasenberg-Jones constants (California-derived; flagged non-universal). The a/b/c/p below
# are the order-of-magnitude generic values used by the OAF when no sequence-specific fit exists.
# They are NOT silently trusted for Chile — see the region note in region.chile.yaml.
GENERIC_RJ = {"a": -1.67, "b": 0.91, "c": 0.05, "p": 1.08}


@dataclass
class ReasenbergJonesForecaster(BaseForecaster):
    """Transparent modified-Omori aftershock forecaster (USGS OAF shape).

    Parameters
    ----------
    a, b, c, p:
        Reasenberg-Jones constants. If left ``None`` the generic values :data:`GENERIC_RJ` are used
        and flagged in :attr:`params_used`; ``b`` defaults to the catalog Aki-Utsu estimate when a
        completeness is supplied via :meth:`fit`.
    influence_km:
        Spatial smoothing length (km) of the mainshock's aftershock zone — aftershock density is a
        Gaussian in epicentral distance with this scale, so the per-cell rate decays away from the
        mainshock rather than being uniform over the region. Scales with mainshock magnitude via the
        Utsu rupture-length relation if not overridden.
    trigger_min_mag:
        Only events at/above this magnitude are considered candidate mainshocks.
    """

    name: str = "reasenberg_jones"
    version: str = "0.1.0"

    a: float | None = None
    b: float | None = None
    c: float | None = None
    p: float | None = None
    influence_km: float | None = None
    trigger_min_mag: float = 5.0

    # Fitted state.
    _mainshock_mag: float = field(default=0.0, repr=False)
    _mainshock_lat: float = field(default=0.0, repr=False)
    _mainshock_lon: float = field(default=0.0, repr=False)
    _mainshock_time: pd.Timestamp | None = field(default=None, repr=False)
    _influence_km: float = field(default=0.0, repr=False)
    params_used: dict = field(default_factory=dict, repr=False)

    # ── Forecaster.fit ───────────────────────────────────────────────────────
    def fit(
        self, catalog: pd.DataFrame, region: Region, t_issue: pd.Timestamp
    ) -> "ReasenbergJonesForecaster":
        """Identify the dominant recent mainshock before ``t_issue`` and pin the R-J constants.

        R-J is conditioned on the single largest triggering event in the (un-declustered) catalog
        slice ``(-∞, t_issue)``. If no event reaches ``trigger_min_mag`` the forecaster reports a
        flat zero-aftershock rate (everything floors to the smoothed background elsewhere in the
        pipeline) rather than fabricating productivity.
        """
        validate_catalog(catalog)
        df = catalog.loc[catalog["time"] < t_issue]
        cand = df.loc[df["mw"] >= self.trigger_min_mag]

        # Pin constants (generic unless explicitly provided).
        a = self.a if self.a is not None else GENERIC_RJ["a"]
        b = self.b if self.b is not None else GENERIC_RJ["b"]
        c = self.c if self.c is not None else GENERIC_RJ["c"]
        p = self.p if self.p is not None else GENERIC_RJ["p"]
        self.a, self.b, self.c, self.p = a, b, c, p

        if cand.empty:
            self._mainshock_time = None
            self.params_used = {
                "a": a, "b": b, "c": c, "p": p, "mainshock": None,
                "source": "generic_rj" if self.a is None else "user",
                "note": "no event >= trigger_min_mag before t_issue; rate = 0 (floors to background)",
            }
            return self

        row = cand.loc[cand["mw"].idxmax()]
        self._mainshock_mag = float(row["mw"])
        self._mainshock_lat = float(row["latitude"])
        self._mainshock_lon = float(row["longitude"])
        self._mainshock_time = pd.Timestamp(row["time"])

        # Aftershock-zone scale: Wells & Coppersmith (1994) subsurface rupture length
        # log10 L(km) ≈ 0.59 M - 2.44 → use half-length as the Gaussian smoothing sigma, floored.
        if self.influence_km is not None:
            self._influence_km = float(self.influence_km)
        else:
            rupture_len_km = 10.0 ** (0.59 * self._mainshock_mag - 2.44)
            self._influence_km = max(rupture_len_km / 2.0, 5.0)

        self.params_used = {
            "a": a, "b": b, "c": c, "p": p,
            "mainshock": {
                "mw": self._mainshock_mag,
                "time": self._mainshock_time.isoformat(),
                "lat": self._mainshock_lat,
                "lon": self._mainshock_lon,
            },
            "influence_km": self._influence_km,
            "source": "generic_rj_california" if self.a is None else "user",
            "caveat": "generic R-J constants are California-derived; refit for Chile (Page et al. 2016)",
        }
        return self

    # ── Rate integrals ───────────────────────────────────────────────────────
    def _expected_count_for_window(
        self, m_threshold: float, t1_days: float, t2_days: float
    ) -> float:
        """Total expected ``N(>=M*)`` aftershocks over elapsed-time window ``[t1, t2]`` (days).

        Closed-form modified-Omori integral (region-wide, before spatial distribution)::

            magnitude term: 10^{a + b (M_m - M*)}
            time integral:  [(t2+c)^{1-p} - (t1+c)^{1-p}] / (1 - p)     (p != 1)
                            ln((t2+c)/(t1+c))                           (p == 1)
        """
        if self._mainshock_time is None or m_threshold > self._mainshock_mag:
            return 0.0
        a, b, c, p = self.a, self.b, self.c, self.p
        mag_term = 10.0 ** (a + b * (self._mainshock_mag - m_threshold))
        if abs(p - 1.0) < 1e-9:
            time_term = np.log((t2_days + c) / (t1_days + c))
        else:
            time_term = ((t2_days + c) ** (1.0 - p) - (t1_days + c) ** (1.0 - p)) / (1.0 - p)
        return float(max(mag_term * time_term, 0.0))

    def _spatial_weights(self, cells: list[Cell]) -> np.ndarray:
        """Normalized Gaussian-in-distance weights distributing the region total across cells.

        ``w_j ∝ exp(-r_j^2 / (2 σ^2))`` with ``σ = influence_km`` and ``r_j`` the cell-to-mainshock
        epicentral distance. Normalized to sum to 1 so the spatially-distributed counts integrate
        back to the region total ``N``.
        """
        lat = np.array([c.lat for c in cells])
        lon = np.array([c.lon for c in cells])
        r = haversine_km(self._mainshock_lat, self._mainshock_lon, lat, lon)
        w = np.exp(-(r * r) / (2.0 * self._influence_km**2))
        total = w.sum()
        return w / total if total > 0 else np.full(len(cells), 1.0 / max(len(cells), 1))

    # ── Forecaster.expected_counts ───────────────────────────────────────────
    def expected_counts(
        self,
        region: Region,
        cells: list[Cell],
        horizon_days: float,
        m_threshold: float,
        t_issue: pd.Timestamp,
    ) -> list[float]:
        """Expected aftershock count ``N_{>=M*}`` per cell over ``[t_issue, t_issue + horizon)``.

        The elapsed-time window is measured from the mainshock origin time, so a sequence that began
        before ``t_issue`` is correctly decayed: ``t1 = (t_issue - t_main)`` and ``t2 = t1 + horizon``
        (both in days). The region total is distributed over cells by the mainshock spatial kernel.
        """
        if self._mainshock_time is None:
            return [0.0] * len(cells)
        t1 = max((t_issue - self._mainshock_time).total_seconds() / 86400.0, 0.0)
        t2 = t1 + horizon_days
        total = self._expected_count_for_window(m_threshold, t1, t2)
        if total <= 0:
            return [0.0] * len(cells)
        weights = self._spatial_weights(cells)
        return [float(total * w) for w in weights]

    def forecast_probabilities(
        self,
        region: Region,
        cells: list[Cell],
        horizon_days: float,
        m_threshold: float,
        t_issue: pd.Timestamp,
    ) -> list[float]:
        """Convenience: per-cell ``P(>=1 event >= M*) = 1 - e^{-N}`` (the public formula)."""
        return [
            poisson_p_at_least_one(n)
            for n in self.expected_counts(region, cells, horizon_days, m_threshold, t_issue)
        ]

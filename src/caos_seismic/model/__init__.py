"""Conditional seismicity forecasters for CAOS_SEISMIC.

Every model here implements the :class:`~caos_seismic.contracts.Forecaster` port (fit → conditional
intensity → expected counts → exceedance probability), so the inference driver and the CSEP harness
can treat them interchangeably:

* :class:`ETASForecaster` — space-time ETAS (Ogata 1998), the primary estimator *and* the reference
  any candidate forecaster must beat in prospective CSEP testing.
* :class:`ReasenbergJonesForecaster` — the transparent modified-Omori aftershock fallback / sanity
  check (USGS OAF shape).
* :class:`SmoothedSeismicityForecaster` — the adaptive smoothed-seismicity Poisson background and the
  *mandatory null* (Helmstetter-Kagan-Jackson 2007); also supplies ``mu(x, y)`` to ETAS.

The short aliases ``ETAS``, ``ReasenbergJones`` and ``SmoothedSeismicity`` are provided for ergonomic
construction at call sites and in configs.
"""

from __future__ import annotations

from .etas import ETASForecaster, ETASStabilityError
from .reasenberg_jones import ReasenbergJonesForecaster
from .smoothed import SmoothedSeismicityForecaster

# Ergonomic aliases (the canonical names are the *Forecaster classes above).
ETAS = ETASForecaster
ReasenbergJones = ReasenbergJonesForecaster
SmoothedSeismicity = SmoothedSeismicityForecaster

__all__ = [
    "ETASForecaster",
    "ETASStabilityError",
    "ReasenbergJonesForecaster",
    "SmoothedSeismicityForecaster",
    "ETAS",
    "ReasenbergJones",
    "SmoothedSeismicity",
]

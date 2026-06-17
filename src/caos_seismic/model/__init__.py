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
* :class:`TiledForecaster` — the GLOBAL adapter: fits ETAS (+ its smoothed background) **per tectonic
  regime / spatial tile** and aggregates the per-tile fields into one global conditional field, so the
  conditional models stay tractable (no global ``O(N^2)``) and physically meaningful (subduction
  parameters never bleed into the stable interior). It still implements the same Forecaster port. The
  regimes + tiling live in :mod:`caos_seismic.model.regime`
  (:func:`~caos_seismic.model.regime.assign_regime`, :func:`~caos_seismic.model.regime.iterate_tiles`).
* :class:`ContextTPPForecaster` — the **gated neural challenger**: a context-conditioned spatio-temporal
  neural temporal point process (Hawkes inductive bias + CNN context encoder). It is **never the
  default** and reaches the public field only if it beats ETAS in prospective CSEP *and* calibrates.
  Its symbols are re-exported here, but importing this subpackage stays torch-free — ``context_tpp``
  imports torch lazily inside its methods, so ``from caos_seismic.model import ContextTPPForecaster``
  works with only the core deps.

The short aliases ``ETAS``, ``ReasenbergJones``, ``SmoothedSeismicity``, ``TiledETAS`` and
``ContextTPP`` are provided for ergonomic construction at call sites and in configs.
"""

from __future__ import annotations

from .context_tpp import (
    ContextTPPConfig,
    ContextTPPForecaster,
    CovariateField,
    CovariateFieldProvider,
)
from .etas import ETASForecaster, ETASStabilityError
from .reasenberg_jones import ReasenbergJonesForecaster
from .regime import (
    REGIME_PRIORS,
    RegimeAssignment,
    RegimePrior,
    TectonicRegime,
    Tile,
    assign_regime,
    assign_regimes,
    dominant_regime,
    iterate_tiles,
    regime_prior,
    tiles_for_region,
)
from .smoothed import SmoothedSeismicityForecaster
from .tiled import TiledForecaster, TileFit

# Ergonomic aliases (the canonical names are the *Forecaster classes above).
ETAS = ETASForecaster
ReasenbergJones = ReasenbergJonesForecaster
SmoothedSeismicity = SmoothedSeismicityForecaster
TiledETAS = TiledForecaster
ContextTPP = ContextTPPForecaster

__all__ = [
    "ETASForecaster",
    "ETASStabilityError",
    "ReasenbergJonesForecaster",
    "SmoothedSeismicityForecaster",
    "TiledForecaster",
    "TileFit",
    "ContextTPPForecaster",
    "ContextTPPConfig",
    "CovariateField",
    "CovariateFieldProvider",
    "ETAS",
    "ReasenbergJones",
    "SmoothedSeismicity",
    "TiledETAS",
    "ContextTPP",
    # Regime / tiling primitives (the global-conditioning layer)
    "TectonicRegime",
    "RegimePrior",
    "RegimeAssignment",
    "REGIME_PRIORS",
    "regime_prior",
    "assign_regime",
    "assign_regimes",
    "dominant_regime",
    "Tile",
    "iterate_tiles",
    "tiles_for_region",
]

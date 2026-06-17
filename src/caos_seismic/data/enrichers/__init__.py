"""Global geophysical enrichers — the worldwide static-context covariate loaders.

Core thesis of CAOS_SEISMIC: **global context conditions short-term local forecasts.** The model
trains on worldwide seismicity plus complementary *global* covariate fields, and any country is a
spatial *view* into that one global field. Each enricher here loads a worldwide dataset and answers
``features_at(lat, lon)`` for any cell on Earth, so the same enrichers serve every region.

Each enricher module exposes the same two-function contract (data-and-pipelines.md §1.3/§1.4):

* ``download(...) -> Provenance`` — fetch the raw global dataset into the gitignored
  ``data/enrichers/<dataset>/`` cache and return a license/citation/provenance record for the public
  credits page. (``tides`` is *computed*, so its ``download`` returns only the tool provenance.)
* ``features_at(lat, lon, ...) -> dict[str, float | None]`` — the per-coordinate covariate(s) for one
  forecast cell (``None`` where the cell falls outside the dataset's footprint, e.g. a slab grid only
  covers subduction margins — which is information, not an error).

The enrichers, ranked by expected lift for a conditional short-term forecast (Slab2 > faults +
plates > GNSS strain > stress > tides; the ranking is a hypothesis, each must clear a prospective
CSEP information-gain gate before it ships in a public number):

============  ==========================================================  ===============
module        covariates                                                  source / license
============  ==========================================================  ===============
``slab2``     slab depth / dip / strike / interface distance              Slab2 (USGS PD)
``faults``    distance-to-fault + slip style                              GEM faults (CC-BY-SA 4.0)
``plates``    distance-to-boundary + type + relative velocity             Bird PB2002 (open)
``gnss``      geodetic strain-rate proxy                                  NGL MIDAS (open + attr)
``stress``    SHmax azimuth + tectonic regime                             WSM (CC-BY 4.0)
``tides``     tidal Coulomb stress / stressing rate / Mf envelope         computed (pygtide + ocean)
============  ==========================================================  ===============

Heavy geospatial dependencies (``geopandas``, ``shapely``, ``netCDF4``, ``xarray``, ``pygtide``) are
imported **lazily inside the functions that need them**, never at module top level, so
``import caos_seismic`` works on the core deps alone (an explicit, hard requirement of the package).

:func:`caos_seismic.catalog.features.build_context_features` joins all of these onto the forecast
grid to produce the global context feature matrix the model ingests; :data:`ENRICHERS` is the
registry it iterates over.
"""

from __future__ import annotations

from types import ModuleType

from . import faults, gnss, plates, slab2, stress, tides
from ._base import (
    ENRICHER_CACHE,
    EnricherResult,
    Provenance,
    cache_dir,
    empty_like,
    features_for_cells,
    http_download,
    require,
)

#: Registry of the enricher modules keyed by dataset id, in expected-lift order. Each value is a
#: module exposing ``download`` / ``features_at`` / ``FEATURE_NAMES``. ``build_context_features``
#: iterates this so adding an enricher is a one-line change here.
ENRICHERS: dict[str, ModuleType] = {
    "slab2": slab2,
    "faults": faults,
    "plates": plates,
    "gnss": gnss,
    "stress": stress,
    "tides": tides,
}

#: All covariate column names produced across every enricher, in registry order (stable for the
#: feature matrix schema and the model's expected input columns).
ALL_FEATURE_NAMES: list[str] = [
    name for mod in ENRICHERS.values() for name in getattr(mod, "FEATURE_NAMES", ())
]


def feature_names_for(*datasets: str) -> list[str]:
    """Return the covariate column names produced by the named enrichers (all of them if none given)."""
    mods = ENRICHERS if not datasets else {k: ENRICHERS[k] for k in datasets}
    return [name for mod in mods.values() for name in getattr(mod, "FEATURE_NAMES", ())]


__all__ = [
    "ENRICHERS",
    "ALL_FEATURE_NAMES",
    "feature_names_for",
    "ENRICHER_CACHE",
    "EnricherResult",
    "Provenance",
    "cache_dir",
    "empty_like",
    "features_for_cells",
    "http_download",
    "require",
    "slab2",
    "faults",
    "plates",
    "gnss",
    "stress",
    "tides",
]

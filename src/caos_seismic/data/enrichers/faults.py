"""GEM Global Active Faults enricher — distance-to-nearest-fault and fault style per cell.

The GEM Global Active Faults Database (Styron & Pagani, 2020) is a worldwide, harmonized vector
catalog of active fault traces with a ``slip_type`` attribute (normal / reverse / dextral /
sinistral / strike-slip, etc.). For a conditional forecaster it supplies two static covariates that
condition the *background* (long-term) term and the spatial smoothing kernel (data-and-pipelines.md
§1.3): proximity to a known active structure, and the kinematic style of the nearest structure.

Per-cell features
-----------------
``fault_dist_km``        great-circle distance to the nearest mapped active fault trace (km).
``fault_style_code``     integer code of the nearest fault's slip type (see :data:`SLIP_TYPE_CODES`;
                         ``0`` = unknown / unclassified).
``fault_is_reverse``     1.0 if the nearest fault is reverse/thrust (megathrust-style), else 0.0.
``fault_is_normal``      1.0 if the nearest fault is normal, else 0.0.
``fault_is_strikeslip``  1.0 if the nearest fault is strike-slip (any sense), else 0.0.

Data & license
--------------
* Source: ``github.com/GEMScienceTools/gem-global-active-faults`` (GeoJSON/GPKG/SHP). The harmonized
  release file is ``gem_active_faults_harmonized.geojson``.
* License: **CC-BY-SA 4.0** (verify the repo ``LICENSE`` per release — share-alike: a redistributed
  derivative must keep the license + attribution). Internal feature-building is unaffected; the
  public credits page must attribute GEM.

Heavy deps (``geopandas`` + ``shapely``) are imported lazily; the module imports on core deps alone.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ...model._common import DEG2KM, haversine_km
from ._base import EnricherResult, Provenance, cache_dir, http_download, require

logger = logging.getLogger(__name__)

DATASET = "faults"

GEM_REPO = "https://github.com/GEMScienceTools/gem-global-active-faults"
#: Raw URL of the harmonized GeoJSON release on the GEM repo's default branch.
GEM_HARMONIZED_GEOJSON = (
    "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/"
    "master/geojson/gem_active_faults_harmonized.geojson"
)

#: Slip-type string → small integer code (stable; do not renumber — it is a model feature).
SLIP_TYPE_CODES: dict[str, int] = {
    "unknown": 0,
    "reverse": 1,
    "thrust": 1,
    "normal": 2,
    "dextral": 3,
    "sinistral": 4,
    "strike-slip": 5,
    "strike_slip": 5,
    "dextral-normal": 6,
    "normal-dextral": 6,
    "sinistral-normal": 7,
    "normal-sinistral": 7,
    "dextral-reverse": 8,
    "reverse-dextral": 8,
    "sinistral-reverse": 9,
    "reverse-sinistral": 9,
    "subduction": 10,
    "spreading_ridge": 11,
    "transform": 12,
}

_REVERSE_CODES = {1, 8, 9, 10}
_NORMAL_CODES = {2, 6, 7}
_STRIKESLIP_CODES = {3, 4, 5, 12}

FEATURE_NAMES = (
    "fault_dist_km",
    "fault_style_code",
    "fault_is_reverse",
    "fault_is_normal",
    "fault_is_strikeslip",
)


def style_code(slip_type: Any) -> int:
    """Map a GEM ``slip_type`` string to its stable integer code (``0`` for unknown/empty)."""
    if not isinstance(slip_type, str) or not slip_type.strip():
        return 0
    return SLIP_TYPE_CODES.get(slip_type.strip().lower(), 0)


def download(
    *,
    url: str | None = None,
    dest: Path | None = None,
    overwrite: bool = False,
    session: Any | None = None,
) -> Provenance:
    """Download the GEM harmonized active-faults GeoJSON to the gitignored cache.

    Defaults to the harmonized GeoJSON on the GEM repo's default branch
    (:data:`GEM_HARMONIZED_GEOJSON`); pass ``url`` to pin a specific release/commit. The whole GEM
    repo can also be ``git clone``d, but a single harmonized GeoJSON is enough for distance queries
    and avoids vendoring the full repo.
    """
    dest = dest or cache_dir(DATASET)
    url = url or GEM_HARMONIZED_GEOJSON
    out = dest / "gem_active_faults_harmonized.geojson"
    http_download(url, out, overwrite=overwrite, session=session)
    return Provenance(
        dataset=DATASET,
        title="GEM Global Active Faults Database (harmonized)",
        version="harmonized release",
        source_url=GEM_REPO,
        license="CC-BY-SA 4.0 (verify repo LICENSE per release — share-alike)",
        attribution="GEM Global Active Faults Database (Styron & Pagani, 2020)",
        citation=(
            "Styron, R., & Pagani, M. (2020). The GEM Global Active Faults Database. "
            "Earthquake Spectra, 36(1_suppl), 160-180. doi:10.1177/8755293020944182"
        ),
        files=[str(out.relative_to(dest.parent.parent.parent))],
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        notes="Vector fault traces with slip_type; CC-BY-SA share-alike must be preserved.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell feature extraction
# ─────────────────────────────────────────────────────────────────────────────


class FaultsEnricher:
    """Lazy-loading nearest-fault feature extractor over the cached GEM faults vector layer.

    Loads the GeoJSON once (via ``geopandas``), densifies each trace into vertices, and answers
    ``features_at`` with the great-circle distance to the nearest vertex and that fault's slip style.
    Vertex distance is an excellent proxy for distance-to-trace at GEM's mapping resolution and keeps
    the query dependency-light (haversine over a vertex array, no per-query shapely projection).
    """

    def __init__(self, dest: Path | None = None) -> None:
        self.dest = dest or cache_dir(DATASET)
        self._lat: np.ndarray | None = None  # vertex latitudes
        self._lon: np.ndarray | None = None  # vertex longitudes
        self._code: np.ndarray | None = None  # per-vertex slip-style code

    def _path(self) -> Path:
        p = self.dest / "gem_active_faults_harmonized.geojson"
        if not p.exists():
            # Accept any single GeoJSON in the cache dir (release file names vary).
            candidates = sorted(self.dest.glob("*.geojson")) + sorted(self.dest.glob("*.gpkg"))
            if candidates:
                return candidates[0]
            raise FileNotFoundError(
                f"no GEM faults layer cached under {self.dest}. Run "
                "caos_seismic.data.enrichers.faults.download() first."
            )
        return p

    def _load(self) -> None:
        if self._lat is not None:
            return
        gpd = require("geopandas")()
        gdf = gpd.read_file(self._path())
        slip_col = next((c for c in ("slip_type", "slip_typ", "fault_type") if c in gdf.columns), None)

        lats: list[float] = []
        lons: list[float] = []
        codes: list[int] = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            code = style_code(row[slip_col]) if slip_col else 0
            for lon, lat in _iter_line_coords(geom):
                lons.append(lon)
                lats.append(lat)
                codes.append(code)

        self._lat = np.asarray(lats, dtype=float)
        self._lon = np.asarray(lons, dtype=float)
        self._code = np.asarray(codes, dtype=int)
        if self._lat.size == 0:
            raise ValueError("GEM faults layer densified to zero vertices — check the source file.")

    def features_at(self, lat: float, lon: float, **_: Any) -> EnricherResult:
        """Return nearest-fault distance + style covariates at ``(lat, lon)``."""
        self._load()
        assert self._lat is not None and self._lon is not None and self._code is not None
        d = haversine_km(lat, lon, self._lat, self._lon)
        j = int(np.argmin(d))
        code = int(self._code[j])
        return {
            "fault_dist_km": float(d[j]),
            "fault_style_code": float(code),
            "fault_is_reverse": 1.0 if code in _REVERSE_CODES else 0.0,
            "fault_is_normal": 1.0 if code in _NORMAL_CODES else 0.0,
            "fault_is_strikeslip": 1.0 if code in _STRIKESLIP_CODES else 0.0,
        }


_DEFAULT_ENRICHER: FaultsEnricher | None = None


def features_at(lat: float, lon: float, **kwargs: Any) -> EnricherResult:
    """Module-level convenience: nearest-fault features at one coordinate (cached enricher)."""
    global _DEFAULT_ENRICHER
    if _DEFAULT_ENRICHER is None:
        _DEFAULT_ENRICHER = FaultsEnricher()
    return _DEFAULT_ENRICHER.features_at(lat, lon, **kwargs)


def _iter_line_coords(geom: Any):
    """Yield ``(lon, lat)`` vertices from a shapely (Multi)LineString without importing shapely.

    Uses the geo-interface so we never import shapely at module top level. Handles LineString,
    MultiLineString, and (defensively) GeometryCollection-like nestings.
    """
    gj = geom.__geo_interface__
    yield from _iter_coords(gj.get("type"), gj.get("coordinates", []))


def _iter_coords(gtype: str | None, coords: Any):
    if gtype == "LineString":
        for c in coords:
            yield float(c[0]), float(c[1])
    elif gtype == "MultiLineString":
        for line in coords:
            for c in line:
                yield float(c[0]), float(c[1])
    elif gtype in ("Polygon",):  # pragma: no cover - GEM faults are lines, but be defensive
        for ring in coords:
            for c in ring:
                yield float(c[0]), float(c[1])
    elif gtype in ("MultiPolygon",):  # pragma: no cover
        for poly in coords:
            for ring in poly:
                for c in ring:
                    yield float(c[0]), float(c[1])


__all__ = [
    "DATASET",
    "SLIP_TYPE_CODES",
    "FEATURE_NAMES",
    "FaultsEnricher",
    "download",
    "features_at",
    "style_code",
    "DEG2KM",
]

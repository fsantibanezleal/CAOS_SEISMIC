"""Slab2 enricher — depth-to-slab, dip, strike, and interface distance at subduction margins.

Slab2 (Hayes et al., 2018, *Science*) is the USGS global subduction-zone geometry model: a set of
per-region 0.05° NetCDF/GMT grids giving the depth, dip, and strike of the subducting slab interface.
For a conditional short-term forecaster it is the **highest-lift static enricher in subduction
regimes** (data-and-pipelines.md §1.3): great-earthquake triggering is anisotropic and slab-geometry
controlled, so a point-source isotropic ETAS kernel under-models it — the slab geometry lets the
model condition on *where in the megathrust* a cell sits.

Per-cell features
-----------------
``slab_depth_km``       depth of the slab interface beneath the cell (km, positive down); ``None``
                        outside any slab footprint (i.e. not a subduction margin).
``slab_dip_deg``        local slab dip (degrees from horizontal).
``slab_strike_deg``     local slab strike (degrees clockwise from north).
``slab_interface_dist_km``
                        3-D distance from a *surface* cell to the nearest slab interface point
                        (great-circle horizontal combined with the vertical depth) — a proxy for how
                        close a shallow cell is to the locked megathrust.

Data & license
--------------
* Source: ScienceBase item ``5aa1b00ee4b0b1c392e86467`` ("Slab2 — A Comprehensive Subduction Zone
  Geometry Model"); mirror at ``github.com/usgs/slab2``. 0.05° grids, NetCDF (``.grd``).
* License: USGS work — **public domain** (cite Hayes et al., 2018). No redistribution restriction.

Heavy deps (``xarray`` + a NetCDF backend, ``scipy`` for the local gradient) are imported lazily;
the module imports on the core deps alone.
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

DATASET = "slab2"

#: ScienceBase item id for the Slab2 release (the canonical, citable source).
SCIENCEBASE_ITEM = "5aa1b00ee4b0b1c392e86467"
SCIENCEBASE_ITEM_URL = f"https://www.sciencebase.gov/catalog/item/{SCIENCEBASE_ITEM}"

#: The Slab2 three-letter region codes (the model ships one grid set per subduction zone).
SLAB2_REGIONS: tuple[str, ...] = (
    "alu", "cal", "cam", "car", "cas", "cot", "hal", "hel", "him", "hin", "izu",
    "ker", "kur", "mak", "man", "mue", "pam", "phi", "png", "puy", "ryu", "sam",
    "sco", "sol", "sul", "sum", "van",
)

#: The grid variables Slab2 publishes per region (suffix in the file name and our feature key).
SLAB2_VARIABLES: dict[str, str] = {
    "dep": "slab_depth_km",     # depth to the slab top (km; published negative-down)
    "dip": "slab_dip_deg",      # dip in degrees
    "str": "slab_strike_deg",   # strike in degrees
}

FEATURE_NAMES = (
    "slab_depth_km",
    "slab_dip_deg",
    "slab_strike_deg",
    "slab_interface_dist_km",
)


def grid_filename(region: str, variable: str) -> str:
    """Conventional Slab2 grid file name, e.g. ``sam_slab2_dep.grd`` (South America depth grid)."""
    return f"{region}_slab2_{variable}.grd"


def download(
    *,
    regions: tuple[str, ...] | list[str] | None = None,
    base_url: str | None = None,
    dest: Path | None = None,
    overwrite: bool = False,
    session: Any | None = None,
) -> Provenance:
    """Download Slab2 NetCDF (``.grd``) grids for the requested subduction regions.

    The ScienceBase item issues per-file download URLs that change between releases, so the exact
    file URL prefix must be passed via ``base_url`` (or the per-region/variable URLs resolved out of
    band). With no ``base_url`` this raises an actionable error pointing at the ScienceBase item and
    the ``github.com/usgs/slab2`` mirror, rather than guessing a brittle hard-coded URL.

    Parameters
    ----------
    regions:
        Slab2 region codes to fetch (default: all of :data:`SLAB2_REGIONS`). For a focused product
        you fetch only the relevant margin (e.g. ``["sam"]`` for Chile/Peru).
    base_url:
        URL prefix such that ``f"{base_url}/{grid_filename(region, var)}"`` is the downloadable grid.
    dest:
        Cache directory (default ``data/enrichers/slab2/``; gitignored).

    Returns
    -------
    Provenance
        License/citation record listing the cached files (for the public credits page).
    """
    regions = tuple(regions) if regions is not None else SLAB2_REGIONS
    dest = dest or cache_dir(DATASET)
    if base_url is None:
        raise ValueError(
            "download(slab2) requires an explicit `base_url` to the current Slab2 grid files. "
            f"Resolve the per-file URLs from the ScienceBase item {SCIENCEBASE_ITEM_URL} "
            "(or the github.com/usgs/slab2 mirror) — they are versioned per release, so no URL is "
            "hard-coded. Then call e.g. download(regions=['sam'], base_url=...)."
        )

    base = base_url.rstrip("/")
    files: list[str] = []
    for region in regions:
        for var in SLAB2_VARIABLES:
            name = grid_filename(region, var)
            try:
                path = http_download(f"{base}/{name}", dest / name, overwrite=overwrite, session=session)
                files.append(str(path.relative_to(dest.parent.parent.parent)))
            except RuntimeError as exc:
                # A region may not publish every variable; log and continue rather than abort.
                logger.warning("slab2: skipping %s (%s)", name, exc)

    return Provenance(
        dataset=DATASET,
        title="Slab2 — A Comprehensive Subduction Zone Geometry Model",
        version="2018 (Hayes et al.)",
        source_url=SCIENCEBASE_ITEM_URL,
        license="USGS — public domain (cite Hayes et al., 2018)",
        attribution="Slab2 / U.S. Geological Survey (Hayes et al., 2018, Science 362:58-61)",
        citation=(
            "Hayes, G.P., et al. (2018). Slab2, a comprehensive subduction zone geometry model. "
            "Science, 362(6410), 58-61. doi:10.1126/science.aat4723"
        ),
        files=files,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        notes="0.05° per-region NetCDF (.grd) grids; depth published negative-down (km).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell feature extraction
# ─────────────────────────────────────────────────────────────────────────────


class Slab2Enricher:
    """Lazy-loading per-coordinate Slab2 feature extractor over the cached region grids.

    Loads each cached ``.grd`` once (via ``xarray``) and caches the in-memory grids. ``features_at``
    finds the region whose footprint covers the query point and bilinearly samples depth/dip/strike;
    cells outside every slab footprint return all-``None`` (correct — most of Earth is not a
    subduction interface).
    """

    def __init__(self, dest: Path | None = None) -> None:
        self.dest = dest or cache_dir(DATASET)
        self._grids: dict[str, dict[str, Any]] | None = None  # region -> {variable_key: DataArray}

    # -- grid loading -------------------------------------------------------

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._grids is not None:
            return self._grids
        xr = require("xarray")()
        grids: dict[str, dict[str, Any]] = {}
        for region in SLAB2_REGIONS:
            region_grids: dict[str, Any] = {}
            for var, key in SLAB2_VARIABLES.items():
                path = self.dest / grid_filename(region, var)
                if not path.exists():
                    continue
                da = xr.open_dataarray(path)
                region_grids[key] = _normalize_lon(da)
            if "slab_depth_km" in region_grids:  # a region is usable only if it has a depth grid
                grids[region] = region_grids
        if not grids:
            raise FileNotFoundError(
                f"no Slab2 grids cached under {self.dest}. Run "
                "caos_seismic.data.enrichers.slab2.download(base_url=...) first."
            )
        self._grids = grids
        return grids

    # -- feature extraction -------------------------------------------------

    def features_at(self, lat: float, lon: float, **_: Any) -> EnricherResult:
        """Return Slab2 covariates at ``(lat, lon)`` (all-``None`` if off every slab footprint)."""
        grids = self._load()
        lon = _wrap_lon(lon)
        out: EnricherResult = {name: None for name in FEATURE_NAMES}

        region = self._region_for(grids, lat, lon)
        if region is None:
            return out

        rg = grids[region]
        for key, da in rg.items():
            out[key] = _bilinear_sample(da, lat, lon)

        depth = out.get("slab_depth_km")
        if depth is not None:
            out["slab_depth_km"] = abs(float(depth))  # publish positive-down regardless of source sign
            out["slab_interface_dist_km"] = abs(float(depth))  # surface cell → vertical to interface
        return out

    def _region_for(self, grids: dict[str, dict[str, Any]], lat: float, lon: float) -> str | None:
        """Pick the slab region whose depth grid covers ``(lat, lon)`` and has a finite value there."""
        for region, rg in grids.items():
            da = rg["slab_depth_km"]
            if _within_bounds(da, lat, lon):
                val = _bilinear_sample(da, lat, lon)
                if val is not None and np.isfinite(val):
                    return region
        return None


# Module-level singleton so repeated catalog joins do not reload the grids.
_DEFAULT_ENRICHER: Slab2Enricher | None = None


def features_at(lat: float, lon: float, **kwargs: Any) -> EnricherResult:
    """Module-level convenience: Slab2 features at one coordinate (uses a cached enricher)."""
    global _DEFAULT_ENRICHER
    if _DEFAULT_ENRICHER is None:
        _DEFAULT_ENRICHER = Slab2Enricher()
    return _DEFAULT_ENRICHER.features_at(lat, lon, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers (xarray DataArray sampling) — no xarray import at module top level
# ─────────────────────────────────────────────────────────────────────────────


def _coord_names(da: Any) -> tuple[str, str]:
    """Return the (lon, lat) coordinate names of a Slab2 DataArray (``x``/``y`` or ``lon``/``lat``)."""
    names = {n.lower(): n for n in da.coords}
    lon = names.get("x") or names.get("lon") or names.get("longitude")
    lat = names.get("y") or names.get("lat") or names.get("latitude")
    if lon is None or lat is None:  # pragma: no cover - defensive for unusual grids
        dims = list(da.dims)
        lon, lat = dims[-1], dims[-2]
    return lon, lat


def _normalize_lon(da: Any):
    """Sort the grid by ascending lon/lat so bilinear interpolation indexing is well-defined."""
    lon_name, lat_name = _coord_names(da)
    return da.sortby([lat_name, lon_name])


def _wrap_lon(lon: float) -> float:
    """Wrap a longitude into [-180, 180); Slab2 region grids use the convention of their margin."""
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lon


def _within_bounds(da: Any, lat: float, lon: float) -> bool:
    lon_name, lat_name = _coord_names(da)
    lons = da[lon_name].values
    lats = da[lat_name].values
    lon_q = lon
    # Slab2 grids spanning the dateline may use 0..360; try both conventions.
    if lon_q < float(lons.min()):
        lon_q += 360.0
    return (
        float(lats.min()) <= lat <= float(lats.max())
        and float(lons.min()) <= lon_q <= float(lons.max())
    )


def _bilinear_sample(da: Any, lat: float, lon: float) -> float | None:
    """Bilinearly sample a DataArray at ``(lat, lon)``; ``None`` if outside or NaN there."""
    lon_name, lat_name = _coord_names(da)
    lons = da[lon_name].values
    lon_q = lon
    if lon_q < float(lons.min()):
        lon_q += 360.0
    try:
        val = da.interp({lat_name: lat, lon_name: lon_q}, method="linear").values
    except Exception:  # pragma: no cover - defensive against odd coordinate layouts
        return None
    v = float(val)
    return v if np.isfinite(v) else None


def interface_distance_km(surface_lat: float, surface_lon: float, enricher: Slab2Enricher | None = None) -> float | None:
    """3-D distance from a surface point to the slab interface directly beneath it (km).

    For a surface cell the nearest interface point is approximately the slab depth directly below
    (the horizontal offset to the true nearest interface point is second-order at 0.05° resolution).
    Returned for completeness / explicit callers; ``features_at`` already populates
    ``slab_interface_dist_km``.
    """
    enr = enricher or Slab2Enricher()
    feats = enr.features_at(surface_lat, surface_lon)
    depth = feats.get("slab_depth_km")
    return None if depth is None else float(depth)


# Re-export the haversine constants used by callers that combine horizontal + vertical distances.
__all__ = [
    "DATASET",
    "SLAB2_REGIONS",
    "SLAB2_VARIABLES",
    "FEATURE_NAMES",
    "Slab2Enricher",
    "download",
    "features_at",
    "grid_filename",
    "interface_distance_km",
    "DEG2KM",
    "haversine_km",
]

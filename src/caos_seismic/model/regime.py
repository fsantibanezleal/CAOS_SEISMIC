"""Tectonic regimes + spatial tiling — the GLOBAL conditioning that makes ETAS tractable per region.

The core thesis of the global re-scope is that **global context conditions short-term local
forecasts**: the model trains on worldwide seismicity, and any country is a *view* into one global
field. But a single global ETAS fit is both intractable (the triggering sum is ``O(N^2)`` in the
catalog size, and a global daily catalog is ``10^5``-``10^6`` events) and *physically wrong* — a
subduction megathrust does not share productivity / Omori / spatial-decay parameters with a
stable continental interior. The fix is to partition the globe twice:

1. **By tectonic regime** (this module's :class:`TectonicRegime`): every point is assigned to one of
   five mechanism classes — subduction *interface*, *intraslab*, *crustal / strike-slip*,
   *intraplate*, *ridge / transform* — using the static geophysical enrichers (Slab2 geometry, the
   Bird 2003 PB2002 plate-boundary model, GEM active faults). Regimes carry **different ETAS priors**
   (Page et al. 2016 global tectonic-regime aftershock statistics), so a thin-data tile *borrows
   strength* from the worldwide pool for its regime rather than inventing a noisy local fit
   (model-design §8 "borrow strength spatially"; methodology §1.4 global tectonic-regime extension).

2. **By spatial tile** (:class:`Tile` / :func:`iterate_tiles`): a regular lon/lat grid (with a small
   halo so triggering near tile edges is not truncated) over which ETAS + smoothed-seismicity are fit
   independently and then **aggregated into one global field**. Tiling caps the per-fit catalog size
   (so the ``O(N^2)`` triggering sum stays bounded) and lets the daily job parallelize trivially.

This module is **enricher-aware but enricher-optional**. The heavy geophysical layers (Slab2 NetCDF,
PB2002 ASCII, GEM faults GeoPackage — see ``data-and-pipelines.md`` §1.3) are loaded *lazily* and
only if present on disk; when they are absent the classifier degrades to a transparent,
self-contained heuristic on ``(lat, lon, depth)`` (depth bands + a coarse built-in subduction-margin
mask) so the package always imports and runs on the core deps alone. Every classification records its
``source`` ("slab2" / "pb2002" / "heuristic") for the provenance manifest, so a forecast never hides
*how* a regime was assigned.

References
----------
Hayes, G. P. et al. (2018), *Science* 362(6410), 58-61, doi:10.1126/science.aat4723 (Slab2).
Bird, P. (2003), *G3* 4(3), 1027, doi:10.1029/2001GC000252 (PB2002 plate-boundary model).
Page, M. T. et al. (2016), *BSSA* 106(5), 2290-2301, doi:10.1785/0120160073
    (three-tectonic-regime global aftershock parameters used by the USGS OAF).
Styron, R. & Pagani, M. (2020), *Seismol. Res. Lett.* 91(1), 130-141, doi:10.1785/0220190058
    (GEM Global Active Faults Database).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

import numpy as np

from ..contracts import BBox, Region

# ─────────────────────────────────────────────────────────────────────────────
# Tectonic regimes
# ─────────────────────────────────────────────────────────────────────────────


class TectonicRegime(str, Enum):
    """The five mechanism classes a point can be assigned to.

    The split follows the productivity/clustering distinctions that matter for short-horizon ETAS:
    subduction *interface* events are the most productive (great megathrust aftershock sequences),
    *intraslab* (deep, in the down-going slab) are less productive and more isolated, *crustal*
    strike-slip/normal/thrust faulting dominates continental plate boundaries, *intraplate* stable
    interiors are very low-rate (cold-start dominates), and *ridge/transform* oceanic-spreading
    seismicity is shallow and swarm-prone.
    """

    SUBDUCTION_INTERFACE = "subduction_interface"
    INTRASLAB = "intraslab"
    CRUSTAL = "crustal"          # crustal / strike-slip / continental active faulting
    INTRAPLATE = "intraplate"    # stable continental interior (very low rate)
    RIDGE = "ridge"             # oceanic ridge / transform (spreading-center, swarm-prone)


#: The depth (km) below which a non-interface event is treated as intraslab rather than crustal.
#: Slab2 interface events are mostly < ~70 km; deeper seismicity inside a slab is intraslab.
INTRASLAB_DEPTH_KM = 70.0

#: Distance to the slab interface (km) within which a shallow event is the megathrust *interface*
#: rather than overriding-plate crustal. ~30 km absorbs Slab2 vertical uncertainty + the seismogenic
#: width of the coupled zone.
INTERFACE_DISTANCE_KM = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-regime ETAS priors (Page et al. 2016 global tectonic-regime aftershock statistics)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RegimePrior:
    """Per-regime ETAS / aftershock prior — the strength a thin tile borrows from the global pool.

    The values are *priors* (regularization centres + the data-informed optimizer start), never
    hard-published parameters: the daily fit still MLE-estimates each tile and only shrinks toward
    these centres when the local data are sparse (empirical-Bayes pooling, model-design §8). They are
    anchored on the USGS three-regime global aftershock study (Page et al. 2016) and the Omori/Utsu
    productivity literature; the regime distinctions (interface > intraslab, ridge swarm-prone,
    intraplate very low rate) are qualitative-but-robust and recorded so a sparse tile is never left
    to a noisy local fit.

    Attributes
    ----------
    productivity_k:
        Prior centre for the Utsu productivity ``K`` (relative aftershock abundance). Interface
        sequences are the most productive; intraplate the least.
    alpha:
        Prior centre for the productivity magnitude-scaling ``alpha`` (must stay below
        ``beta = b ln 10`` for finite branching — gate 1 in :mod:`caos_seismic.model.etas`).
    p, c:
        Modified-Omori decay exponent / offset (days) prior centres.
    b_value:
        Typical Gutenberg-Richter ``b`` for the regime (ridges run high ~1.1-1.3, interiors ~0.9-1.0).
        Used only as a fallback prior; the per-tile ``b`` is always estimated when data allow.
    n_neighbors:
        Smoothed-seismicity adaptive-kernel neighbour count tuned for the regime's spatial density
        (dense interface zones sharpen, sparse interiors broaden).
    """

    productivity_k: float
    alpha: float
    p: float
    c: float
    b_value: float
    n_neighbors: int


#: Regime-keyed ETAS priors. Centres are order-of-magnitude, deliberately conservative, and exist to
#: regularize sparse tiles toward the worldwide behaviour of their regime — NOT to bypass the MLE.
REGIME_PRIORS: dict[TectonicRegime, RegimePrior] = {
    TectonicRegime.SUBDUCTION_INTERFACE: RegimePrior(
        productivity_k=0.15, alpha=1.0, p=1.10, c=0.02, b_value=1.0, n_neighbors=8
    ),
    TectonicRegime.INTRASLAB: RegimePrior(
        productivity_k=0.07, alpha=0.9, p=1.08, c=0.03, b_value=1.0, n_neighbors=6
    ),
    TectonicRegime.CRUSTAL: RegimePrior(
        productivity_k=0.10, alpha=0.9, p=1.08, c=0.05, b_value=1.0, n_neighbors=6
    ),
    TectonicRegime.INTRAPLATE: RegimePrior(
        productivity_k=0.05, alpha=0.8, p=1.05, c=0.05, b_value=0.95, n_neighbors=5
    ),
    TectonicRegime.RIDGE: RegimePrior(
        productivity_k=0.12, alpha=0.9, p=1.15, c=0.05, b_value=1.2, n_neighbors=6
    ),
}


def regime_prior(regime: TectonicRegime | str) -> RegimePrior:
    """Return the :class:`RegimePrior` for a regime (accepts the enum or its string value)."""
    key = TectonicRegime(regime) if not isinstance(regime, TectonicRegime) else regime
    return REGIME_PRIORS[key]


# ─────────────────────────────────────────────────────────────────────────────
# Regime assignment
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RegimeAssignment:
    """One point's regime classification + the evidence that produced it (for the manifest)."""

    regime: TectonicRegime
    source: str                       # "slab2" | "pb2002" | "heuristic"
    depth_to_interface_km: float | None = None
    distance_to_boundary_km: float | None = None
    boundary_type: str | None = None  # "subduction" | "ridge" | "transform" | "continental" | None


# Coarse built-in subduction-margin boxes (lon_min, lon_max, lat_min, lat_max) used ONLY when Slab2
# is not on disk. They are intentionally generous polygons over the world's circum-Pacific + key
# subduction margins so the heuristic does not silently mislabel a megathrust as intraplate. This is
# a fallback, not the authoritative geometry — Slab2 supersedes it whenever present.
_SUBDUCTION_MARGIN_BOXES: tuple[tuple[float, float, float, float], ...] = (
    (-80.0, -68.0, -56.0, 6.0),     # South America (Nazca/Antarctic → SAM): Chile, Peru, Colombia, Ecuador
    (-106.0, -83.0, 7.0, 20.0),     # Central America (Cocos → Caribbean)
    (-160.0, -147.0, 50.0, 62.0),   # Alaska–Aleutians (east)
    (165.0, 180.0, 50.0, 62.0),     # Aleutians (west, dateline)
    (140.0, 165.0, 30.0, 56.0),     # Kuril–Kamchatka–NE Japan
    (130.0, 145.0, 24.0, 40.0),     # SW Japan / Nankai / Ryukyu (north)
    (120.0, 135.0, 10.0, 26.0),     # Ryukyu (south) / Taiwan / Luzon
    (118.0, 128.0, -11.0, 8.0),     # Philippines / Sulawesi
    (95.0, 120.0, -11.0, 6.0),      # Sumatra–Java (Sunda)
    (150.0, 170.0, -12.0, 0.0),     # New Britain / Solomon
    (165.0, 180.0, -23.0, -12.0),   # Vanuatu
    (-180.0, -170.0, -40.0, -14.0), # Tonga–Kermadec (dateline west side)
    (172.0, 180.0, -40.0, -14.0),   # Tonga–Kermadec (dateline east side)
    (172.0, 179.0, -48.0, -38.0),   # Hikurangi (New Zealand)
    (20.0, 30.0, 33.0, 40.0),       # Hellenic (Aegean)
    (-90.0, -60.0, 10.0, 20.0),     # Lesser Antilles / Caribbean (rough)
    (122.0, 134.0, -9.0, -3.0),     # Banda
)

# Mid-ocean-ridge latitude/longitude corridors (very coarse) for the heuristic ridge label when no
# PB2002 model is present. Only used to avoid calling obvious oceanic-spreading seismicity
# "intraplate"; PB2002 supersedes it whenever present.
_RIDGE_BOXES: tuple[tuple[float, float, float, float], ...] = (
    (-45.0, -10.0, -60.0, 70.0),    # Mid-Atlantic Ridge corridor (very rough)
    (-120.0, -100.0, -60.0, 0.0),   # East Pacific Rise (south)
    (-115.0, -98.0, 0.0, 23.0),     # East Pacific Rise (north) / Gulf of California
    (55.0, 75.0, -40.0, -10.0),     # Central / SW Indian Ridge (rough)
)


def _point_in_boxes(lat: float, lon: float, boxes: tuple[tuple[float, float, float, float], ...]) -> bool:
    """True if ``(lat, lon)`` falls in any (lon_min, lon_max, lat_min, lat_max) box."""
    for lon_min, lon_max, lat_min, lat_max in boxes:
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return True
    return False


class _Enrichers:
    """Lazy holder for the optional static geophysical layers (Slab2 / PB2002 / GEM faults).

    Loaded once on first use and cached. Heavy deps (``xarray``/``netCDF4`` for Slab2, ``geopandas``/
    ``shapely`` for faults/plates) are imported *inside* the loaders so importing this module needs
    only numpy. Missing data on disk is not an error — the classifier just falls back to the
    heuristic and records ``source="heuristic"``.

    The expected on-disk layout (gitignored; rebuilt by the fetch stage) is::

        data/enrichers/slab2/*_dep.grd        # Slab2 depth-to-interface grids (per zone)
        data/enrichers/pb2002/PB2002_boundaries.dig.txt
        data/enrichers/gem_faults/gem_active_faults.gpkg
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._slab2_loaded = False
        self._slab2 = None  # list of (xarray.DataArray) depth grids, or None
        self._pb2002_loaded = False
        self._pb2002 = None  # list of (boundary_type, np.ndarray[lat, lon]) polylines, or None

    # -- Slab2 ----------------------------------------------------------------
    def slab_depth_km(self, lat: float, lon: float) -> float | None:
        """Depth (km, positive down) to the subduction interface at a point, or ``None`` if no Slab2.

        Returns ``None`` both when Slab2 is not installed and when the point is outside every loaded
        slab grid (i.e. not in a subduction zone). A finite value means "this point sits above a
        modelled megathrust at this depth."
        """
        grids = self._ensure_slab2()
        if not grids:
            return None
        for grid in grids:
            val = self._sample_grid(grid, lat, lon)
            if val is not None and np.isfinite(val):
                # Slab2 depth grids are negative-down (depth below sea level); return positive km.
                return float(abs(val))
        return None

    def _ensure_slab2(self):
        if self._slab2_loaded:
            return self._slab2
        self._slab2_loaded = True
        slab_dir = self.root / "data" / "enrichers" / "slab2"
        if not slab_dir.is_dir():
            self._slab2 = None
            return None
        grd_files = sorted(slab_dir.glob("*dep*.grd")) + sorted(slab_dir.glob("*dep*.nc"))
        if not grd_files:
            self._slab2 = None
            return None
        try:
            import xarray as xr  # lazy heavy dep
        except ModuleNotFoundError:
            self._slab2 = None
            return None
        grids = []
        for path in grd_files:
            try:
                ds = xr.open_dataarray(path)
                grids.append(ds)
            except Exception:
                continue
        self._slab2 = grids or None
        return self._slab2

    @staticmethod
    def _sample_grid(grid, lat: float, lon: float) -> float | None:
        """Nearest-neighbour sample of an xarray depth grid at (lat, lon); ``None`` if out of bounds."""
        try:
            # Slab2 grids index on (y=latitude, x=longitude); longitudes may be 0..360.
            lon_q = lon % 360.0 if float(grid["x"].max()) > 180.0 else lon
            sel = grid.sel(x=lon_q, y=lat, method="nearest")
            val = float(sel.values)
            return val if np.isfinite(val) else None
        except Exception:
            return None

    # -- PB2002 plate boundaries ---------------------------------------------
    def nearest_boundary(self, lat: float, lon: float) -> tuple[float, str] | None:
        """(distance_km, boundary_type) to the nearest PB2002 plate boundary, or ``None`` if no model.

        ``boundary_type`` is one of ``"subduction" | "ridge" | "transform" | "continental"`` mapped
        from the PB2002 step classes. Distance is the great-circle distance to the nearest boundary
        vertex (a vertex-level proxy for the segment distance — fine for regime classification at the
        0.1° grid scale).
        """
        boundaries = self._ensure_pb2002()
        if not boundaries:
            return None
        from ._common import haversine_km  # local import keeps the module light at import time

        best_d = np.inf
        best_type = "continental"
        for btype, verts in boundaries:
            d = haversine_km(lat, lon, verts[:, 0], verts[:, 1])
            dmin = float(np.min(d))
            if dmin < best_d:
                best_d = dmin
                best_type = btype
        return best_d, best_type

    def _ensure_pb2002(self):
        if self._pb2002_loaded:
            return self._pb2002
        self._pb2002_loaded = True
        path = self.root / "data" / "enrichers" / "pb2002" / "PB2002_boundaries.dig.txt"
        if not path.is_file():
            self._pb2002 = None
            return None
        try:
            self._pb2002 = _parse_pb2002_dig(path)
        except Exception:
            self._pb2002 = None
        return self._pb2002


def _parse_pb2002_dig(path: Path):
    """Parse Bird (2003) PB2002 ``*_boundaries.dig.txt`` into [(boundary_type, verts[lat, lon]), ...].

    The ``.dig`` format is a sequence of named polylines: a header line carrying the two-plate code
    and a step class, then ``lon, lat`` vertex lines, terminated by ``*** end of line segment ***``.
    The PB2002 step class letter is mapped to our coarse boundary types: ``S``→subduction,
    ``R``/``O``→ridge (oceanic spreading), ``T``→transform, everything else→continental. Parsing is
    pure-Python (no geopandas) so it stays on the core deps.
    """
    boundaries: list[tuple[str, np.ndarray]] = []
    cur_type = "continental"
    cur: list[tuple[float, float]] = []

    def _flush():
        if len(cur) >= 1:
            boundaries.append((cur_type, np.asarray(cur, dtype=float)))

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("***"):  # end of a segment
                _flush()
                cur = []
                cur_type = "continental"
                continue
            parts = line.replace(",", " ").split()
            # A vertex line is two floats (lon, lat); anything else is a header → derive the type.
            if len(parts) >= 2 and _is_float(parts[0]) and _is_float(parts[1]):
                lon, lat = float(parts[0]), float(parts[1])
                cur.append((lat, lon))
            else:
                _flush()
                cur = []
                cur_type = _pb2002_type_from_header(line)
    _flush()
    return boundaries or None


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _pb2002_type_from_header(header: str) -> str:
    """Map a PB2002 boundary-step header to {subduction, ridge, transform, continental}."""
    h = header.upper()
    if "SUB" in h or h.endswith("S") or " S " in h:
        return "subduction"
    if "OSR" in h or "RID" in h or " R " in h or h.endswith("R"):
        return "ridge"
    if "OTF" in h or "TRANSF" in h or " T " in h or h.endswith("T"):
        return "transform"
    return "continental"


# Module-level enricher cache keyed by repo root, so repeated calls reuse the parsed layers.
_ENRICHERS: dict[Path, _Enrichers] = {}


def _get_enrichers(root: Path | None = None) -> _Enrichers:
    if root is None:
        from ..config import REPO_ROOT

        root = REPO_ROOT
    root = Path(root)
    if root not in _ENRICHERS:
        _ENRICHERS[root] = _Enrichers(root)
    return _ENRICHERS[root]


def assign_regime(
    lat: float,
    lon: float,
    depth_km: float | None = None,
    *,
    root: Path | None = None,
) -> RegimeAssignment:
    """Classify one point into a :class:`TectonicRegime`, using enrichers when present, else heuristics.

    Decision order (most authoritative first):

    1. **Slab2** — if the point sits above a modelled subduction interface, use the interface depth:
       shallow + close to interface → ``SUBDUCTION_INTERFACE``; deep (> :data:`INTRASLAB_DEPTH_KM`)
       → ``INTRASLAB``.
    2. **PB2002** — otherwise the nearest plate-boundary type within a tolerance fixes ridge /
       transform / subduction-margin / continental-crustal; far from any boundary → ``INTRAPLATE``.
    3. **Heuristic** — with no enrichers on disk, a coarse built-in subduction/ridge mask plus the
       event depth give a transparent fallback (recorded ``source="heuristic"``).

    Parameters
    ----------
    lat, lon:
        Epicentre in degrees (WGS84). ``lon`` may be in [-180, 180] or [0, 360]; both are handled.
    depth_km:
        Hypocentral depth (km, positive down). When ``None`` the depth-dependent interface/intraslab
        split cannot use the event depth and the classifier leans on geometry alone.
    root:
        Repo root override (tests); defaults to the package :data:`~caos_seismic.config.REPO_ROOT`.

    Returns
    -------
    RegimeAssignment
        The regime plus the supporting evidence (interface depth, boundary distance/type, source).
    """
    lon180 = ((float(lon) + 180.0) % 360.0) - 180.0
    enr = _get_enrichers(root)

    # 1) Slab2 interface geometry (the authoritative subduction discriminator).
    slab_depth = enr.slab_depth_km(lat, lon180)
    if slab_depth is not None:
        if depth_km is not None and depth_km > INTRASLAB_DEPTH_KM and depth_km > slab_depth + INTERFACE_DISTANCE_KM:
            return RegimeAssignment(
                TectonicRegime.INTRASLAB, source="slab2", depth_to_interface_km=slab_depth
            )
        # Shallow / near the interface → megathrust interface.
        if depth_km is None or abs(depth_km - slab_depth) <= INTERFACE_DISTANCE_KM or depth_km <= INTRASLAB_DEPTH_KM:
            return RegimeAssignment(
                TectonicRegime.SUBDUCTION_INTERFACE, source="slab2", depth_to_interface_km=slab_depth
            )
        return RegimeAssignment(
            TectonicRegime.INTRASLAB, source="slab2", depth_to_interface_km=slab_depth
        )

    # 2) PB2002 plate boundary (ridge / transform / subduction margin / continental).
    nb = enr.nearest_boundary(lat, lon180)
    if nb is not None:
        dist, btype = nb
        if dist <= 150.0:  # within a plate-boundary corridor
            if btype == "ridge":
                return RegimeAssignment(
                    TectonicRegime.RIDGE, source="pb2002",
                    distance_to_boundary_km=dist, boundary_type=btype,
                )
            if btype == "subduction":
                regime = (
                    TectonicRegime.INTRASLAB
                    if depth_km is not None and depth_km > INTRASLAB_DEPTH_KM
                    else TectonicRegime.SUBDUCTION_INTERFACE
                )
                return RegimeAssignment(
                    regime, source="pb2002",
                    distance_to_boundary_km=dist, boundary_type=btype,
                )
            # transform / continental boundary → crustal active faulting.
            return RegimeAssignment(
                TectonicRegime.CRUSTAL, source="pb2002",
                distance_to_boundary_km=dist, boundary_type=btype,
            )
        # Far from any plate boundary → stable interior.
        return RegimeAssignment(
            TectonicRegime.INTRAPLATE, source="pb2002",
            distance_to_boundary_km=dist, boundary_type="interior",
        )

    # 3) Heuristic fallback (no enrichers on disk).
    return _heuristic_regime(lat, lon180, depth_km)


def _heuristic_regime(lat: float, lon: float, depth_km: float | None) -> RegimeAssignment:
    """Self-contained regime guess from coarse masks + depth — the no-enricher fallback.

    Uses the built-in subduction-margin / ridge boxes plus the event depth: in a subduction box a
    deep event is intraslab and a shallow one is interface; in a ridge corridor it is ridge; otherwise
    a shallow event near a (very coarse) margin is crustal and a deep stable-interior event is
    intraplate. The intent is to never mislabel a megathrust as intraplate, not to be precise.
    """
    if _point_in_boxes(lat, lon, _SUBDUCTION_MARGIN_BOXES):
        if depth_km is not None and depth_km > INTRASLAB_DEPTH_KM:
            return RegimeAssignment(TectonicRegime.INTRASLAB, source="heuristic")
        return RegimeAssignment(TectonicRegime.SUBDUCTION_INTERFACE, source="heuristic")
    if _point_in_boxes(lat, lon, _RIDGE_BOXES):
        return RegimeAssignment(TectonicRegime.RIDGE, source="heuristic")
    # Continental vs. oceanic-interior split by a crude depth/latitude rule: shallow → crustal active
    # faulting; otherwise stable intraplate interior.
    if depth_km is not None and depth_km > INTRASLAB_DEPTH_KM:
        return RegimeAssignment(TectonicRegime.INTRASLAB, source="heuristic")
    return RegimeAssignment(TectonicRegime.CRUSTAL, source="heuristic")


def assign_regimes(
    lat: np.ndarray,
    lon: np.ndarray,
    depth_km: np.ndarray | None = None,
    *,
    root: Path | None = None,
) -> list[RegimeAssignment]:
    """Vectorized convenience: classify many points (one :class:`RegimeAssignment` per element)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    n = lat.size
    depth = (
        np.asarray(depth_km, dtype=float)
        if depth_km is not None
        else np.full(n, np.nan)
    )
    out: list[RegimeAssignment] = []
    for i in range(n):
        d = float(depth[i]) if np.isfinite(depth[i]) else None
        out.append(assign_regime(float(lat[i]), float(lon[i]), d, root=root))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Spatial tiling
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Tile:
    """One spatial tile: an interior bbox plus a halo bbox for edge-correct triggering.

    ETAS/smoothed-seismicity are *fit* on the events inside ``halo`` (so a parent just outside the
    interior still triggers offspring that land inside it — no truncation at tile edges), but each
    tile *owns* only the cells whose centres lie in ``interior`` when the per-tile fields are
    aggregated into the global field (so cells are never double-counted across overlapping halos).

    Attributes
    ----------
    id:
        Stable tile key ``"r{row}_c{col}"`` for manifests + caching.
    interior:
        The non-overlapping bbox this tile owns for aggregation.
    halo:
        The (larger) bbox whose events are used to *fit* this tile.
    """

    id: str
    interior: BBox
    halo: BBox

    def contains_interior(self, lat: float, lon: float) -> bool:
        """True if a cell centre ``(lat, lon)`` is owned by this tile (lies in ``interior``)."""
        b = self.interior
        return b.lat_min <= lat < b.lat_max and b.lon_min <= lon < b.lon_max

    def contains_halo(self, lat: float, lon: float) -> bool:
        """True if ``(lat, lon)`` lies within this tile's fitting halo."""
        b = self.halo
        return b.lat_min <= lat <= b.lat_max and b.lon_min <= lon <= b.lon_max


def iterate_tiles(
    region: Region | BBox,
    *,
    tile_deg: float = 10.0,
    halo_deg: float = 1.0,
) -> Iterator[Tile]:
    """Partition a region's bbox into a regular grid of :class:`Tile` objects (interior + halo).

    The interior tiles tessellate the bbox exactly (no gaps / no overlap); each one is grown by
    ``halo_deg`` on every side to form its fitting halo. ``tile_deg`` bounds the per-fit catalog size
    so the ETAS triggering sum stays tractable (a 10° tile of a busy subduction margin holds at most a
    few thousand M≥Mc events per few years — well within the ``O(N^2)`` budget), while ``halo_deg``
    (≈ a large-event aftershock-zone radius) keeps triggering continuous across tile boundaries.

    Parameters
    ----------
    region:
        A :class:`~caos_seismic.contracts.Region` (its ``bbox`` is used) or a raw :class:`BBox`. To
        tile the **whole globe**, pass ``BBox(lat_min=-90, lat_max=90, lon_min=-180, lon_max=180)``.
    tile_deg:
        Edge length of each interior tile in degrees (square in lat/lon; the physical area shrinks
        with latitude, which is acceptable since the fit cost scales with event count, not area).
    halo_deg:
        Halo width added on every side for edge-correct triggering (degrees).

    Yields
    ------
    Tile
        Tiles in row-major (south→north, west→east) order, clipped to the bbox at the edges.
    """
    bb = region.bbox if isinstance(region, Region) else region
    if tile_deg <= 0:
        raise ValueError("tile_deg must be positive")

    lat_edges = _edges(bb.lat_min, bb.lat_max, tile_deg)
    lon_edges = _edges(bb.lon_min, bb.lon_max, tile_deg)

    for r in range(len(lat_edges) - 1):
        lat0, lat1 = lat_edges[r], lat_edges[r + 1]
        for c in range(len(lon_edges) - 1):
            lon0, lon1 = lon_edges[c], lon_edges[c + 1]
            interior = BBox(lat_min=lat0, lat_max=lat1, lon_min=lon0, lon_max=lon1)
            halo = BBox(
                lat_min=max(lat0 - halo_deg, -90.0),
                lat_max=min(lat1 + halo_deg, 90.0),
                lon_min=lon0 - halo_deg,  # longitudes may run past ±180 in the halo; callers wrap
                lon_max=lon1 + halo_deg,
            )
            yield Tile(id=f"r{r}_c{c}", interior=interior, halo=halo)


def _edges(lo: float, hi: float, step: float) -> list[float]:
    """Tile edges from ``lo`` to ``hi`` in ``step`` increments, with the last edge clipped to ``hi``."""
    if hi <= lo:
        return [lo, hi]
    n = int(np.ceil((hi - lo) / step))
    edges = [lo + i * step for i in range(n)]
    edges.append(hi)
    return edges


def tiles_for_region(
    region: Region | BBox,
    *,
    tile_deg: float = 10.0,
    halo_deg: float = 1.0,
) -> list[Tile]:
    """Eager :func:`iterate_tiles` — the materialized tile list (convenient for length / indexing)."""
    return list(iterate_tiles(region, tile_deg=tile_deg, halo_deg=halo_deg))


def dominant_regime(
    tile: Tile,
    catalog_lat: np.ndarray,
    catalog_lon: np.ndarray,
    catalog_depth: np.ndarray | None = None,
    *,
    root: Path | None = None,
    default: TectonicRegime = TectonicRegime.CRUSTAL,
) -> TectonicRegime:
    """The modal tectonic regime of a tile, from the regimes of the events that fall in its halo.

    Used to pick the per-tile ETAS prior (:func:`regime_prior`): a tile is regularized toward the
    regime most of its seismicity belongs to. With no events in the halo the tile takes ``default``
    (crustal — the most generic regime). Empty/degenerate inputs degrade gracefully to ``default``.
    """
    lat = np.asarray(catalog_lat, dtype=float)
    lon = np.asarray(catalog_lon, dtype=float)
    if lat.size == 0:
        return default
    in_halo = np.array([tile.contains_halo(la, lo) for la, lo in zip(lat, lon)], dtype=bool)
    if not np.any(in_halo):
        return default
    depth = (
        np.asarray(catalog_depth, dtype=float)[in_halo]
        if catalog_depth is not None
        else None
    )
    assigns = assign_regimes(lat[in_halo], lon[in_halo], depth, root=root)
    counts: dict[TectonicRegime, int] = {}
    for a in assigns:
        counts[a.regime] = counts.get(a.regime, 0) + 1
    if not counts:
        return default
    return max(counts.items(), key=lambda kv: kv[1])[0]

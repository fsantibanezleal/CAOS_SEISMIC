"""Bird (2003) PB2002 plate-boundary enricher — distance-to-boundary, boundary type, relative velocity.

PB2002 (Bird, 2003, *G-cubed*) is a global model of present-day plate boundaries: digitized boundary
segments, each tagged with a two-letter **boundary class** and the relative motion of the two
bounding plates. For a conditional forecaster it conditions the long-term background term on tectonic
setting (data-and-pipelines.md §1.3): a cell on a fast convergent boundary has a very different prior
rate of large events than a cell on a slow intracontinental one.

Per-cell features
-----------------
``plate_boundary_dist_km``   great-circle distance to the nearest PB2002 boundary point (km).
``plate_boundary_type_code`` integer code of the nearest boundary's class (see
                             :data:`BOUNDARY_TYPE_CODES`; ``0`` = unknown).
``plate_rel_velocity_mm_yr`` relative plate velocity at the nearest boundary point (mm/yr) — from the
                             PB2002 boundary file's per-step velocity, when present.
``plate_is_convergent``      1.0 if the nearest boundary is a subduction/convergent class, else 0.0.
``plate_is_divergent``       1.0 if it is an oceanic-spreading/divergent class, else 0.0.
``plate_is_transform``       1.0 if it is a transform/fracture-zone class, else 0.0.

Data & license
--------------
* Source: ``peterbird.name/publications/2003_pb2002/`` — ASCII files
  ``PB2002_boundaries.dig.txt`` (boundary geometry, two-letter class per segment) and
  ``PB2002_steps.dat`` (per-step boundary class + relative velocity).
* License: open for research with citation (Bird, 2003). No redistribution restriction.

This enricher parses **ASCII** and so needs only the core deps (numpy); no heavy import is required.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ...model._common import DEG2KM, haversine_km
from ._base import EnricherResult, Provenance, cache_dir, http_download

logger = logging.getLogger(__name__)

DATASET = "plates"

PB2002_BASE = "http://peterbird.name/publications/2003_pb2002"
PB2002_BOUNDARIES = "PB2002_boundaries.dig.txt"
PB2002_STEPS = "PB2002_steps.dat"

#: PB2002 two-letter boundary class → stable integer code + kinematic family.
#: OSR=oceanic spreading ridge, OTF=oceanic transform fault, OCB=oceanic convergent boundary,
#: CRB=continental rift boundary, CTF=continental transform fault, CCB=continental convergent
#: boundary, SUB=subduction zone.
BOUNDARY_TYPE_CODES: dict[str, int] = {
    "OSR": 1,  # oceanic spreading ridge   (divergent)
    "OTF": 2,  # oceanic transform fault    (transform)
    "OCB": 3,  # oceanic convergent boundary(convergent)
    "CRB": 4,  # continental rift boundary  (divergent)
    "CTF": 5,  # continental transform fault(transform)
    "CCB": 6,  # continental convergent     (convergent)
    "SUB": 7,  # subduction zone            (convergent)
}

_DIVERGENT = {1, 4}
_TRANSFORM = {2, 5}
_CONVERGENT = {3, 6, 7}

FEATURE_NAMES = (
    "plate_boundary_dist_km",
    "plate_boundary_type_code",
    "plate_rel_velocity_mm_yr",
    "plate_is_convergent",
    "plate_is_divergent",
    "plate_is_transform",
)


def type_code(boundary_class: Any) -> int:
    """Map a PB2002 two-letter boundary class (e.g. ``"SUB"``) to its stable integer code."""
    if not isinstance(boundary_class, str) or not boundary_class.strip():
        return 0
    return BOUNDARY_TYPE_CODES.get(boundary_class.strip().upper(), 0)


def download(
    *,
    base_url: str | None = None,
    dest: Path | None = None,
    overwrite: bool = False,
    session: Any | None = None,
) -> Provenance:
    """Download the PB2002 boundary geometry + steps ASCII files to the gitignored cache."""
    dest = dest or cache_dir(DATASET)
    base = (base_url or PB2002_BASE).rstrip("/")
    files: list[str] = []
    for name in (PB2002_BOUNDARIES, PB2002_STEPS):
        try:
            out = http_download(f"{base}/{name}", dest / name, overwrite=overwrite, session=session)
            files.append(str(out.relative_to(dest.parent.parent.parent)))
        except RuntimeError as exc:
            logger.warning("plates: could not fetch %s (%s)", name, exc)

    return Provenance(
        dataset=DATASET,
        title="PB2002 plate-boundary model (Bird, 2003)",
        version="PB2002",
        source_url=f"{PB2002_BASE}/",
        license="Open for research with citation (Bird, 2003)",
        attribution="PB2002 plate-boundary model (Bird, 2003)",
        citation=(
            "Bird, P. (2003). An updated digital model of plate boundaries. "
            "Geochemistry, Geophysics, Geosystems, 4(3), 1027. doi:10.1029/2001GC000252"
        ),
        files=files,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        notes="ASCII boundary digitization + per-step relative velocity; parsed with core deps only.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ASCII parsing (core deps only)
# ─────────────────────────────────────────────────────────────────────────────


def _parse_boundaries(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Parse ``PB2002_boundaries.dig.txt`` → (lat[], lon[], per-vertex boundary-class strings).

    The ``.dig`` format is a sequence of named segments: a header line carrying the segment name
    (which embeds the two adjacent plates, e.g. ``PA-NA``), then ``lon, lat`` lines, terminated by a
    ``*** end of line segment ***`` marker. The two-letter kinematic class is taken from the steps
    file (richer); here we keep the geometry and the plate-pair tag for fallback typing.
    """
    lats: list[float] = []
    lons: list[float] = []
    classes: list[str] = []
    current_class = ""
    text = path.read_text(encoding="latin-1", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("***"):  # end-of-segment marker
            current_class = ""
            continue
        coord = _coord_pair(line)
        if coord is None:
            # A segment header line (e.g. "PA-NA" or a name); reset class context.
            current_class = ""
            continue
        lon, lat = coord
        lons.append(lon)
        lats.append(lat)
        classes.append(current_class)
    return np.asarray(lats, dtype=float), np.asarray(lons, dtype=float), classes


def _parse_steps(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse ``PB2002_steps.dat`` → (lat[], lon[], type_code[], rel_velocity_mm_yr[]).

    The steps file lists, per boundary step, the midpoint coordinates, the two-letter class, and the
    relative velocity components (mm/yr). We use the midpoint as the boundary sample, the class for
    typing, and the velocity magnitude as the per-cell ``plate_rel_velocity_mm_yr`` feature.
    """
    lats: list[float] = []
    lons: list[float] = []
    codes: list[int] = []
    vels: list[float] = []
    text = path.read_text(encoding="latin-1", errors="replace")
    cls_re = re.compile(r"\b(OSR|OTF|OCB|CRB|CTF|CCB|SUB)\b")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        nums = _floats(line)
        if len(nums) < 4:
            continue
        # PB2002_steps columns include longitudes/latitudes of the step endpoints and a velocity
        # azimuth/magnitude; the robust, format-version-independent read is: first lon/lat pair as
        # the sample point, the boundary class token, and the largest plausible velocity magnitude.
        lon, lat = nums[0], nums[1]
        if not (-360.0 <= lon <= 360.0 and -90.0 <= lat <= 90.0):
            continue
        m = cls_re.search(line)
        code = type_code(m.group(1)) if m else 0
        vel = _velocity_mm_yr(nums)
        lons.append(_wrap_lon(lon))
        lats.append(lat)
        codes.append(code)
        vels.append(vel)
    return (
        np.asarray(lats, dtype=float),
        np.asarray(lons, dtype=float),
        np.asarray(codes, dtype=int),
        np.asarray(vels, dtype=float),
    )


def _coord_pair(line: str) -> tuple[float, float] | None:
    """Parse a ``lon, lat`` (or whitespace-separated) coordinate line; ``None`` if not a coord line."""
    nums = _floats(line)
    if len(nums) < 2:
        return None
    lon, lat = nums[0], nums[1]
    if -360.0 <= lon <= 360.0 and -90.0 <= lat <= 90.0:
        return _wrap_lon(lon), lat
    return None


def _floats(line: str) -> list[float]:
    out: list[float] = []
    for tok in re.split(r"[,\s]+", line.strip()):
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def _velocity_mm_yr(nums: list[float]) -> float:
    """Best-effort relative-velocity magnitude from a PB2002 steps row (mm/yr).

    PB2002 velocities are O(0..200 mm/yr). We take the largest value in the row that falls in a
    plausible velocity range and is not obviously a coordinate, giving a robust magnitude without
    hard-coding a column index that varies between distributed copies of the file.
    """
    cands = [v for v in nums[2:] if 0.0 < v <= 250.0]
    return float(max(cands)) if cands else float("nan")


def _wrap_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell feature extraction
# ─────────────────────────────────────────────────────────────────────────────


class PlatesEnricher:
    """Nearest plate-boundary feature extractor over the cached PB2002 ASCII files.

    Prefers the steps file (carries class + velocity); falls back to the boundary geometry for
    distance only. Loads once and caches the boundary-point arrays in memory.
    """

    def __init__(self, dest: Path | None = None) -> None:
        self.dest = dest or cache_dir(DATASET)
        self._lat: np.ndarray | None = None
        self._lon: np.ndarray | None = None
        self._code: np.ndarray | None = None
        self._vel: np.ndarray | None = None

    def _load(self) -> None:
        if self._lat is not None:
            return
        steps = self.dest / PB2002_STEPS
        bnds = self.dest / PB2002_BOUNDARIES
        if steps.exists():
            lat, lon, code, vel = _parse_steps(steps)
            if lat.size:
                self._lat, self._lon, self._code, self._vel = lat, lon, code, vel
                return
        if bnds.exists():
            lat, lon, _classes = _parse_boundaries(bnds)
            if lat.size:
                self._lat, self._lon = lat, lon
                self._code = np.zeros(lat.size, dtype=int)
                self._vel = np.full(lat.size, np.nan)
                return
        raise FileNotFoundError(
            f"no PB2002 files cached under {self.dest}. Run "
            "caos_seismic.data.enrichers.plates.download() first."
        )

    def features_at(self, lat: float, lon: float, **_: Any) -> EnricherResult:
        """Return nearest plate-boundary distance / type / relative-velocity covariates."""
        self._load()
        assert self._lat is not None and self._lon is not None
        d = haversine_km(lat, lon, self._lat, self._lon)
        j = int(np.argmin(d))
        code = int(self._code[j]) if self._code is not None else 0
        vel = float(self._vel[j]) if self._vel is not None else float("nan")
        return {
            "plate_boundary_dist_km": float(d[j]),
            "plate_boundary_type_code": float(code),
            "plate_rel_velocity_mm_yr": None if not np.isfinite(vel) else vel,
            "plate_is_convergent": 1.0 if code in _CONVERGENT else 0.0,
            "plate_is_divergent": 1.0 if code in _DIVERGENT else 0.0,
            "plate_is_transform": 1.0 if code in _TRANSFORM else 0.0,
        }


_DEFAULT_ENRICHER: PlatesEnricher | None = None


def features_at(lat: float, lon: float, **kwargs: Any) -> EnricherResult:
    """Module-level convenience: nearest plate-boundary features at one coordinate."""
    global _DEFAULT_ENRICHER
    if _DEFAULT_ENRICHER is None:
        _DEFAULT_ENRICHER = PlatesEnricher()
    return _DEFAULT_ENRICHER.features_at(lat, lon, **kwargs)


def boundary_points(dest: Path | None = None) -> Iterator[tuple[float, float, int, float]]:
    """Yield ``(lat, lon, type_code, velocity_mm_yr)`` for every cached PB2002 boundary point."""
    enr = PlatesEnricher(dest)
    enr._load()
    assert enr._lat is not None and enr._lon is not None
    code = enr._code if enr._code is not None else np.zeros(enr._lat.size, dtype=int)
    vel = enr._vel if enr._vel is not None else np.full(enr._lat.size, np.nan)
    for i in range(enr._lat.size):
        yield float(enr._lat[i]), float(enr._lon[i]), int(code[i]), float(vel[i])


__all__ = [
    "DATASET",
    "BOUNDARY_TYPE_CODES",
    "FEATURE_NAMES",
    "PlatesEnricher",
    "download",
    "features_at",
    "type_code",
    "boundary_points",
    "DEG2KM",
]

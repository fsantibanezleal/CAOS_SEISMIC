"""NGL / MIDAS GNSS enricher — geodetic strain-rate proxy at each cell.

The Nevada Geodetic Laboratory publishes MIDAS (Blewitt et al., 2016) trend velocities for the global
GNSS station network: a robust, outlier-resistant secular velocity per station in the IGS14 frame.
For a conditional forecaster the geodetic velocity field is the cleanest available proxy for the
**tectonic loading rate** (data-and-pipelines.md §1.3): cells in regions of high horizontal strain
rate accumulate stress faster and carry a higher long-term background. We turn the discrete station
velocities into a per-cell **strain-rate proxy** via the velocity gradient among nearby stations.

Per-cell features
-----------------
``gnss_n_stations``        number of MIDAS stations within the search radius.
``gnss_speed_mm_yr``       inverse-distance-weighted horizontal speed of nearby stations (mm/yr).
``gnss_strain_rate_nstrain_yr``
                           second-invariant strain-rate proxy from the local velocity gradient
                           (nanostrain/yr); ``None`` if too few stations to estimate a gradient.

Data & license
--------------
* Source: ``geodesy.unr.edu`` — the MIDAS combined file ``midas.IGS14.txt`` (one row per station:
  id, lon, lat, east/north/up velocity + uncertainties) and per-station ``.tenv3`` time series.
* License: open with attribution (Nevada Geodetic Laboratory / Blewitt et al., 2018).

Parsing the MIDAS table needs only the core deps (numpy/pandas); no heavy import. The optional
``.tenv3`` per-station time-series helper is provided for completeness.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...model._common import DEG2KM, haversine_km
from ._base import EnricherResult, Provenance, cache_dir, http_download

logger = logging.getLogger(__name__)

DATASET = "gnss"

NGL_BASE = "https://geodesy.unr.edu"  # HTTPS: the plain-HTTP (port 80) endpoint times out from many networks
MIDAS_URL = f"{NGL_BASE}/velocities/midas.IGS14.txt"

FEATURE_NAMES = (
    "gnss_n_stations",
    "gnss_speed_mm_yr",
    "gnss_strain_rate_nstrain_yr",
)


def download(
    *,
    url: str | None = None,
    dest: Path | None = None,
    overwrite: bool = False,
    session: Any | None = None,
) -> Provenance:
    """Download the NGL MIDAS combined velocity table (``midas.IGS14.txt``) to the cache."""
    dest = dest or cache_dir(DATASET)
    url = url or MIDAS_URL
    out = dest / "midas.IGS14.txt"
    http_download(url, out, overwrite=overwrite, session=session)
    return Provenance(
        dataset=DATASET,
        title="Nevada Geodetic Laboratory — MIDAS GNSS velocities (IGS14)",
        version="MIDAS combined (rolling)",
        source_url=f"{NGL_BASE}/velocities/",
        license="Open with attribution (Nevada Geodetic Laboratory)",
        attribution="Nevada Geodetic Laboratory, University of Nevada, Reno (Blewitt et al.)",
        citation=(
            "Blewitt, G., Hammond, W. C., & Kreemer, C. (2018). Harnessing the GPS data explosion "
            "for interdisciplinary science. Eos, 99. doi:10.1029/2018EO104623; "
            "Blewitt, G., Kreemer, C., Hammond, W. C., & Gazeaux, J. (2016). MIDAS robust trend "
            "estimator for accurate GPS station velocities. JGR Solid Earth, 121, 2054-2068."
        ),
        files=[str(out.relative_to(dest.parent.parent.parent))],
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        notes="One row per station: id, lon, lat, E/N/U velocity (m/yr) + uncertainties.",
    )


def download_station_tenv3(
    station: str,
    *,
    frame: str = "IGS14",
    dest: Path | None = None,
    overwrite: bool = False,
    session: Any | None = None,
) -> Path:
    """Download one NGL ``.tenv3`` station time series (optional; for per-station analysis).

    The ``.tenv3`` per-station series under ``geodesy.unr.edu/gps_timeseries/tenv3/<frame>/`` is used
    for transient / time-dependent geodetic features. The daily forecast uses the MIDAS *trend* table
    (a static secular velocity field); this helper is here for the deeper geodetic enricher work.
    """
    dest = dest or (cache_dir(DATASET) / "tenv3")
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{NGL_BASE}/gps_timeseries/tenv3/{frame}/{station.upper()}.tenv3"
    return http_download(url, dest / f"{station.upper()}.tenv3", overwrite=overwrite, session=session)


# ─────────────────────────────────────────────────────────────────────────────
# MIDAS table parsing (core deps only)
# ─────────────────────────────────────────────────────────────────────────────


def load_midas(path: Path) -> pd.DataFrame:
    """Parse ``midas.IGS14.txt`` into a frame with ``lat, lon, ve_mm_yr, vn_mm_yr`` (mm/yr).

    The MIDAS combined file is whitespace-delimited with the station code first and the longitude /
    latitude and east/north velocity (m/yr) among the leading numeric columns. We read it robustly:
    the first all-numeric columns after the station id give lon/lat, then the E/N velocities (NGL's
    documented column order), and convert velocities m/yr → mm/yr.
    """
    rows = []
    text = path.read_text(encoding="latin-1", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) < 12:
            continue
        # NGL MIDAS combined columns (fixed layout): station label t_first t_last dt n_epochs n_good
        # n_pairs  VE VN VU  sve svn svu  ... and the reference position (LAT LON HEIGHT) is the LAST
        # three columns. East/north secular velocities are columns 8/9 (m/yr). Using fixed positions is
        # robust; the earlier value-range heuristic mis-read the small VE/VN pair as lon/lat -> garbage.
        try:
            ve = float(parts[8])
            vn = float(parts[9])
            lat = float(parts[-3])
            lon = float(parts[-2])
        except (ValueError, IndexError):
            continue
        if not (-90.0 <= lat <= 90.0):
            continue
        lon = ((lon + 180.0) % 360.0) - 180.0  # normalize to [-180, 180)
        rows.append(
            {
                "station": parts[0],
                "lat": lat,
                "lon": lon,
                "ve_mm_yr": ve * 1000.0,
                "vn_mm_yr": vn * 1000.0,
            }
        )
    df = pd.DataFrame(rows, columns=["station", "lat", "lon", "ve_mm_yr", "vn_mm_yr"])
    if df.empty:
        raise ValueError(f"parsed zero MIDAS stations from {path} — check the file format.")
    return df


def _is_float(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _find_lonlat(nums: list[float]) -> tuple[float | None, float | None]:
    """Locate the lon/lat pair in a MIDAS numeric row by their value ranges (lon in 0..360 or ±180)."""
    for i in range(len(nums) - 1):
        a, b = nums[i], nums[i + 1]
        if -90.0 <= b <= 90.0 and (0.0 <= a <= 360.0 or -180.0 <= a <= 180.0) and abs(a) > 0.0:
            lon = ((a + 180.0) % 360.0) - 180.0
            return lon, b
    return None, None


def _find_en_velocity(nums: list[float]) -> tuple[float, float]:
    """Pick the E/N secular velocities (m/yr): the first small-magnitude pair (|v| < 1 m/yr)."""
    small = [v for v in nums if abs(v) < 1.0]
    if len(small) >= 2:
        return small[0], small[1]
    return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell feature extraction
# ─────────────────────────────────────────────────────────────────────────────


class GnssEnricher:
    """Per-cell GNSS strain-rate proxy from the cached MIDAS velocity field.

    For a query cell, finds MIDAS stations within ``radius_km`` and estimates a local horizontal
    velocity gradient by least squares (planar fit of ve, vn vs. local east/north coordinates). The
    second invariant of the strain-rate tensor is reported as a nanostrain/yr proxy. Sparse-network
    regions return ``None`` for the strain rate (honest: the gradient is unconstrained).
    """

    def __init__(self, dest: Path | None = None, *, radius_km: float = 150.0, min_stations: int = 4) -> None:
        self.dest = dest or cache_dir(DATASET)
        self.radius_km = float(radius_km)
        self.min_stations = int(min_stations)
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        path = self.dest / "midas.IGS14.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"no MIDAS table cached under {self.dest}. Run "
                "caos_seismic.data.enrichers.gnss.download() first."
            )
        self._df = load_midas(path)
        return self._df

    def features_at(self, lat: float, lon: float, **_: Any) -> EnricherResult:
        """Return GNSS station count, weighted speed, and strain-rate proxy near ``(lat, lon)``."""
        df = self._load()
        d = haversine_km(lat, lon, df["lat"].to_numpy(), df["lon"].to_numpy())
        sel = d <= self.radius_km
        n = int(np.count_nonzero(sel))
        out: EnricherResult = {
            "gnss_n_stations": float(n),
            "gnss_speed_mm_yr": None,
            "gnss_strain_rate_nstrain_yr": None,
        }
        if n == 0:
            out["gnss_speed_mm_yr"] = 0.0
            return out

        dd = d[sel]
        ve = df["ve_mm_yr"].to_numpy()[sel]
        vn = df["vn_mm_yr"].to_numpy()[sel]
        speed = np.hypot(ve, vn)
        w = 1.0 / np.maximum(dd, 1.0)
        out["gnss_speed_mm_yr"] = float(np.sum(w * speed) / np.sum(w))

        if n >= self.min_stations:
            out["gnss_strain_rate_nstrain_yr"] = _strain_rate_second_invariant(
                lat, lon, df["lat"].to_numpy()[sel], df["lon"].to_numpy()[sel], ve, vn
            )
        return out


def _strain_rate_second_invariant(
    lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray, ve: np.ndarray, vn: np.ndarray
) -> float | None:
    """Second invariant of the horizontal strain-rate tensor (nanostrain/yr) from a planar velocity fit.

    Local east/north coordinates (km) are formed about the query cell; ve, vn (mm/yr) are regressed
    on (x_km, y_km) by least squares to get the velocity gradient tensor ∂v/∂x. The strain-rate
    tensor is its symmetric part; we return its second invariant. Units: (mm/yr)/km = 1e-6/yr =
    1 microstrain/yr → ×1000 = nanostrain/yr.
    """
    x = (lons - lon0) * DEG2KM * np.cos(np.radians(lat0))
    y = (lats - lat0) * DEG2KM
    a = np.column_stack([x, y, np.ones_like(x)])
    try:
        cx, *_ = np.linalg.lstsq(a, ve, rcond=None)  # ve = cx[0]*x + cx[1]*y + cx[2]
        cy, *_ = np.linalg.lstsq(a, vn, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate station geometry
        return None
    dve_dx, dve_dy = cx[0], cx[1]
    dvn_dx, dvn_dy = cy[0], cy[1]
    exx = dve_dx
    eyy = dvn_dy
    exy = 0.5 * (dve_dy + dvn_dx)
    second_invariant = float(np.sqrt(exx**2 + eyy**2 + 2.0 * exy**2))  # (mm/yr)/km = microstrain/yr
    return second_invariant * 1000.0  # → nanostrain/yr


_DEFAULT_ENRICHER: GnssEnricher | None = None


def features_at(lat: float, lon: float, **kwargs: Any) -> EnricherResult:
    """Module-level convenience: GNSS strain-rate proxy features at one coordinate."""
    global _DEFAULT_ENRICHER
    if _DEFAULT_ENRICHER is None:
        _DEFAULT_ENRICHER = GnssEnricher()
    return _DEFAULT_ENRICHER.features_at(lat, lon, **kwargs)


__all__ = [
    "DATASET",
    "FEATURE_NAMES",
    "GnssEnricher",
    "download",
    "download_station_tenv3",
    "features_at",
    "load_midas",
    "DEG2KM",
]

"""Stress enricher — crustal stress orientation/regime from the World Stress Map (+ focal mechanisms).

The World Stress Map (WSM; Heidbach et al., 2016) is the global compilation of present-day crustal
stress indicators: each record carries the azimuth of the maximum horizontal compressive stress
(SHmax) and a tectonic-regime class (normal / strike-slip / thrust / unknown) with a quality flag.
For a conditional forecaster this conditions the *triggering geometry* (data-and-pipelines.md §1.3):
the stress regime and SHmax orientation set the sign and magnitude of Coulomb stress transfer that
governs aftershock productivity and directivity. Focal-mechanism catalogs (GCMT, regional MTs) are
the event-driven complement and can be folded into the same per-cell summary.

Per-cell features
-----------------
``shmax_azimuth_deg``    inverse-distance-weighted SHmax azimuth of nearby WSM records (degrees,
                         0-180; orientation is modulo 180°, handled by circular averaging on 2θ).
``stress_regime_code``   modal tectonic-regime code of nearby records (see :data:`REGIME_CODES`).
``stress_is_thrust``     1.0 if the local modal regime is thrust/reverse, else 0.0.
``stress_is_normal``     1.0 if the local modal regime is normal, else 0.0.
``stress_is_strikeslip`` 1.0 if the local modal regime is strike-slip, else 0.0.
``stress_n_records``     number of WSM records within the search radius.

Data & license
--------------
* Source: World Stress Map (``world-stress-map.org`` / GFZ Data Services). The CSV release carries
  per-record ``LAT, LON, AZI`` (SHmax azimuth) and ``REGIME`` + ``QUALITY``.
* License: CC-BY 4.0 (verify per release); attribute the WSM / GFZ. Cite Heidbach et al. (2016, 2018).

Parsing the WSM CSV needs only the core deps; no heavy import.
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

DATASET = "stress"

WSM_HOME = "https://www.world-stress-map.org"

#: WSM tectonic-regime letter → stable code. NF=normal, SS=strike-slip, TF=thrust, plus the
#: transitional NS/TS classes mapped to their dominant family; U/blank → unknown.
REGIME_CODES: dict[str, int] = {
    "NF": 1, "NS": 1,  # normal (and normal-strike-slip transitional)
    "SS": 2,
    "TF": 3, "TS": 3,  # thrust (and thrust-strike-slip transitional)
    "U": 0, "": 0,
}

_NORMAL = {1}
_STRIKESLIP = {2}
_THRUST = {3}

FEATURE_NAMES = (
    "shmax_azimuth_deg",
    "stress_regime_code",
    "stress_is_thrust",
    "stress_is_normal",
    "stress_is_strikeslip",
    "stress_n_records",
)


def regime_code(regime: Any) -> int:
    """Map a WSM regime letter (``NF``/``SS``/``TF``/…) to its stable integer code (``0`` unknown)."""
    if not isinstance(regime, str):
        return 0
    return REGIME_CODES.get(regime.strip().upper(), 0)


def download(
    *,
    url: str | None = None,
    dest: Path | None = None,
    overwrite: bool = False,
    session: Any | None = None,
) -> Provenance:
    """Download a World Stress Map CSV release to the gitignored cache.

    The WSM database file is versioned and distributed via GFZ Data Services with per-release URLs,
    so ``url`` must point at the current CSV. With no ``url`` this raises an actionable error naming
    the WSM home and the expected columns (``LAT, LON, AZI, REGIME, QUALITY``) rather than guessing a
    brittle hard-coded link.
    """
    dest = dest or cache_dir(DATASET)
    if url is None:
        raise ValueError(
            "download(stress) requires an explicit `url` to the current World Stress Map CSV "
            f"(obtain it from {WSM_HOME} / GFZ Data Services — versioned per release). The CSV must "
            "carry LAT, LON, AZI (SHmax azimuth), REGIME, and QUALITY columns."
        )
    out = dest / "world_stress_map.csv"
    http_download(url, out, overwrite=overwrite, session=session)
    return Provenance(
        dataset=DATASET,
        title="World Stress Map database",
        version="release CSV",
        source_url=f"{WSM_HOME}/",
        license="CC-BY 4.0 (verify per release)",
        attribution="World Stress Map Project / GFZ German Research Centre for Geosciences",
        citation=(
            "Heidbach, O., et al. (2018). The World Stress Map database release 2016: "
            "Crustal stress pattern across scales. Tectonophysics, 744, 484-498. "
            "doi:10.1016/j.tecto.2018.07.007"
        ),
        files=[str(out.relative_to(dest.parent.parent.parent))],
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        notes="Per-record SHmax azimuth (0-360, orientation mod 180) + tectonic regime + quality.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# WSM CSV parsing (core deps only)
# ─────────────────────────────────────────────────────────────────────────────


def load_wsm(path: Path) -> pd.DataFrame:
    """Parse a WSM CSV into ``lat, lon, azi_deg, regime_code`` (case-insensitive column matching)."""
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", low_memory=False)
    cols = {c.lower().strip(): c for c in df.columns}
    lat_c = cols.get("lat") or cols.get("latitude")
    lon_c = cols.get("lon") or cols.get("longitude") or cols.get("long")
    azi_c = cols.get("azi") or cols.get("azimuth") or cols.get("shmax")
    reg_c = cols.get("regime") or cols.get("regm")
    if lat_c is None or lon_c is None:
        raise ValueError(f"WSM CSV at {path} has no recognizable LAT/LON columns (got {list(df.columns)})")
    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(df[lat_c], errors="coerce"),
            "lon": pd.to_numeric(df[lon_c], errors="coerce"),
            "azi_deg": pd.to_numeric(df[azi_c], errors="coerce") if azi_c else np.nan,
            "regime_code": (df[reg_c].map(regime_code) if reg_c else 0),
        }
    ).dropna(subset=["lat", "lon"])
    out["lon"] = ((out["lon"] + 180.0) % 360.0) - 180.0
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell feature extraction
# ─────────────────────────────────────────────────────────────────────────────


class StressEnricher:
    """Per-cell stress orientation + regime from the cached World Stress Map records.

    For a query cell, aggregates WSM records within ``radius_km``: SHmax azimuth via inverse-distance
    **circular** averaging on the doubled angle (stress axes are 180°-ambiguous), and the modal
    tectonic regime. Cells with no nearby record return all-``None`` for the orientation/regime and a
    zero record count.
    """

    def __init__(self, dest: Path | None = None, *, radius_km: float = 200.0) -> None:
        self.dest = dest or cache_dir(DATASET)
        self.radius_km = float(radius_km)
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        path = self.dest / "world_stress_map.csv"
        if not path.exists():
            candidates = sorted(self.dest.glob("*.csv"))
            if candidates:
                path = candidates[0]
            else:
                raise FileNotFoundError(
                    f"no World Stress Map CSV cached under {self.dest}. Run "
                    "caos_seismic.data.enrichers.stress.download(url=...) first."
                )
        self._df = load_wsm(path)
        return self._df

    def features_at(self, lat: float, lon: float, **_: Any) -> EnricherResult:
        """Return SHmax azimuth + modal regime covariates near ``(lat, lon)``."""
        df = self._load()
        d = haversine_km(lat, lon, df["lat"].to_numpy(), df["lon"].to_numpy())
        sel = d <= self.radius_km
        n = int(np.count_nonzero(sel))
        out: EnricherResult = {name: None for name in FEATURE_NAMES}
        out["stress_n_records"] = float(n)
        out["stress_is_thrust"] = 0.0
        out["stress_is_normal"] = 0.0
        out["stress_is_strikeslip"] = 0.0
        if n == 0:
            return out

        dd = d[sel]
        w = 1.0 / np.maximum(dd, 1.0)
        azi = df["azi_deg"].to_numpy()[sel]
        regimes = df["regime_code"].to_numpy()[sel]

        finite = np.isfinite(azi)
        if np.any(finite):
            out["shmax_azimuth_deg"] = _circular_mean_axis(azi[finite], w[finite])

        modal = _modal_regime(regimes, w)
        out["stress_regime_code"] = float(modal)
        out["stress_is_thrust"] = 1.0 if modal in _THRUST else 0.0
        out["stress_is_normal"] = 1.0 if modal in _NORMAL else 0.0
        out["stress_is_strikeslip"] = 1.0 if modal in _STRIKESLIP else 0.0
        return out


def _circular_mean_axis(azi_deg: np.ndarray, weights: np.ndarray) -> float:
    """Weighted circular mean of orientation data (mod 180°), returned in [0, 180).

    Stress axes have a 180° ambiguity, so the mean is taken on the **doubled** angle 2θ and halved
    back (standard axial statistics; Mardia & Jupp, 2000).
    """
    two_theta = np.radians(2.0 * azi_deg)
    s = float(np.sum(weights * np.sin(two_theta)))
    c = float(np.sum(weights * np.cos(two_theta)))
    mean = np.degrees(np.arctan2(s, c)) / 2.0
    return float(mean % 180.0)


def _modal_regime(codes: np.ndarray, weights: np.ndarray) -> int:
    """Weighted mode of the regime codes among nearby records (ignoring unknown=0 when possible)."""
    known = codes != 0
    use_codes = codes[known] if np.any(known) else codes
    use_w = weights[known] if np.any(known) else weights
    if use_codes.size == 0:
        return 0
    totals: dict[int, float] = {}
    for code, w in zip(use_codes.tolist(), use_w.tolist()):
        totals[int(code)] = totals.get(int(code), 0.0) + float(w)
    return max(totals.items(), key=lambda kv: kv[1])[0]


_DEFAULT_ENRICHER: StressEnricher | None = None


def features_at(lat: float, lon: float, **kwargs: Any) -> EnricherResult:
    """Module-level convenience: World Stress Map covariates at one coordinate."""
    global _DEFAULT_ENRICHER
    if _DEFAULT_ENRICHER is None:
        _DEFAULT_ENRICHER = StressEnricher()
    return _DEFAULT_ENRICHER.features_at(lat, lon, **kwargs)


__all__ = [
    "DATASET",
    "REGIME_CODES",
    "FEATURE_NAMES",
    "StressEnricher",
    "download",
    "features_at",
    "regime_code",
    "load_wsm",
    "DEG2KM",
]

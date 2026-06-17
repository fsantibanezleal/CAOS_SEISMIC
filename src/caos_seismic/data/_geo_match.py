"""Spatio-temporal event association — match catalog rows to a reference catalog.

Used by :func:`caos_seismic.data.clean.build_mw_anchor` to pair network events with their
Mw-homogenized ISC-GEM/GCMT counterparts (the overlap that anchors the TLS magnitude conversion).
Kept tiny and dependency-light (numpy/pandas + the shared haversine) so it stays on the core deps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..model._common import haversine_km


def associate_by_time_space(
    catalog: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    max_dt_s: float = 30.0,
    max_dist_km: float = 50.0,
) -> pd.DataFrame:
    """Greedy nearest-in-time-then-space matching of ``catalog`` rows to ``reference`` rows.

    For each catalog event (with a finite native ``mag``), the nearest unused reference event within
    ``max_dt_s`` origin-time **and** ``max_dist_km`` epicentral distance is its match; the match's
    moment magnitude ``mw`` becomes the anchor ``mw_ref``. Each reference event is consumed at most
    once (greedy, smallest time gap first), so two network detections do not both claim one reference.

    Returns a frame with columns ``m_native`` (the catalog native magnitude), ``mag_type`` (the
    *normalized* native family, via :func:`caos_seismic.data.clean.normalize_mag_type`), and
    ``mw_ref`` (the reference moment magnitude). Empty input → empty frame with those columns.
    """
    # Local import avoids a circular import at module load (clean imports this module).
    from .clean import normalize_mag_type

    out_cols = ["m_native", "mag_type", "mw_ref"]
    if catalog.empty or reference.empty:
        return pd.DataFrame(columns=out_cols)

    cat = catalog.dropna(subset=["time", "latitude", "longitude", "mag"]).copy()
    ref = reference.dropna(subset=["time", "latitude", "longitude"]).copy()
    # The reference must carry a moment magnitude (mw); fall back to native mag if it is already Mw.
    ref_mw = pd.to_numeric(ref["mw"], errors="coerce")
    ref_mw = ref_mw.fillna(pd.to_numeric(ref["mag"], errors="coerce"))
    ref = ref.assign(_mw=ref_mw).dropna(subset=["_mw"])
    if cat.empty or ref.empty:
        return pd.DataFrame(columns=out_cols)

    cat = cat.sort_values("time").reset_index(drop=True)
    ref = ref.sort_values("time").reset_index(drop=True)

    cat_t = cat["time"].to_numpy()
    ref_t = ref["time"].to_numpy()
    cat_lat = cat["latitude"].to_numpy(dtype=float)
    cat_lon = cat["longitude"].to_numpy(dtype=float)
    ref_lat = ref["latitude"].to_numpy(dtype=float)
    ref_lon = ref["longitude"].to_numpy(dtype=float)
    ref_mw_arr = ref["_mw"].to_numpy(dtype=float)
    cat_mag = cat["mag"].to_numpy(dtype=float)
    cat_fam = cat["mag_type"].map(normalize_mag_type).to_numpy()

    used = np.zeros(len(ref), dtype=bool)
    dt = np.timedelta64(int(max_dt_s * 1000), "ms")

    rows: list[dict] = []
    lo = 0  # left edge of the reference time window (both arrays are time-sorted)
    for i in range(len(cat)):
        t = cat_t[i]
        # Advance the window's left edge past references older than t - max_dt_s.
        while lo < len(ref) and (t - ref_t[lo]) > dt:
            lo += 1
        best_j = -1
        best_gap = None
        j = lo
        while j < len(ref) and (ref_t[j] - t) <= dt:
            if not used[j]:
                d = float(haversine_km(cat_lat[i], cat_lon[i], ref_lat[j], ref_lon[j]))
                if d <= max_dist_km:
                    gap = abs(ref_t[j] - t)
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best_j = j
            j += 1
        if best_j >= 0:
            used[best_j] = True
            rows.append(
                {
                    "m_native": float(cat_mag[i]),
                    "mag_type": str(cat_fam[i]),
                    "mw_ref": float(ref_mw_arr[best_j]),
                }
            )

    return pd.DataFrame(rows, columns=out_cols)

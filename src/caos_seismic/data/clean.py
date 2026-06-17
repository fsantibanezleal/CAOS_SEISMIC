"""Stage B — clean / homogenize the catalog.

This module turns the raw, multi-provider event stream from :mod:`caos_seismic.data.fetch` into the
single clean catalog every downstream stage consumes. It implements §3 step 2 of the methodology and
the **CLEAN / HOMOGENIZE** node of the pipeline DAG (``docs/data-and-pipelines.md`` §3–§4):

1. **Cross-provider dedupe by preferred id.** The same earthquake appears in ComCat, the regional
   network (CSN/SCEDC/…), EMSC and ISC under different ids. We keep one row per physical event,
   preferring the most authoritative provider and the row carrying the best (moment) magnitude.

2. **Magnitude homogenization to Mw.** Catalogs mix ``ML``/``mb``/``Ms``/``Md``/``Mw`` — different
   saturation, different physics — so mixing them silently distorts the Gutenberg–Richter tail and
   every rate forecast. We convert the native magnitude to a moment-magnitude equivalent with a
   **total-least-squares (orthogonal) regression**, *not* OLS, because **both axes carry measurement
   error** (the native magnitude *and* the reference Mw). The conversion is anchored on the
   **ISC-GEM / GCMT overlap** (events present in both a network and a Mw-homogenized reference) and is
   fit **per native magnitude type** (one ``ML→Mw`` line, one ``mb→Mw`` line, …). Native magnitudes
   are **kept** alongside the homogenized ``mw`` column — nothing is dropped.

Why TLS / orthogonal regression (not OLS)
-----------------------------------------
OLS minimizes vertical residuals and assumes the predictor (here the native magnitude) is error-free.
Both magnitudes are noisy estimates, so OLS attenuates the slope toward zero (the classic
errors-in-variables bias) and a wrong slope shifts the *whole* GR distribution. TLS minimizes the
**perpendicular** distance to the line, which is the maximum-likelihood fit when both variables have
(comparable) Gaussian error. The closed-form Deming/TLS slope used here is, for the error-variance
ratio ``δ = var(ε_y) / var(ε_x)`` (``δ = 1`` ⇒ orthogonal regression)::

    s_xx = Σ (x - x̄)² / (n-1),  s_yy = Σ (y - ȳ)² / (n-1),  s_xy = Σ (x - x̄)(y - ȳ) / (n-1)
    slope = ( s_yy - δ s_xx + sqrt( (s_yy - δ s_xx)² + 4 δ s_xy² ) ) / ( 2 s_xy )
    intercept = ȳ - slope · x̄

This is the standard orthogonal-regression solution (Deming 1943; Markovsky & Van Huffel 2007) and is
the same estimator the seismology literature uses for inter-magnitude conversions (e.g. Castellaro,
Mulargia & Kagan 2006, *GJI* 165, 245–255, doi:10.1111/j.1365-246X.2006.02902.x; Lolli & Gasperini
2012). We expose the fitted coefficients so the *clean* manifest can version the conversion — a wrong
conversion is silent and global, so it must be auditable.

Only the core deps (``numpy`` / ``pandas`` / ``scipy``) are used, so cleaning runs on the ComCat spine
without any heavy geophysics stack.

References
----------
* Deming, W. E. (1943). *Statistical Adjustment of Data.* Wiley (errors-in-both-variables fit).
* Castellaro, S., Mulargia, F. & Kagan, Y. Y. (2006). *GJI* 165, 245–255,
  doi:10.1111/j.1365-246X.2006.02902.x (TLS/orthogonal magnitude regression — both axes have error).
* Markovsky, I. & Van Huffel, S. (2007). *Signal Processing* 87(10), 2283–2302 (TLS overview).
* Methodology synthesis §3 step 2 (homogenize to Mw, TLS, ISC-GEM/GCMT anchor, keep native + Mw).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contracts import CATALOG_COLUMNS, validate_catalog

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Magnitude-type normalization
# ─────────────────────────────────────────────────────────────────────────────

#: Provider authority order for dedupe — earlier = preferred preferred-origin/id. ISC-GEM and GCMT
#: are Mw-homogenized references and win on magnitude quality; the regional network wins on location
#: completeness; ComCat is the global spine; EMSC is the independent cross-check.
DEFAULT_SOURCE_PRIORITY: tuple[str, ...] = (
    "isc_gem",
    "gcmt",
    "csn",
    "scedc",
    "ncedc",
    "geonet",
    "ingv",
    "isc",
    "usgs_comcat",
    "emsc",
)


def normalize_mag_type(mag_type: object) -> str:
    """Collapse a raw ``magType`` string to a canonical family key (``mw``/``mb``/``ml``/``ms``/``md``).

    Providers spell the same scale many ways (``Mww``, ``mwc``, ``MLv``, ``mb_Lg`` …). We map each to
    the family that shares a saturation/conversion behaviour. Moment magnitudes (anything starting
    ``mw``) collapse to ``"mw"`` and need **no** conversion. Unknown/empty types return ``"unknown"``
    and are treated conservatively (no conversion applied; native value carried into ``mw`` only if it
    is already moment magnitude).
    """
    if not isinstance(mag_type, str):
        return "unknown"
    mt = mag_type.strip().lower()
    if not mt:
        return "unknown"
    if mt.startswith("mw") or mt in {"w", "mwc", "mww", "mwb", "mwr", "mwp"}:
        return "mw"
    if mt.startswith("mb"):  # mb, mb_Lg, mbLg, mB
        return "mb"
    if mt.startswith("ms"):  # Ms, Ms_20, MS, mB? — surface-wave
        return "ms"
    if mt.startswith("ml") or mt in {"m", "ml(maxc)"}:  # ML, MLv, ml, MLr
        return "ml"
    if mt.startswith("md") or mt.startswith("mc"):  # duration / coda
        return "md"
    if mt.startswith("me"):  # energy magnitude — rare; leave as its own family
        return "me"
    return "unknown"


def _is_moment(mag_type_family: str) -> bool:
    return mag_type_family == "mw"


# ─────────────────────────────────────────────────────────────────────────────
# Total-least-squares (orthogonal / Deming) regression
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TLSFit:
    """A fitted orthogonal (total-least-squares) ``native → Mw`` magnitude conversion line.

    Attributes
    ----------
    slope, intercept:
        The line ``mw = slope * m_native + intercept``.
    delta:
        Error-variance ratio ``var(ε_native) / var(ε_mw)`` assumed in the Deming fit. ``1.0`` is the
        orthogonal/TLS case (equal error on both axes) — the default the methodology specifies.
    n:
        Number of overlap pairs the line was fit on.
    rms:
        Root-mean-square orthogonal residual (a fit-quality diagnostic recorded in the manifest).
    mag_type:
        Native magnitude family the line applies to (``"ml"``, ``"mb"``, …).
    """

    slope: float
    intercept: float
    delta: float
    n: int
    rms: float
    mag_type: str = ""

    def apply(self, m_native: np.ndarray | float) -> np.ndarray | float:
        """Convert native magnitudes to Mw with this line."""
        return self.slope * np.asarray(m_native, dtype=float) + self.intercept

    def to_dict(self) -> dict:
        """Serializable coefficients for the clean manifest (versions the conversion)."""
        return {
            "mag_type": self.mag_type,
            "slope": self.slope,
            "intercept": self.intercept,
            "delta": self.delta,
            "n": self.n,
            "rms": self.rms,
            "method": "total_least_squares_deming",
        }


def tls_regression(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    *,
    delta: float = 1.0,
    mag_type: str = "",
) -> TLSFit:
    """Total-least-squares (Deming/orthogonal) fit of ``y ≈ slope·x + intercept``.

    Both ``x`` (native magnitude) and ``y`` (reference Mw) are noisy, so we minimize the
    **perpendicular** distance to the line rather than the vertical OLS residual. With the
    error-variance ratio ``δ = var(ε_x)/var(ε_y)`` the closed-form Deming slope is::

        slope = ( s_yy - δ s_xx + sqrt( (s_yy - δ s_xx)² + 4 δ s_xy² ) ) / ( 2 s_xy )

    and ``δ = 1`` recovers ordinary orthogonal (TLS) regression — the methodology default, since the
    native and moment magnitudes have comparable scatter. Returns a :class:`TLSFit`.

    Raises
    ------
    ValueError
        If fewer than 3 finite, paired samples are available, or the points are degenerate
        (``s_xy == 0`` — no covariance to define a slope).
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    n = int(xa.size)
    if n < 3:
        raise ValueError(f"TLS regression needs >= 3 paired samples, got {n}")
    if delta <= 0:
        raise ValueError(f"delta (error-variance ratio) must be > 0, got {delta}")

    x_bar = float(xa.mean())
    y_bar = float(ya.mean())
    dx = xa - x_bar
    dy = ya - y_bar
    # Sample (co)variances with the (n-1) normalization; the slope is scale-invariant in it but we
    # keep it explicit so the formula matches the docstring exactly.
    denom = n - 1
    s_xx = float(np.dot(dx, dx) / denom)
    s_yy = float(np.dot(dy, dy) / denom)
    s_xy = float(np.dot(dx, dy) / denom)
    if s_xy == 0.0:
        raise ValueError(
            "degenerate TLS fit: zero covariance between native and reference magnitudes "
            "(no usable inter-magnitude relation)"
        )

    discriminant = (s_yy - delta * s_xx) ** 2 + 4.0 * delta * s_xy * s_xy
    slope = (s_yy - delta * s_xx + math.sqrt(discriminant)) / (2.0 * s_xy)
    intercept = y_bar - slope * x_bar

    # Orthogonal RMS residual: perpendicular distance from each point to the line.
    perp = (slope * xa - ya + intercept) / math.sqrt(slope * slope + 1.0)
    rms = float(np.sqrt(np.mean(perp * perp)))
    return TLSFit(
        slope=float(slope),
        intercept=float(intercept),
        delta=float(delta),
        n=n,
        rms=rms,
        mag_type=mag_type,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Building the ISC-GEM / GCMT overlap anchor and fitting one line per native type
# ─────────────────────────────────────────────────────────────────────────────


def build_mw_anchor(
    catalog: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    max_dt_s: float = 30.0,
    max_dist_km: float = 50.0,
) -> pd.DataFrame:
    """Match network events to a Mw-homogenized reference (ISC-GEM/GCMT) into ``(m_native, mw_ref)`` pairs.

    Each event in ``catalog`` (with native ``mag``/``mag_type``) is matched to the nearest reference
    event in **time then space** (a reference event is consumed at most once). The matched reference's
    moment magnitude becomes the anchor ``mw_ref`` for that native reading. The returned frame has the
    columns the per-type TLS fit needs: ``m_native``, ``mag_type`` (normalized family), ``mw_ref``.

    Parameters
    ----------
    catalog:
        Network catalog with native magnitudes (the rows we want to convert).
    reference:
        Mw-homogenized reference catalog (ISC-GEM and/or GCMT; ``mw`` populated, ``mag_type`` moment).
    max_dt_s, max_dist_km:
        Association gates — an event matches a reference only if within this origin-time and epicentral
        distance. Defaults are deliberately tight (global Mw references are sparse and well-located).

    Notes
    -----
    Pure-numpy ``O(N·M)`` association is fine here: the Mw reference for a region is at most a few
    thousand events. A KD-tree refinement is deferred until profiling shows it is needed.
    """
    from ._geo_match import associate_by_time_space  # local import keeps module surface small

    return associate_by_time_space(
        catalog, reference, max_dt_s=max_dt_s, max_dist_km=max_dist_km
    )


def fit_conversions(
    anchor_pairs: pd.DataFrame,
    *,
    delta: float = 1.0,
    min_pairs: int = 20,
) -> dict[str, TLSFit]:
    """Fit one TLS ``native → Mw`` line **per native magnitude family** from the overlap pairs.

    ``anchor_pairs`` is the output of :func:`build_mw_anchor` (columns ``m_native``, ``mag_type``,
    ``mw_ref``). A line is fit only for families with at least ``min_pairs`` overlap samples; families
    with too few pairs are skipped (the homogenizer then leaves those native readings without a
    conversion and flags them, rather than fitting a noisy, untrustworthy line).

    Moment-magnitude readings (family ``"mw"``) are intentionally **not** fit — they are already Mw.
    """
    fits: dict[str, TLSFit] = {}
    if anchor_pairs.empty:
        return fits
    for fam, grp in anchor_pairs.groupby("mag_type"):
        if _is_moment(str(fam)) or str(fam) == "unknown":
            continue
        if len(grp) < min_pairs:
            logger.info(
                "skip %s→Mw conversion: only %d overlap pairs (< %d)", fam, len(grp), min_pairs
            )
            continue
        try:
            fit = tls_regression(
                grp["m_native"].to_numpy(),
                grp["mw_ref"].to_numpy(),
                delta=delta,
                mag_type=str(fam),
            )
        except ValueError as exc:
            logger.warning("skip %s→Mw conversion: %s", fam, exc)
            continue
        fits[str(fam)] = fit
        logger.info(
            "fitted %s→Mw: mw = %.4f·m %+.4f  (n=%d, rms=%.3f)",
            fam, fit.slope, fit.intercept, fit.n, fit.rms,
        )
    return fits


# ─────────────────────────────────────────────────────────────────────────────
# Homogenization to Mw
# ─────────────────────────────────────────────────────────────────────────────


def homogenize_to_mw(
    catalog: pd.DataFrame,
    conversions: dict[str, TLSFit] | None = None,
    *,
    keep_native: bool = True,
) -> pd.DataFrame:
    """Fill the ``mw`` column from native magnitudes using the per-type TLS conversions.

    Rule per row:

    * if the native type is already moment magnitude (``mw`` family) → ``mw = mag`` (identity);
    * else if a fitted conversion exists for the native family → ``mw = slope·mag + intercept``;
    * else → ``mw`` is left ``NaN`` (no trustworthy conversion; the row is flagged, never silently
      mis-converted).

    The native ``mag`` and ``mag_type`` columns are **preserved** (``keep_native=True``) — the
    contract requires both the native value+type and the homogenized ``mw`` to survive (a wrong or
    missing conversion must remain auditable). Returns a **new** validated DataFrame.

    Parameters
    ----------
    catalog:
        Catalog with ``mag``/``mag_type`` populated (raw or deduped).
    conversions:
        Mapping ``family → TLSFit`` from :func:`fit_conversions`. If ``None`` or empty, only the
        already-moment magnitudes are carried into ``mw`` (everything else stays ``NaN``).
    """
    df = catalog.copy()
    fam = df["mag_type"].map(normalize_mag_type)
    mag = pd.to_numeric(df["mag"], errors="coerce")

    mw = pd.Series(np.nan, index=df.index, dtype=float)
    # 1) already moment magnitude → identity.
    is_mw = fam.eq("mw")
    mw.loc[is_mw] = mag.loc[is_mw]

    # 2) convert each non-moment family that has a fitted line.
    if conversions:
        for family, fit in conversions.items():
            sel = fam.eq(family) & ~is_mw & mag.notna()
            if sel.any():
                mw.loc[sel] = fit.apply(mag.loc[sel].to_numpy())

    df["mw"] = mw.astype(float)
    if not keep_native:  # never used by the pipeline, but explicit
        df = df.drop(columns=["mag", "mag_type"], errors="ignore")
    return validate_catalog(df)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-provider dedupe
# ─────────────────────────────────────────────────────────────────────────────


def dedupe_events(
    catalog: pd.DataFrame,
    *,
    source_priority: tuple[str, ...] = DEFAULT_SOURCE_PRIORITY,
    max_dt_s: float = 16.0,
    max_dist_km: float = 100.0,
) -> pd.DataFrame:
    """Collapse duplicate detections of the same physical earthquake to one row.

    Two passes, in order:

    1. **Exact id** — rows sharing ``event_id`` are duplicates (e.g. overlapping fetch tiles); keep
       the highest-priority provider's row.
    2. **Spatio-temporal** — across providers an earthquake has *different* ids, so we cluster rows
       whose origin times are within ``max_dt_s`` and epicentres within ``max_dist_km`` and keep one
       representative per cluster.

    The kept representative is chosen by **provider authority** (``source_priority``) first, then by
    magnitude quality (a moment-magnitude reading beats a local-magnitude reading), so the surviving
    row carries the best location *and* the best magnitude available for that event. This is the
    "preferred id / preferred origin" dedupe of §3 step 1.

    The input is sorted by time internally; the output is sorted by time and re-indexed.
    """
    if catalog.empty:
        return validate_catalog(catalog.copy())

    df = catalog.copy()
    df = df.dropna(subset=["time", "latitude", "longitude"]).reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Rank each row: lower is better. Provider authority, then moment-magnitude preference.
    prio = {s: i for i, s in enumerate(source_priority)}
    df["_src_rank"] = df["source"].map(lambda s: prio.get(str(s), len(source_priority)))
    fam = df["mag_type"].map(normalize_mag_type)
    df["_mag_rank"] = np.where(fam.eq("mw"), 0, 1)  # prefer rows already in Mw
    df["_has_mag"] = (~pd.to_numeric(df["mag"], errors="coerce").isna()).astype(int)

    # Pass 1 — exact id.
    df = df.sort_values(
        ["event_id", "_src_rank", "_mag_rank", "_has_mag"],
        ascending=[True, True, True, False],
    )
    df = df.drop_duplicates(subset="event_id", keep="first")

    # Pass 2 — spatio-temporal clustering across providers.
    df = df.sort_values("time").reset_index(drop=True)
    keep_mask = _spatiotemporal_dedupe_mask(df, max_dt_s=max_dt_s, max_dist_km=max_dist_km)
    df = df.loc[keep_mask].copy()

    df = df.drop(columns=["_src_rank", "_mag_rank", "_has_mag"], errors="ignore")
    df = df.sort_values("time").reset_index(drop=True)
    return validate_catalog(df)


def _spatiotemporal_dedupe_mask(
    df: pd.DataFrame, *, max_dt_s: float, max_dist_km: float
) -> np.ndarray:
    """Boolean keep-mask: one representative per spatio-temporal cluster (df sorted by time).

    Sweeps the time-sorted catalog. For each not-yet-assigned event it gathers all later events within
    ``max_dt_s`` (the sweep can stop early on time) and ``max_dist_km``, picks the best-ranked member
    of that cluster as the survivor, and marks the rest as duplicates. ``O(N·k)`` with ``k`` the local
    temporal neighbourhood — cheap for daily regional catalogs.
    """
    from ..model._common import haversine_km

    n = len(df)
    keep = np.ones(n, dtype=bool)
    assigned = np.zeros(n, dtype=bool)
    times = df["time"].to_numpy()
    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)
    rank = list(zip(df["_src_rank"].to_numpy(), df["_mag_rank"].to_numpy(), -df["_has_mag"].to_numpy()))
    dt = np.timedelta64(int(max_dt_s * 1000), "ms")

    for i in range(n):
        if assigned[i]:
            continue
        # Candidate window forward in time (sorted), bounded by max_dt_s.
        j = i + 1
        cluster = [i]
        while j < n and (times[j] - times[i]) <= dt:
            if not assigned[j]:
                d = float(haversine_km(lat[i], lon[i], lat[j], lon[j]))
                if d <= max_dist_km:
                    cluster.append(j)
            j += 1
        if len(cluster) == 1:
            assigned[i] = True
            continue
        # Keep the best-ranked member; drop the rest.
        survivor = min(cluster, key=lambda idx: rank[idx])
        for idx in cluster:
            assigned[idx] = True
            if idx != survivor:
                keep[idx] = False
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Multi-provider merge (the input to dedupe)
# ─────────────────────────────────────────────────────────────────────────────


def merge_providers(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate per-provider catalogs into the single merged stream the dedupe consumes.

    The global pipeline pulls several providers independently — the worldwide ComCat spine, each
    country **view**'s regional network (CSN/SCEDC/GeoNet/INGV), the EMSC cross-check, and the
    ISC-GEM/GCMT Mw anchors. Each returns a CATALOG_COLUMNS frame; this stacks them (dropping empties),
    coerces ``time`` to UTC, and returns one validated frame ready for :func:`dedupe_events`. It does
    **not** dedupe — that is dedupe's job, after the merge, using the provider authority order.

    Empty / all-empty inputs return a typed-empty CATALOG_COLUMNS frame.
    """
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        empty = pd.DataFrame({c: pd.Series(dtype="object") for c in CATALOG_COLUMNS})
        empty["time"] = pd.to_datetime(empty["time"], utc=True)
        for c in ("latitude", "longitude", "depth_km", "mag", "mw"):
            empty[c] = pd.to_numeric(empty[c], errors="coerce")
        return validate_catalog(empty[list(CATALOG_COLUMNS)])
    for f in usable:
        validate_catalog(f)
    merged = pd.concat([f[list(CATALOG_COLUMNS)] for f in usable], ignore_index=True)
    merged["time"] = pd.to_datetime(merged["time"], utc=True)
    return validate_catalog(merged)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL space-time Mc handling hook
# ─────────────────────────────────────────────────────────────────────────────


def global_mc_grid(
    catalog: pd.DataFrame,
    *,
    cell_deg: float = 5.0,
    window_days: float = 365.0,
    step_days: float | None = None,
    dm: float = 0.1,
    correction: float = 0.2,
    min_events: int = 50,
    global_default: float | None = 4.5,
) -> pd.DataFrame:
    """Estimate ``Mc(x, y, t)`` on a coarse GLOBAL space-time grid — the global completeness hook.

    A worldwide catalog has wildly non-stationary completeness: ``Mc`` differs by network era and by
    region (a dense regional view sees ``Mc≈1``; the open ocean only ``Mc≈4.5+``). A single global
    ``Mc`` injects fake non-stationarity into the Gutenberg–Richter tail and every rate, so the global
    pipeline estimates ``Mc`` **per spatial cell and per time epoch** (synthesis §3 step 1). This hook
    bins the catalog into ``cell_deg`` lat/lon cells and runs the existing rolling-time estimator
    (:func:`caos_seismic.catalog.completeness.rolling_mc`, MAXC + GFT cross-check) inside each cell.

    The result is the per-cell-per-epoch ``Mc`` table that stage (C) cuts events below before
    declustering/feature-building. It is a *hook*: a coarse, leakage-safe (right-labelled windows)
    first cut that the full stage (C) Mc artifact refines per region view — here we provide the global
    field so the spine is never cut at a single planet-wide ``Mc``.

    Parameters
    ----------
    catalog:
        A clean (Mw-homogenized) global catalog with ``time``/``latitude``/``longitude``/``mw``.
    cell_deg:
        Spatial cell size in degrees (default 5° — coarse on purpose; the global field is smooth and a
        finer grid starves cells of the ``min_events`` an Mc estimate needs).
    window_days, step_days, dm, correction, min_events:
        Passed through to the per-cell rolling Mc estimator.
    global_default:
        Conservative worldwide floor used where a cell-epoch has too few events to estimate ``Mc``
        (default 4.5 — the same homogeneity floor the global ComCat spine is pulled at).

    Returns
    -------
    DataFrame with columns ``cell_lat``, ``cell_lon`` (the cell's lower-left corner), ``window_start``,
    ``window_end``, ``mc``, ``maxc_raw``, ``n_events``, ``method``. Empty catalog → empty frame with
    those columns. Core deps only (the estimator is numpy/pandas/scipy).
    """
    from ..catalog.completeness import rolling_mc

    cols = [
        "cell_lat", "cell_lon", "window_start", "window_end",
        "mc", "maxc_raw", "n_events", "method",
    ]
    if catalog.empty:
        return pd.DataFrame(columns=cols)

    df = catalog.dropna(subset=["time", "latitude", "longitude", "mw"]).copy()
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    # Floor each event into its grid cell (the cell's lower-left corner identifies it).
    df["cell_lat"] = (np.floor(lat / cell_deg) * cell_deg).astype(float)
    df["cell_lon"] = (np.floor(lon / cell_deg) * cell_deg).astype(float)

    out_rows: list[pd.DataFrame] = []
    for (clat, clon), grp in df.groupby(["cell_lat", "cell_lon"], sort=True):
        roll = rolling_mc(
            grp,
            window_days=window_days,
            step_days=step_days,
            dm=dm,
            correction=correction,
            min_events=min_events,
            regional_default=global_default,
            mag_col="mw",
            time_col="time",
        )
        if roll.empty:
            continue
        roll = roll.copy()
        roll.insert(0, "cell_lon", float(clon))
        roll.insert(0, "cell_lat", float(clat))
        out_rows.append(roll)

    if not out_rows:
        return pd.DataFrame(columns=cols)
    grid = pd.concat(out_rows, ignore_index=True)
    return grid[cols]


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end clean
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CleanResult:
    """The cleaned catalog plus the provenance the clean manifest records.

    Attributes
    ----------
    catalog:
        Deduped, Mw-homogenized catalog (native ``mag``/``mag_type`` kept; ``mw`` filled where a
        trustworthy conversion exists).
    conversions:
        The fitted ``family → TLSFit`` lines (serialize via :meth:`TLSFit.to_dict` into the manifest).
    stats:
        Dedupe + conversion statistics (counts in/out, n unconverted, per-type fit quality).
    """

    catalog: pd.DataFrame
    conversions: dict[str, TLSFit] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def manifest_outputs(self) -> dict:
        """The conversion + dedupe provenance block for a ``stage="clean"`` :class:`Manifest`."""
        return {
            "conversions": {fam: fit.to_dict() for fam, fit in self.conversions.items()},
            "stats": self.stats,
        }


def clean_catalog(
    catalog: pd.DataFrame,
    *,
    reference: pd.DataFrame | None = None,
    conversions: dict[str, TLSFit] | None = None,
    source_priority: tuple[str, ...] = DEFAULT_SOURCE_PRIORITY,
    dedupe_dt_s: float = 16.0,
    dedupe_dist_km: float = 100.0,
    tls_delta: float = 1.0,
    min_pairs: int = 20,
    anchor_dt_s: float = 30.0,
    anchor_dist_km: float = 50.0,
) -> CleanResult:
    """Run the full clean stage: cross-provider dedupe → Mw homogenization (TLS) → validation.

    This is the function ``scripts/build-features`` calls for stage (B). It:

    1. de-duplicates the merged multi-provider catalog by preferred id + spatio-temporal proximity
       (:func:`dedupe_events`);
    2. obtains the ``native → Mw`` TLS conversions — either the pre-fitted ``conversions`` passed in,
       or fit fresh from the ISC-GEM/GCMT ``reference`` overlap (:func:`build_mw_anchor` +
       :func:`fit_conversions`); both axes have error, so the fit is orthogonal/TLS, never OLS;
    3. fills ``mw`` per row (:func:`homogenize_to_mw`), keeping the native ``mag``/``mag_type``;
    4. returns a :class:`CleanResult` with the validated catalog, the fitted lines, and stats for the
       clean manifest.

    Parameters
    ----------
    catalog:
        The merged raw catalog across providers (ComCat spine + regional + cross-check).
    reference:
        A Mw-homogenized reference (ISC-GEM and/or GCMT) used to fit the conversions when
        ``conversions`` is not supplied. If both are ``None``, only already-moment magnitudes get a
        ``mw`` value (everything else stays ``NaN`` and is counted in ``stats``).
    conversions:
        Pre-fitted conversions (e.g. versioned from a previous run's manifest) to reuse verbatim — the
        production path so the daily delta is converted with the *same* line as the base catalog.
    """
    n_in = int(len(catalog))
    deduped = dedupe_events(
        catalog,
        source_priority=source_priority,
        max_dt_s=dedupe_dt_s,
        max_dist_km=dedupe_dist_km,
    )
    n_dedup = int(len(deduped))

    fits: dict[str, TLSFit] = dict(conversions) if conversions else {}
    if not fits and reference is not None and not reference.empty:
        # Fit the conversion from the *native-magnitude network rows of the full input* matched to the
        # reference — NOT the deduped catalog. Dedupe replaces a network ML reading with its better
        # (reference Mw) row, which would erase exactly the native→Mw overlap pairs the line needs.
        # We exclude rows that are themselves from the reference's source(s) so we only learn the
        # network → reference relation.
        ref_sources = set(reference["source"].dropna().astype(str).unique())
        network = catalog.loc[~catalog["source"].astype(str).isin(ref_sources)].copy()
        pairs = build_mw_anchor(
            network, reference, max_dt_s=anchor_dt_s, max_dist_km=anchor_dist_km
        )
        fits = fit_conversions(pairs, delta=tls_delta, min_pairs=min_pairs)

    clean = homogenize_to_mw(deduped, fits, keep_native=True)

    fam = clean["mag_type"].map(normalize_mag_type)
    n_unconverted = int(clean["mw"].isna().sum())
    stats = {
        "n_in": n_in,
        "n_after_dedupe": n_dedup,
        "n_duplicates_removed": n_in - n_dedup,
        "n_with_mw": int(clean["mw"].notna().sum()),
        "n_unconverted": n_unconverted,
        "native_type_counts": {
            str(k): int(v) for k, v in fam.value_counts(dropna=False).items()
        },
        "conversion_families": sorted(fits.keys()),
    }
    if n_unconverted:
        logger.warning(
            "%d/%d events left without a Mw value (no trustworthy conversion for their type)",
            n_unconverted, len(clean),
        )
    return CleanResult(catalog=clean, conversions=fits, stats=stats)


# ─────────────────────────────────────────────────────────────────────────────
# Clean-catalog store (gitignored Parquet) — the handoff to inference/back-analysis
# ─────────────────────────────────────────────────────────────────────────────


def clean_catalog_path(region_id: str, base_dir=None):
    """Path of the cleaned-catalog Parquet store for a region (``data/clean/<region>.parquet``).

    The store is gitignored — only configs + manifests + code are versioned; the catalog is
    rebuildable from them. Kept here so the fetch→clean stage and the inference loader agree on
    one location.
    """
    from pathlib import Path

    from ..config import REPO_ROOT

    base = Path(base_dir) if base_dir is not None else (REPO_ROOT / "data" / "clean")
    return base / f"{region_id}.parquet"


def save_clean_catalog(catalog: pd.DataFrame, region_id: str, base_dir=None):
    """Write the cleaned, Mw-homogenized catalog to the gitignored Parquet store; return the path.

    Only the contract columns are persisted (extra feature columns are rebuildable). The directory
    is created on demand. Used by :func:`caos_seismic.catalog.features.run_build_features`.
    """
    validate_catalog(catalog)
    path = clean_catalog_path(region_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in CATALOG_COLUMNS if c in catalog.columns]
    catalog[cols].to_parquet(path, index=False)
    logger.info("wrote clean catalog store: %s (%d events)", path, len(catalog))
    return path


def load_clean_catalog(region, base_dir=None) -> pd.DataFrame:
    """Load the cleaned catalog Parquet store for a region (the loader the daily inference calls).

    ``region`` may be a :class:`~caos_seismic.contracts.Region` or a region id string. Raises a
    clear :class:`FileNotFoundError` (pointing at ``caos-seismic fetch`` + ``build-features``) when
    the store has not been built yet — the inference stage surfaces that as an actionable message.
    Returns a validated catalog DataFrame with a tz-aware UTC ``time`` column.
    """
    region_id = region.id if hasattr(region, "id") else str(region)
    path = clean_catalog_path(region_id, base_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"no cleaned catalog store at {path}. Build it first:\n"
            f"    caos-seismic fetch --region {region_id}\n"
            f"    caos-seismic build-features --region {region_id}"
        )
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return validate_catalog(df)

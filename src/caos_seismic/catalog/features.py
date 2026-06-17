"""Stage (B+C) entry point — clean + completeness + declustering, into the inference-ready store.

This is the module the ``caos-seismic build-features`` command delegates to. It chains the already
built primitives into one runnable stage and writes the cleaned catalog store the daily inference
loads:

1. **Load** the raw multi-provider catalog written by :mod:`caos_seismic.data.fetch`
   (``data/raw/comcat_<region>.parquet``; gitignored).
2. **Clean / homogenize** to Mw via :func:`caos_seismic.data.clean.clean_catalog` (cross-provider
   dedupe + total-least-squares ``native → Mw`` conversion; native columns kept).
3. **Completeness + b-value** via :func:`caos_seismic.catalog.completeness.mc_estimate` and
   :func:`~caos_seismic.catalog.completeness.aki_utsu_b_value` (MAXC + GFT cross-check; the b-value
   is *estimated*, never hard-coded — methodology §3).
4. **Dual catalog** via :func:`caos_seismic.catalog.decluster.dual_catalog` — the declustered
   background (for the smoothed null) and the full un-declustered catalog (for ETAS / scoring). The
   **dual-catalog rule** is enforced by the primitive; this stage only records which view is which.
5. **Persist** the cleaned, below-Mc-cut catalog to the gitignored store
   (:func:`caos_seismic.data.clean.save_clean_catalog`) and write a ``stage="mc_decluster"``
   provenance :class:`~caos_seismic.contracts.Manifest`.

Only the core deps are needed (numpy / pandas / scipy / pyarrow / pydantic). The heavy
declustering ZBZ pass is the same core-only code as :mod:`caos_seismic.catalog.decluster`; nothing
here imports an optional science dependency. Inputs/outputs are versioned by manifest, not by
committing the catalog — the catalog is rebuildable from the configs + manifests + code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import REPO_ROOT, load, load_region
from ..contracts import Region, validate_catalog
from ..data.clean import clean_catalog, save_clean_catalog
from ..inference.provenance import build_manifest, snapshot_id, write_manifest
from .completeness import aki_utsu_b_value, mc_estimate
from .decluster import dual_catalog

logger = logging.getLogger(__name__)


def _raw_catalog_path(region_id: str) -> Path:
    """Path of the raw ComCat store written by the fetch stage (``data/raw/comcat_<region>.parquet``)."""
    return REPO_ROOT / "data" / "raw" / f"comcat_{region_id}.parquet"


def _load_raw(region: Region, catalog: pd.DataFrame | None) -> pd.DataFrame:
    """Return the raw catalog: the one passed in, or the fetched Parquet store (clear error if absent)."""
    if catalog is not None:
        return validate_catalog(catalog)
    path = _raw_catalog_path(region.id)
    if not path.exists():
        raise FileNotFoundError(
            f"no raw catalog at {path}. Run `caos-seismic fetch --region {region.id}` first "
            "(it writes the gitignored raw store), or pass `catalog=` directly."
        )
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return validate_catalog(df)


def run_build_features(
    *,
    region: Region | str = "chile",
    catalog: pd.DataFrame | None = None,
    reference: pd.DataFrame | None = None,
    write_store: bool = True,
    write_manifest_file: bool = True,
) -> dict[str, Any]:
    """Stage entry the ``caos-seismic build-features`` command calls.

    Parameters
    ----------
    region:
        A :class:`Region` or region id (``configs/region.<id>.yaml``).
    catalog:
        Optional in-memory raw catalog (skips loading the Parquet store — used by ``check`` and
        tests so the stage runs offline).
    reference:
        Optional Mw-homogenized reference (ISC-GEM/GCMT) to fit the TLS conversions; when absent only
        already-moment magnitudes get an ``mw`` value (the ComCat spine is mostly Mw/mb at M≥3.5).
    write_store, write_manifest_file:
        Persist the cleaned store / provenance manifest (both default on; disabled by tests).

    Returns
    -------
    dict
        ``{"region", "n_in", "n_clean", "n_below_mc_cut", "mc", "b_value", "b_uncertainty",
        "n_background", "n_conditional", "declustering", "clean_store", "manifest"}`` — the CLI
        prints this summary.
    """
    reg = load_region(region) if isinstance(region, str) else region
    completeness_cfg = load("completeness")
    declustering_cfg = load("declustering")
    grid_cfg = load("grid")

    mc_cfg = completeness_cfg.get("mc", {})
    dm = float(grid_cfg.get("fit", {}).get("mag_bin", 0.1))
    correction = float(mc_cfg.get("maxc_correction", 0.2))
    min_events = int(mc_cfg.get("min_events", 50))
    regional_default = float(mc_cfg.get("regional_default", 3.5))

    raw = _load_raw(reg, catalog)
    n_in = int(len(raw))

    # 2) Clean + homogenize to Mw (TLS conversion; native columns preserved).
    clean_result = clean_catalog(raw, reference=reference)
    clean = clean_result.catalog

    # 3) Completeness + b-value on events with a usable Mw (estimated, never hard-coded).
    mw = pd.to_numeric(clean["mw"], errors="coerce")
    with_mw = clean.loc[mw.notna()].copy()
    mc_est = mc_estimate(
        with_mw["mw"].to_numpy(),
        dm=dm,
        correction=correction,
        min_events=min_events,
        regional_default=regional_default,
    )
    mc = float(mc_est.mc)
    try:
        b_est = aki_utsu_b_value(with_mw["mw"].to_numpy(), mc, dm=dm)
        b_value, b_unc = float(b_est.b), float(b_est.b_uncertainty)
    except ValueError as exc:  # thin / degenerate FMD — keep going with a flagged default
        logger.warning("b-value estimation failed (%s); recording b=1.0 as a flagged placeholder", exc)
        b_value, b_unc = 1.0, float("nan")

    # Cut to the completeness threshold — every downstream rate is defined for events >= Mc.
    complete = with_mw.loc[with_mw["mw"] >= mc - 1e-9].reset_index(drop=True)
    complete = validate_catalog(complete)
    n_clean = int(len(complete))

    # 4) Dual catalog (the dual-catalog rule). The ZBZ pass is O(N^2); skip it for very large
    #    catalogs where only the Gardner–Knopoff background is needed downstream.
    compute_nnd = n_clean <= 20000
    dual = dual_catalog(
        complete,
        b=b_value,
        df_fractal=float(declustering_cfg.get("features", {}).get("fractal_dimension", 1.6)),
        q=float(declustering_cfg.get("features", {}).get("q", 0.5)),
        compute_nnd=compute_nnd,
    )
    n_background = int(len(dual.background))

    # 5) Persist the cleaned (un-declustered, scoring + conditional) catalog + provenance manifest.
    store_path = None
    if write_store:
        store_path = str(save_clean_catalog(dual.full, reg.id))

    issued_at = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest_path = None
    if write_manifest_file:
        manifest = build_manifest(
            stage="mc_decluster",
            region_id=reg.id,
            t_issue=issued_at,
            input_snapshot_id=snapshot_id(dual.full, reg.id, issued_at),
            mc_grid_version=f"maxc+{correction:g}/{mc_est.method}",
            declustering=str(declustering_cfg.get("background", {}).get("method", "gardner_knopoff")),
            model_name="catalog_hygiene",
            model_version="0.1.0",
            model_params={"mc": mc, "b_value": b_value, "delta_m": dm},
            inputs={
                "n_raw": n_in,
                "n_after_clean": int(len(clean)),
                "clean_stats": clean_result.stats,
            },
            outputs={
                "n_complete": n_clean,
                "n_background": n_background,
                "n_conditional": n_clean,
                "mc": mc,
                "b_value": b_value,
                "b_uncertainty": b_unc,
                "clean_store": store_path,
            },
            stats={"decluster": dual.manifest_stats()},
        )
        manifest_path = str(write_manifest(manifest))

    return {
        "region": reg.id,
        "n_in": n_in,
        "n_clean": n_clean,
        "n_below_mc_cut": int(len(with_mw)) - n_clean,
        "mc": mc,
        "b_value": b_value,
        "b_uncertainty": b_unc,
        "n_background": n_background,
        "n_conditional": n_clean,
        "declustering": "gardner_knopoff" + ("+zaliapin_ben_zion" if compute_nnd else ""),
        "clean_store": store_path,
        "manifest": manifest_path,
    }

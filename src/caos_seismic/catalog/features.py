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
from ..contracts import Cell, Region, validate_catalog
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


# ─────────────────────────────────────────────────────────────────────────────
# Global context feature matrix — join every enricher onto the forecast grid
# ─────────────────────────────────────────────────────────────────────────────


def build_context_features(
    grid: "list[Cell] | pd.DataFrame",
    *,
    enrichers: "list[str] | None" = None,
    t_issue: "str | pd.Timestamp | None" = None,
    write_store: bool = False,
    region_id: str | None = None,
    enricher_kwargs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Join the global geophysical enrichers onto the forecast grid → the context feature matrix.

    This is the bridge from the **global static context** to the model. Each cell of the forecast
    grid is a *view* into the worldwide covariate field (slab geometry, faults, plate boundaries,
    geodetic strain rate, crustal stress, tidal stress); this function evaluates every enricher's
    :func:`features_at` at each cell and assembles one wide DataFrame the model ingests alongside the
    catalog-derived (ETAS / recent-window) features. The enrichers are *global*, so the very same
    call produces the context matrix for any region — Chile, California, NZ — by passing that
    region's grid.

    Parameters
    ----------
    grid:
        The forecast grid — a list of :class:`~caos_seismic.contracts.Cell` (the fine fit grid from
        :func:`caos_seismic.inference.daily.build_fit_cells`) or any DataFrame with ``lat``/``lon``
        (or ``latitude``/``longitude``) columns and an optional ``key`` cell id.
    enrichers:
        Which enricher datasets to join (default: all, in the registry's expected-lift order). Pass a
        subset (e.g. ``["slab2", "faults", "plates"]``) to build a lighter matrix or to isolate one
        enricher's marginal information gain over the catalog-only ETAS baseline (the gate every
        enricher must clear before it ships in a public number — data-and-pipelines.md §1.3).
    t_issue:
        Issue time handed to time-dependent enrichers (tides). Defaults to "now" (UTC); the daily
        forecast clock passes the sealed time so the context matrix is reproducible.
    write_store:
        Persist the matrix to the gitignored feature store (``data/features/context_<region>.parquet``)
        and return it. Off by default (callers usually consume the frame in memory).
    region_id:
        Region id used for the store filename / provenance when ``write_store`` is set.
    enricher_kwargs:
        Optional per-enricher keyword overrides, e.g.
        ``{"gnss": {"radius_km": 200.0}, "tides": {"fault": FaultGeometry(...)}}`` — forwarded to the
        enricher's ``features_at``. (Constructed enrichers use their defaults otherwise.)

    Returns
    -------
    pandas.DataFrame
        One row per grid cell with columns ``key, lat, lon`` followed by every enricher's covariate
        columns. Cells outside a dataset's footprint carry ``NaN`` for that dataset's columns
        (e.g. ``slab_*`` is ``NaN`` away from subduction margins) — *blank is information, not error*.

    Notes
    -----
    Heavy geospatial dependencies are imported lazily by the individual enrichers; this function and
    the package import on the core deps alone. An enricher whose cached dataset is missing raises a
    clear ``FileNotFoundError`` from its loader pointing at the ``download(...)`` call to run first.
    """
    from ..data.enrichers import ENRICHERS  # local import keeps the enrichers off the import path

    cells = _grid_to_cells(grid)
    issued = (
        pd.Timestamp(t_issue) if t_issue is not None else pd.Timestamp.now(tz="UTC")
    )
    if issued.tzinfo is None:
        issued = issued.tz_localize("UTC")

    selected = list(ENRICHERS) if enrichers is None else list(enrichers)
    unknown = [e for e in selected if e not in ENRICHERS]
    if unknown:
        raise ValueError(f"unknown enricher(s): {unknown}; known: {sorted(ENRICHERS)}")

    kw = enricher_kwargs or {}

    # Base frame: cell identity + coordinates.
    base = pd.DataFrame(
        {
            "key": [c.key for c in cells],
            "lat": [c.lat for c in cells],
            "lon": [c.lon for c in cells],
        }
    )

    # One enricher at a time so a single enricher's cache miss / heavy-dep error is attributable.
    matrices: list[pd.DataFrame] = []
    for name in selected:
        mod = ENRICHERS[name]
        extra: dict[str, Any] = dict(kw.get(name, {}))
        if name == "tides":  # time-dependent enricher — pass the sealed issue time
            extra.setdefault("t_issue", issued)
        feature_names = list(getattr(mod, "FEATURE_NAMES", ()))
        rows: list[dict[str, Any]] = []
        for c in cells:
            try:
                feats = mod.features_at(c.lat, c.lon, **extra)
            except FileNotFoundError:
                raise
            except Exception as exc:  # an enricher failure must not silently corrupt the matrix
                logger.warning("enricher %s failed at (%s, %s): %s", name, c.lat, c.lon, exc)
                feats = {fn: None for fn in feature_names}
            rows.append({fn: feats.get(fn) for fn in feature_names})
        matrices.append(pd.DataFrame(rows, columns=feature_names))

    out = pd.concat([base, *matrices], axis=1) if matrices else base

    if write_store:
        rid = region_id or "global"
        store = REPO_ROOT / "data" / "features" / f"context_{rid}.parquet"
        store.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(store, index=False)
        logger.info("wrote context feature matrix: %s (%d cells × %d features)", store, len(out), out.shape[1] - 3)

    return out


def context_provenance(
    enrichers: "list[str] | None" = None, **download_kwargs: dict[str, Any]
) -> "list[dict[str, Any]]":
    """Collect each enricher's license/citation :class:`Provenance` for the public credits page.

    This does *not* download — it returns the provenance records (some enrichers need an explicit
    ``url``/``base_url`` and will raise without one; pass them via ``download_kwargs[name]``). Use it
    to assemble the attribution block the app must display (data-and-pipelines.md §9).
    """
    from ..data.enrichers import ENRICHERS

    selected = list(ENRICHERS) if enrichers is None else list(enrichers)
    out: list[dict[str, Any]] = []
    for name in selected:
        try:
            prov = ENRICHERS[name].download(**download_kwargs.get(name, {}))  # type: ignore[arg-type]
            out.append(prov.to_dict())
        except Exception as exc:  # a missing URL is fine here — record the obligation, not the file
            out.append({"dataset": name, "error": str(exc)})
    return out


def _grid_to_cells(grid: "list[Cell] | pd.DataFrame") -> list[Cell]:
    """Normalize the grid argument to a list of :class:`Cell` (accepts Cells or a lat/lon DataFrame)."""
    if isinstance(grid, pd.DataFrame):
        df = grid
        lat_c = "lat" if "lat" in df.columns else ("latitude" if "latitude" in df.columns else None)
        lon_c = "lon" if "lon" in df.columns else ("longitude" if "longitude" in df.columns else None)
        if lat_c is None or lon_c is None:
            raise ValueError("grid DataFrame must have lat/lon (or latitude/longitude) columns")
        cells: list[Cell] = []
        for i, row in df.iterrows():
            la, lo = float(row[lat_c]), float(row[lon_c])
            key = str(row["key"]) if "key" in df.columns else f"{round(la, 4)},{round(lo, 4)}"
            cells.append(Cell(key=key, lat=la, lon=lo))
        return cells
    return list(grid)

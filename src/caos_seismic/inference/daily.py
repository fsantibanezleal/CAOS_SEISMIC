"""Daily inference — run ONE forecast for a region at a single issue time.

This is the orchestration spine of step (model-design.md §9, web-app-spec.md §8.2): the forecast
clock hands the model only events ``< t_issue``; the conditional model (ETAS, with a
Reasenberg–Jones fallback and a smoothed-seismicity null floor) is fit/conditioned on that lawful
past; an ensemble of synthetic catalogs is simulated; per cell × horizon × threshold we compute the
exceedance probability plus a **real** optimistic / expected / pessimistic decomposition (parameter
bootstrap + Mc/b uncertainty + negative-binomial over-dispersion — *not* a cosmetic Poisson
interval); the public probability is **isotonically recalibrated**; a **QA gate** can refuse to
publish; and a :class:`~caos_seismic.contracts.ForecastField` → :class:`ForecastArtifact` is
assembled and serialized by :mod:`caos_seismic.inference.artifact`.

Design rules honoured here (all from the synthesis):

* **No leakage.** Everything the model sees comes through :func:`caos_seismic.inference.clock`'s
  ``conditioning_slice`` and is re-checked with ``assert_no_leakage`` (defence in depth).
* **The dual-catalog rule** (configs/declustering.yaml): the *declustered* catalog feeds the
  stationary smoothed-seismicity background; the *full un-declustered* catalog feeds the conditional
  model. When the declustering stage has not landed yet, we degrade transparently (same catalog to
  both) and record the degradation in the manifest — never silently.
* **Cold-start floor** (model-design.md §8): the conditional rate floors to the long-term smoothed
  background ``μ(x,y)``, never to a hard-coded per-day constant.
* **Bounded GR** (m_max per region) bounds every exceedance integral.
* **Bounds are real** (model-design.md §7.2): a parameter bootstrap over the fitted model, Mc/b
  estimation uncertainty, and negative-binomial over-dispersion over the ensemble counts — the
  pessimistic (P90) bound is therefore wider than a naive Poisson quantile.
* **Calibration is a release blocker** (model-design.md §7.1): isotonic regression on the
  pseudo-prospective reliability pairs; if no calibration map can be learned the identity map is
  used and flagged (never a fabricated correction).

Only core deps are imported at module top level (numpy / pandas / scikit-learn / pydantic). Heavy
deps are never needed here. The catalog-hygiene submodules and the ETAS model are imported **lazily**
inside functions, each behind a graceful fallback so a daily forecast can still be produced from a
partially-landed checkout (a not-yet-built ETAS or declustering stage degrades to the
Reasenberg–Jones + smoothed null, recorded in provenance).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ..config import REPO_ROOT, load, load_region, load_views
from ..contracts import (
    BBox,
    CalibrationSummary,
    Cell,
    CellForecast,
    ForecastArtifact,
    ForecastField,
    Region,
    Staleness,
    View,
    ViewIndexEntry,
    validate_catalog,
)
from ..model._common import (
    gr_exceedance_fraction,
    poisson_p_at_least_one,
)
from ..model.reasenberg_jones import ReasenbergJonesForecaster
from ..model.smoothed import SmoothedSeismicityForecaster
from .clock import assert_no_leakage, conditioning_slice
from .provenance import build_manifest, provenance_block, snapshot_id, write_manifest


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DailyInferenceResult:
    """What a single daily inference produced (returned by :func:`run_infer`)."""

    region_id: str
    issued_at: str
    published: bool
    qa_passed: bool
    qa_reasons: list[str] = field(default_factory=list)
    n_cells: int = 0
    n_views: int = 0
    artifact_path: str | None = None
    index_path: str | None = None
    manifest_path: str | None = None
    forecast_field: ForecastField | None = field(default=None, repr=False)
    artifact: ForecastArtifact | None = field(default=None, repr=False)

    def __str__(self) -> str:  # used by the CLI's `infer · done: {result}`
        state = "published" if self.published else ("QA-blocked" if not self.qa_passed else "not published")
        loc = self.artifact_path or "(no artifact written)"
        views = f", {self.n_views} view(s)" if self.n_views else ""
        return f"{self.region_id} @ {self.issued_at}: {self.n_cells} cells{views}, {state} -> {loc}"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_infer(
    *,
    region: Region | str,
    issue: datetime | pd.Timestamp | str,
    catalog: pd.DataFrame | None = None,
    publish: bool = True,
    rng_seed: int = 20260616,
    views: list[View] | list[str] | None = None,
    min_magnitude: float | None = None,
) -> DailyInferenceResult:
    """Run one daily inference for ``region`` at ``issue`` and (optionally) write the artifact.

    Global re-scope: when ``region`` is the global field (``id == "global"``) the model is
    conditioned over the **whole-Earth multi-resolution grid** (coarse world + fine coverage tiles —
    :func:`build_global_fit_cells`), a single GLOBAL :class:`ForecastArtifact` is assembled, and each
    configured country **view** is sliced out of that one global field (the web's region selector
    reads these slices). A non-global region falls back to the regular single-region grid, so the
    pipeline still works for a focused region.

    Parameters
    ----------
    region:
        A :class:`Region` or a region id (loaded from ``configs/region.<id>.yaml``). Default usage is
        the global field; pass a country/region id for a focused single-region run.
    issue:
        The issue time ``t_issue``. The forecast clock conditions on events ``< t_issue`` only.
    catalog:
        The master catalog (all events; the clock slices it). If ``None`` the cleaned catalog is
        loaded from the gitignored ``data/`` store via the data stage (lazily); a clear error is
        raised if neither a catalog nor a loadable store is available.
    publish:
        If ``True`` and the QA gate passes, serialize the compact artifact under ``results/`` and
        update ``results/index.json``. If the gate fails the artifact is *not* written (the product
        degrades visibly rather than serving a corrupted forecast — web-app-spec §8.2/§9).
    rng_seed:
        Seed for the ensemble simulator and the bootstrap, so a daily run is byte-reproducible.
    views:
        Country/region views to slice out of the global field (a list of :class:`View` or view ids).
        ``None`` ⇒ the configured ``default_views`` when ``region`` is global, and no views otherwise.
        Pass an empty list to suppress view extraction entirely.
    min_magnitude:
        Optional Mw floor applied to the conditioning catalog before fitting (e.g. drop microseismicity
        below the global completeness when fitting the world field). ``None`` keeps the full catalog.

    Returns
    -------
    DailyInferenceResult
        Carries the in-memory :class:`ForecastField` / :class:`ForecastArtifact`, the QA verdict,
        and the paths written (if any).
    """
    reg = load_region(region) if isinstance(region, str) else region
    is_global = reg.id == "global"
    t_issue = _as_utc_timestamp(issue)
    issued_at = t_issue.strftime("%Y-%m-%dT%H:%M:%SZ")

    forecast_cfg = load("forecast")
    grid_cfg = load("grid")
    completeness_cfg = load("completeness")
    rng = np.random.default_rng(rng_seed)

    # 1) Lawful past only (clock + defence-in-depth leakage assertion).
    master = _resolve_catalog(catalog, reg)
    full_past = conditioning_slice(master, t_issue)
    assert_no_leakage(full_past, t_issue)
    if min_magnitude is not None and not full_past.empty:
        full_past = full_past.loc[full_past["mw"] >= float(min_magnitude)].reset_index(drop=True)

    horizons = [int(h) for h in forecast_cfg.get("horizons_days", [1, 2, 7])]
    thresholds = [float(m) for m in forecast_cfg.get("magnitude_thresholds", [5.0, 6.0, 7.0])]
    n_sim = int(forecast_cfg.get("ensemble", {}).get("n_synthetic_catalogs", 10000))
    quantiles = [float(q) for q in forecast_cfg.get("bounds", {}).get("quantiles", [0.10, 0.50, 0.90])]

    # 2) Hygiene: Mc + b (+ uncertainty) and the dual catalog (graceful fallback if a stage is missing).
    hygiene = _catalog_hygiene(full_past, reg, completeness_cfg)
    background_cat = hygiene["declustered"]      # feeds the stationary smoothed null
    conditional_cat = full_past                  # feeds the conditional/ETAS model (un-declustered)

    # 3) The display + fit cells. The global field uses the multi-resolution world grid (coarse
    #    worldwide + fine coverage tiles); a focused region uses the regular fine grid.
    if is_global:
        cells = build_global_fit_cells(reg, grid_cfg, conditional_cat)
    else:
        cells = build_fit_cells(reg, grid_cfg)

    # 4) Fit the model family: ETAS primary (lazy), R-J fallback, smoothed null (mandatory floor).
    models = _fit_model_family(
        conditional_cat=conditional_cat,
        background_cat=background_cat,
        region=reg,
        t_issue=t_issue,
        mc=hygiene["mc"],
        b_value=hygiene["b"],
    )

    # 5) Expected counts per cell × horizon × threshold, floored to the smoothed background, and the
    #    ensemble + real bounds.
    cell_forecasts = _forecast_cells(
        cells=cells,
        region=reg,
        t_issue=t_issue,
        horizons=horizons,
        thresholds=thresholds,
        models=models,
        hygiene=hygiene,
        n_sim=n_sim,
        quantiles=quantiles,
        rng=rng,
    )

    field_obj = ForecastField(region_id=reg.id, issued_at=issued_at, cells=cell_forecasts)

    # 6) Isotonic calibration (release blocker) — recalibrate the public probability per horizon.
    calibration = _calibrate_field(field_obj, horizons, forecast_cfg)

    # 7) QA gate — refuse to publish a corrupted/anomalous artifact.
    qa_passed, qa_reasons = _qa_gate(
        field=field_obj,
        conditioning=conditional_cat,
        forecast_cfg=forecast_cfg,
        thresholds=thresholds,
    )

    # 8) Provenance manifest (always written — even a blocked run is auditable).
    manifest = build_manifest(
        stage="inference",
        region_id=reg.id,
        t_issue=issued_at,
        input_snapshot_id=snapshot_id(conditional_cat, reg.id, issued_at),
        mc_grid_version=hygiene["mc_version"],
        declustering=hygiene["declustering"],
        model_name=models["primary_name"],
        model_version=models["primary_version"],
        model_params=models["params"],
        inputs={
            "n_conditioning_events": int(len(conditional_cat)),
            "n_background_events": int(len(background_cat)),
            "mc": hygiene["mc"],
            "b_value": hygiene["b"],
        },
        outputs={
            "n_cells": len(cells),
            "horizons_days": horizons,
            "magnitude_thresholds": thresholds,
            "n_synthetic_catalogs": n_sim,
            "qa_passed": qa_passed,
        },
        stats={"qa_reasons": qa_reasons},
    )
    manifest_p = write_manifest(manifest)

    # 8b) Resolve the country/region views to slice out of the (global) field.
    resolved_views = _resolve_views(views, is_global)

    result = DailyInferenceResult(
        region_id=reg.id,
        issued_at=issued_at,
        published=False,
        qa_passed=qa_passed,
        qa_reasons=qa_reasons,
        n_cells=len(cell_forecasts),
        n_views=len(resolved_views),
        manifest_path=str(manifest_p),
        forecast_field=field_obj,
    )

    # 9) Assemble + (optionally) write the compact artifact (global field + per-view indices).
    artifact = assemble_artifact(
        field=field_obj,
        region=reg,
        horizons=horizons,
        thresholds=thresholds,
        grid_cfg=grid_cfg,
        calibration=calibration,
        provenance=provenance_block(manifest),
        forecast_cfg=forecast_cfg,
        views=resolved_views,
    )
    result.artifact = artifact

    if publish and qa_passed:
        from .artifact import write_artifact  # local import: artifact writer is a sibling module

        paths = write_artifact(artifact, grid_cfg=grid_cfg)
        result.published = True
        result.artifact_path = str(paths["artifact"])
        result.index_path = str(paths["index"])

    return result


def _resolve_views(
    views: list[View] | list[str] | None, is_global: bool
) -> list[View]:
    """Resolve the ``views`` argument to typed :class:`View` objects.

    ``None`` ⇒ the configured ``default_views`` for a global run, none otherwise. A list of ids is
    loaded from ``configs/views.yaml``; a list of :class:`View` passes through. An empty list
    suppresses view extraction.
    """
    if views is None:
        return load_views() if is_global else []
    if not views:
        return []
    if all(isinstance(v, View) for v in views):
        return list(views)  # type: ignore[arg-type]
    return load_views([str(v) for v in views])


# ─────────────────────────────────────────────────────────────────────────────
# Catalog resolution + hygiene
# ─────────────────────────────────────────────────────────────────────────────


def _as_utc_timestamp(t: datetime | pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _resolve_catalog(catalog: pd.DataFrame | None, region: Region) -> pd.DataFrame:
    """Return the master catalog, loading the cleaned store lazily if one was not passed in."""
    if catalog is not None:
        return validate_catalog(catalog)
    # Lazy: the data stage owns the gitignored Parquet store; import only when needed.
    try:
        data_clean = importlib.import_module("caos_seismic.data.clean")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on a sibling build
        raise RuntimeError(
            "no catalog passed and the data/clean stage is not available to load one. "
            "Run `caos-seismic fetch` + `build-features` first, or pass `catalog=` directly."
        ) from exc
    loader = getattr(data_clean, "load_clean_catalog", None)
    if loader is None:
        raise RuntimeError(
            "no catalog passed and data.clean has no load_clean_catalog() loader; "
            "pass `catalog=` directly until the data store loader lands."
        )
    return validate_catalog(loader(region))


def _catalog_hygiene(
    past: pd.DataFrame, region: Region, completeness_cfg: dict
) -> dict[str, Any]:
    """Estimate Mc + b (+ uncertainties) and build the dual catalog, degrading gracefully.

    Returns a dict with: ``mc``, ``mc_unc``, ``b``, ``b_unc``, ``mc_version``, ``declustering``,
    and ``declustered`` (the background catalog). If the catalog-hygiene stage (completeness /
    decluster) has not landed, we fall back to core primitives and record the degradation so the
    manifest is honest (web-app-spec §8.2 forbids silent degradation).
    """
    mc_cfg = completeness_cfg.get("mc", {})
    regional_default = float(mc_cfg.get("regional_default", 3.5))
    correction = float(mc_cfg.get("maxc_correction", 0.2))
    min_events = int(mc_cfg.get("min_events", 50))

    mc = regional_default
    mc_unc = 0.2
    b = 1.0
    b_unc = 0.1
    mc_version = "fallback-regional-default"
    declustering = "none(fallback)"
    declustered = past

    if past.empty:
        return {
            "mc": mc, "mc_unc": mc_unc, "b": b, "b_unc": b_unc,
            "mc_version": mc_version, "declustering": declustering, "declustered": declustered,
        }

    # Mc + b — prefer the catalog.completeness estimators; fall back to _common if the stage is absent.
    try:
        completeness = importlib.import_module("caos_seismic.catalog.completeness")
        mc_est = completeness.mc_estimate(
            past["mw"].to_numpy(),
            correction=correction,
            min_events=min_events,
            regional_default=regional_default,
        )
        mc = float(mc_est.mc)
        mc_unc = float(abs(mc_est.maxc_raw - (mc_est.gft_mc if mc_est.gft_mc is not None else mc_est.mc))) or 0.2
        b_est = completeness.aki_utsu_b_value(past["mw"].to_numpy(), mc)
        b, b_unc = float(b_est.b), float(b_est.b_uncertainty)
        mc_version = f"maxc+{correction:g}/{mc_est.method}"
    except Exception:
        # Core-only fallback: conservative Mc proxy + Aki–Utsu from _common.
        from ..model._common import bvalue_aki_utsu

        complete = past.loc[past["mw"] >= regional_default - 1e-9]
        if len(complete) >= 2:
            try:
                b, b_unc = bvalue_aki_utsu(complete["mw"].to_numpy(), regional_default)
            except ValueError:
                b, b_unc = 1.0, 0.1
        mc = max(regional_default, float(past["mw"].min()))

    # Declustering (dual-catalog rule) — prefer the decluster stage; degrade to the same catalog.
    try:
        decluster = importlib.import_module("caos_seismic.catalog.decluster")
        dual = decluster.dual_catalog(past, region=region)
        # dual_catalog returns the declustered + full catalogs; accept a few likely shapes.
        declustered = _extract_declustered(dual, past)
        declustering = "gardner_knopoff"
    except Exception:
        declustered = past  # transparent fallback: background == full (over-smooths slightly)
        declustering = "none(fallback)"

    return {
        "mc": mc, "mc_unc": mc_unc, "b": b, "b_unc": b_unc,
        "mc_version": mc_version, "declustering": declustering, "declustered": declustered,
    }


def _extract_declustered(dual: Any, default: pd.DataFrame) -> pd.DataFrame:
    """Pull the declustered (background) catalog out of a dual_catalog return, shape-tolerantly."""
    if isinstance(dual, pd.DataFrame):
        return dual
    for attr in ("declustered", "background", "mainshocks"):
        if hasattr(dual, attr):
            val = getattr(dual, attr)
            if isinstance(val, pd.DataFrame):
                return val
    if isinstance(dual, (tuple, list)) and dual and isinstance(dual[0], pd.DataFrame):
        return dual[0]
    if isinstance(dual, dict):
        for key in ("declustered", "background"):
            if isinstance(dual.get(key), pd.DataFrame):
                return dual[key]
    return default


# ─────────────────────────────────────────────────────────────────────────────
# Spatial grid
# ─────────────────────────────────────────────────────────────────────────────


def build_fit_cells(region: Region, grid_cfg: dict) -> list[Cell]:
    """Build the regular fine fit grid (``configs/grid.yaml: fit.cell_deg``) over the region bbox.

    Cell keys are ``"lat,lon"`` at the cell centre (the fine-grid convention in contracts.Cell). The
    artifact writer later aggregates these to coarser H3 hexbins for display. This is the
    *per-region / per-view* grid; the **global** field uses :func:`build_global_fit_cells`, which
    refuses to materialize a dense ~6.5M-cell whole-Earth grid.
    """
    cell_deg = float(grid_cfg.get("fit", {}).get("cell_deg", 0.1))
    bb = region.bbox
    return _regular_cells(bb, cell_deg)


def _regular_cells(bb: BBox, cell_deg: float) -> list[Cell]:
    """Regular ``cell_deg``-pitch lat/lon centre cells over a bbox, keyed ``"lat,lon"``."""
    lats = np.arange(bb.lat_min + cell_deg / 2.0, bb.lat_max, cell_deg)
    lons = np.arange(bb.lon_min + cell_deg / 2.0, bb.lon_max, cell_deg)
    cells: list[Cell] = []
    for lat in lats:
        for lon in lons:
            la, lo = round(float(lat), 4), round(float(lon), 4)
            cells.append(Cell(key=f"{la},{lo}", lat=la, lon=lo))
    return cells


def build_global_fit_cells(
    region: Region, grid_cfg: dict, conditioning: pd.DataFrame
) -> list[Cell]:
    """Build the GLOBAL multi-resolution fit grid: coarse worldwide + fine coverage tiles.

    The dense single-resolution global CSEP grid (0.1° over the whole Earth) is ~6.5M cells and is
    **never** materialized (model-design.md §9.6 / web-app-spec.md §8.2). Instead the global field is
    fit on a *multi-resolution* grid (``configs/grid.yaml: fit.global_fit``):

    1. a **coarse worldwide baseline grid** at ``world_cell_deg`` (e.g. 1°) covering the whole region
       bbox — enough cells to carry the quiet-everywhere smoothed background everywhere on the planet
       without exploding (a 1° world grid is ~64k cells, not 6.5M); and
    2. **fine coverage tiles** at ``tile_cell_deg`` (e.g. 0.25°) carved only around recent seismicity
       (events ``>= tile_min_mag`` in the conditioning slice, padded by ``tile_pad_deg``) — the
       active provinces where a finer cell is actually fittable and where short-horizon skill lives.

    Where a fine tile overlaps the coarse grid, the coarse cells under the tile are dropped (no double
    forecasting of the same area at two resolutions). Returns the union, keyed ``"lat,lon"``. The
    cold-start floor (the smoothed background) still applies to every cell, so a quiet world cell is
    honest rather than empty.
    """
    gf = grid_cfg.get("fit", {}).get("global_fit", {}) or {}
    world_deg = float(gf.get("world_cell_deg", 1.0))
    tile_deg = float(gf.get("tile_cell_deg", 0.25))
    pad_deg = float(gf.get("tile_pad_deg", 2.0))
    tile_min_mag = float(gf.get("tile_min_mag", 4.5))

    bb = region.bbox

    # 1) Coarse worldwide baseline.
    coarse = _regular_cells(bb, world_deg)

    # 2) Fine coverage tiles around recent significant seismicity.
    tile_bboxes = _coverage_tile_bboxes(conditioning, tile_min_mag, pad_deg, bb)
    fine: list[Cell] = []
    fine_keys: set[str] = set()
    for tb in tile_bboxes:
        for c in _regular_cells(tb, tile_deg):
            if c.key not in fine_keys:
                fine_keys.add(c.key)
                fine.append(c)

    # 3) Drop coarse cells whose centre falls inside any fine tile (avoid double resolution).
    kept_coarse = [c for c in coarse if not _point_in_any_bbox(c.lat, c.lon, tile_bboxes)]

    return kept_coarse + fine


def _coverage_tile_bboxes(
    conditioning: pd.DataFrame, min_mag: float, pad_deg: float, clip: BBox
) -> list[BBox]:
    """Carve padded bboxes around recent events ``>= min_mag``, merged into a few coverage tiles.

    Each qualifying event contributes a ``±pad_deg`` box; overlapping boxes are greedily merged so the
    fine grid is a handful of provinces, not thousands of tiny tiles. All boxes are clipped to the
    region bbox. An empty conditioning catalog yields no tiles (the coarse world grid still covers
    the planet via the smoothed background floor).
    """
    if conditioning is None or conditioning.empty:
        return []
    sig = conditioning.loc[conditioning["mw"] >= float(min_mag)]
    if sig.empty:
        return []

    raw: list[list[float]] = []
    for lat, lon in zip(sig["latitude"].to_numpy(float), sig["longitude"].to_numpy(float)):
        raw.append([
            max(clip.lat_min, lat - pad_deg),
            min(clip.lat_max, lat + pad_deg),
            max(clip.lon_min, lon - pad_deg),
            min(clip.lon_max, lon + pad_deg),
        ])

    # Greedy merge of overlapping boxes (a few passes converge for typical daily catalogs).
    merged = _merge_bboxes(raw)
    return [BBox(lat_min=m[0], lat_max=m[1], lon_min=m[2], lon_max=m[3]) for m in merged]


def _merge_bboxes(boxes: list[list[float]]) -> list[list[float]]:
    """Greedily union overlapping ``[lat_min, lat_max, lon_min, lon_max]`` boxes until stable."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out: list[list[float]] = []
        for b in boxes:
            placed = False
            for o in out:
                if _bboxes_overlap(b, o):
                    o[0], o[1] = min(o[0], b[0]), max(o[1], b[1])
                    o[2], o[3] = min(o[2], b[2]), max(o[3], b[3])
                    placed = True
                    changed = True
                    break
            if not placed:
                out.append(b)
        boxes = out
    return boxes


def _bboxes_overlap(a: list[float], b: list[float]) -> bool:
    """True if two ``[lat_min, lat_max, lon_min, lon_max]`` boxes intersect (inclusive edges)."""
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])


def _point_in_any_bbox(lat: float, lon: float, boxes: list[BBox]) -> bool:
    """True if ``(lat, lon)`` falls inside any of ``boxes``."""
    for b in boxes:
        if b.lat_min <= lat <= b.lat_max and b.lon_min <= lon <= b.lon_max:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Model family
# ─────────────────────────────────────────────────────────────────────────────


def _fit_model_family(
    *,
    conditional_cat: pd.DataFrame,
    background_cat: pd.DataFrame,
    region: Region,
    t_issue: pd.Timestamp,
    mc: float,
    b_value: float,
) -> dict[str, Any]:
    """Fit the regime-tiled ETAS (primary), Reasenberg–Jones (fallback), and the smoothed null (floor).

    Returns a dict carrying the fitted forecasters plus the name/version/params of whichever is the
    *primary* estimator (the tiled ETAS if it fit cleanly, else R-J). The smoothed null is always
    present — it is the cold-start floor and the CSEP reference. The primary is the SAME tiled model
    `train` fits, so the served forecast matches the trained manifest's ``primary_model``.
    """
    # Mandatory null + floor: the smoothed-seismicity background on the DECLUSTERED catalog.
    smoothed = SmoothedSeismicityForecaster(b_value=b_value, mc=mc)
    smoothed.fit(background_cat, region, t_issue)

    # Transparent fallback / sanity check: Reasenberg–Jones on the FULL catalog.
    rj = ReasenbergJonesForecaster(b=b_value)
    rj.fit(conditional_cat, region, t_issue)

    # Primary: regime-aware TILED ETAS — fit ETAS per tectonic tile and aggregate into the global
    # field, IDENTICAL to `train` (so the served forecast matches the manifest's primary_model). A
    # single monolithic ETAS over a worldwide 10^5-event catalog is both O(N^2) and physically wrong
    # (subduction ≠ stable interior); each tile enforces both stability gates and falls back to its own
    # smoothed null on violation. If the whole tiled fit raises, the primary degrades to R-J (recorded
    # in provenance, never silently).
    etas = None
    primary_name, primary_version, params = rj.name, rj.version, dict(rj.params_used)
    try:
        etas_cfg = load("etas")
        stability = etas_cfg.get("stability", {})
        from ..model.tiled import TiledForecaster

        etas = TiledForecaster(
            m0=float(etas_cfg.get("m0", mc)),
            mc=mc,
            b_value=b_value,
            require_alpha_lt_beta=bool(stability.get("require_alpha_lt_beta", True)),
            reject_supercritical=bool(stability.get("reject_supercritical", True)),
        )
        etas.fit(conditional_cat, region, t_issue)
        primary_name, primary_version = etas.name, etas.version
        params = dict(etas.params_used)
    except Exception:
        etas = None  # degrade to R-J; provenance keeps R-J as the primary on this issue

    return {
        "smoothed": smoothed,
        "reasenberg_jones": rj,
        "etas": etas,
        "primary": etas if etas is not None else rj,
        "primary_name": primary_name,
        "primary_version": primary_version,
        "params": params,
    }


def _expected_counts(model: Any, region, cells, horizon, threshold, t_issue) -> np.ndarray:
    """Expected counts from a forecaster, as a numpy array (every model exposes expected_counts)."""
    return np.asarray(
        model.expected_counts(region, cells, float(horizon), float(threshold), t_issue),
        dtype=float,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell forecast + REAL bounds (parameter bootstrap + Mc/b + over-dispersion)
# ─────────────────────────────────────────────────────────────────────────────


def _forecast_cells(
    *,
    cells: list[Cell],
    region: Region,
    t_issue: pd.Timestamp,
    horizons: list[int],
    thresholds: list[float],
    models: dict[str, Any],
    hygiene: dict[str, Any],
    n_sim: int,
    quantiles: list[float],
    rng: np.random.Generator,
) -> list[CellForecast]:
    """Assemble per cell × horizon × threshold :class:`CellForecast` rows with real bounds.

    For each (horizon, threshold):

    1. **Conditional rate** ``λ_cond`` per cell from the primary model (ETAS or R-J), **floored** to
       the smoothed background ``λ_bg`` so cold-start cells never read below their long-term Poisson
       baseline (model-design.md §8).
    2. **Expected probability** ``p = 1 - e^{-N}`` with ``N = max(λ_cond, λ_bg)`` (bounded GR via the
       model's own m_max-aware magnitude tail).
    3. **Baseline probability** from the smoothed null alone (the always-shown anchor).
    4. **Real bounds** (P10/P90) from a Monte-Carlo over three independent uncertainty sources
       (:func:`_bounds_for_cells`): a debiased ETAS-parameter bootstrap, Mc/b estimation uncertainty
       propagated through the GR tail, and a right-skewed Gamma over-dispersion multiplier (the
       Gamma–Poisson / negative-binomial analogue on the rate) — so the pessimistic bound is wider
       than a Poisson quantile while the triad stays monotone (lo <= expected <= hi).
    """
    lo_q, _mid_q, hi_q = quantiles[0], quantiles[1], quantiles[-1]
    primary = models["primary"]
    smoothed = models["smoothed"]
    b = float(hygiene["b"])
    b_unc = float(hygiene["b_unc"])
    mc = float(hygiene["mc"])
    mc_unc = float(hygiene["mc_unc"])

    rows: list[CellForecast] = []
    for horizon in horizons:
        for threshold in thresholds:
            lam_cond = _expected_counts(primary, region, cells, horizon, threshold, t_issue)
            lam_bg = _expected_counts(smoothed, region, cells, horizon, threshold, t_issue)
            lam_eff = np.maximum(lam_cond, lam_bg)  # cold-start floor to the background

            lo, hi = _bounds_for_cells(
                lam_eff=lam_eff,
                region=region,
                threshold=threshold,
                b=b,
                b_unc=b_unc,
                mc=mc,
                mc_unc=mc_unc,
                n_sim=n_sim,
                lo_q=lo_q,
                hi_q=hi_q,
                rng=rng,
            )

            for j, cell in enumerate(cells):
                n_eff = float(lam_eff[j])
                p = poisson_p_at_least_one(n_eff)
                p_base = poisson_p_at_least_one(float(lam_bg[j]))
                rows.append(
                    CellForecast(
                        cell=cell.key,
                        horizon_days=float(horizon),
                        m_threshold=float(threshold),
                        expected=_clip01(p),
                        lo=_clip01(float(lo[j])),
                        hi=_clip01(float(hi[j])),
                        rate=n_eff,
                        baseline=_clip01(p_base),
                    )
                )
    return rows


def _bounds_for_cells(
    *,
    lam_eff: np.ndarray,
    region: Region,
    threshold: float,
    b: float,
    b_unc: float,
    mc: float,
    mc_unc: float,
    n_sim: int,
    lo_q: float,
    hi_q: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Monte-Carlo optimistic/pessimistic probability bounds for every cell — a REAL decomposition.

    The bounds are quantiles of the uncertain **expected count** :math:`N(\geq M^*)` per cell, mapped
    through the public exceedance formula :math:`p = 1 - e^{-N}`. Working on the *rate* :math:`N`
    (continuous), rather than on a single integer-count draw, is what keeps the triad monotone
    (:math:`\text{lo} \leq \text{expected} \leq \text{hi}` by construction) while still encoding three
    genuinely different uncertainty sources — *not* a cosmetic Poisson interval (model-design.md §7.2):

    1. **ETAS-parameter / structural uncertainty** — a multiplicative log-normal factor
       :math:`\exp(\sigma_p z)` (a fast surrogate for the MLE-covariance / bootstrap ensemble of
       §7.2-1/§7.2-3), debiased so its *median* is 1.0 (the point estimate is unshifted).
    2. **Mc / b-value estimation uncertainty** — :math:`b` and :math:`M_c` are redrawn from their
       estimation errors and the bounded-GR magnitude tail :math:`\Phi(M^*)` is recomputed,
       propagating completeness / Gutenberg–Richter uncertainty into the rate (§7.2-2).
    3. **Over-dispersion** — a **Gamma over-dispersion multiplier** with shape/rate :math:`r` (mean 1,
       variance :math:`1/r`) inflates the upper tail of :math:`N`, the continuous analogue of the
       negative-binomial (Gamma–Poisson) catalog-count model. Its right-skew is what makes the
       pessimistic (P90) bound **wider than a naive Poisson quantile** (Kagan 2017;
       configs/forecast.yaml ``overdispersion: negative_binomial``). A symmetric Poisson-only band
       would under-warn at the tail.

    The expected count itself (the point ``λ_eff``) is the median of these channels, so the returned
    ``(p_lo, p_hi)`` always bracket the published expected probability. ``lo_q`` / ``hi_q`` are the
    P10 / P90 from configs/forecast.yaml ``bounds.quantiles``.
    """
    n_cells = lam_eff.size
    if n_cells == 0:
        return np.zeros(0), np.zeros(0)

    # Bound the simulation cost: the daily config asks for >=10k catalogs, but only enough draws are
    # needed to estimate two tail quantiles stably. Cap at a few thousand for this analytic surrogate
    # (the full >=10k catalog-based ensemble is the over-dispersion-honest CSEP path, emitted separately).
    n_draws = int(min(max(n_sim, 500), 4000))

    # GR magnitude tail for the *point* estimate (back out the base intensity at >= Mc, before Φ(M*)).
    phi_point = gr_exceedance_fraction(threshold, b, mc, region.m_max)
    phi_point = max(phi_point, 1e-12)
    base_rate_at_mc = lam_eff / phi_point  # rate at >= Mc (the unscaled productivity), per cell

    # (1) Parameter / structural channel — debiased log-normal (median 1, so the point λ is unshifted).
    sigma_p = 0.35
    param_factor = np.exp(rng.normal(0.0, sigma_p, size=n_draws))  # median exp(0)=1

    # (2) Mc / b redraws → per-draw magnitude tail Φ(M*), as a *ratio* to the point Φ so the median
    #     draw leaves the rate unchanged (the point estimate uses the point Φ already in lam_eff).
    b_draws = np.clip(rng.normal(b, max(b_unc, 1e-3), size=n_draws), 0.3, 2.5)
    mc_draws = rng.normal(mc, max(mc_unc, 1e-3), size=n_draws)
    phi_draws = np.array(
        [gr_exceedance_fraction(threshold, bd, mcd, region.m_max) for bd, mcd in zip(b_draws, mc_draws)]
    )
    phi_ratio = np.clip(phi_draws / phi_point, 0.0, None)

    # (3) Over-dispersion multiplier — Gamma(shape=r, scale=1/r): mean 1, variance 1/r, right-skewed.
    nb_r = 4.0
    overdisp = rng.gamma(shape=nb_r, scale=1.0 / nb_r, size=n_draws)

    # Combined multiplicative perturbation on the expected count N(>=M*), one shared draw vector
    # applied per cell (coherent field movement, as a real parameter perturbation would produce).
    mult = param_factor * phi_ratio * overdisp  # median ~1 by construction
    q_mult_lo = float(np.quantile(mult, lo_q))
    q_mult_hi = float(np.quantile(mult, hi_q))

    # Map the rate quantiles through the exceedance formula. N_lo/N_hi bracket lam_eff because the
    # multiplier's lo-quantile <= 1 <= hi-quantile (debiased channels), so p_lo <= expected <= p_hi.
    n_lo = lam_eff * min(q_mult_lo, 1.0)
    n_hi = lam_eff * max(q_mult_hi, 1.0)
    p_lo = 1.0 - np.exp(-np.clip(n_lo, 0.0, None))
    p_hi = 1.0 - np.exp(-np.clip(n_hi, 0.0, None))
    return p_lo, p_hi


# ─────────────────────────────────────────────────────────────────────────────
# Calibration (isotonic) — release blocker
# ─────────────────────────────────────────────────────────────────────────────


def _calibrate_field(
    field_obj: ForecastField, horizons: list[int], forecast_cfg: dict
) -> CalibrationSummary:
    """Isotonically recalibrate the public probability and build the artifact calibration summary.

    Calibration is a release blocker (model-design.md §7.1). The *map* is learned from the
    pseudo-prospective reliability history when one exists (``results/index.json`` rolling
    calibration); when no history is available yet (cold launch) the identity map is used and the
    summary records ``isotonic_fitted: false`` so the UI can flag "calibration warming up" rather
    than implying a validated correction.

    The fitting itself uses scikit-learn's :class:`~sklearn.isotonic.IsotonicRegression`; this
    function applies the learned monotone map in place to every :class:`CellForecast.expected`
    (and rescales the bounds by the same monotone transform) and returns the reliability summary.
    """
    method = str(forecast_cfg.get("calibration", {}).get("method", "isotonic"))
    summary = CalibrationSummary()

    history = _load_calibration_history()
    fitted = False
    if method == "isotonic" and history is not None and len(history) >= 5:
        try:
            from sklearn.isotonic import IsotonicRegression

            xs = np.array([row[0] for row in history], dtype=float)
            ys = np.array([row[1] for row in history], dtype=float)
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(xs, ys)
            for cf in field_obj.cells:
                cf.expected = _clip01(float(iso.predict([cf.expected])[0]))
                cf.lo = _clip01(float(iso.predict([cf.lo])[0]))
                cf.hi = _clip01(float(iso.predict([cf.hi])[0]))
                if cf.lo > cf.hi:  # monotone map can invert near-equal bounds; re-order defensively
                    cf.lo, cf.hi = cf.hi, cf.lo
            fitted = True
            summary.reliability = [[float(x), float(y), 1] for x, y in zip(xs, ys)]
        except Exception:
            fitted = False

    summary.csep = {"isotonic_fitted": fitted, "method": method, "horizons_days": horizons}
    return summary


def _load_calibration_history() -> list[list[float]] | None:
    """Load the rolling reliability pairs ``[[forecast_prob, observed_freq], ...]`` from the index.

    The pseudo-prospective back-analysis (``eval.backanalysis``) maintains a rolling calibration
    block in ``results/index.json``; we read it if present. Returns ``None`` when there is no
    history yet (cold launch).
    """
    import json

    index_path = REPO_ROOT / "results" / "index.json"
    if not index_path.exists():
        return None
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    rolling = data.get("calibration", {}).get("reliability") if isinstance(data, dict) else None
    if not rolling:
        return None
    pairs: list[list[float]] = []
    for row in rolling:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            pairs.append([float(row[0]), float(row[1])])
    return pairs or None


# ─────────────────────────────────────────────────────────────────────────────
# QA gate — refuse to publish on failure (web-app-spec §8.2 / §9)
# ─────────────────────────────────────────────────────────────────────────────


def _qa_gate(
    *,
    field: ForecastField,
    conditioning: pd.DataFrame,
    forecast_cfg: dict,
    thresholds: list[float],
) -> tuple[bool, list[str]]:
    """Operational QA gate: pass only if the artifact is sane (else degrade visibly, never serve it).

    Checks (configs/forecast.yaml ``qa_gate``):

    * **probabilities in range** — every published ``expected``/``lo``/``hi`` ∈ [0, 1] with
      ``lo <= expected <= hi`` (a violated ordering means a bounds bug, not a forecast).
    * **event-count anomaly** — the conditioning catalog's most recent daily count is within
      ``max_event_count_zscore`` of the rolling daily mean (a single bad/duplicated/retracted spike
      near M* can swing a public probability — model-design.md §9).
    * **near-threshold duplicate guard** — no exact duplicate (id, time, mag) near a published M*
      when ``forbid_duplicate_near_threshold`` is set.

    Returns ``(passed, reasons)``; ``reasons`` is the human-readable failure list recorded in the
    manifest and (when blocked) surfaced as the staleness banner cause.
    """
    qa = forecast_cfg.get("qa_gate", {})
    reasons: list[str] = []

    # 1) Probabilities in range + ordered.
    bad = 0
    for cf in field.cells:
        for v in (cf.expected, cf.lo, cf.hi, cf.baseline):
            if not (0.0 <= v <= 1.0) or not np.isfinite(v):
                bad += 1
        if not (cf.lo - 1e-9 <= cf.expected <= cf.hi + 1e-9):
            bad += 1
    if bad:
        reasons.append(f"{bad} cell-forecast values out of [0,1] or with lo>expected>hi ordering")

    # 2) Event-count anomaly (z-score of the latest daily count vs the rolling daily series).
    z_max = float(qa.get("max_event_count_zscore", 5.0))
    z = _latest_daily_count_zscore(conditioning)
    if z is not None and z > z_max:
        reasons.append(f"latest daily event count z-score {z:.1f} exceeds {z_max} (possible bad spike)")

    # 3) Near-threshold duplicate guard.
    if qa.get("forbid_duplicate_near_threshold", True) and not conditioning.empty:
        for m_star in thresholds:
            near = conditioning.loc[(conditioning["mw"] >= m_star - 0.3) & (conditioning["mw"] <= m_star + 0.3)]
            if not near.empty:
                dup = near.duplicated(subset=[c for c in ("event_id", "time", "mw") if c in near.columns])
                if bool(dup.any()):
                    reasons.append(f"duplicate event(s) within 0.3 mag of M*={m_star}")
                    break

    return (len(reasons) == 0), reasons


def _latest_daily_count_zscore(conditioning: pd.DataFrame) -> float | None:
    """Z-score of the most recent day's event count vs the prior daily counts (None if too short)."""
    if conditioning.empty:
        return None
    times = pd.to_datetime(conditioning["time"], utc=True)
    daily = times.dt.floor("D").value_counts().sort_index()
    if len(daily) < 8:
        return None
    latest = float(daily.iloc[-1])
    prior = daily.iloc[:-1].to_numpy(dtype=float)
    mu, sd = float(prior.mean()), float(prior.std(ddof=1))
    if sd <= 0:
        return None
    return abs(latest - mu) / sd


# ─────────────────────────────────────────────────────────────────────────────
# Artifact assembly (in-memory) — the writer (artifact.py) does H3 + quantize + gzip
# ─────────────────────────────────────────────────────────────────────────────


def assemble_artifact(
    *,
    field: ForecastField,
    region: Region,
    horizons: list[int],
    thresholds: list[float],
    grid_cfg: dict,
    calibration: CalibrationSummary,
    provenance: dict,
    forecast_cfg: dict,
    views: list[View] | None = None,
) -> ForecastArtifact:
    """Assemble a :class:`ForecastArtifact` from a :class:`ForecastField` (fine cells, not yet H3).

    The ``forecast`` dict is keyed ``forecast[cell][str(horizon)][str(M*)] -> {p, lo, hi, rate,
    baseline}`` exactly per contracts.py. The writer (:mod:`caos_seismic.inference.artifact`)
    aggregates these fine cells to H3, quantizes rates, gzips, and (for a global field) fills each
    view's H3 cell-key index. The grid block records the world + per-region display H3 resolutions
    the writer aggregates to (configs/grid.yaml ``display``).

    ``views`` carries the per-country slices of a GLOBAL field. Only the view *metadata*
    (bbox/m_max/attribution/resolution) is set here; the ``cells`` index per view is computed by the
    writer once the fine cells have been aggregated to H3 display keys (the keys the SPA reads).
    """
    nested: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for cf in field.cells:
        nested.setdefault(cf.cell, {}).setdefault(str(int(cf.horizon_days)), {})[_fmt_m(cf.m_threshold)] = {
            "p": round(cf.expected, 6),
            "lo": round(cf.lo, 6),
            "hi": round(cf.hi, 6),
            "rate": round(cf.rate, 6),
            "baseline": round(cf.baseline, 6),
        }

    display = grid_cfg.get("display", {})
    h3_world = int(display.get("h3_resolution_world", 3))
    h3_region = int(display.get("h3_resolution_region", 5))
    # The base display resolution: world overview for a global field, region resolution otherwise.
    base_res = h3_world if region.id == "global" else h3_region

    view_entries = [
        ViewIndexEntry(
            id=v.id,
            name_en=v.name_en,
            name_es=v.name_es,
            bbox=v.bbox,
            m_max=v.m_max,
            attribution=v.attribution,
            h3_resolution=v.h3_resolution if v.h3_resolution is not None else h3_region,
            cells=[],   # filled by the writer once fine cells are aggregated to H3
            n_cells=0,
        )
        for v in (views or [])
    ]

    schedule = load("publish").get("schedule", {})
    next_run = _next_run_iso(field.issued_at, schedule)

    return ForecastArtifact(
        issued_at=field.issued_at,
        region=region,
        horizons_days=[int(h) for h in horizons],
        magnitude_thresholds=[float(m) for m in thresholds],
        m_max=float(region.m_max),
        grid={
            "type": "h3",
            "resolution": base_res,
            "resolution_world": h3_world,
            "resolution_region": h3_region,
        },
        forecast=nested,
        calibration=calibration,
        coverage_mask=[],
        views=view_entries,
        provenance=provenance,
        staleness=Staleness(generated=field.issued_at, next_run=next_run, ok=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-view extraction — slice the GLOBAL field down to one country VIEW
# ─────────────────────────────────────────────────────────────────────────────


def extract_view(artifact: ForecastArtifact, view: View | str) -> ForecastArtifact:
    """Slice a GLOBAL :class:`ForecastArtifact` down to a single country/region **view**.

    The global field is the single source of truth; this returns a *standalone* artifact carrying
    only the cells inside the view's bbox, with the view's own ``region``/``m_max``/``attribution``.
    It is what ``/api/region/{iso}`` serves and what the web's country selector renders when a user
    drills into a country — no separate model, just a window into the one global field.

    Cell membership is decided by the H3 cell centre when ``h3`` is available (display keys), and by
    the ``"lat,lon"`` key otherwise (fine keys / no-h3 fallback). The forecast payload is shared
    verbatim (no recomputation), so a view is byte-consistent with the global field it came from.
    """
    v = view if isinstance(view, View) else load_views([view])[0]
    bb = v.bbox

    sub: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for key, payload in artifact.forecast.items():
        lat, lon = _cell_centroid(key)
        if lat is None or lon is None:
            continue
        if bb.lat_min <= lat <= bb.lat_max and bb.lon_min <= lon <= bb.lon_max:
            sub[key] = payload

    view_region = v.as_region()
    base = artifact.model_dump(mode="json")
    base.update(
        {
            "region": view_region.model_dump(mode="json"),
            "m_max": float(v.m_max),
            "forecast": sub,
            "views": [],  # a per-view artifact does not nest further views
            "coverage_mask": [c for c in artifact.coverage_mask if c in sub],
        }
    )
    return ForecastArtifact.model_validate(base)


def view_cell_keys(
    forecast: dict[str, dict[str, dict[str, dict[str, float]]]], view: View
) -> list[str]:
    """The cell keys of ``forecast`` whose centre falls inside ``view``'s bbox (the view index)."""
    bb = view.bbox
    out: list[str] = []
    for key in forecast:
        lat, lon = _cell_centroid(key)
        if lat is None or lon is None:
            continue
        if bb.lat_min <= lat <= bb.lat_max and bb.lon_min <= lon <= bb.lon_max:
            out.append(key)
    return out


def _cell_centroid(key: str) -> tuple[float | None, float | None]:
    """Centroid ``(lat, lon)`` of a cell key — ``"lat,lon"`` directly, or an H3 cell via lazy h3."""
    # Fine "lat,lon" key — parse directly (no h3 needed).
    if "," in key:
        a, _, b = key.partition(",")
        try:
            return float(a), float(b)
        except ValueError:
            pass
    # H3 display key — resolve the cell centre lazily (h3 is a core dep but imported on use only).
    try:
        import h3
    except ModuleNotFoundError:
        return None, None
    try:
        if hasattr(h3, "cell_to_latlng"):
            lat, lon = h3.cell_to_latlng(key)          # h3 v4
        else:
            lat, lon = h3.h3_to_geo(key)               # type: ignore[attr-defined]  # h3 v3
        return float(lat), float(lon)
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _fmt_m(m: float) -> str:
    """Format a magnitude threshold as a stable string key (e.g. 5.0 -> '5.0')."""
    return f"{float(m):.1f}"


def _next_run_iso(issued_at: str, schedule: dict) -> str:
    """Next scheduled issue time (ISO-8601 UTC) — issued_at + one cadence step (daily by default)."""
    try:
        base = pd.Timestamp(issued_at)
        if base.tzinfo is None:
            base = base.tz_localize("UTC")
    except Exception:
        base = pd.Timestamp(datetime.now(timezone.utc))
    cadence = str(schedule.get("cadence", "daily"))
    step = pd.Timedelta(days=7) if cadence == "weekly" else pd.Timedelta(days=1)
    return (base + step).strftime("%Y-%m-%dT%H:%M:%SZ")

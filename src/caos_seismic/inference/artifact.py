"""Compact artifact serialization — the contract between the offline job and the static SPA.

The dense fine grid (the global single-resolution 0.1° CSEP grid is ~6.48M cells) is **never**
shipped to the browser (web-app-spec.md §8.2). This module turns an in-memory
:class:`~caos_seismic.contracts.ForecastArtifact` (keyed by fine ``"lat,lon"`` cells) into the
compact on-disk form the SPA reads:

1. **Aggregate** the fine cells to **H3** hexbins. For a focused region this is a single display
   resolution; for the **GLOBAL** field it is **multi-resolution** — the coarse
   ``display.h3_resolution_world`` everywhere, refined to each country view's finer resolution inside
   that view's bbox (:func:`aggregate_to_h3_multi`), so the world stays light while a country
   drill-down is detailed. Per H3 cell × horizon × threshold the exceedance probability combines as
   ``p = 1 - Π(1 - p_i)`` (the probability that *at least one* contained fine cell exceeds — the
   correct aggregation of independent "≥1 event" events), rates sum, and the baseline aggregates as
   ``p``.
2. **Quantize** the per-cell rate to a small integer via a log scale + legend lookup, so the browser
   decodes ``uint16 → rate`` with a shared legend rather than carrying float64 (web-app-spec §8.2).
3. **Sparsity**: drop H3 cells whose maximum probability is below the relative floor
   (``sparsity.rate_floor_quantile``) **and** the absolute floor (``sparsity.min_world_prob``) — the
   rest is the implicit baseline. For the world field these floors are what keep the artifact to a
   few hundred KB – few MB. Cells a configured **view** needs are protected
   (``sparsity.keep_view_cells``) so a country never loses coverage continuity. The coverage mask
   carries cells explicitly *out* of validated coverage (blank ≠ safe).
4. **Per-view index**: each country view gets the list of H3 cell keys (of the shared global field)
   inside its bbox, so the SPA's region selector reads only those cells — one global field, many
   country slices, no duplicated payload.
5. **gzip** the JSON to ``results/forecast-<region>-YYYY-MM-DD.json.gz`` (a few hundred KB – few MB)
   and update ``results/index.json`` (the ``latest`` pointer, the per-view list, + rolling
   calibration history).

A loader (:func:`load_artifact`) round-trips a written file back to a :class:`ForecastArtifact`,
de-quantizing rates via the embedded legend, so the back-analysis and tests can re-read what shipped.

Only core deps at module top level. ``h3`` is a declared core dependency but is imported **lazily**
inside :func:`_to_h3` with a clear, actionable error if it is missing — and when the fine cell keys
are already H3 indices (not ``"lat,lon"``) the H3 library is not needed at all.
"""

from __future__ import annotations

import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT, load
from ..contracts import (
    ARTIFACT_SCHEMA_VERSION,
    BBox,
    CalibrationSummary,
    ForecastArtifact,
    Region,
    Staleness,
    ViewIndexEntry,
)

RESULTS_DIR = REPO_ROOT / "results"

# ─────────────────────────────────────────────────────────────────────────────
# Rate quantization legend (log scale → uint16). Shared by writer + loader + SPA.
# ─────────────────────────────────────────────────────────────────────────────

#: Smallest non-zero rate we encode; anything below maps to the implicit-baseline bin (code 0).
_RATE_MIN = 1e-6
#: Largest rate we encode before clamping (an aftershock burst can exceed this; clamped at the top).
_RATE_MAX = 1e3
#: Number of quantization codes (uint16 head-room; the legend interpolates log-linearly between).
_RATE_LEVELS = 65535


def _rate_legend() -> dict[str, Any]:
    """The legend the SPA uses to decode quantized rate codes back to floats (embedded in the file)."""
    return {
        "kind": "log_uint16",
        "rate_min": _RATE_MIN,
        "rate_max": _RATE_MAX,
        "levels": _RATE_LEVELS,
        "note": "code 0 = below rate_min (implicit baseline); code k>0 = rate_min*(rate_max/rate_min)^((k-1)/(levels-1))",
    }


def quantize_rate(rate: float) -> int:
    """Encode a non-negative rate to a uint16 code on a log scale (0 ⇒ below ``_RATE_MIN``)."""
    if not math.isfinite(rate) or rate < _RATE_MIN:
        return 0
    r = min(rate, _RATE_MAX)
    frac = math.log(r / _RATE_MIN) / math.log(_RATE_MAX / _RATE_MIN)
    return 1 + int(round(frac * (_RATE_LEVELS - 2)))


def dequantize_rate(code: int) -> float:
    """Decode a uint16 code back to a rate (inverse of :func:`quantize_rate`; code 0 ⇒ 0.0)."""
    if code <= 0:
        return 0.0
    frac = (code - 1) / (_RATE_LEVELS - 2)
    return float(_RATE_MIN * (_RATE_MAX / _RATE_MIN) ** frac)


# ─────────────────────────────────────────────────────────────────────────────
# H3 aggregation
# ─────────────────────────────────────────────────────────────────────────────


def _looks_like_latlon_key(key: str) -> bool:
    """True if a cell key is the fine-grid ``"lat,lon"`` form (vs an H3 index string)."""
    if "," not in key:
        return False
    a, _, b = key.partition(",")
    try:
        float(a)
        float(b)
        return True
    except ValueError:
        return False


def _to_h3(lat: float, lon: float, resolution: int) -> str:
    """Map a lat/lon to its H3 cell at ``resolution`` (lazy import; clear error if h3 is absent)."""
    try:
        import h3
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without h3 installed
        raise ModuleNotFoundError(
            "the 'h3' package is required to aggregate the fine forecast grid to H3 hexbins for the "
            "compact artifact. It is a core dependency — install it with `pip install h3` (or "
            "`pip install -e .`). If your cell keys are already H3 indices, no aggregation is needed."
        ) from exc
    # h3 v4 (latlng_to_cell) and v3 (geo_to_h3) both exist in the wild — support either.
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lon, resolution)
    return h3.geo_to_h3(lat, lon, resolution)  # type: ignore[attr-defined]


def _combine_at_least_one(probs: list[float]) -> float:
    """Probability that AT LEAST ONE of independent ``≥1 event`` outcomes occurs: ``1 - Π(1 - p_i)``.

    This is the correct way to aggregate fine-cell "≥1 event ≥ M*" probabilities into a coarse hexbin
    (the events are in disjoint cells, so independence is the right first-order assumption).
    """
    prod = 1.0
    for p in probs:
        prod *= max(0.0, 1.0 - min(max(p, 0.0), 1.0))
    return float(min(max(1.0 - prod, 0.0), 1.0))


def aggregate_to_h3(
    forecast: dict[str, dict[str, dict[str, dict[str, float]]]],
    resolution: int,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Aggregate the fine ``"lat,lon"`` forecast dict to H3 hexbins at ``resolution``.

    Per H3 cell × horizon × threshold: probabilities (``p``, ``lo``, ``hi``, ``baseline``) combine
    via :func:`_combine_at_least_one`; the expected count ``rate`` sums. Keys that are already H3
    indices pass through unchanged (no double aggregation, no h3 import needed).
    """
    # Group fine cells by their H3 parent.
    members: dict[str, list[str]] = {}
    passthrough: dict[str, str] = {}
    for cell_key in forecast:
        if _looks_like_latlon_key(cell_key):
            lat_s, _, lon_s = cell_key.partition(",")
            h3_key = _to_h3(float(lat_s), float(lon_s), resolution)
            members.setdefault(h3_key, []).append(cell_key)
        else:
            passthrough[cell_key] = cell_key  # already an H3 (or opaque) key

    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for h3_key, fine_keys in members.items():
        agg: dict[str, dict[str, dict[str, float]]] = {}
        for fine in fine_keys:
            for horizon, by_thr in forecast[fine].items():
                for m_star, vals in by_thr.items():
                    bucket = agg.setdefault(horizon, {}).setdefault(m_star, {
                        "_p": [], "_lo": [], "_hi": [], "_rate": 0.0, "_baseline": []
                    })
                    bucket["_p"].append(float(vals.get("p", 0.0)))
                    bucket["_lo"].append(float(vals.get("lo", 0.0)))
                    bucket["_hi"].append(float(vals.get("hi", 0.0)))
                    bucket["_baseline"].append(float(vals.get("baseline", 0.0)))
                    bucket["_rate"] += float(vals.get("rate", 0.0))
        out[h3_key] = {
            horizon: {
                m_star: {
                    "p": round(_combine_at_least_one(b["_p"]), 6),
                    "lo": round(_combine_at_least_one(b["_lo"]), 6),
                    "hi": round(_combine_at_least_one(b["_hi"]), 6),
                    "rate": round(b["_rate"], 6),
                    "baseline": round(_combine_at_least_one(b["_baseline"]), 6),
                }
                for m_star, b in by_thr.items()
            }
            for horizon, by_thr in agg.items()
        }

    # Pass any already-H3 cells straight through.
    for key in passthrough:
        out[key] = forecast[key]
    return out


def aggregate_to_h3_multi(
    forecast: dict[str, dict[str, dict[str, dict[str, float]]]],
    world_resolution: int,
    refinements: list[tuple[BBox, int]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Multi-resolution H3 aggregation: coarse worldwide + finer inside refinement bboxes.

    The GLOBAL field is far too large to ship at one fine resolution (web-app-spec.md §8.2). Instead
    each fine ``"lat,lon"`` cell is binned at:

    * the **world resolution** (coarse) by default — the planet-wide overview the SPA renders first; or
    * a **finer resolution** when the cell's centre falls inside a refinement bbox (a configured view,
      e.g. a country drill-down) — so a user zooming into a country sees the higher-resolution field
      while the world stays light.

    ``refinements`` is ``[(bbox, resolution), ...]``; the **highest** resolution among the bboxes a
    cell falls inside wins (a country inside a region inside the world resolves to the country's
    detail). Aggregation within each H3 parent is the same as :func:`aggregate_to_h3` (``≥1``-event
    probabilities combine via ``1 - Π(1-p)``, rates sum). Already-H3 keys pass through unchanged.
    """
    members: dict[str, list[str]] = {}
    passthrough: dict[str, str] = {}
    for cell_key in forecast:
        if not _looks_like_latlon_key(cell_key):
            passthrough[cell_key] = cell_key
            continue
        lat_s, _, lon_s = cell_key.partition(",")
        lat, lon = float(lat_s), float(lon_s)
        res = world_resolution
        for bbox, r in refinements:
            if _point_in_bbox(lat, lon, bbox) and r > res:
                res = r
        h3_key = _to_h3(lat, lon, res)
        members.setdefault(h3_key, []).append(cell_key)

    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for h3_key, fine_keys in members.items():
        out[h3_key] = _combine_fine_cells(forecast, fine_keys)
    for key in passthrough:
        out[key] = forecast[key]
    return out


def _combine_fine_cells(
    forecast: dict[str, dict[str, dict[str, dict[str, float]]]],
    fine_keys: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Combine several fine cells into one H3 parent (shared by the single/multi aggregators)."""
    agg: dict[str, dict[str, dict[str, float]]] = {}
    for fine in fine_keys:
        for horizon, by_thr in forecast[fine].items():
            for m_star, vals in by_thr.items():
                bucket = agg.setdefault(horizon, {}).setdefault(m_star, {
                    "_p": [], "_lo": [], "_hi": [], "_rate": 0.0, "_baseline": []
                })
                bucket["_p"].append(float(vals.get("p", 0.0)))
                bucket["_lo"].append(float(vals.get("lo", 0.0)))
                bucket["_hi"].append(float(vals.get("hi", 0.0)))
                bucket["_baseline"].append(float(vals.get("baseline", 0.0)))
                bucket["_rate"] += float(vals.get("rate", 0.0))
    return {
        horizon: {
            m_star: {
                "p": round(_combine_at_least_one(b["_p"]), 6),
                "lo": round(_combine_at_least_one(b["_lo"]), 6),
                "hi": round(_combine_at_least_one(b["_hi"]), 6),
                "rate": round(b["_rate"], 6),
                "baseline": round(_combine_at_least_one(b["_baseline"]), 6),
            }
            for m_star, b in by_thr.items()
        }
        for horizon, by_thr in agg.items()
    }


def _point_in_bbox(lat: float, lon: float, bbox: BBox) -> bool:
    """True if ``(lat, lon)`` falls inside ``bbox`` (inclusive edges)."""
    return bbox.lat_min <= lat <= bbox.lat_max and bbox.lon_min <= lon <= bbox.lon_max


# ─────────────────────────────────────────────────────────────────────────────
# Compaction (quantize + sparsify)
# ─────────────────────────────────────────────────────────────────────────────


def _max_prob_over_cell(by_horizon: dict[str, dict[str, dict[str, float]]]) -> float:
    """Maximum expected probability across all horizons/thresholds for one cell (for sparsity)."""
    best = 0.0
    for by_thr in by_horizon.values():
        for vals in by_thr.values():
            best = max(best, float(vals.get("p", 0.0)))
    return best


def compact_forecast(
    h3_forecast: dict[str, dict[str, dict[str, dict[str, float]]]],
    rate_floor_quantile: float,
    *,
    min_abs_prob: float = 0.0,
    protected_keys: set[str] | None = None,
) -> tuple[dict[str, dict[str, dict[str, dict[str, float]]]], int]:
    """Quantize rates to uint16 codes and drop near-zero cells (sparsity), returning ``(compact, n_dropped)``.

    The probability triad stays as rounded floats (6 dp — small and human-auditable); only the
    *rate* is quantized, since it spans many orders of magnitude and dominates the byte count.

    A cell is dropped to the implicit baseline only if it is below BOTH thresholds and is not
    protected:

    * ``rate_floor_quantile`` — the empirical quantile of per-cell max-probability (the existing
      relative floor; ``0`` keeps everything). For the WORLD field this is the dominant control (most
      of the planet is quiet, so dropping the quietest fraction is what keeps the artifact small).
    * ``min_abs_prob`` — an absolute floor (``grid.yaml: sparsity.min_world_prob``) so that even a
      field where the quantile is loose never ships cells whose max ``P(>=1)`` is negligibly small.
    * ``protected_keys`` — H3 keys a configured view needs (``sparsity.keep_view_cells``); never
      dropped, so a country drill-down keeps coverage continuity even where the world floor would
      have removed a quiet cell. **Blank never means safe** — the coverage mask, not deletion, marks
      out-of-coverage.
    """
    if not h3_forecast:
        return {}, 0

    protected = protected_keys or set()

    # Relative floor: the empirical quantile of per-cell max-probability.
    maxes = sorted(_max_prob_over_cell(v) for v in h3_forecast.values())
    q = min(max(float(rate_floor_quantile), 0.0), 1.0)
    if q <= 0.0 or len(maxes) == 0:
        rel_floor = -1.0  # keep everything (relative test passes for all)
    else:
        idx = min(int(q * len(maxes)), len(maxes) - 1)
        rel_floor = maxes[idx]

    abs_floor = max(float(min_abs_prob), 0.0)

    compact: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    dropped = 0
    for key, by_horizon in h3_forecast.items():
        if key not in protected:
            cell_max = _max_prob_over_cell(by_horizon)
            if cell_max < rel_floor and cell_max < abs_floor:
                dropped += 1
                continue
        compact[key] = {
            horizon: {
                m_star: {
                    "p": round(float(vals.get("p", 0.0)), 6),
                    "lo": round(float(vals.get("lo", 0.0)), 6),
                    "hi": round(float(vals.get("hi", 0.0)), 6),
                    "baseline": round(float(vals.get("baseline", 0.0)), 6),
                    "q": quantize_rate(float(vals.get("rate", 0.0))),  # quantized rate code
                }
                for m_star, vals in by_thr.items()
            }
            for horizon, by_thr in by_horizon.items()
        }
    return compact, dropped


# ─────────────────────────────────────────────────────────────────────────────
# Serialize / write
# ─────────────────────────────────────────────────────────────────────────────


def artifact_filename(region_id: str, issued_at: str) -> str:
    """Compact file name ``forecast-<region>-YYYY-MM-DD.json.gz`` (UTC date of the issue time)."""
    day = issued_at[:10]
    return f"forecast-{region_id}-{day}.json.gz"


def serialize_artifact(
    artifact: ForecastArtifact,
    grid_cfg: dict | None = None,
) -> dict[str, Any]:
    """Build the compact JSON-able dict (H3-aggregated, quantized, sparsified) WITHOUT writing it.

    Separated from :func:`write_artifact` so callers/tests can inspect or size the payload in memory.
    The returned dict embeds the rate legend so a loader can de-quantize without external state.

    Global field: when the artifact carries country **views**, the fine cells are aggregated
    *multi-resolution* — the world resolution everywhere, refined to each view's finer resolution
    inside that view's bbox — and each view's H3 cell-key index (+ ``n_cells``) is filled from the
    aggregated keys, so the SPA's country selector reads only the relevant cells of the one global
    field. Sparsity drops the quietest world cells (relative + absolute floor) but never a cell a
    view needs.
    """
    grid_cfg = grid_cfg or load("grid")
    display = grid_cfg.get("display", {})
    sparsity = grid_cfg.get("sparsity", {})

    grid_block = artifact.grid or {}
    world_res = int(grid_block.get("resolution_world", display.get("h3_resolution_world", 3)))
    region_res = int(grid_block.get("resolution_region", display.get("h3_resolution_region", 5)))
    base_res = int(grid_block.get("resolution", region_res))

    rate_floor_q = float(sparsity.get("rate_floor_quantile", 0.0))
    min_world_prob = float(sparsity.get("min_world_prob", 0.0))
    keep_view_cells = bool(sparsity.get("keep_view_cells", True))

    is_global = artifact.region.id == "global" or bool(artifact.views)

    if is_global and artifact.views:
        # Multi-resolution: refine each view's bbox to its own resolution (falls back to region_res).
        refinements: list[tuple[BBox, int]] = [
            (v.bbox, int(v.h3_resolution) if v.h3_resolution is not None else region_res)
            for v in artifact.views
        ]
        h3_forecast = aggregate_to_h3_multi(artifact.forecast, world_res, refinements)
        out_res = world_res
    else:
        h3_forecast = aggregate_to_h3(artifact.forecast, base_res)
        out_res = base_res

    # Compute each view's cell-key index from the aggregated (display) keys BEFORE sparsity so a view
    # keeps its full footprint, then protect those keys from being dropped.
    view_indices: dict[str, list[str]] = {}
    protected: set[str] = set()
    for v in artifact.views:
        keys = _cells_in_bbox(h3_forecast, v.bbox)
        view_indices[v.id] = keys
        if keep_view_cells:
            protected.update(keys)

    compact, n_dropped = compact_forecast(
        h3_forecast,
        rate_floor_q,
        min_abs_prob=min_world_prob,
        protected_keys=protected,
    )

    # Fill the view entries' cell lists (intersected with what actually survived compaction).
    surviving = set(compact)
    filled_views: list[dict[str, Any]] = []
    for v in artifact.views:
        keys = [k for k in view_indices[v.id] if k in surviving]
        entry = ViewIndexEntry(
            id=v.id,
            name_en=v.name_en,
            name_es=v.name_es,
            bbox=v.bbox,
            m_max=v.m_max,
            attribution=v.attribution,
            h3_resolution=v.h3_resolution if v.h3_resolution is not None else region_res,
            cells=keys,
            n_cells=len(keys),
        )
        filled_views.append(entry.model_dump(mode="json"))

    payload = artifact.model_dump(mode="json")
    payload["forecast"] = compact
    payload["grid"] = {
        "type": "h3",
        "resolution": out_res,
        "resolution_world": world_res,
        "resolution_region": region_res,
    }
    payload["views"] = filled_views
    payload["rate_legend"] = _rate_legend()
    payload["compaction"] = {
        "n_h3_cells": len(compact),
        "n_dropped_below_floor": n_dropped,
        "rate_floor_quantile": rate_floor_q,
        "min_world_prob": min_world_prob,
        "multi_resolution": bool(is_global and artifact.views),
        "n_views": len(filled_views),
    }
    return payload


def _cells_in_bbox(
    h3_forecast: dict[str, dict[str, dict[str, dict[str, float]]]], bbox: BBox
) -> list[str]:
    """H3 (or ``"lat,lon"``) cell keys whose centre falls inside ``bbox`` — one view's cell index.

    Used to build each view's index into the shared global forecast dict. H3 keys are resolved to
    their centre via the lazy h3 import; ``"lat,lon"`` keys are parsed directly (no h3 needed).
    """
    out: list[str] = []
    for key in h3_forecast:
        lat, lon = _key_centroid(key)
        if lat is None or lon is None:
            continue
        if _point_in_bbox(lat, lon, bbox):
            out.append(key)
    return out


def _key_centroid(key: str) -> tuple[float | None, float | None]:
    """Centroid ``(lat, lon)`` of a cell key — ``"lat,lon"`` parsed directly, or an H3 cell centre."""
    if _looks_like_latlon_key(key):
        a, _, b = key.partition(",")
        return float(a), float(b)
    try:
        import h3
    except ModuleNotFoundError:
        return None, None
    try:
        if hasattr(h3, "cell_to_latlng"):
            lat, lon = h3.cell_to_latlng(key)
        else:
            lat, lon = h3.h3_to_geo(key)  # type: ignore[attr-defined]
        return float(lat), float(lon)
    except Exception:
        return None, None


def write_artifact(
    artifact: ForecastArtifact,
    grid_cfg: dict | None = None,
    results_dir: Path | None = None,
) -> dict[str, Path]:
    """Serialize + gzip the compact artifact to ``results/`` and update ``results/index.json``.

    Returns ``{"artifact": <path to .json.gz>, "index": <path to index.json>}``. The dense fine grid
    is never written — only the H3-aggregated, quantized, sparsified payload. The write is atomic
    (temp file + replace) so a crashed run cannot leave a half-written artifact the SPA might serve.
    """
    out_dir = results_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = serialize_artifact(artifact, grid_cfg=grid_cfg)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    fname = artifact_filename(artifact.region.id, artifact.issued_at)
    path = out_dir / fname
    tmp = out_dir / (fname + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=9) as fh:
        fh.write(raw)
    tmp.replace(path)

    index_path = update_index(artifact, path, out_dir, payload_size_bytes=path.stat().st_size)
    return {"artifact": path, "index": index_path}


def update_index(
    artifact: ForecastArtifact,
    artifact_path: Path,
    results_dir: Path,
    payload_size_bytes: int | None = None,
) -> Path:
    """Update ``results/index.json``: the ``latest`` pointer, the rolling file list, and calibration.

    The index is the SPA's directory: ``/api/forecast/latest`` resolves through it. It keeps a
    bounded rolling history (one entry per daily file) and a ``calibration`` block (the reliability
    pairs the next day's isotonic fit reads back). Written atomically.
    """
    index_path = results_dir / "index.json"
    index: dict[str, Any] = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}

    entry = {
        "issued_at": artifact.issued_at,
        "region": artifact.region.id,
        "file": artifact_path.name,
        "horizons_days": artifact.horizons_days,
        "magnitude_thresholds": artifact.magnitude_thresholds,
        "size_bytes": int(payload_size_bytes or 0),
        "staleness_ok": artifact.staleness.ok,
        "generated": artifact.staleness.generated,
        "next_run": artifact.staleness.next_run,
        # The country views available as slices of this (global) field — the SPA's region selector.
        "views": [
            {"id": v.id, "name_en": v.name_en, "n_cells": v.n_cells}
            for v in artifact.views
        ],
    }

    forecasts = [e for e in index.get("forecasts", []) if isinstance(e, dict)]
    # Replace any existing entry for the same region+date (re-runs overwrite, never duplicate).
    forecasts = [
        e for e in forecasts
        if not (e.get("region") == entry["region"] and str(e.get("issued_at", ""))[:10] == artifact.issued_at[:10])
    ]
    forecasts.append(entry)
    forecasts.sort(key=lambda e: str(e.get("issued_at", "")))
    # Bound the rolling history (per region) to a sane window for the "forecast from {date}" picker.
    forecasts = forecasts[-400:]

    latest_by_region: dict[str, dict[str, Any]] = {}
    for e in forecasts:
        latest_by_region[str(e.get("region"))] = e
    latest = max(forecasts, key=lambda e: str(e.get("issued_at", "")), default=entry)

    # Carry forward / refresh the rolling calibration reliability block from the shipped artifact.
    calibration = index.get("calibration", {}) if isinstance(index.get("calibration"), dict) else {}
    if artifact.calibration.reliability:
        calibration = {
            "reliability": [list(map(float, row[:2])) for row in artifact.calibration.reliability],
            "updated_at": artifact.issued_at,
            "csep": artifact.calibration.csep,
        }

    index.update(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "product": "CAOS_SEISMIC",
            "updated_at": _utc_now_iso(),
            "latest": latest,
            "latest_by_region": latest_by_region,
            "forecasts": forecasts,
            "calibration": calibration,
        }
    )

    tmp = index_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(index_path)
    return index_path


# ─────────────────────────────────────────────────────────────────────────────
# Load (round-trip)
# ─────────────────────────────────────────────────────────────────────────────


def load_artifact(path: Path | str) -> ForecastArtifact:
    """Load a written ``forecast-*.json.gz`` (or plain ``.json``) back into a :class:`ForecastArtifact`.

    De-quantizes each cell's rate code (``q``) back to a float ``rate`` via the embedded legend, so
    the returned artifact's ``forecast`` dict matches the in-memory schema ({p, lo, hi, rate,
    baseline}). The back-analysis and round-trip tests read what actually shipped, not the source
    field.
    """
    p = Path(path)
    raw = gzip.open(p, "rb").read() if p.suffix == ".gz" else p.read_bytes()
    data = json.loads(raw.decode("utf-8"))

    # Re-hydrate the quantized rate codes back to float rates.
    forecast = data.get("forecast", {})
    rehydrated: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for cell, by_horizon in forecast.items():
        rehydrated[cell] = {}
        for horizon, by_thr in by_horizon.items():
            rehydrated[cell][horizon] = {}
            for m_star, vals in by_thr.items():
                row = {
                    "p": float(vals.get("p", 0.0)),
                    "lo": float(vals.get("lo", 0.0)),
                    "hi": float(vals.get("hi", 0.0)),
                    "baseline": float(vals.get("baseline", 0.0)),
                }
                if "q" in vals:
                    row["rate"] = dequantize_rate(int(vals["q"]))
                elif "rate" in vals:
                    row["rate"] = float(vals["rate"])
                else:
                    row["rate"] = 0.0
                rehydrated[cell][horizon][m_star] = row

    views = [ViewIndexEntry(**v) for v in data.get("views", []) if isinstance(v, dict)]

    return ForecastArtifact(
        schema_version=data.get("schema_version", ARTIFACT_SCHEMA_VERSION),
        product=data.get("product", "CAOS_SEISMIC"),
        issued_at=data["issued_at"],
        region=Region(**data["region"]),
        horizons_days=[int(h) for h in data.get("horizons_days", [])],
        magnitude_thresholds=[float(m) for m in data.get("magnitude_thresholds", [])],
        m_max=float(data.get("m_max", data["region"].get("m_max", 0.0))),
        grid=data.get("grid", {"type": "h3", "resolution": 5}),
        forecast=rehydrated,
        calibration=CalibrationSummary(**data.get("calibration", {})),
        coverage_mask=list(data.get("coverage_mask", [])),
        views=views,
        provenance=data.get("provenance", {}),
        staleness=Staleness(**data["staleness"]),
    )


def load_index(results_dir: Path | None = None) -> dict[str, Any]:
    """Load ``results/index.json`` (returns ``{}`` if it does not exist yet)."""
    index_path = (results_dir or RESULTS_DIR) / "index.json"
    if not index_path.exists():
        return {}
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

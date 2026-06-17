"""Compact artifact serialization — the contract between the offline job and the static SPA.

The dense fine grid (0.1° cells over a region; the global single-resolution CSEP grid is ~6.48M
cells) is **never** shipped to the browser (web-app-spec.md §8.2). This module turns an in-memory
:class:`~caos_seismic.contracts.ForecastArtifact` (keyed by fine ``"lat,lon"`` cells) into the
compact on-disk form the SPA reads:

1. **Aggregate** the fine cells to **H3** hexbins at the display resolution (configs/grid.yaml
   ``display.h3_resolution_region``). Per H3 cell × horizon × threshold the exceedance probability is
   combined as ``p = 1 - Π(1 - p_i)`` (the probability that *at least one* contained fine cell
   exceeds — the correct aggregation of independent "≥1 event" events), rates are summed, and the
   baseline is aggregated the same way as ``p``.
2. **Quantize** the per-cell rate to a small integer via a log scale + legend lookup, so the browser
   decodes ``uint16 → rate`` with a shared legend rather than carrying float64 (web-app-spec §8.2).
3. **Sparsity**: drop H3 cells whose maximum probability across all horizons/thresholds is below a
   rate floor (``configs/grid.yaml: sparsity.rate_floor_quantile``) — the rest is the implicit
   baseline. The coverage mask carries cells explicitly *out* of validated coverage (blank ≠ safe).
4. **gzip** the JSON to ``results/forecast-<region>-YYYY-MM-DD.json.gz`` (a few hundred KB – few MB)
   and update ``results/index.json`` (the ``latest`` pointer + rolling calibration history).

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
    CalibrationSummary,
    ForecastArtifact,
    Region,
    Staleness,
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
) -> tuple[dict[str, dict[str, dict[str, dict[str, float]]]], int]:
    """Quantize rates to uint16 codes and drop near-zero cells (sparsity), returning ``(compact, n_dropped)``.

    The probability triad stays as rounded floats (6 dp — small and human-auditable); only the
    *rate* is quantized, since it spans many orders of magnitude and dominates the byte count. Cells
    whose max probability is at/below the ``rate_floor_quantile`` of the field are dropped to the
    implicit baseline (``rate_floor_quantile == 0`` keeps everything, the config default).
    """
    if not h3_forecast:
        return {}, 0

    # Threshold for sparsity: the empirical quantile of per-cell max-probability.
    maxes = sorted(_max_prob_over_cell(v) for v in h3_forecast.values())
    q = min(max(float(rate_floor_quantile), 0.0), 1.0)
    if q <= 0.0 or len(maxes) == 0:
        floor = -1.0  # keep everything
    else:
        idx = min(int(q * len(maxes)), len(maxes) - 1)
        floor = maxes[idx]

    compact: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    dropped = 0
    for key, by_horizon in h3_forecast.items():
        if _max_prob_over_cell(by_horizon) < floor:
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
    """
    grid_cfg = grid_cfg or load("grid")
    resolution = int(artifact.grid.get("resolution", grid_cfg.get("display", {}).get("h3_resolution_region", 5)))
    rate_floor_q = float(grid_cfg.get("sparsity", {}).get("rate_floor_quantile", 0.0))

    h3_forecast = aggregate_to_h3(artifact.forecast, resolution)
    compact, n_dropped = compact_forecast(h3_forecast, rate_floor_q)

    payload = artifact.model_dump(mode="json")
    payload["forecast"] = compact
    payload["grid"] = {"type": "h3", "resolution": resolution}
    payload["rate_legend"] = _rate_legend()
    payload["compaction"] = {
        "n_h3_cells": len(compact),
        "n_dropped_below_floor": n_dropped,
        "rate_floor_quantile": rate_floor_q,
    }
    return payload


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

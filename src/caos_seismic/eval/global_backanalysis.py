"""Global, multi-view pseudo-prospective CSEP back-analysis — the THESIS measurement.

The re-scoped product is a single **global** context-conditioned forecaster; any country is a *view*
into that global field (see :mod:`caos_seismic.eval.views`). This driver runs the leakage-free
forecast-clock back-analysis (:func:`caos_seismic.eval.backanalysis.run_back_analysis`) through every
pre-registered country view AND a global view, then reduces the per-view ledgers to the two headline
numbers the whole re-scoping exists to report:

1. **Context gain over catalog-only ETAS** — per view × horizon and pooled globally. ETAS already
   reproduces Omori/Utsu clustering, so a positive, significant information gain (in **nats**) over
   it is *not* "I predicted aftershocks" — it quantifies how much the **global context** (worldwide
   seismicity + complementary covariates) adds to the *local* short-term forecast. This is the direct
   answer to "how much does context contribute?". When the context channel has not yet landed (the
   enricher stack is feature-flagged off, model-design §6.2), this gain is ~0 by construction and the
   summary says so honestly via ``context_channel_active`` — it is never faked into a positive value.

2. **High-vs-low-seismicity bias** — the same skill/calibration metrics computed separately over the
   HIGH-seismicity views (active plate boundaries: Chile, Japan, California, NZ) and the
   LOW-seismicity views (stable interiors: C&E US, W Europe, E Australia), with their difference. A
   single pooled global number is dominated by the loud subduction margins; this partition asks the
   adversarial question — *does the model only look good because it over-fits high-seismicity zones?*
   The gap (``high − low``) in N-test pass rate, IGPE vs null, context gain, and Brier is the bias
   metric. A large positive gap is a red flag the summary surfaces rather than hides.

Everything is reported honestly including failures and underperformance (evaluation-plan §4.5/§7):
a view × horizon where the model fails to beat its baselines, or where the context gain is negative,
is emitted as such — selective reporting is the exact selection-bias trap CSEP exists to prevent.

The driver imports only core deps at module top level; the per-view back-analysis (which lazily
imports the inference machinery) is called inside :func:`run_global_back_analysis`, so importing this
module stays light.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from ..contracts import Region, validate_catalog
from .backanalysis import BackAnalysisConfig, BackAnalysisResult, run_back_analysis
from .views import CountryView, all_views

logger = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "results"

#: The per-horizon scalar fields aggregated across views for the high-vs-low bias comparison. Each
#: maps a name → (per_horizon key, "higher is better" direction). Brier is the only "lower is better".
_BIAS_FIELDS: dict[str, tuple[str, bool]] = {
    "n_test_pass_rate": ("n_test_pass_rate", True),
    "mean_igpe_vs_null_nats": ("mean_igpe_vs_null_nats", True),
    "mean_context_gain_vs_etas_nats": ("mean_context_gain_vs_etas_nats", True),
    "brier": ("brier", False),
}


# ─────────────────────────────────────────────────────────────────────────────
# Config + result containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GlobalBackAnalysisConfig:
    """Inputs defining one global, multi-view back-analysis run.

    Attributes
    ----------
    start, end:
        First and last issue dates (UTC), shared by every view so the comparison is apples-to-apples.
    horizons_days, magnitude_thresholds, reliability_threshold:
        Forwarded to each per-view :class:`~caos_seismic.eval.backanalysis.BackAnalysisConfig`. Frozen
        across all views and the whole test split (evaluation-plan §2 — no per-view snooping).
    include_global:
        Run the GLOBAL view (the whole field) in addition to the country views (default on).
    views:
        Explicit view list override; defaults to the pre-registered set
        (:func:`caos_seismic.eval.views.all_views`).
    issue_hour_utc, rng_seed:
        Forwarded to each per-view run so a global back-analysis is byte-reproducible.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    horizons_days: list[int] = field(default_factory=lambda: [1, 2, 7])
    magnitude_thresholds: list[float] = field(default_factory=lambda: [5.0, 6.0, 7.0])
    reliability_threshold: float = 5.0
    include_global: bool = True
    views: list[CountryView] | None = None
    issue_hour_utc: int = 0
    rng_seed: int = 20260616
    #: Per-view back-analysis SCORING grid size (degrees). ``None`` ⇒ :data:`DEFAULT_SCORING_CELL_DEG`
    #: (0.5°, coarser than the 0.1° production fit grid — M≥5 events are sparse, so a coarser scoring
    #: grid gives better-powered CSEP tests AND ~25× cheaper daily inference). The global view coarsens
    #: further automatically. Set finer (e.g. 0.25) for a denser study at a higher per-day cost.
    scoring_cell_deg: float | None = None
    #: Full-MLE refit cadence (days) forwarded to each per-view :class:`BackAnalysisConfig`.
    refit_every_days: int = 7


@dataclass
class ViewResult:
    """One view's back-analysis result + its pre-registered seismicity class and tectonic setting."""

    view_id: str
    name_en: str
    seismicity_class: str
    plate_setting: str
    fit_cell_deg: float | None = None  # effective fit-grid cell size (None ⇒ configured fine grid)
    result: BackAnalysisResult | None = field(default=None, repr=False)
    error: str | None = None  # a view that could not be scored at all (kept, never dropped)


@dataclass
class GlobalBackAnalysisResult:
    """The accumulated global back-analysis across all views + the cross-view reductions.

    ``per_view`` is one block per country/global view (the per-horizon ledger the web app drills
    into). ``context_gain`` is the THESIS headline reduced per view × horizon and pooled. ``bias``
    is the HIGH-vs-LOW-seismicity comparison (skill/calibration in each class and their gap).
    """

    start: str
    end: str
    view_ids: list[str]
    horizons_days: list[int]
    per_view: list[dict[str, Any]] = field(default_factory=list)
    context_gain: dict[str, Any] = field(default_factory=dict)
    bias: dict[str, Any] = field(default_factory=dict)
    summary_path: str | None = None
    pycsep_used: bool = False

    def __str__(self) -> str:
        loc = self.summary_path or "(not written)"
        n_ok = sum(1 for v in self.per_view if not v.get("error"))
        return (
            f"global back-analysis {self.start[:10]}→{self.end[:10]}: "
            f"{n_ok}/{len(self.view_ids)} views scored -> {loc}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


def run_global_back_analysis(
    config: GlobalBackAnalysisConfig,
    *,
    catalog: pd.DataFrame | None = None,
    write_summary: bool = True,
    results_dir: Path | None = None,
) -> GlobalBackAnalysisResult:
    """Run the multi-view back-analysis and reduce it to the context-gain + high-vs-low-bias summary.

    For each pre-registered view, the master catalog is sliced to the view's bounding box (so the
    same global model is scored through that country window) and the leakage-free per-view
    back-analysis runs. The per-view per-horizon ledgers are then reduced across views to:

    * the **context gain** vs catalog-only ETAS (per view × horizon and pooled), and
    * the **HIGH-vs-LOW-seismicity bias** (skill/calibration per class + the gap).

    A compact JSON for the web app is written into ``results/`` (unless ``write_summary=False``).

    Parameters
    ----------
    config:
        The :class:`GlobalBackAnalysisConfig` (period, horizons, thresholds, views, cadence).
    catalog:
        The GLOBAL master catalog (all events worldwide). If ``None`` each view loads its own cleaned
        store lazily via the per-view driver. Passing one global catalog is preferred: it is sliced
        per view here so every view is a true window into the *same* field.
    write_summary:
        Write the compact global JSON summary into ``results/`` (default on).
    results_dir:
        Override the output directory (defaults to ``results/``).
    """
    views = config.views if config.views is not None else all_views(include_global=config.include_global)
    master = validate_catalog(catalog) if catalog is not None else None
    cell_deg_by_area = _grid_cell_deg_resolver(scoring_cell_deg=config.scoring_cell_deg)

    view_results: list[ViewResult] = []
    pycsep_used = False
    for cv in views:
        reg = cv.region
        view_catalog = _slice_to_view(master, reg) if master is not None else None
        view_cell_deg = cell_deg_by_area(reg)
        vr = ViewResult(
            view_id=reg.id,
            name_en=reg.name_en,
            seismicity_class=cv.seismicity_class,
            plate_setting=cv.plate_setting,
            fit_cell_deg=view_cell_deg,
        )
        try:
            sub_cfg = BackAnalysisConfig(
                region=reg,
                start=config.start,
                end=config.end,
                horizons_days=[int(h) for h in config.horizons_days],
                magnitude_thresholds=[float(m) for m in config.magnitude_thresholds],
                reliability_threshold=float(config.reliability_threshold),
                issue_hour_utc=int(config.issue_hour_utc),
                rng_seed=int(config.rng_seed),
                # Back-analysis SCORING grid (DEFAULT_SCORING_CELL_DEG = 0.5°, coarser than the 0.1°
                # production fit grid — M≥5 events are sparse, so a coarser grid powers the CSEP tests
                # better AND cuts the dominant per-day inference cost ~25×). The whole-Earth GLOBAL window
                # coarsens further (the dense 0.1° global grid is ~6.5M cells, never materialized).
                cell_deg=view_cell_deg,
                # Fit the per-tile MLE weekly + recondition daily (mirrors the live cadence).
                refit_every_days=int(config.refit_every_days),
            )
            # Per-view summaries are written too (one file per view) so each view is independently
            # auditable; the global summary is the cross-view reduction on top.
            vr.result = run_back_analysis(
                sub_cfg,
                catalog=view_catalog,
                write_summary=write_summary,
                results_dir=results_dir or RESULTS_DIR,
            )
            pycsep_used = pycsep_used or bool(vr.result.pycsep_used)
        except Exception as exc:  # a view we could not score at all — RECORD it, never drop it
            logger.warning("view %s could not be scored: %s", reg.id, exc)
            vr.error = f"{type(exc).__name__}: {exc}"
        view_results.append(vr)

    per_view = [_view_block(vr) for vr in view_results]
    context_gain = _reduce_context_gain(view_results, config.horizons_days)
    bias = _reduce_high_low_bias(view_results, config.horizons_days)

    result = GlobalBackAnalysisResult(
        start=config.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end=config.end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        view_ids=[vr.view_id for vr in view_results],
        horizons_days=[int(h) for h in config.horizons_days],
        per_view=per_view,
        context_gain=context_gain,
        bias=bias,
        pycsep_used=pycsep_used,
    )

    if write_summary:
        result.summary_path = str(_write_global_summary(result, config, results_dir or RESULTS_DIR))
    return result


def run_global_backanalysis(
    *,
    start: datetime | pd.Timestamp | str,
    end: datetime | pd.Timestamp | str,
    catalog: pd.DataFrame | None = None,
    horizons_days: list[int] | None = None,
    magnitude_thresholds: list[float] | None = None,
    include_global: bool = True,
    **kwargs: Any,
) -> GlobalBackAnalysisResult:
    """CLI-friendly entry point — build a :class:`GlobalBackAnalysisConfig` and run.

    Horizons / thresholds default to ``configs/forecast.yaml`` when not given.
    """
    from ..config import load

    forecast_cfg = load("forecast")
    horizons = horizons_days or [int(h) for h in forecast_cfg.get("horizons_days", [1, 2, 7])]
    thresholds = magnitude_thresholds or [
        float(m) for m in forecast_cfg.get("magnitude_thresholds", [5.0, 6.0, 7.0])
    ]
    config = GlobalBackAnalysisConfig(
        start=_as_utc(start),
        end=_as_utc(end),
        horizons_days=[int(h) for h in horizons],
        magnitude_thresholds=[float(m) for m in thresholds],
        reliability_threshold=min(thresholds) if thresholds else 5.0,
        include_global=include_global,
    )
    return run_global_back_analysis(config, catalog=catalog, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-view reductions
# ─────────────────────────────────────────────────────────────────────────────


def _view_block(vr: ViewResult) -> dict[str, Any]:
    """One view's compact block for the global summary (per-horizon ledger + class + setting)."""
    block: dict[str, Any] = {
        "view_id": vr.view_id,
        "name_en": vr.name_en,
        "seismicity_class": vr.seismicity_class,
        "plate_setting": vr.plate_setting,
        "fit_cell_deg": vr.fit_cell_deg,  # None ⇒ configured fine grid; a number ⇒ coarsened (large bbox)
    }
    if vr.error is not None or vr.result is None:
        block["error"] = vr.error or "no result"
        block["per_horizon"] = []
        return block
    r = vr.result
    block.update(
        {
            "n_issue_days": r.n_issue_days,
            "n_scored_days": r.n_scored_days,
            "n_failed_days": r.n_failed_days,
            "per_horizon": r.per_horizon,
            "reliability": r.reliability,
            "summary_path": Path(r.summary_path).name if r.summary_path else None,
        }
    )
    return block


def _reduce_context_gain(view_results: list[ViewResult], horizons: list[int]) -> dict[str, Any]:
    """THESIS headline: context gain over catalog-only ETAS, per view × horizon and pooled.

    For each view we surface the per-horizon ``mean_context_gain_vs_etas_nats`` and whether the
    context channel was active. The pooled (across-view) gain per horizon is the **scored-day-count
    weighted** mean of the per-view gains, so a view with more scored days carries proportionally
    more weight (rather than letting a tiny, barely-scored view dominate). Honest framing: if no view
    has an active context channel, the pooled gain is ~0 and ``context_channel_active`` is False.
    """
    per_view: dict[str, Any] = {}
    any_active = False
    # weighted accumulator per horizon: sum(gain * weight), sum(weight)
    pooled_num: dict[int, float] = {int(h): 0.0 for h in horizons}
    pooled_den: dict[int, float] = {int(h): 0.0 for h in horizons}

    for vr in view_results:
        if vr.result is None:
            continue
        by_h: dict[str, Any] = {}
        for ph in vr.result.per_horizon:
            h = int(ph["horizon_days"])
            gain = ph.get("mean_context_gain_vs_etas_nats")
            active = bool(ph.get("context_channel_active"))
            any_active = any_active or active
            n_scored = int(ph.get("n_scored", 0))
            by_h[str(h)] = {
                "mean_context_gain_vs_etas_nats": gain,
                "context_channel_active": active,
                "context_gain_positive": bool(ph.get("context_gain_positive")),
                "n_scored": n_scored,
            }
            if gain is not None and h in pooled_num:
                pooled_num[h] += float(gain) * n_scored
                pooled_den[h] += n_scored
        per_view[vr.view_id] = {
            "seismicity_class": vr.seismicity_class,
            "by_horizon": by_h,
        }

    pooled: dict[str, Any] = {}
    for h in horizons:
        h = int(h)
        den = pooled_den[h]
        mean = (pooled_num[h] / den) if den > 0 else None
        pooled[str(h)] = {
            "mean_context_gain_vs_etas_nats": round(mean, 6) if mean is not None else None,
            "context_gain_positive": bool(mean is not None and mean > 0.0),
            "n_scored_weight": int(den),
        }

    return {
        "definition": (
            "Information gain (nats) of the context-conditioned model over catalog-only ETAS. "
            "Positive ⇒ the global context (worldwide seismicity + covariates) improves the local "
            "short-term forecast beyond the self-exciting catalog model. Quiet periods sit near 0; "
            "active sequences can spike. Pooled values are scored-day weighted across views."
        ),
        "context_channel_active": any_active,
        "channel_note": (
            "context channel active in at least one view"
            if any_active
            else "context channel NOT yet landed (enricher stack feature-flagged off, model-design "
            "§6.2): gain is ~0 by construction, not a measured null — reported honestly"
        ),
        "per_view": per_view,
        "pooled": pooled,
    }


def _reduce_high_low_bias(view_results: list[ViewResult], horizons: list[int]) -> dict[str, Any]:
    """HIGH-vs-LOW-seismicity bias: skill/calibration per class + the gap (does it over-fit loud zones?).

    For each seismicity class (high / low) and each horizon we pool the per-view per-horizon scalars
    (N-test pass rate, IGPE vs null, context gain, Brier) as a scored-day-weighted mean, then report
    the gap ``high − low``. The direction of "good" is encoded per field (Brier is lower-is-better):
    a large positive *advantage* of high over low (skill higher / Brier lower in high) is the
    over-fit-to-high-seismicity signal — surfaced, not hidden.
    """
    classes = {"high": [], "low": []}  # type: dict[str, list[ViewResult]]
    for vr in view_results:
        if vr.result is None:
            continue
        # The whole-Earth GLOBAL view is the aggregate field, NOT a regional class member — it must not
        # pollute the high-vs-low COUNTRY comparison (it carries the highest IGPE by construction and
        # would inflate the "high" pool). Its own skill is reported in per_view / context_gain instead.
        if vr.view_id == "global":
            continue
        if vr.seismicity_class in classes:
            classes[vr.seismicity_class].append(vr)

    per_class: dict[str, Any] = {}
    for cls, members in classes.items():
        per_class[cls] = {
            "view_ids": [vr.view_id for vr in members],
            "by_horizon": _pool_class_by_horizon(members, horizons),
        }

    # Gap (high − low) per horizon per field, with the bias direction made explicit.
    gap: dict[str, Any] = {}
    for h in horizons:
        h = int(h)
        hi = per_class.get("high", {}).get("by_horizon", {}).get(str(h), {})
        lo = per_class.get("low", {}).get("by_horizon", {}).get(str(h), {})
        fields: dict[str, Any] = {}
        for name, (_, higher_better) in _BIAS_FIELDS.items():
            hv = hi.get(name)
            lv = lo.get(name)
            if hv is None or lv is None:
                fields[name] = {"high": hv, "low": lv, "gap_high_minus_low": None}
                continue
            diff = float(hv) - float(lv)
            # "advantage of high over low" in the good direction (skill up / Brier down).
            advantage = diff if higher_better else -diff
            fields[name] = {
                "high": round(float(hv), 6),
                "low": round(float(lv), 6),
                "gap_high_minus_low": round(diff, 6),
                "high_better_by": round(advantage, 6),
            }
        gap[str(h)] = fields

    return {
        "definition": (
            "Skill/calibration computed separately over the pre-registered HIGH-seismicity views "
            "(active plate boundaries) and LOW-seismicity views (stable interiors), with the gap. A "
            "large advantage of HIGH over LOW (higher pass rate / IGPE / context gain, lower Brier) "
            "is the 'model over-fits high-seismicity zones' signal — pre-registered partition, not a "
            "post-hoc split."
        ),
        "field_directions": {k: ("higher_better" if v[1] else "lower_better") for k, v in _BIAS_FIELDS.items()},
        "per_class": per_class,
        "gap": gap,
    }


def _pool_class_by_horizon(members: list[ViewResult], horizons: list[int]) -> dict[str, Any]:
    """Scored-day-weighted pool of the per-horizon scalars across the views in one seismicity class."""
    out: dict[str, Any] = {}
    for h in horizons:
        h = int(h)
        acc: dict[str, list[tuple[float, float]]] = {k: [] for k in _BIAS_FIELDS}  # (value, weight)
        n_scored_total = 0
        for vr in members:
            if vr.result is None:
                continue
            for ph in vr.result.per_horizon:
                if int(ph["horizon_days"]) != h:
                    continue
                w = float(ph.get("n_scored", 0))
                n_scored_total += int(ph.get("n_scored", 0))
                for name, (key, _) in _BIAS_FIELDS.items():
                    val = ph.get(key)
                    if val is not None and w > 0:
                        acc[name].append((float(val), w))
        block: dict[str, Any] = {"n_scored": n_scored_total}
        for name, pairs in acc.items():
            if pairs:
                num = sum(v * w for v, w in pairs)
                den = sum(w for _, w in pairs)
                block[name] = round(num / den, 6) if den > 0 else None
            else:
                block[name] = None
        out[str(h)] = block
    return out


# ─────────────────────────────────────────────────────────────────────────────
# IO + helpers
# ─────────────────────────────────────────────────────────────────────────────


#: Scoring-grid cell size (degrees) for the pseudo-prospective back-analysis. Deliberately COARSER than
#: the 0.1° production fit grid: the back-analysis scores M≥5 events, which are spatially sparse, so a
#: 0.1° grid is almost entirely empty cells — the CSEP S/L spatial tests are then under-powered AND there
#: are ~25× more cells to infer over EVERY issue day (the dominant cost once the MLE is cadenced). A 0.5°
#: (~55 km) grid gives better-populated cells (sounder CSEP tests for sparse large events) and a ~25×
#: cheaper daily inference. The published daily FORECAST still uses the fine grid; only multi-day skill
#: scoring coarsens. Override per run via GlobalBackAnalysisConfig.scoring_cell_deg.
DEFAULT_SCORING_CELL_DEG: float = 0.5


def _grid_cell_deg_resolver(max_country_cells: int = 200_000, scoring_cell_deg: float | None = None):
    """Return ``region -> cell_deg`` giving each view its back-analysis SCORING grid size.

    A bounded country view scores on ``scoring_cell_deg`` (:data:`DEFAULT_SCORING_CELL_DEG` = 0.5°) —
    coarser than the 0.1° production fit grid on purpose (see that constant). The whole-Earth GLOBAL
    view — and any view whose cell count at ``scoring_cell_deg`` would still exceed ``max_country_cells``
    — coarsens further toward ``grid.yaml: fit.global_fit.world_cell_deg`` (1.0°) and beyond, so the
    global window never materializes the ~6.5M-cell dense world grid (grid.yaml).
    """
    from ..config import load

    grid_cfg = load("grid")
    fit = grid_cfg.get("fit", {}) if isinstance(grid_cfg, dict) else {}
    world_deg = float(fit.get("global_fit", {}).get("world_cell_deg", 1.0))
    base_deg = float(scoring_cell_deg if scoring_cell_deg is not None else DEFAULT_SCORING_CELL_DEG)

    def _n_cells(region: Region, deg: float) -> float:
        bb = region.bbox
        return ((bb.lat_max - bb.lat_min) / deg) * ((bb.lon_max - bb.lon_min) / deg)

    def resolve(region: Region) -> float:
        deg = base_deg
        if _n_cells(region, deg) > max_country_cells:
            # Large-bbox view (the whole-Earth window): coarsen toward the world grid and beyond until
            # the cell count is under the cap — a safety valve so the global window never explodes.
            deg = max(deg, world_deg)
            while _n_cells(region, deg) > max_country_cells and deg < 90.0:
                deg *= 2.0
        return deg

    return resolve


def _slice_to_view(master: pd.DataFrame, region: Region) -> pd.DataFrame:
    """Slice the global master catalog to a view's bounding box (a window into the global field).

    Keeps every event inside ``[lat_min, lat_max] × [lon_min, lon_max]``. The longitude test is the
    plain inclusive interval — the pre-registered view boxes do not cross the antimeridian, so no
    wrap-around handling is needed (the GLOBAL view's box stops just short of ±180°).
    """
    validate_catalog(master)
    bb = region.bbox
    lat = pd.to_numeric(master["latitude"], errors="coerce")
    lon = pd.to_numeric(master["longitude"], errors="coerce")
    mask = (lat >= bb.lat_min) & (lat <= bb.lat_max) & (lon >= bb.lon_min) & (lon <= bb.lon_max)
    return master.loc[mask].reset_index(drop=True)


def _write_global_summary(
    result: GlobalBackAnalysisResult, config: GlobalBackAnalysisConfig, results_dir: Path
) -> Path:
    """Write the compact global back-analysis JSON the web app reads. Atomic (temp + replace)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    fname = f"backanalysis-global-{result.start[:10]}_{result.end[:10]}.json"
    path = results_dir / fname

    payload = {
        "product": "CAOS_SEISMIC",
        "kind": "backanalysis_global",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "period": {"start": result.start, "end": result.end},
        "issue_cadence": "daily",
        "horizons_days": result.horizons_days,
        "magnitude_thresholds": [float(m) for m in config.magnitude_thresholds],
        "reliability_threshold": config.reliability_threshold,
        "views": result.view_ids,
        "per_view": result.per_view,
        # THESIS headline + adversarial bias check.
        "context_gain": result.context_gain,
        "high_vs_low_bias": result.bias,
        "pycsep_used": result.pycsep_used,
        "framing": (
            "Global, leakage-free pseudo-prospective back-analysis (forecast clock). One global "
            "context-conditioned model is scored through each country VIEW and globally. The "
            "headline measurement is the information gain over catalog-only ETAS (context_gain) — "
            "how much the global context adds to the local forecast. The high_vs_low_bias block is "
            "the adversarial check: does the model only look good on high-seismicity margins? Every "
            "view × horizon is reported including failures and underperformance (no post-hoc "
            "selection). This complements official OEF systems; it is not a civil-protection alarm."
        ),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.info("wrote global back-analysis summary: %s", path)
    return path


def _as_utc(t: datetime | pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

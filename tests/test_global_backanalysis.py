"""Unit tests for the global, multi-view back-analysis reductions (core deps only, no fitting).

These pin the two headline measurements the global re-scoping adds, *without* running any model fit
or forecast-clock step (those are exercised end-to-end elsewhere and need a catalog). Each test
builds tiny in-memory :class:`BackAnalysisResult` objects with hand-set ``per_horizon`` blocks and
checks that the cross-view reductions compute the right thing:

1. **Context gain over catalog-only ETAS** — per view × horizon and the scored-day-weighted pool.
2. **HIGH-vs-LOW-seismicity bias** — per-class pools and the ``high − low`` gap, with the bias
   direction respected (Brier is lower-is-better).
3. **Config-driven views** — the pre-registered high/low partition loads from ``configs/views.yaml``
   and the whole-Earth GLOBAL view coarsens its fit grid (never the dense ~6.5M-cell world grid).
"""

from __future__ import annotations

from caos_seismic.eval import views
from caos_seismic.eval.backanalysis import BackAnalysisResult
from caos_seismic.eval.global_backanalysis import (
    GlobalBackAnalysisResult,
    ViewResult,
    _grid_cell_deg_resolver,
    _reduce_context_gain,
    _reduce_high_low_bias,
    _view_block,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — hand-built per-horizon blocks (the shape backanalysis._reduce_per_horizon emits)
# ─────────────────────────────────────────────────────────────────────────────


def _per_horizon(
    horizon: int,
    *,
    n_scored: int,
    n_test_pass_rate: float,
    igpe_null: float,
    context_gain: float | None,
    context_active: bool,
    brier: float,
) -> dict:
    return {
        "horizon_days": horizon,
        "n_scored": n_scored,
        "n_failed": 0,
        "n_test_pass_rate": n_test_pass_rate,
        "mean_igpe_vs_null_nats": igpe_null,
        "skill_over_null_positive": igpe_null > 0.0,
        "mean_context_gain_vs_etas_nats": context_gain,
        "context_gain_positive": bool(context_gain is not None and context_gain > 0.0),
        "context_channel_active": context_active,
        "brier": brier,
        "n_reliability_pairs": n_scored,
        "by_threshold": {},
        "note": "",
    }


def _view_result(
    view_id: str, seismicity_class: str, per_horizon: list[dict], n_scored_days: int = 10
) -> ViewResult:
    r = BackAnalysisResult(
        region_id=view_id,
        start="2020-01-01T00:00:00Z",
        end="2020-01-10T00:00:00Z",
        n_issue_days=n_scored_days,
        n_scored_days=n_scored_days,
        n_failed_days=0,
        horizons_days=[ph["horizon_days"] for ph in per_horizon],
        magnitude_thresholds=[5.0],
        per_horizon=per_horizon,
        reliability=[],
        summary_path=f"backanalysis-{view_id}-2020-01-01_2020-01-10.json",
    )
    return ViewResult(
        view_id=view_id,
        name_en=view_id,
        seismicity_class=seismicity_class,
        plate_setting="test",
        result=r,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1) Context gain reduction (per view + scored-day-weighted pool)
# ─────────────────────────────────────────────────────────────────────────────


def test_context_gain_pool_is_scored_day_weighted():
    # Two views at horizon 1: a loud view (8 scored days, gain 1.0) and a quiet view (2 days, 0.0).
    loud = _view_result(
        "CL",
        "high",
        [_per_horizon(1, n_scored=8, n_test_pass_rate=1.0, igpe_null=0.5,
                      context_gain=1.0, context_active=True, brier=0.1)],
    )
    quiet = _view_result(
        "AU-E",
        "low",
        [_per_horizon(1, n_scored=2, n_test_pass_rate=1.0, igpe_null=0.0,
                      context_gain=0.0, context_active=True, brier=0.01)],
    )
    cg = _reduce_context_gain([loud, quiet], horizons=[1])
    # Pool weight = 8 + 2 = 10; weighted mean = (1.0*8 + 0.0*2)/10 = 0.8.
    assert cg["pooled"]["1"]["mean_context_gain_vs_etas_nats"] == 0.8
    assert cg["pooled"]["1"]["n_scored_weight"] == 10
    assert cg["pooled"]["1"]["context_gain_positive"] is True
    assert cg["context_channel_active"] is True
    # Per-view block keeps each view's own gain + active flag.
    assert cg["per_view"]["CL"]["by_horizon"]["1"]["mean_context_gain_vs_etas_nats"] == 1.0
    assert cg["per_view"]["AU-E"]["by_horizon"]["1"]["mean_context_gain_vs_etas_nats"] == 0.0


def test_context_gain_inactive_channel_is_reported_honestly():
    # When NO view has an active context channel, the pooled gain is 0 and the flag is False.
    v = _view_result(
        "CL",
        "high",
        [_per_horizon(1, n_scored=5, n_test_pass_rate=1.0, igpe_null=0.2,
                      context_gain=0.0, context_active=False, brier=0.1)],
    )
    cg = _reduce_context_gain([v], horizons=[1])
    assert cg["context_channel_active"] is False
    assert "NOT yet landed" in cg["channel_note"]
    assert cg["pooled"]["1"]["mean_context_gain_vs_etas_nats"] == 0.0
    assert cg["pooled"]["1"]["context_gain_positive"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 2) High-vs-low-seismicity bias reduction
# ─────────────────────────────────────────────────────────────────────────────


def test_high_low_bias_gap_and_direction():
    # HIGH view: high pass rate, high IGPE, but WORSE (higher) Brier than the LOW view.
    high = _view_result(
        "CL",
        "high",
        [_per_horizon(1, n_scored=10, n_test_pass_rate=0.9, igpe_null=0.4,
                      context_gain=0.6, context_active=True, brier=0.20)],
    )
    low = _view_result(
        "AU-E",
        "low",
        [_per_horizon(1, n_scored=10, n_test_pass_rate=0.6, igpe_null=0.1,
                      context_gain=0.0, context_active=True, brier=0.05)],
    )
    bias = _reduce_high_low_bias([high, low], horizons=[1])
    assert set(bias["per_class"].keys()) == {"high", "low"}
    g = bias["gap"]["1"]
    # Pass-rate gap (higher is better): high 0.9 − low 0.6 = +0.3, advantage to high.
    assert g["n_test_pass_rate"]["gap_high_minus_low"] == round(0.9 - 0.6, 6)
    assert g["n_test_pass_rate"]["high_better_by"] == round(0.9 - 0.6, 6)
    # Brier gap (LOWER is better): high 0.20 − low 0.05 = +0.15, but that means HIGH is WORSE, so the
    # "advantage of high" must be NEGATIVE (direction respected).
    assert g["brier"]["gap_high_minus_low"] == round(0.20 - 0.05, 6)
    assert g["brier"]["high_better_by"] == round(-(0.20 - 0.05), 6)
    # Field directions are declared for the consumer.
    assert bias["field_directions"]["brier"] == "lower_better"
    assert bias["field_directions"]["n_test_pass_rate"] == "higher_better"


def test_bias_handles_missing_class_gracefully():
    # Only a HIGH view present: the LOW pool is empty and the gap is None (never a crash).
    high = _view_result(
        "JP",
        "high",
        [_per_horizon(1, n_scored=4, n_test_pass_rate=1.0, igpe_null=0.3,
                      context_gain=0.0, context_active=False, brier=0.1)],
    )
    bias = _reduce_high_low_bias([high], horizons=[1])
    assert bias["per_class"]["low"]["view_ids"] == []
    assert bias["gap"]["1"]["n_test_pass_rate"]["gap_high_minus_low"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 3) View block + config-driven views + grid coarsening
# ─────────────────────────────────────────────────────────────────────────────


def test_view_block_records_failure_without_dropping():
    vr = ViewResult(
        view_id="XX", name_en="broken", seismicity_class="low", plate_setting="test",
        error="RuntimeError: no catalog",
    )
    block = _view_block(vr)
    assert block["error"].startswith("RuntimeError")
    assert block["per_horizon"] == []  # failure is reported, not silently omitted


def test_pre_registered_views_partition_from_config():
    high = [v.region.id for v in views.high_seismicity_views()]
    low = [v.region.id for v in views.low_seismicity_views()]
    # The pre-registered set has BOTH classes (so the bias comparison is meaningful).
    assert high and low
    # The classes are disjoint and the global view is its own thing.
    assert not (set(high) & set(low))
    assert views.GLOBAL_VIEW.region.id == "global"


def test_global_view_coarsens_fit_grid_but_country_views_do_not():
    resolve = _grid_cell_deg_resolver()
    # A bounded country view keeps the fine grid (None ⇒ configured fit.cell_deg).
    cl = views.view_by_id("CL")
    assert resolve(cl.region) is None
    # The whole-Earth GLOBAL window MUST coarsen (never the dense ~6.5M-cell 0.1° world grid).
    deg = resolve(views.GLOBAL_VIEW.region)
    assert deg is not None and deg >= 1.0


def test_global_result_str_reports_scored_view_count():
    res = GlobalBackAnalysisResult(
        start="2020-01-01T00:00:00Z",
        end="2020-01-10T00:00:00Z",
        view_ids=["CL", "AU-E"],
        horizons_days=[1],
        per_view=[{"view_id": "CL"}, {"view_id": "AU-E", "error": "boom"}],
        summary_path="results/backanalysis-global-x.json",
    )
    s = str(res)
    assert "1/2 views scored" in s

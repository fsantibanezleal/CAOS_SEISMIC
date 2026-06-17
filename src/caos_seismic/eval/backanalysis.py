"""Pseudo-prospective CSEP back-analysis driver — the credibility record the web app renders.

This module steps the **forecast clock** (the *same* code the live daily product runs, so the
back-analysis and production cannot diverge) across a region × period, day by day. At each issue
date ``t`` the model is conditioned on **only** the catalog slice ``(-∞, t)``, the forecast is
sealed, and it is scored against the target window ``[t, t + H)`` — which the model never saw. The
per-issue scores are accumulated and reduced to a compact JSON summary written into ``results/`` for
the web app's Back-analysis section (evaluation-plan §9).

What this driver guarantees, straight from the evaluation plan:

* **No temporal leakage** (§4.1) — the clock hands the model a causal slice; a defence-in-depth
  assertion re-checks it.
* **Score on the non-declustered catalog** (§5) — the target includes aftershocks; the product
  deliberately forecasts clustering. (The dual-catalog rule applies to *inputs*, not the target.)
* **Report every region × horizon cell, including failures** (§4.5, §7) — a cell where the model
  fails to beat its baselines, or where a day could not be scored, is emitted as such, never
  silently dropped. Selective reporting is the exact selection-bias trap CSEP exists to prevent.
* **Skill lives in the comparison test** (§6.2) — per-horizon we record the consistency **N-test**
  (calibration of one model) *and* the **information gain per earthquake (nats)** of the model vs
  the smoothed-seismicity null (the comparison test where skill is actually established), plus the
  **reliability** pairs (§6.4) the live isotonic recalibration reads back.

Scoring defers to :mod:`caos_seismic.eval.csep`, which uses **pyCSEP** when installed and falls back
to dependency-free numpy closed forms (N-test, IGPE in nats, Brier, reliability bins) otherwise — so
the back-analysis runs on the core stack. The driver itself imports only core deps at module top;
the inference machinery (forecast clock + the model family) is imported lazily inside
:func:`run_back_analysis` so ``import caos_seismic.eval`` stays light.

Public API (mirrors what :mod:`caos_seismic.eval` re-exports):

* :class:`BackAnalysisConfig` — region, period, horizons, thresholds, the M* the binary
  reliability/Brier output is computed at, and the issue cadence.
* :class:`ScoredForecast` — one issued forecast's scores at one issue date.
* :class:`BackAnalysisResult` — the accumulated, per-horizon-reduced summary + the JSON written.
* :func:`run_back_analysis` / :func:`run_backanalysis` — the driver (the second name is the CLI's).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import REPO_ROOT, config_hash, load, load_region
from ..contracts import Region

logger = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Config + result containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BackAnalysisConfig:
    """Inputs that define one pseudo-prospective back-analysis run.

    Attributes
    ----------
    region:
        The :class:`Region` scored.
    start, end:
        First and last **issue** dates (UTC). The clock issues one forecast per day in ``[start, end]``.
    horizons_days:
        Forecast horizons scored independently (configs/forecast.yaml ``horizons_days``).
    magnitude_thresholds:
        Magnitude bands the rate/exceedance is evaluated at.
    reliability_threshold:
        The single M* the binary exceedance outcome (for the reliability diagram + Brier score) is
        computed at — the most populated band, so the diagram is well-sampled (evaluation-plan §6.4).
    issue_hour_utc:
        Issue time of day (UTC) — matches the live ``publish.yaml`` cadence.
    rng_seed:
        Seed for the per-issue bound simulator, so a back-analysis is byte-reproducible.
    cell_deg:
        Optional fit-grid cell size (degrees) override. When ``None`` the configured
        ``grid.yaml: fit.cell_deg`` (0.1°) is used — correct for a spatially-bounded country view.
        A large-bbox view (the whole-Earth GLOBAL window) MUST pass a coarse value (e.g. the
        ``grid.yaml: fit.global_fit.world_cell_deg`` = 1.0°): the dense 0.1° grid over the whole
        Earth is ~6.5M cells and is never materialized (grid.yaml). The global driver sets this.
    """

    region: Region
    start: pd.Timestamp
    end: pd.Timestamp
    horizons_days: list[int] = field(default_factory=lambda: [1, 2, 7])
    magnitude_thresholds: list[float] = field(default_factory=lambda: [5.0, 6.0, 7.0])
    reliability_threshold: float = 5.0
    issue_hour_utc: int = 0
    rng_seed: int = 20260616
    cell_deg: float | None = None


@dataclass
class ScoredForecast:
    """The scores of ONE issued forecast at ONE issue date (one row of the back-analysis ledger).

    ``ok=False`` with a ``reason`` records a day that could not be scored (e.g. the model could not
    be conditioned because the lawful past was empty) — those days are *kept*, never dropped, so the
    published record cannot be selection-biased.

    Two information-gain channels are recorded per row, both in **nats** (never bits):

    * ``igpe_vs_null_nats`` — gain of the context-conditioned model over the smoothed-seismicity
      null. This is the classic "does conditioning on history help at all" number.
    * ``igpe_vs_etas_nats`` — gain of the context-conditioned model over a **catalog-only ETAS**
      baseline. This is the THESIS headline: ETAS already reproduces Omori/Utsu clustering, so a
      positive, significant gain here is *not* "I predicted aftershocks" — it is "the global context
      (covariates, worldwide seismicity) makes the local short-term forecast better than the standard
      self-exciting model can on the catalog alone". When no separate context channel has landed yet
      (the enricher stack is feature-flagged in model-design §6.2), the primary *is* catalog-only
      ETAS, so this gain is ~0 and ``context_channel_active`` is ``False`` — reported honestly, never
      faked into a positive number.
    """

    issued_at: str
    horizon_days: int
    m_threshold: float
    n_forecast: float          # region-total expected count from the model (sum over cells)
    n_forecast_null: float     # region-total expected count from the smoothed null
    n_observed: int            # observed target events >= M* in [t, t+H)
    n_test_quantile: float | None = None
    n_test_passed: bool | None = None
    igpe_vs_null_nats: float | None = None
    n_forecast_etas: float | None = None  # region-total expected count from catalog-only ETAS
    igpe_vs_etas_nats: float | None = None  # THESIS: context gain over catalog-only ETAS
    context_channel_active: bool | None = None  # False while the primary IS catalog-only ETAS
    exceedance_prob: float | None = None  # region P(>=1 >= M*) (for the reliability pair at M_rel)
    observed_any: int | None = None       # 1 if >=1 observed >= M* (the reliability outcome)
    brier_term: float | None = None       # (p - y)^2 for the reliability pair (for a pooled Brier)
    ok: bool = True
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v != ""}


@dataclass
class BackAnalysisResult:
    """The accumulated back-analysis: every scored row + a per-horizon reduction + the JSON path.

    ``per_horizon`` is the compact block the web app renders (one entry per horizon): the N-test pass
    rate, the mean information gain over the null (nats), the reliability diagram, the Brier score,
    and the count of scored vs failed issue days. Failures are surfaced explicitly (``n_failed``).
    """

    region_id: str
    start: str
    end: str
    n_issue_days: int
    n_scored_days: int
    n_failed_days: int
    horizons_days: list[int]
    magnitude_thresholds: list[float]
    per_horizon: list[dict[str, Any]] = field(default_factory=list)
    reliability: list[list[float]] = field(default_factory=list)
    scored: list[ScoredForecast] = field(default_factory=list, repr=False)
    summary_path: str | None = None
    pycsep_used: bool = False

    def __str__(self) -> str:  # used by the CLI's `backanalysis · done: {result}`
        loc = self.summary_path or "(not written)"
        return (
            f"{self.region_id} {self.start[:10]}→{self.end[:10]}: "
            f"{self.n_scored_days}/{self.n_issue_days} days scored "
            f"({self.n_failed_days} failed) -> {loc}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


def run_back_analysis(
    config: BackAnalysisConfig,
    *,
    catalog: pd.DataFrame | None = None,
    write_summary: bool = True,
    results_dir: Path | None = None,
) -> BackAnalysisResult:
    """Run the pseudo-prospective back-analysis described by ``config`` and (optionally) write the JSON.

    Steps each daily issue date with the forecast clock, conditions the model family on the lawful
    past, scores the sealed forecast against the target window via :mod:`caos_seismic.eval.csep`,
    accumulates per (issue × horizon × threshold) rows (including failures), reduces to a per-horizon
    summary, and writes ``results/backanalysis-<region>-<start>_<end>.json`` for the web app.

    Parameters
    ----------
    config:
        The :class:`BackAnalysisConfig` (region, period, horizons, thresholds, cadence).
    catalog:
        The master catalog (all events; the clock slices it). If ``None`` the cleaned store is loaded
        lazily for the region; a clear error is raised if neither is available.
    write_summary:
        Write the compact JSON summary into ``results/`` (default on).
    results_dir:
        Override the output directory (defaults to ``results/``).
    """
    # Lazy: the inference machinery (clock + model family) is heavier than the eval primitives.
    from ..inference.clock import ForecastClock, assert_no_leakage, target_slice
    from ..inference.daily import (
        _catalog_hygiene,
        _fit_model_family,
        build_fit_cells,
        _expected_counts,
    )
    from . import csep

    reg = config.region
    master = _resolve_master_catalog(catalog, reg)
    clock = ForecastClock(master)
    grid_cfg = load("grid")
    completeness_cfg = load("completeness")
    # Fit-grid cell size: a spatially-bounded country view uses the configured fine 0.1° grid; a
    # large-bbox view (whole-Earth GLOBAL) MUST coarsen (the dense global 0.1° grid is ~6.5M cells and
    # is never materialized — grid.yaml). `config.cell_deg` overrides when set (the global driver does).
    if config.cell_deg is not None:
        grid_cfg = {**grid_cfg, "fit": {**grid_cfg.get("fit", {}), "cell_deg": float(config.cell_deg)}}
    cells = build_fit_cells(reg, grid_cfg)

    horizons = [int(h) for h in config.horizons_days]
    thresholds = [float(m) for m in config.magnitude_thresholds]
    m_rel = float(config.reliability_threshold)

    scored: list[ScoredForecast] = []
    # reliability pairs accumulate (forecast_prob, observed_any) at m_rel for the longest horizon.
    rel_pairs: list[tuple[float, int]] = []
    rel_horizon = max(horizons) if horizons else 1

    n_issue_days = 0
    n_scored_days = 0
    n_failed_days = 0

    for t_issue, past in clock.daily_issues(
        config.start, config.end, issue_hour_utc=config.issue_hour_utc
    ):
        n_issue_days += 1
        issued_at = t_issue.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Condition the model family on the lawful past (defence-in-depth leakage assertion).
        try:
            assert_no_leakage(past, t_issue)
            if past.empty or len(past) < 3:
                raise ValueError(f"only {len(past)} lawful events before issue")
            hygiene = _catalog_hygiene(past, reg, completeness_cfg)
            models = _fit_model_family(
                conditional_cat=past,
                background_cat=hygiene["declustered"],
                region=reg,
                t_issue=t_issue,
                mc=hygiene["mc"],
                b_value=hygiene["b"],
            )
            primary = models["primary"]
            null = models["smoothed"]
            # Catalog-only ETAS baseline for the THESIS context-gain channel. When the context
            # channel is feature-flagged off, `primary` IS catalog-only ETAS, so the gain is ~0 and
            # `context_active` is False — surfaced honestly, never faked positive.
            etas_baseline, context_active = _catalog_only_etas_baseline(models)
        except Exception as exc:  # a day we could not score — RECORD it, never drop it
            n_failed_days += 1
            for horizon in horizons:
                for m_star in thresholds:
                    scored.append(
                        ScoredForecast(
                            issued_at=issued_at,
                            horizon_days=int(horizon),
                            m_threshold=float(m_star),
                            n_forecast=0.0,
                            n_forecast_null=0.0,
                            n_observed=0,
                            ok=False,
                            reason=f"could not condition model: {exc}",
                        )
                    )
            continue

        n_scored_days += 1
        for horizon in horizons:
            target = target_slice(master, t_issue, float(horizon))
            target_mw = pd.to_numeric(target["mw"], errors="coerce")
            for m_star in thresholds:
                lam_null = _expected_counts(null, reg, cells, horizon, m_star, t_issue)
                # The primary's published rate is floored to the smoothed background (cold-start floor,
                # daily.py §_forecast_cells). The catalog-only ETAS baseline must receive the *same*
                # floor, otherwise the floor difference alone (not context) shows up as spurious gain.
                lam_primary = np.maximum(
                    _expected_counts(primary, reg, cells, horizon, m_star, t_issue), lam_null
                )
                if context_active:
                    lam_etas = np.maximum(
                        _expected_counts(etas_baseline, reg, cells, horizon, m_star, t_issue), lam_null
                    )
                else:
                    # Context channel not landed: the baseline IS the primary, so the gain is exactly
                    # 0 by construction (not a measured null). Use the identical floored rate.
                    lam_etas = lam_primary
                obs = target.loc[target_mw >= m_star - 1e-9]
                n_obs = int(len(obs))

                n_test = csep.n_test_poisson(float(lam_primary.sum()), n_obs)
                omega = _bin_counts_to_cells(obs, cells)
                igpe_null, _ = csep.information_gain_per_earthquake(lam_primary, lam_null, omega)
                igpe_etas, _ = csep.information_gain_per_earthquake(lam_primary, lam_etas, omega)

                row = ScoredForecast(
                    issued_at=issued_at,
                    horizon_days=int(horizon),
                    m_threshold=float(m_star),
                    n_forecast=round(float(lam_primary.sum()), 6),
                    n_forecast_null=round(float(lam_null.sum()), 6),
                    n_observed=n_obs,
                    n_test_quantile=n_test.quantile,
                    n_test_passed=bool(n_test.passed),
                    igpe_vs_null_nats=round(float(igpe_null), 6),
                    n_forecast_etas=round(float(lam_etas.sum()), 6),
                    igpe_vs_etas_nats=round(float(igpe_etas), 6),
                    context_channel_active=bool(context_active),
                )
                # Region exceedance probability at the reliability threshold/horizon → a reliability pair.
                if abs(m_star - m_rel) < 1e-9:
                    p_region = float(1.0 - np.exp(-np.clip(lam_primary, 0.0, None)).prod())
                    row.exceedance_prob = round(p_region, 6)
                    row.observed_any = int(n_obs > 0)
                    row.brier_term = round((p_region - float(n_obs > 0)) ** 2, 6)
                    if int(horizon) == rel_horizon:
                        rel_pairs.append((p_region, int(n_obs > 0)))
                scored.append(row)

    # Reduce to the per-horizon summary the web app renders.
    per_horizon = _reduce_per_horizon(scored, horizons, csep)
    reliability = _reliability_from_pairs(rel_pairs, csep)

    result = BackAnalysisResult(
        region_id=reg.id,
        start=config.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end=config.end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_issue_days=n_issue_days,
        n_scored_days=n_scored_days,
        n_failed_days=n_failed_days,
        horizons_days=horizons,
        magnitude_thresholds=thresholds,
        per_horizon=per_horizon,
        reliability=reliability,
        scored=scored,
        pycsep_used=csep.pycsep_available(),
    )

    if write_summary:
        result.summary_path = str(_write_summary(result, config, results_dir or RESULTS_DIR))
    return result


def run_backanalysis(
    *,
    region: Region | str,
    start: datetime | pd.Timestamp | str,
    end: datetime | pd.Timestamp | str,
    catalog: pd.DataFrame | None = None,
    **kwargs: Any,
) -> BackAnalysisResult:
    """CLI entry point (``caos-seismic backanalysis``) — build a config from the region + period and run.

    A thin adapter over :func:`run_back_analysis` matching the CLI's keyword call
    (``region=``, ``start=``, ``end=``). Horizons / thresholds default to ``configs/forecast.yaml``.
    """
    reg = load_region(region) if isinstance(region, str) else region
    forecast_cfg = load("forecast")
    horizons = [int(h) for h in forecast_cfg.get("horizons_days", [1, 2, 7])]
    thresholds = [float(m) for m in forecast_cfg.get("magnitude_thresholds", [5.0, 6.0, 7.0])]
    config = BackAnalysisConfig(
        region=reg,
        start=_as_utc(start),
        end=_as_utc(end),
        horizons_days=horizons,
        magnitude_thresholds=thresholds,
        reliability_threshold=min(thresholds) if thresholds else 5.0,
    )
    return run_back_analysis(config, catalog=catalog, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Reductions + IO
# ─────────────────────────────────────────────────────────────────────────────


def _reduce_per_horizon(scored: list[ScoredForecast], horizons: list[int], csep) -> list[dict[str, Any]]:
    """Per-horizon block: N-test pass rate, both IGPE channels, totals, Brier, and failure counts.

    Pools across thresholds and issue dates for the headline N-test pass rate / mean information
    gain, and keeps a per-threshold breakdown so the web app can drill down. Two IGPE channels are
    reduced: ``mean_igpe_vs_null_nats`` (over the smoothed null) and the THESIS
    ``mean_context_gain_vs_etas_nats`` (over catalog-only ETAS — how much the global context adds).
    Failed (un-scorable) rows are counted but excluded from the rate means — and surfaced via
    ``n_failed``.
    """
    out: list[dict[str, Any]] = []
    for horizon in horizons:
        rows = [s for s in scored if s.horizon_days == horizon]
        ok_rows = [s for s in rows if s.ok]
        failed = [s for s in rows if not s.ok]
        n_pass = sum(1 for s in ok_rows if s.n_test_passed)
        igpes = [s.igpe_vs_null_nats for s in ok_rows if s.igpe_vs_null_nats is not None]
        ctx_gains = [s.igpe_vs_etas_nats for s in ok_rows if s.igpe_vs_etas_nats is not None]
        brier_terms = [s.brier_term for s in ok_rows if s.brier_term is not None]
        context_active = any(bool(s.context_channel_active) for s in ok_rows)
        by_threshold: dict[str, Any] = {}
        for m_star in sorted({s.m_threshold for s in ok_rows}):
            sub = [s for s in ok_rows if abs(s.m_threshold - m_star) < 1e-9]
            sub_pass = sum(1 for s in sub if s.n_test_passed)
            sub_igpe = [s.igpe_vs_null_nats for s in sub if s.igpe_vs_null_nats is not None]
            sub_ctx = [s.igpe_vs_etas_nats for s in sub if s.igpe_vs_etas_nats is not None]
            by_threshold[f"{m_star:.1f}"] = {
                "n": len(sub),
                "n_test_pass_rate": round(sub_pass / len(sub), 4) if sub else None,
                "mean_igpe_vs_null_nats": round(float(np.mean(sub_igpe)), 6) if sub_igpe else None,
                "mean_context_gain_vs_etas_nats": (
                    round(float(np.mean(sub_ctx)), 6) if sub_ctx else None
                ),
                "n_observed_total": int(sum(s.n_observed for s in sub)),
                "n_forecast_total": round(sum(s.n_forecast for s in sub), 4),
            }
        mean_ctx = float(np.mean(ctx_gains)) if ctx_gains else None
        out.append(
            {
                "horizon_days": int(horizon),
                "n_scored": len(ok_rows),
                "n_failed": len(failed),
                "n_test_pass_rate": round(n_pass / len(ok_rows), 4) if ok_rows else None,
                "mean_igpe_vs_null_nats": round(float(np.mean(igpes)), 6) if igpes else None,
                "skill_over_null_positive": bool(igpes and float(np.mean(igpes)) > 0.0),
                # THESIS headline: how much the global context adds over catalog-only ETAS (nats).
                "mean_context_gain_vs_etas_nats": (round(mean_ctx, 6) if mean_ctx is not None else None),
                "context_gain_positive": bool(mean_ctx is not None and mean_ctx > 0.0),
                "context_channel_active": context_active,
                "brier": round(float(np.mean(brier_terms)), 6) if brier_terms else None,
                "n_reliability_pairs": len(brier_terms),
                "by_threshold": by_threshold,
                "note": (
                    "Poisson grid tests over-reject during aftershock sequences; pair with the "
                    "catalog-based result. Skill is the comparison-test win vs ETAS (eval.csep), "
                    "not the consistency pass rate. context_gain_vs_etas is the headline thesis "
                    "measurement — when context_channel_active is false the context stack has not "
                    "yet landed and the gain is ~0 by construction (not a measured null)."
                ),
            }
        )
    return out


def _reliability_from_pairs(pairs: list[tuple[float, int]], csep) -> list[list[float]]:
    """Binned reliability rows ``[[forecast_prob, observed_freq, n], ...]`` from (prob, outcome) pairs."""
    if not pairs:
        return []
    probs = [p for p, _ in pairs]
    outcomes = [y for _, y in pairs]
    diagram = csep.reliability_diagram(probs, outcomes, n_bins=10)
    return diagram.as_rows()


def _write_summary(result: BackAnalysisResult, config: BackAnalysisConfig, results_dir: Path) -> Path:
    """Write the compact back-analysis JSON the web app reads. Atomic (temp + replace)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    fname = f"backanalysis-{result.region_id}-{result.start[:10]}_{result.end[:10]}.json"
    path = results_dir / fname

    payload = {
        "product": "CAOS_SEISMIC",
        "kind": "backanalysis",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "region": {"id": result.region_id, "name_en": config.region.name_en},
        "period": {"start": result.start, "end": result.end},
        "issue_cadence": "daily",
        "n_issue_days": result.n_issue_days,
        "n_scored_days": result.n_scored_days,
        "n_failed_days": result.n_failed_days,
        "horizons_days": result.horizons_days,
        "magnitude_thresholds": result.magnitude_thresholds,
        "reliability_threshold": config.reliability_threshold,
        "per_horizon": result.per_horizon,
        "reliability": result.reliability,
        "pycsep_used": result.pycsep_used,
        "config_hash": _safe_config_hash("forecast", "etas", "completeness", "declustering", "grid"),
        "framing": (
            "Pseudo-prospective, leakage-free (forecast clock). Every region×horizon cell is "
            "reported including failures (no post-hoc selection). Consistency tests calibrate one "
            "model; skill is established only by the comparison test (information gain vs ETAS). "
            "The headline thesis measurement is the information gain of the context-conditioned "
            "model over catalog-only ETAS (mean_context_gain_vs_etas_nats), in nats — it quantifies "
            "how much global context adds beyond the self-exciting catalog model. This complements "
            "official OEF systems; it is not a civil-protection alarm."
        ),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.info("wrote back-analysis summary: %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_master_catalog(catalog: pd.DataFrame | None, region: Region) -> pd.DataFrame:
    """Return the master catalog, loading the cleaned store lazily if one was not passed in."""
    if catalog is not None:
        from ..contracts import validate_catalog

        return validate_catalog(catalog)
    from ..data.clean import load_clean_catalog

    return load_clean_catalog(region)


def _catalog_only_etas_baseline(models: dict[str, Any]) -> tuple[Any, bool]:
    """Pick the **catalog-only ETAS** baseline for the context-gain channel + a context-active flag.

    The THESIS headline is the information gain of the *context-conditioned* model (``models[
    "primary"]``) over a catalog-only ETAS that has seen no global context. Two honest cases:

    * **Context channel active** — a distinct context-conditioned primary landed and a separate
      catalog-only ETAS (``models["etas"]``) is available: the baseline is that ETAS, and the gain
      measures exactly what the global context adds. Returns ``(etas, True)``.
    * **Context channel not yet landed** — the enricher stack is feature-flagged off (model-design
      §6.2), so the primary IS catalog-only ETAS (or its Reasenberg–Jones fallback). The baseline is
      then the same estimator as the primary, the gain is ~0 by construction, and the flag is
      ``False`` so the reduction labels the context contribution as "not yet measured" rather than
      implying a real zero. Returns ``(primary, False)``.

    Detecting "the primary is a distinct context model" without coupling to a concrete class: the
    context channel is active only when ``models["etas"]`` exists AND is a *different object* from
    ``models["primary"]`` (a separate context-conditioned model would replace the primary while
    keeping the plain ETAS around as the baseline).
    """
    primary = models["primary"]
    etas = models.get("etas")
    if etas is not None and etas is not primary:
        return etas, True
    return primary, False


def _bin_counts_to_cells(observed: pd.DataFrame, cells) -> np.ndarray:
    """Count observed events into the nearest fit cell (array aligned to ``cells``) for the IGPE sum."""
    omega = np.zeros(len(cells), dtype=float)
    if observed.empty or not cells:
        return omega
    lats = np.array([c.lat for c in cells])
    lons = np.array([c.lon for c in cells])
    for _, ev in observed.iterrows():
        d2 = (lats - float(ev["latitude"])) ** 2 + (lons - float(ev["longitude"])) ** 2
        omega[int(np.argmin(d2))] += 1.0
    return omega


def _as_utc(t: datetime | pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _safe_config_hash(*names: str) -> str | None:
    try:
        return config_hash(*names)
    except Exception:  # pragma: no cover - config optional
        return None

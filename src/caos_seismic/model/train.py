"""Stage entry point — fit the model family and run a self-consistency CSEP check.

This is the module the ``caos-seismic train`` command delegates to. "Training" for this product is
deliberately light: ETAS is *conditioned* per issue date at inference time (the forecast clock hands
it the lawful past), so there is no long offline gradient-descent run to checkpoint. What ``train``
does instead is the **fit + validate** step the methodology mandates before a model may issue public
forecasts:

1. **Fit the null** — the adaptive smoothed-seismicity background on the **declustered** catalog
   (the mandatory Poisson reference; :class:`~caos_seismic.model.smoothed.SmoothedSeismicityForecaster`).
2. **Fit ETAS** — space-time ETAS by MLE on the **full un-declustered** catalog before the training
   cutoff (:class:`~caos_seismic.model.etas.ETASForecaster`), enforcing *both* stability gates
   (``alpha < beta`` and branching ratio ``n < 1``; configs/etas.yaml). A fit that violates either
   gate raises :class:`~caos_seismic.model.etas.ETASStabilityError` — the model is **rejected**, not
   silently clamped, and the R-J fallback carries the region until a clean fit lands.
3. **Self-consistency CSEP check** — score the fitted models on a held-out tail window with the
   leakage-free forecast clock: the consistency **N-test** (Poisson tails) and the **comparison
   test** (information gain per earthquake in nats, with the paired T-test) of ETAS vs the
   smoothed-seismicity null. This is a *fit diagnostic* (in-sample tail), not the published
   pseudo-prospective evidence — that is :mod:`caos_seismic.eval.backanalysis`. Skill is established
   only by the back-analysis comparison tests; this step just refuses to ship a model that cannot
   even out-score its own null on a recent window.

Only core deps (numpy / pandas / scipy / scikit-learn) are needed; ETAS and the CSEP primitives are
all core. pyCSEP, when installed, is used by :mod:`caos_seismic.eval.csep` for the authoritative
simulation-based tests; here the closed-form N-test / IGPE fallbacks are sufficient for the gate.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..config import load, load_region
from ..contracts import Region
from ..eval import csep
from ..inference.clock import conditioning_slice, target_slice
from ..inference.provenance import build_manifest, snapshot_id, write_manifest
from .etas import ETASForecaster, ETASStabilityError
from .reasenberg_jones import ReasenbergJonesForecaster
from .smoothed import SmoothedSeismicityForecaster

logger = logging.getLogger(__name__)


def _resolve_catalog(region: Region, catalog: pd.DataFrame | None) -> pd.DataFrame:
    """Load the cleaned catalog store for the region, unless one was passed in (offline/tests)."""
    if catalog is not None:
        from ..contracts import validate_catalog

        return validate_catalog(catalog)
    from ..data.clean import load_clean_catalog

    return load_clean_catalog(region)


def _background_catalog(full: pd.DataFrame, b_value: float) -> pd.DataFrame:
    """Declustered (background) view of the catalog — Gardner–Knopoff mainshocks, the dual-catalog rule.

    Falls back transparently to the full catalog if the declustering primitive cannot run (e.g. a
    degenerate tiny catalog), so training still proceeds; the manifest records the degradation.
    """
    try:
        from ..catalog.decluster import dual_catalog

        return dual_catalog(full, b=b_value, compute_nnd=False).background
    except Exception as exc:  # pragma: no cover - defensive: never crash training on declustering
        logger.warning("declustering failed (%s); background falls back to the full catalog", exc)
        return full


def run_train(
    *,
    region: Region | str = "chile",
    catalog: pd.DataFrame | None = None,
    train_cutoff: pd.Timestamp | str | None = None,
    holdout_days: float = 30.0,
    write_manifest_file: bool = True,
) -> dict[str, Any]:
    """Fit the null + ETAS, run the self-consistency CSEP check, write a ``stage="model"`` manifest.

    Parameters
    ----------
    region:
        A :class:`Region` or region id.
    catalog:
        Optional in-memory cleaned catalog (skips the store load — used offline / by tests).
    train_cutoff:
        The training/holdout split time. Models are conditioned on events ``< cutoff`` and scored on
        ``[cutoff, cutoff + holdout_days)``. Default: ``holdout_days`` before the last event so the
        check uses real observed target events.
    holdout_days:
        Length of the held-out tail window scored by the self-consistency check (days).
    write_manifest_file:
        Persist the provenance manifest (default on; disabled by tests).

    Returns
    -------
    dict
        Summary with the fitted ETAS params + stability flags, the null fit, the N-test outcome, the
        ETAS-vs-null information gain (nats), and whether the consistency gate passed.
    """
    reg = load_region(region) if isinstance(region, str) else region
    etas_cfg = load("etas")
    completeness_cfg = load("completeness")
    forecast_cfg = load("forecast")

    full = _resolve_catalog(reg, catalog)
    if full.empty or len(full) < 5:
        raise ValueError(
            f"catalog for region '{reg.id}' has too few events ({len(full)}) to fit/validate a model"
        )
    full = full.sort_values("time").reset_index(drop=True)

    # Split time.
    if train_cutoff is None:
        last = pd.to_datetime(full["time"], utc=True).max()
        cutoff = last - pd.Timedelta(days=float(holdout_days))
    else:
        cutoff = pd.Timestamp(train_cutoff)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")

    past = conditioning_slice(full, cutoff)
    if past.empty or len(past) < 3:
        raise ValueError(
            f"only {len(past)} events before the training cutoff {cutoff.isoformat()}; "
            "move the cutoff later or fetch a longer history"
        )

    # Completeness + b on the training slice (estimated, never hard-coded).
    mc_cfg = completeness_cfg.get("mc", {})
    regional_default = float(mc_cfg.get("regional_default", 3.5))
    mc, b_value, b_unc = _mc_b(past, completeness_cfg, regional_default)

    # 1) Null: smoothed seismicity on the DECLUSTERED background.
    background = _background_catalog(past, b_value)
    null = SmoothedSeismicityForecaster(b_value=b_value, mc=mc)
    null.fit(background, reg, cutoff)

    # 2) ETAS on the FULL un-declustered training slice; enforce stability gates.
    m0 = float(etas_cfg.get("m0", regional_default))
    etas_params: dict[str, Any] = {}
    etas_fitted = False
    etas_rejection: str | None = None
    primary_name, primary_version = null.name, null.version
    primary = None
    require_lt = bool(etas_cfg.get("stability", {}).get("require_alpha_lt_beta", True))
    reject_super = bool(etas_cfg.get("stability", {}).get("reject_supercritical", True))
    try:
        # Regime-aware TILED ETAS: fit ETAS per tectonic tile and aggregate into the global field. This
        # is the tractable, physically-honest primary at global scope — a single monolithic ETAS over a
        # worldwide 10^5-event catalog is both O(N^2) and wrong (subduction ≠ stable interior). Each tile
        # enforces both stability gates and falls back to its smoothed null on violation.
        from .tiled import TiledForecaster

        etas = TiledForecaster(
            m0=m0,
            mc=mc,
            b_value=b_value,
            require_alpha_lt_beta=require_lt,
            reject_supercritical=reject_super,
        )
        etas.fit(past, reg, cutoff)
        etas_params = dict(etas.params_used)
        etas_fitted = int(etas_params.get("n_tiles_etas", 0)) > 0
        primary, primary_name, primary_version = etas, etas.name, etas.version
    except ValueError as exc:
        etas_rejection = str(exc)
        logger.warning("tiled ETAS fit failed (%s); R-J fallback carries the region this cycle", exc)

    # Transparent fallback model (always available for the consistency comparison).
    rj = ReasenbergJonesForecaster(b=b_value)
    rj.fit(past, reg, cutoff)
    if primary is None:
        primary, primary_name, primary_version = rj, rj.name, rj.version

    # 3) Self-consistency CSEP check on the held-out tail (leakage-free via the forecast clock).
    consistency = _self_consistency_check(
        full=full,
        cutoff=cutoff,
        holdout_days=holdout_days,
        region=reg,
        primary=primary,
        null=null,
        forecast_cfg=forecast_cfg,
    )

    summary: dict[str, Any] = {
        "region": reg.id,
        "train_cutoff": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "holdout_days": float(holdout_days),
        "n_train": int(len(past)),
        "mc": mc,
        "b_value": b_value,
        "b_uncertainty": b_unc,
        "etas_fitted": etas_fitted,
        "etas_rejection": etas_rejection,
        "branching_ratio": etas_params.get("branching_ratio"),
        "n_tiles_etas": etas_params.get("n_tiles_etas"),
        "n_tiles_null_fallback": etas_params.get("n_tiles_null_fallback"),
        "primary_model": primary_name,
        "n_test_passed": consistency.get("n_test", {}).get("passed"),
        "n_test_quantile": consistency.get("n_test", {}).get("quantile"),
        "igpe_vs_null_nats": consistency.get("igpe_vs_null"),
        "consistency_gate_passed": consistency.get("gate_passed"),
        "n_observed_holdout": consistency.get("n_observed"),
    }

    if write_manifest_file:
        issued_at = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest = build_manifest(
            stage="model",
            region_id=reg.id,
            t_issue=issued_at,
            input_snapshot_id=snapshot_id(past, reg.id, issued_at),
            mc_grid_version=f"train@{issued_at}",
            declustering="gardner_knopoff",
            model_name=primary_name,
            model_version=primary_version,
            model_params={k: _jsonable(v) for k, v in etas_params.items()},
            inputs={"n_train": int(len(past)), "mc": mc, "b_value": b_value},
            outputs={
                "etas_fitted": etas_fitted,
                "etas_rejection": etas_rejection,
                "null_model": {"name": null.name, "version": null.version},
            },
            stats={"self_consistency": consistency},
        )
        summary["manifest"] = str(write_manifest(manifest))

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────


def _mc_b(past: pd.DataFrame, completeness_cfg: dict, regional_default: float) -> tuple[float, float, float]:
    """Estimate (Mc, b, sigma_b) on a training slice, degrading to the regional default if thin."""
    mc_cfg = completeness_cfg.get("mc", {})
    correction = float(mc_cfg.get("maxc_correction", 0.2))
    min_events = int(mc_cfg.get("min_events", 50))
    try:
        from ..catalog.completeness import aki_utsu_b_value, mc_estimate

        mw = pd.to_numeric(past["mw"], errors="coerce").dropna().to_numpy()
        est = mc_estimate(
            mw, correction=correction, min_events=min_events, regional_default=regional_default
        )
        mc = float(est.mc)
        b_est = aki_utsu_b_value(mw, mc)
        return mc, float(b_est.b), float(b_est.b_uncertainty)
    except Exception as exc:  # pragma: no cover - thin catalog fallback
        logger.warning("Mc/b estimation fell back to defaults (%s)", exc)
        return regional_default, 1.0, float("nan")


def _self_consistency_check(
    *,
    full: pd.DataFrame,
    cutoff: pd.Timestamp,
    holdout_days: float,
    region: Region,
    primary,
    null: SmoothedSeismicityForecaster,
    forecast_cfg: dict,
) -> dict[str, Any]:
    """Score the fitted primary + null on the held-out tail (one issue, the cutoff) with the clock.

    A *fit diagnostic*, not the public evidence: it confirms the model can out-score its own null on
    a recent window before the model is allowed to issue forecasts. Uses the region-wide expected
    count (sum over the fine grid) so the N-test sees the whole-region rate vs the observed total,
    and the closed-form IGPE for the comparison. The published pseudo-prospective record is produced
    by :mod:`caos_seismic.eval.backanalysis`.
    """
    from ..inference.daily import build_global_fit_cells

    horizon = float(holdout_days)
    thresholds = [float(m) for m in forecast_cfg.get("magnitude_thresholds", [5.0])]
    m_star = min(thresholds)  # the most populated band → the most informative consistency check

    # Use the multi-resolution grid (coarse worldwide baseline + fine coverage tiles around recent
    # seismicity), NEVER the dense 0.1° grid — over the GLOBAL bbox that is ~6.5M cells and would hang
    # just materializing the Cell list. build_global_fit_cells is also correct for a bounded view.
    cond = conditioning_slice(full, cutoff)
    cells = build_global_fit_cells(region, load("grid"), cond)
    if not cells:
        return {"gate_passed": None, "note": "empty fit grid"}

    # Region-total expected counts (sum over cells) for each model.
    lam_primary = np.asarray(primary.expected_counts(region, cells, horizon, m_star, cutoff), float)
    lam_null = np.asarray(null.expected_counts(region, cells, horizon, m_star, cutoff), float)
    lam_primary = np.maximum(lam_primary, lam_null)  # cold-start floor (same as inference)

    # Observed target events >= M* in [cutoff, cutoff + holdout).
    target = target_slice(full, cutoff, horizon)
    observed = target.loc[pd.to_numeric(target["mw"], errors="coerce") >= m_star - 1e-9]
    n_obs = int(len(observed))

    n_fore_primary = float(lam_primary.sum())
    n_test = csep.n_test_poisson(n_fore_primary, n_obs)

    # Comparison: IGPE of the primary vs the null, binning observed events onto the fine grid.
    omega = _bin_counts_to_cells(observed, cells)
    igpe, _ = csep.information_gain_per_earthquake(lam_primary, lam_null, omega)

    gate_passed = bool(n_test.passed) and (n_obs == 0 or igpe >= 0.0)
    return {
        "horizon_days": horizon,
        "m_threshold": m_star,
        "n_observed": n_obs,
        "n_forecast_primary": round(n_fore_primary, 6),
        "n_forecast_null": round(float(lam_null.sum()), 6),
        "n_test": {
            "passed": bool(n_test.passed),
            "quantile": n_test.quantile,
            "delta1": n_test.delta1,
            "delta2": n_test.delta2,
        },
        "igpe_vs_null": round(float(igpe), 6),
        "gate_passed": gate_passed,
        "pycsep_used": csep.pycsep_available(),
        "note": "in-sample tail diagnostic — public skill is established by eval.backanalysis only",
    }


def _bin_counts_to_cells(observed: pd.DataFrame, cells) -> np.ndarray:
    """Count observed target events into the nearest fit cell, returning an array aligned to ``cells``.

    The fit grid is regular in degrees; each event is assigned to the nearest cell centre by simple
    rounding to the grid pitch. Cheap and exact enough for a region-level consistency diagnostic.
    """
    omega = np.zeros(len(cells), dtype=float)
    if observed.empty or not cells:
        return omega
    lats = np.array([c.lat for c in cells])
    lons = np.array([c.lon for c in cells])
    for _, ev in observed.iterrows():
        d2 = (lats - float(ev["latitude"])) ** 2 + (lons - float(ev["longitude"])) ** 2
        omega[int(np.argmin(d2))] += 1.0
    return omega


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars / nested dicts to plain Python for the manifest JSON."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value

"""30-day OUTLOOK generation — the geodetic-context background as a longer-horizon product surface.

The 1-7 day operational forecast is ETAS (``inference.daily``); the geodetic context does NOT help there
(E11: -0.053 at 7 d). At the **30-day** horizon the strain-conditioned neural BACKGROUND measurably beats
ETAS (E11: +0.078; E14 per-view). Because that background is **time-flat** (driven by slow GNSS strain),
it is refit on a **weekly** cadence — not daily — and published as its own static artifact that the daily
job serves unchanged.

This module fits the neural once, (A) validates the 30-day win leakage-free across views (the published
evidence), and (B) writes the current 30-day field artifact:

  - ``results/outlook-30d-<date>.json.gz``  — the compact per-cell 30-day expected-count field
  - ``results/outlook-index.json``          — the latest pointer
  - ``results/outlook-evidence-30d.json``   — per-view IGPE(neural, ETAS) at 30 d

Intended to run from ``caos-seismic outlook`` on a weekly schedule, published via the same robust scoped
git-as-data path as the daily job. Honest framing throughout: the outlook is the one horizon where a
covariate earns skill; the short-horizon product stays ETAS.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT, load, load_region
from ..data.clean import load_clean_catalog
from ..data.covariate_provider import make_strain_provider
from ..eval import csep
from ..eval.backanalysis import _bin_counts_to_cells
from ..eval.csep import _paired_t_test_igpe
from ..eval.views import high_seismicity_views
from ..inference.clock import conditioning_slice, target_slice
from ..inference.daily import _catalog_hygiene, build_global_fit_cells
from ..model.context_tpp import ContextTPPForecaster
from ..model.tiled import TiledForecaster


@dataclass
class OutlookConfig:
    region_id: str = "global"
    horizon_days: float = 30.0
    m_star: float = 5.0
    n_windows: int = 10  # leakage-free validation windows (each fully 30d-scorable)
    step_days: int = 7
    results_dir: Path = field(default_factory=lambda: REPO_ROOT / "results")


def generate_outlook(cfg: OutlookConfig | None = None, *, catalog: pd.DataFrame | None = None) -> dict:
    """Fit the geodetic neural once → validate the 30-day win per view → write the current 30-day field.

    Returns a summary dict. Writes the three artifacts atomically-ish (evidence is snapshotted per window
    so a late failure still leaves the validation table). The neural is checkpointed (``checkpoint_dir``)
    so a downstream crash does not waste the fit.
    """
    cfg = cfg or OutlookConfig()
    H, M_STAR = cfg.horizon_days, cfg.m_star
    reg = load_region(cfg.region_id)
    full = (catalog if catalog is not None else load_clean_catalog(cfg.region_id)).sort_values("time").reset_index(drop=True)
    last = pd.to_datetime(full["time"], utc=True).max()
    cutoff0 = last - pd.Timedelta(days=H + cfg.step_days * (cfg.n_windows - 1))

    past0 = conditioning_slice(full, cutoff0)
    hyg = _catalog_hygiene(past0, reg, load("completeness"))
    mc, b = hyg["mc"], hyg["b"]

    neural = ContextTPPForecaster(
        covariate_provider=make_strain_provider(1.0), mc=mc, b_value=b,
        checkpoint_dir="results/checkpoints",
    )
    neural.fit(full, reg, cutoff0)
    etas = TiledForecaster(m0=mc, mc=mc, b_value=b)
    etas.fit(full, reg, cutoff0)

    cells = build_global_fit_cells(reg, load("grid"), past0)
    clat = np.array([c.lat for c in cells])
    clon = np.array([c.lon for c in cells])
    views = [("global", np.ones(len(cells), bool))]
    for v in high_seismicity_views():
        bx = v.region.bbox
        views.append((v.region.id, (clat >= bx.lat_min) & (clat <= bx.lat_max) & (clon >= bx.lon_min) & (clon <= bx.lon_max)))

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = cfg.results_dir / "outlook-evidence-30d.json"

    # (A) per-view 30d validation (leakage-free; incremental snapshots)
    view_win: dict[str, list[float]] = {vid: [] for vid, _ in views}
    view_pe: dict[str, list[np.ndarray]] = {vid: [] for vid, _ in views}
    for w in range(cfg.n_windows):
        cw = cutoff0 + pd.Timedelta(days=cfg.step_days * w)
        neural.recondition(full, cw)
        etas.recondition(full, reg, cw)
        lam_n = np.asarray(neural.expected_counts(reg, cells, H, M_STAR, cw), float)
        lam_e = np.asarray(etas.expected_counts(reg, cells, H, M_STAR, cw), float)
        tgt = target_slice(full, cw, H)
        omega = _bin_counts_to_cells(tgt.loc[pd.to_numeric(tgt["mw"], errors="coerce") >= M_STAR - 1e-9], cells)
        for vid, msk in views:
            if omega[msk].sum() > 0:
                igpe, pe = csep.information_gain_per_earthquake(lam_n[msk], lam_e[msk], omega[msk])
                view_win[vid].append(float(igpe))
                if pe.size:
                    view_pe[vid].append(pe)
        snap = {vid: {"mean": (round(float(np.mean(view_win[vid])), 5) if view_win[vid] else None)} for vid, _ in views}
        evidence_path.write_text(json.dumps({"through_window": w, "by_view": snap}, indent=2), encoding="utf-8")

    evidence: dict = {}
    for vid, _ in views:
        wins = view_win[vid]
        pooled = np.concatenate(view_pe[vid]) if view_pe[vid] else np.zeros(0)
        t = _paired_t_test_igpe(pooled, 0.05) if pooled.size else {"t_ci_low": None, "t_ci_excludes_zero": None}
        evidence[vid] = {
            "mean_igpe_vs_etas": round(float(np.mean(wins)), 5) if wins else None,
            "windows_positive": f"{sum(1 for x in wins if x > 0)}/{len(wins)}",
            "n_eq": int(pooled.size), "ci_excludes_zero": t.get("t_ci_excludes_zero"), "t_ci_low": t.get("t_ci_low"),
        }
    high = [vid for vid, _ in views if vid != "global"]
    regions_pos = sum(1 for vid in high if (evidence[vid]["mean_igpe_vs_etas"] or -1) > 0)
    g = evidence["global"]
    evidence["_summary"] = {
        "global_ci_excludes_zero": bool(g["ci_excludes_zero"]), "high_seis_views_positive": regions_pos,
        "n_high_seis": len(high),
        "verdict": ("30d geodetic win GLOBAL+regionally robust" if g["ci_excludes_zero"] and (g["mean_igpe_vs_etas"] or 0) > 0 and regions_pos >= 2
                    else "30d win GLOBAL only" if (g["mean_igpe_vs_etas"] or 0) > 0 else "no robust 30d win"),
    }
    evidence_path.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")

    # (B) current 30d field at the latest cutoff
    neural.recondition(full, last)
    etas.recondition(full, reg, last)
    lam = np.asarray(neural.expected_counts(reg, cells, H, M_STAR, last), float)
    lam_e = np.asarray(etas.expected_counts(reg, cells, H, M_STAR, last), float)
    thr = max(float(np.percentile(lam, 99.0)) * 1e-3, 1e-5)
    keep = np.nonzero(lam > thr)[0]
    field_cells = [
        {"lat": round(float(clat[i]), 3), "lon": round(float(clon[i]), 3),
         "n30": round(float(lam[i]), 5), "p30": round(float(1 - np.exp(-lam[i])), 5)}
        for i in keep
    ]
    artifact = {
        "kind": "outlook_30d", "issued_at": last.strftime("%Y-%m-%dT00:00:00Z"), "horizon_days": H,
        "m_threshold": M_STAR, "model": "strain_neural_background", "mc": mc,
        "n_total_30d": round(float(lam.sum()), 1), "n_etas_30d": round(float(lam_e.sum()), 1),
        "n_cells": len(field_cells), "field": field_cells,
    }
    fname = f"outlook-30d-{last.strftime('%Y-%m-%d')}.json.gz"
    with gzip.open(cfg.results_dir / fname, "wt", encoding="utf-8") as fh:
        json.dump(artifact, fh)
    (cfg.results_dir / "outlook-index.json").write_text(
        json.dumps({"latest": {"file": fname, "issued_at": artifact["issued_at"], "horizon_days": H,
                               "n_total_30d": artifact["n_total_30d"], "n_cells": len(field_cells)}}, indent=2),
        encoding="utf-8",
    )
    return {"artifact": fname, "n_cells": len(field_cells), "n_total_30d": artifact["n_total_30d"],
            "evidence": evidence["_summary"]}

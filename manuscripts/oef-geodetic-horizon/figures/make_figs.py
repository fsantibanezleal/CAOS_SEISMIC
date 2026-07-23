"""Deterministically generate the validation figures for the OEF paper from the
committed CAOS_SEISMIC artifacts. Every number is read from results/*.json; nothing
is fabricated. Run with the figures venv (matplotlib). Outputs vector PDFs next to this file."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
RES = HERE.parents[2] / "results"          # CAOS_SEISMIC/results
BLUE, ORANGE, GRAY, INK = "#2b6cb0", "#c05621", "#718096", "#1a202c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#4a5568",
                     "axes.linewidth": 0.8, "svg.fonttype": "none"})


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


# ---- Fig 1: the horizon flip (E11 neural pseudo-prospective, per-window IGPE) ----
def fig_horizon():
    d = load("neural_pseudoprospective.json")
    w7 = d["by_horizon"]["7"]["per_window_igpe"]; m7 = d["by_horizon"]["7"]["mean_igpe_vs_etas_nats"]
    w30 = d["by_horizon"]["30"]["per_window_igpe"]; m30 = d["by_horizon"]["30"]["mean_igpe_vs_etas_nats"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.axhline(0, color=GRAY, lw=1, ls="--", zorder=1)
    import random  # not used for values; jitter is deterministic below
    jit7 = [(-1) ** i * 0.05 * (i % 3) for i in range(len(w7))]
    jit30 = [(-1) ** i * 0.05 * (i % 3) for i in range(len(w30))]
    ax.scatter([1 + j for j in jit7], w7, s=42, color=ORANGE, edgecolor="w", lw=0.6, zorder=3, label="per window")
    ax.scatter([2 + j for j in jit30], w30, s=42, color=BLUE, edgecolor="w", lw=0.6, zorder=3)
    ax.plot([0.7, 1.3], [m7, m7], color=ORANGE, lw=2.4, zorder=4)
    ax.plot([1.7, 2.3], [m30, m30], color=BLUE, lw=2.4, zorder=4)
    ax.annotate(f"mean {m7:+.3f}\n({d['by_horizon']['7']['n_windows_positive']}/{len(w7)} +)",
                (1.32, m7), fontsize=8, color=ORANGE, va="center")
    ax.annotate(f"mean {m30:+.3f}\n({d['by_horizon']['30']['n_windows_positive']}/{len(w30)} +)",
                (2.32, m30), fontsize=8, color=BLUE, va="center")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["7-day horizon", "30-day horizon"])
    ax.set_xlim(0.5, 2.9); ax.set_ylabel("IGPE, neural vs ETAS (nats)")
    ax.set_title("Geodetic context: loses at 1-7 d, wins at 30 d", fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(HERE / "fig-horizon-igpe.pdf"); plt.close(fig)


# ---- Fig 2: per-region 30-day IGPE (E14 outlook-evidence) ----
def fig_region():
    d = load("outlook-evidence-30d.json")
    order = [("global", "Global"), ("CL", "Chile"), ("JP", "Japan"), ("US-CA", "California"), ("NZ", "New Zealand")]
    vals = [d[k]["mean_igpe_vs_etas"] for k, _ in order]
    labs = [f"{name}\n(n={d[k]['n_eq']}, {d[k]['windows_positive']})" for k, name in order]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    y = range(len(order))
    ax.barh(list(y), vals, color=[INK if k == "global" else BLUE for k, _ in order], height=0.62, zorder=3)
    ax.axvline(0, color=GRAY, lw=1)
    for i, v in enumerate(vals):
        ax.text(v + 0.03, i, f"{v:+.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels(labs, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("mean IGPE vs ETAS at 30 d (nats)"); ax.set_xlim(0, max(vals) * 1.25)
    ax.set_title("30-day geodetic win: global and regionally robust", fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(HERE / "fig-region-30d.pdf"); plt.close(fig)


# ---- Fig 3: over-dispersion + secondary cascade (E13 catalog-based) ----
def fig_overdisp():
    d = load("e13_catalog_based.json")
    nfore, simmean, obs = d["n_fore_frozen"], d["sim_mean_with_cascade"], d["n_observed"]
    cb = d["catalog_n_test"]; p05, p95 = cb["sim_p05"], cb["sim_p95"]
    # Poisson 90% interval around N_fore (sqrt(N) sd, +-1.645) for the visual contrast
    import math
    lo_p, hi_p = nfore - 1.645 * math.sqrt(nfore), nfore + 1.645 * math.sqrt(nfore)
    fig, ax = plt.subplots(figsize=(4.9, 2.5))
    ax.add_patch(Rectangle((lo_p, 0.62), hi_p - lo_p, 0.16, color=GRAY, alpha=0.35, zorder=2))
    ax.add_patch(Rectangle((p05, 0.30), p95 - p05, 0.16, color=BLUE, alpha=0.30, zorder=2))
    ax.plot([nfore, nfore], [0.24, 0.84], color=ORANGE, lw=2, zorder=4)
    ax.plot([simmean, simmean], [0.24, 0.52], color=BLUE, lw=2, ls="--", zorder=4)
    ax.plot(obs, 0.38, marker="D", ms=9, color=INK, zorder=5)
    ax.text(nfore, 0.88, f"frozen N_fore={nfore:.1f}", ha="center", fontsize=8, color=ORANGE)
    ax.text((lo_p + hi_p) / 2, 0.70, f"Poisson 90%: FAIL (q={d['poisson_n_test']['q']:.4f})", ha="center", fontsize=7.5, color="#2d3748")
    ax.text((p05 + p95) / 2, 0.38, f"catalog-based 90%: PASS (q={cb['quantile']:.2f})", ha="center", fontsize=7.5, color=BLUE)
    ax.text(simmean, 0.18, f"cascade mean {simmean:.1f} (+{d['secondary_cascade_lift_pct']:.0f}%)", ha="center", fontsize=7.5, color=BLUE)
    ax.text(obs, 0.30, f"observed {obs}", ha="center", fontsize=8, color=INK)
    ax.set_xlim(min(lo_p, p05) - 5, max(hi_p, p95) + 5); ax.set_ylim(0.1, 1.0)
    ax.set_yticks([]); ax.set_xlabel("in-window M>=Mc count (30-day global window)")
    ax.set_title("Over-dispersion + secondary cascade, not a rate bias", fontsize=9.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(HERE / "fig-overdispersion.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_horizon(); fig_region(); fig_overdisp()
    print("figures written:", [p.name for p in sorted(HERE.glob("*.pdf"))])

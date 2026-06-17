"""Generate the bundled SAMPLE back-analysis report for the web app.

This is the MOCK at the data boundary ONLY (evaluation-plan.md §9). It writes a small,
schema-conforming back-analysis report (`back-analysis-sample.json`) so the Back-analysis
page renders before any real pyCSEP results exist. Real results produced by the offline
evaluation job replace this file byte-for-byte — the web code path is identical.

The numbers are ILLUSTRATIVE, clearly labelled `"sample": true`. They are NOT real CSEP
scores. They are constructed only to exercise the page: the N/M/S/L/CL consistency tests
(gridded AND catalog-based), the T/W comparison vs smoothed-seismicity AND ETAS baselines,
a reliability diagram per region × horizon, the expected-vs-observed time series, and — as
the evaluation plan requires — at least one HONEST FAILURE (a region × horizon where the
model does NOT beat ETAS), plus a Poisson-grid over-rejection paired with its catalog result.

Determinism: a fixed RNG seed, so the file regenerates identically. Outputs land under
`app/public/data/`. The shape mirrors `app/src/data/backanalysis.ts`.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "app" / "public" / "data"
OUT_FILE = "back-analysis-sample.json"

HORIZONS = [1, 2, 7]
SCHEMA_VERSION = "1.0"


def _consistency(rng: random.Random, *, passing: bool) -> dict:
    """A consistency block whose quantiles fall inside (passing) or outside the band."""

    def q() -> float:
        # In-band quantiles sit comfortably in (0.1, 0.9); failing ones hug a tail.
        if passing:
            return round(0.20 + 0.60 * rng.random(), 3)
        return round(0.02 + 0.06 * rng.random(), 3)

    n_d1 = q()
    return {
        "N": {"quantile": n_d1, "quantile2": round(min(0.95, n_d1 + 0.25), 3), "pass": passing},
        "M": {"quantile": q(), "pass": passing},
        "S": {"quantile": q(), "pass": passing},
        "L": {"quantile": q(), "pass": passing},
        "CL": {"quantile": q(), "pass": passing},
    }


def _reliability(rng: random.Random) -> list[list[float]]:
    """Reliability points hugging the diagonal, with realistic per-bin sample counts."""
    bins = [0.005, 0.02, 0.05, 0.1, 0.2, 0.4]
    pts: list[list[float]] = []
    for p in bins:
        # observed frequency = forecast prob + small zero-mean noise, clamped to [0,1]
        obs = min(1.0, max(0.0, p + (rng.random() - 0.5) * 0.04 * (p + 0.02)))
        n = round(600.0 * math.exp(-6.0 * p) + 25.0, 0)
        pts.append([round(p, 4), round(obs, 4), n])
    return pts


def _comparison(rng: random.Random, *, beats_etas: bool) -> list[dict]:
    """Comparison vs smoothed-seismicity (always beaten) AND ETAS (honest: sometimes not)."""
    # vs smoothed seismicity: the model reliably adds skill (IGPE positive, CI excludes 0).
    ig_sm = round(0.15 + 0.20 * rng.random(), 3)
    sm = {
        "baseline": "smoothed",
        "igpe_nats": ig_sm,
        "t_ci": [round(ig_sm - 0.10, 3), round(ig_sm + 0.10, 3)],
        "w_pvalue": round(0.001 + 0.01 * rng.random(), 4),
        "skill": True,
    }
    # vs ETAS: small gain, and only sometimes significant — the honest, hard comparison.
    if beats_etas:
        ig_et = round(0.04 + 0.06 * rng.random(), 3)
        ci_lo = round(ig_et - 0.03, 3)  # excludes 0 -> skill
        et = {
            "baseline": "etas",
            "igpe_nats": ig_et,
            "t_ci": [ci_lo, round(ig_et + 0.05, 3)],
            "w_pvalue": round(0.01 + 0.03 * rng.random(), 4),
            "skill": ci_lo > 0,
        }
    else:
        # near-zero / slightly negative: NO skill over ETAS — a published failure.
        ig_et = round(-0.02 + 0.04 * rng.random(), 3)
        et = {
            "baseline": "etas",
            "igpe_nats": ig_et,
            "t_ci": [round(ig_et - 0.06, 3), round(ig_et + 0.06, 3)],  # spans 0
            "w_pvalue": round(0.25 + 0.40 * rng.random(), 4),
            "skill": False,
        }
    return [sm, et]


def _scoring(rng: random.Random) -> dict:
    return {
        "brier": round(0.010 + 0.010 * rng.random(), 4),
        "log_score": round(0.06 + 0.04 * rng.random(), 4),
        "crps": round(0.020 + 0.015 * rng.random(), 4),
        "area_skill_score": round(0.58 + 0.20 * rng.random(), 3),
    }


def _expected_vs_observed(rng: random.Random, scale: float) -> list[dict]:
    months = [f"2021-{m:02d}" for m in range(1, 13)]
    out = []
    for mo in months:
        exp = round(scale * (0.6 + 0.8 * rng.random()), 2)
        # observed is Poisson-ish around expected, with one clustered spike to look real
        obs = max(0, round(exp + (rng.random() - 0.45) * exp * 1.4))
        out.append({"period": mo, "expected": exp, "observed": obs})
    # inject one aftershock-sequence spike (observed >> expected) to motivate over-dispersion
    spike = out[6]
    spike["observed"] = round(spike["expected"] * 3.4)
    return out


def _region(
    rng: random.Random,
    region_id: str,
    name_en: str,
    name_es: str,
    catalog: str,
    rationale_en: str,
    rationale_es: str,
    m_min: float,
    *,
    fail_horizon: int | None,
    overreject_horizon: int | None,
    evo_scale: float,
) -> dict:
    cells = []
    for h in HORIZONS:
        beats_etas = h != fail_horizon
        over_rejected = h == overreject_horizon
        cell = {
            "horizon_days": h,
            # During a sequence the Poisson grid test can over-reject -> mark that block failing,
            # but the catalog-based block (over-dispersion honest) passes; they are PAIRED.
            "consistency_gridded": _consistency(rng, passing=not over_rejected),
            "consistency_catalog": _consistency(rng, passing=True),
            "comparison": _comparison(rng, beats_etas=beats_etas),
            "reliability": _reliability(rng),
            "scoring": _scoring(rng),
            "expected_vs_observed": _expected_vs_observed(rng, evo_scale * h),
        }
        if over_rejected:
            cell["poisson_over_rejected"] = True
        cells.append(cell)
    return {
        "region_id": region_id,
        "name_en": name_en,
        "name_es": name_es,
        "catalog": catalog,
        "rationale_en": rationale_en,
        "rationale_es": rationale_es,
        "train_period": ["2007-01-01", "2017-12-31"],
        "test_period": ["2018-01-01", "2021-12-31"],
        "m_min": m_min,
        "cells": cells,
    }


def build_report() -> dict:
    rng = random.Random(20260616)
    regions = [
        _region(
            rng, "chile", "Chile (subduction margin)", "Chile (margen de subducción)",
            "Centro Sismológico Nacional (CSN), Universidad de Chile",
            "Subduction megathrust; the likely target region; supports a strong region-specific ETAS.",
            "Megacabalgamiento de subducción; región objetivo probable; permite un ETAS regional fuerte.",
            4.5, fail_horizon=7, overreject_horizon=1, evo_scale=2.2,
        ),
        _region(
            rng, "california", "California", "California",
            "ANSS / ComCat (SCEDC / NCEDC)",
            "Direct comparison to the 25 published CSEP next-day models (Serafini et al. 2025).",
            "Comparación directa con los 25 modelos CSEP de día siguiente publicados (Serafini et al. 2025).",
            3.95, fail_horizon=None, overreject_horizon=2, evo_scale=3.0,
        ),
        _region(
            rng, "japan", "Japan", "Japón",
            "JMA (research use)",
            "Dense network, very low Mc, abundant sequences — a demanding calibration test.",
            "Red densa, Mc muy bajo, secuencias abundantes — una prueba de calibración exigente.",
            4.0, fail_horizon=2, overreject_horizon=None, evo_scale=4.5,
        ),
        _region(
            rng, "new-zealand", "New Zealand", "Nueva Zelanda",
            "GeoNet",
            "Mixed subduction / crustal tectonics with an established CSEP testing history.",
            "Tectónica mixta de subducción / cortical con un historial CSEP establecido.",
            4.0, fail_horizon=None, overreject_horizon=7, evo_scale=2.0,
        ),
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "product": "CAOS_SEISMIC",
        "generated_at": datetime(2026, 6, 16, 6, 0, 0, tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "tooling": {
            "pycsep_version": "0.6.x",
            "note_en": (
                "All consistency and comparison tests use pyCSEP (Savran et al. 2022); reviewers can "
                "dispute the model, not the test code. Poisson grid tests over-reject during aftershock "
                "sequences, so every grid-test failure is paired with its catalog-based result."
            ),
            "note_es": (
                "Todas las pruebas de consistencia y comparación usan pyCSEP (Savran et al. 2022); se "
                "puede cuestionar el modelo, no el código de prueba. Las pruebas de grilla Poisson "
                "sobre-rechazan durante secuencias de réplicas; cada fallo de grilla se reporta junto a "
                "su resultado basado en catálogo."
            ),
        },
        "sample": True,
        "regions": regions,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    path = OUT_DIR / OUT_FILE
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_cells = sum(len(r["cells"]) for r in report["regions"])
    print(f"wrote {path}  ({len(report['regions'])} regions, {n_cells} cells, {path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

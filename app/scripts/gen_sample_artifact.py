"""Generate the bundled SAMPLE forecast artifact + index for the web app.

This is the MOCK at the data boundary ONLY (web-app-spec.md §8.3). It writes a small,
schema-conforming `ForecastArtifact` (and the `index.json` pointer) so the SPA runs and
previews before any real daily artifact exists. Real artifacts produced by the offline
git-as-data publish job replace these files byte-for-byte — the web client code path is
identical.

The numbers are ILLUSTRATIVE, clearly labelled SAMPLE in the artifact provenance. They
are NOT a forecast and must never be read as one. They are constructed only to exercise:
the sparse forecast tree, the bounds triad (lo/expected/hi), the baseline companion, the
coverage mask, the calibration/reliability fields, and the staleness banner.

Determinism: a fixed RNG seed + real H3 (v4) resolution-3 indices over the Chilean margin
so the same files regenerate identically. Run from anywhere; outputs land under
`app/public/data/`.

Validation: the constructed dict is round-tripped through the real Pydantic
`ForecastArtifact` from `src/caos_seismic/contracts.py`, so the sample cannot drift from
the schema the SPA's TypeScript types mirror.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h3

# Make the package importable when run from the repo without installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from caos_seismic.contracts import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    CalibrationSummary,
    ForecastArtifact,
    Region,
    Staleness,
)
from caos_seismic.config import load, load_region  # noqa: E402

H3_RES_WORLD = None  # filled from grid.yaml
OUT_DIR = REPO_ROOT / "app" / "public" / "data"


def _seeded_rng(seed: int):
    """A tiny deterministic LCG → floats in [0,1). Avoids a numpy dependency for the sample."""
    state = seed & 0xFFFFFFFF

    def nxt() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    return nxt


def _exceedance_p(rate: float) -> float:
    """P(>=1 event) = 1 - e^{-N} (Poisson), the governing exceedance map (contracts/forecast)."""
    import math

    return 1.0 - math.exp(-rate)


def build_artifact() -> ForecastArtifact:
    region: Region = load_region("chile")
    grid_cfg = load("grid")
    forecast_cfg = load("forecast")

    global H3_RES_WORLD
    H3_RES_WORLD = int(grid_cfg["display"]["h3_resolution_world"])
    horizons: list[int] = [int(h) for h in forecast_cfg["horizons_days"]]
    thresholds: list[float] = [float(m) for m in forecast_cfg["magnitude_thresholds"]]

    bbox = region.bbox
    # Cover the Chilean margin with real H3 r3 cells on a coarse lat/lon sweep, dedup by index.
    rnd = _seeded_rng(20260616)
    cell_seed: dict[str, tuple[float, float]] = {}
    lat = bbox.lat_min
    while lat <= bbox.lat_max:
        lon = bbox.lon_min
        while lon <= bbox.lon_max:
            idx = h3.latlng_to_cell(lat, lon, H3_RES_WORLD)
            if idx not in cell_seed:
                clat, clon = h3.cell_to_latlng(idx)
                cell_seed[idx] = (clat, clon)
            lon += 0.9
        lat += 0.9

    cells = sorted(cell_seed)  # stable order

    # Build the sparse forecast tree. One "active sequence" hotspot near the coast (north-central
    # Chile) makes the bounds triad and ratio-to-baseline visibly meaningful; the rest sits at a
    # low, well-constrained background — the realistic shape (most cells quiet).
    hot_lat, hot_lon = -30.0, -71.5  # illustrative elevated cell centroid
    forecast: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for idx in cells:
        clat, clon = cell_seed[idx]
        dist = ((clat - hot_lat) ** 2 + (clon - hot_lon) ** 2) ** 0.5
        # Elevation multiplier: large near the hotspot, ~1 far away (background only).
        elevate = 1.0 + 8.0 * pow(2.71828, -(dist * dist) / 2.0)

        per_h: dict[str, dict[str, dict[str, float]]] = {}
        for h in horizons:
            per_t: dict[str, dict[str, float]] = {}
            for m in thresholds:
                # Long-term background daily rate for M>=m (GR-like falloff per +1 magnitude),
                # scaled by horizon. Illustrative magnitudes only.
                base_daily = {5.0: 0.020, 6.0: 0.0020, 7.0: 0.00020}.get(m, 0.0005)
                baseline_rate = base_daily * h
                baseline_p = _exceedance_p(baseline_rate)

                # Conditional (forecast) rate = baseline * elevation, with a little cell jitter.
                jitter = 0.85 + 0.30 * rnd()
                fc_rate = baseline_rate * elevate * jitter
                p = _exceedance_p(fc_rate)
                # Over-dispersed bounds (wider than a naive Poisson interval): P10/P90 spread grows
                # with elevation to mimic real ETAS-parameter + Mc/b + structural uncertainty.
                spread = 0.35 + 0.25 * (elevate - 1.0) / 8.0
                lo = max(0.0, p * (1.0 - spread))
                hi = min(1.0, p * (1.0 + spread))

                per_t[f"{m}"] = {
                    "p": round(p, 6),
                    "lo": round(lo, 6),
                    "hi": round(hi, 6),
                    "rate": round(fc_rate, 6),
                    "baseline": round(baseline_p, 6),
                }
            per_h[str(h)] = per_t
        forecast[idx] = per_h

    # Coverage mask: a couple of southern offshore cells declared OUT of validated coverage so the
    # UI can exercise the "out of coverage" hatch (blank != safe). Take the 2 southernmost cells.
    southern = sorted(cells, key=lambda i: cell_seed[i][0])[:2]
    coverage_mask = southern

    # Illustrative reliability points (perfectly-on-diagonal-ish) per horizon-bin, and CSEP scores
    # inside the consistency band — the sample is honestly labelled "within consistency" so the badge
    # renders green for the demo. n is a fake sample count.
    reliability = [
        [0.01, 0.012, 480.0],
        [0.05, 0.047, 260.0],
        [0.10, 0.103, 150.0],
        [0.20, 0.191, 90.0],
        [0.40, 0.402, 40.0],
    ]
    csep = {
        "N": 0.52,
        "M": 0.61,
        "S": 0.48,
        "L": 0.55,
        "CL": 0.57,
        "pass": {"N": True, "M": True, "S": True, "L": True, "CL": True},
    }
    calibration = CalibrationSummary(
        reliability=reliability,
        csep=csep,
        info_gain_vs_poisson_nats=0.18,
        info_gain_vs_etas_nats=0.0,  # honest: matches ETAS, no claimed skill over it in the sample
    )

    now = datetime(2026, 6, 16, 6, 0, 0, tzinfo=timezone.utc)
    next_run = now + timedelta(days=1)
    staleness = Staleness(
        generated=now.isoformat().replace("+00:00", "Z"),
        next_run=next_run.isoformat().replace("+00:00", "Z"),
        ok=True,
    )

    artifact = ForecastArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        product="CAOS_SEISMIC",
        issued_at=now.isoformat().replace("+00:00", "Z"),
        region=region,
        horizons_days=horizons,
        magnitude_thresholds=thresholds,
        m_max=region.m_max,
        grid={"type": "h3", "resolution": H3_RES_WORLD},
        forecast=forecast,
        calibration=calibration,
        coverage_mask=coverage_mask,
        provenance={
            "sample": True,
            "note": (
                "ILLUSTRATIVE SAMPLE — not a real forecast. Generated by "
                "app/scripts/gen_sample_artifact.py to let the SPA preview before real "
                "daily artifacts exist. Replaced byte-for-byte by the daily publish job."
            ),
            "generator": "gen_sample_artifact.py",
            "catalog_versions": {"usgs_comcat": "SAMPLE", "isc_gem": "SAMPLE"},
            "model": {"name": "sample-placeholder", "version": "0.0.0"},
        },
        staleness=staleness,
    )
    return artifact


def build_index(artifact: ForecastArtifact, sample_file: str) -> dict:
    date = artifact.issued_at[:10]
    return {
        "schema_version": artifact.schema_version,
        "product": artifact.product,
        "updated_at": artifact.staleness.generated,
        "latest": sample_file,
        "gzipped": False,
        "history": [
            {
                "date": date,
                "file": sample_file,
                "gzipped": False,
                "issued_at": artifact.issued_at,
            }
        ],
        "calibration": artifact.calibration.model_dump(mode="json"),
        "staleness": artifact.staleness.model_dump(mode="json"),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact()
    sample_file = "forecast-sample.json"

    artifact_path = OUT_DIR / sample_file
    index_path = OUT_DIR / "index.json"

    artifact_json = json.dumps(artifact.model_dump_compact(), ensure_ascii=False, indent=2)
    index_json = json.dumps(build_index(artifact, sample_file), ensure_ascii=False, indent=2)

    artifact_path.write_text(artifact_json + "\n", encoding="utf-8")
    index_path.write_text(index_json + "\n", encoding="utf-8")

    n_cells = len(artifact.forecast)
    print(f"wrote {artifact_path}  ({n_cells} cells, {artifact_path.stat().st_size} bytes)")
    print(f"wrote {index_path}      ({index_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

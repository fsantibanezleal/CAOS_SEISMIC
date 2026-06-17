"""Inference-time machinery for CAOS_SEISMIC.

This subpackage holds the pieces a daily run shares with the back-analysis harness, so the
live product and the retrospective evaluation execute *identical* code:

* :class:`ForecastClock` and the ``conditioning_slice`` / ``target_slice`` /
  ``assert_no_leakage`` helpers make temporal leakage structurally impossible — a forecaster
  only ever sees the catalog slice strictly before ``t_issue`` (clock.py).
* The provenance helpers (``build_manifest``, ``write_manifest``, ``read_manifest``,
  ``provenance_block``, ``snapshot_id``, ``code_git_sha``) pin exactly what produced each
  forecast, so any past artifact is byte-reproducible months later (provenance.py).
* :mod:`caos_seismic.inference.daily` runs ONE daily inference (``run_infer``): clock → Mc/b +
  dual-catalog hygiene → fit ETAS (primary) / Reasenberg–Jones (fallback) / smoothed null (floor) →
  ensemble + **real** P10/median/P90 bounds → **isotonic** calibration → **QA gate** → assemble a
  :class:`~caos_seismic.contracts.ForecastField` then a :class:`~caos_seismic.contracts.ForecastArtifact`.
* :mod:`caos_seismic.inference.artifact` serializes that artifact to the compact on-disk form
  (``write_artifact``): H3 aggregation + rate quantization + sparsity + gzip to
  ``results/forecast-<region>-YYYY-MM-DD.json.gz`` and a ``results/index.json`` update; a loader
  (``load_artifact``) round-trips it.

The clock and provenance modules import core deps only and are eagerly re-exported. The heavier
``daily``/``artifact`` symbols (``run_infer``, ``write_artifact``, ``load_artifact``) are exposed
**lazily** via PEP 562 so ``import caos_seismic.inference`` stays cheap and a partially-landed
checkout still imports. Nothing here imports heavy optional deps; everything runs on the core stack.
"""

from __future__ import annotations

from .clock import (
    ForecastClock,
    assert_no_leakage,
    conditioning_slice,
    target_slice,
)
from .provenance import (
    build_manifest,
    code_git_sha,
    manifest_path,
    provenance_block,
    read_manifest,
    snapshot_id,
    write_manifest,
)

__all__ = [
    "ForecastClock",
    "assert_no_leakage",
    "conditioning_slice",
    "target_slice",
    "build_manifest",
    "code_git_sha",
    "manifest_path",
    "provenance_block",
    "read_manifest",
    "snapshot_id",
    "write_manifest",
    # daily / artifact — exposed lazily (see __getattr__) to keep package import light
    "run_infer",
    "DailyInferenceResult",
    "write_artifact",
    "load_artifact",
]


def __getattr__(name: str):  # PEP 562 — lazy re-export of the heavier daily/artifact symbols
    """Expose the daily/artifact entry points without importing them at package-import time.

    Keeps ``import caos_seismic.inference`` to the core-only clock + provenance modules while still
    allowing ``from caos_seismic.inference import run_infer`` / ``write_artifact`` / ``load_artifact``.
    """
    if name in {"run_infer", "DailyInferenceResult"}:
        from . import daily

        return getattr(daily, name)
    if name in {"write_artifact", "load_artifact", "serialize_artifact", "load_index"}:
        from . import artifact

        return getattr(artifact, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Evaluation subpackage — CSEP-style scoring of issued forecasts.

This is the credibility backbone. A forecast is only as honest as its prospective scoring
(methodology.md, "Evaluation framework"). Two layers live here:

  * :mod:`caos_seismic.eval.csep` — the *scoring primitives*: consistency tests
    (N / M / S / L / conditional-L), comparison tests (paired **T-test** + **W-test** on
    **information-gain-per-earthquake in nats**) against BOTH a Poisson/smoothed null AND ETAS,
    plus the reliability diagram. pyCSEP (Savran et al. 2022, doi:10.1785/0220220033) is the
    authoritative implementation and is used when installed; pure-numpy fallbacks cover the
    N-test, the Poisson log-likelihood / information gain (nats), the Brier score, and the
    reliability bins so the package scores without the heavy optional dependency.

  * :mod:`caos_seismic.eval.backanalysis` — the *pseudo-prospective driver*: it steps the daily
    forecast clock over a region × period, scores each issued forecast against the catalog slice
    it could not have seen, accumulates the consistency + comparison + reliability results, and
    emits a compact JSON summary into ``results/`` (per region × period × horizon, failures
    included) for the web app's Back-analysis section.

Framing (non-negotiable): skill is established **only** by winning comparison tests against real
baselines (smoothed-seismicity *and* ETAS); passing consistency tests is necessary but not
sufficient. AUC is **not** a skill metric here and is deliberately absent (DeVries trap;
methodology.md §2.4). Information gain is always reported in **nats**, never bits.
"""

from __future__ import annotations

from .backanalysis import (
    BackAnalysisConfig,
    BackAnalysisResult,
    ScoredForecast,
    run_back_analysis,
)
from .csep import (
    ComparisonResult,
    ConsistencyResult,
    ReliabilityDiagram,
    brier_score,
    comparison_tests,
    consistency_tests,
    information_gain_per_earthquake,
    n_test_poisson,
    poisson_joint_log_likelihood,
    pycsep_available,
    reliability_diagram,
)

__all__ = [
    # csep — scoring primitives
    "ConsistencyResult",
    "ComparisonResult",
    "ReliabilityDiagram",
    "consistency_tests",
    "comparison_tests",
    "reliability_diagram",
    "n_test_poisson",
    "poisson_joint_log_likelihood",
    "information_gain_per_earthquake",
    "brier_score",
    "pycsep_available",
    # backanalysis — pseudo-prospective driver
    "BackAnalysisConfig",
    "BackAnalysisResult",
    "ScoredForecast",
    "run_back_analysis",
]

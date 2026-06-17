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

  * :mod:`caos_seismic.eval.backanalysis` — the *pseudo-prospective driver* for ONE region/view: it
    steps the daily forecast clock over a region × period, scores each issued forecast against the
    catalog slice it could not have seen, accumulates the consistency + comparison + reliability
    results (including the THESIS **context gain** vs catalog-only ETAS), and emits a compact JSON
    summary into ``results/`` (per region × period × horizon, failures included).

  * :mod:`caos_seismic.eval.views` — the **country VIEWs** into the single global field: a
    pre-registered high-seismicity set (active plate boundaries) and low-seismicity set (stable
    interiors) plus the GLOBAL view, each a plain :class:`~caos_seismic.contracts.Region`.

  * :mod:`caos_seismic.eval.global_backanalysis` — the **multi-view + global driver**: it runs the
    per-view back-analysis through every country view and globally, then reduces to the two headline
    numbers — the **information gain over catalog-only ETAS** (how much global context contributes,
    per view + pooled) and the **HIGH-vs-LOW-seismicity bias** (does the model over-fit loud zones?).

Framing (non-negotiable): skill is established **only** by winning comparison tests against real
baselines (smoothed-seismicity *and* ETAS); passing consistency tests is necessary but not
sufficient. The re-scoped headline is the information gain of the context-conditioned model over
**catalog-only ETAS** — both already reproduce Omori clustering, so a positive gain is the global
context's contribution, not "I predicted aftershocks". AUC is **not** a skill metric here and is
deliberately absent (DeVries trap; methodology.md §2.4). Information gain is always reported in
**nats**, never bits.
"""

from __future__ import annotations

from .backanalysis import (
    BackAnalysisConfig,
    BackAnalysisResult,
    ScoredForecast,
    run_back_analysis,
)
from .global_backanalysis import (
    GlobalBackAnalysisConfig,
    GlobalBackAnalysisResult,
    ViewResult,
    run_global_back_analysis,
    run_global_backanalysis,
)
from .views import (
    CountryView,
    GLOBAL_VIEW,
    all_views,
    country_views,
    high_seismicity_views,
    low_seismicity_views,
    view_by_id,
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
    # backanalysis — pseudo-prospective driver (single region/view)
    "BackAnalysisConfig",
    "BackAnalysisResult",
    "ScoredForecast",
    "run_back_analysis",
    # global_backanalysis — multi-view + global driver (context gain + high/low bias)
    "GlobalBackAnalysisConfig",
    "GlobalBackAnalysisResult",
    "ViewResult",
    "run_global_back_analysis",
    "run_global_backanalysis",
    # views — country windows into the global field
    "CountryView",
    "GLOBAL_VIEW",
    "all_views",
    "country_views",
    "high_seismicity_views",
    "low_seismicity_views",
    "view_by_id",
]

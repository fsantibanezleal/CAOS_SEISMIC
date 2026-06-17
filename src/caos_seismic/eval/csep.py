"""CSEP scoring primitives — consistency tests, comparison tests, reliability diagram.

This module is the *scoring* half of the evaluation backbone. It speaks the field-standard CSEP
language (Schorlemmer et al. 2007, *SRL* 78:17-29, doi:10.1785/gssrl.78.1.17; Zechar, Gerstenberger
& Rhoades 2010, *BSSA* 100:1184-1195, doi:10.1785/0120090192) and defers to **pyCSEP** (Savran et
al. 2022, *SRL* 93:2858-2870, doi:10.1785/0220220033) as the authoritative implementation whenever
it is installed. pyCSEP is the path reviewers can dispute least: they can argue with our model, not
with peer-reviewed test code.

When pyCSEP is **absent**, this module still scores via dependency-free numpy fallbacks for the
subset that has a closed form: the **N-test** (Poisson tails), the **Poisson joint log-likelihood**
and **information-gain-per-earthquake in nats**, the **Brier score**, and the **reliability bins**.
The fallbacks carry an explicit ``pycsep_used: False`` flag so a consumer never mistakes them for
the authoritative result. The M-test, S-test, and L/CL-tests require simulation machinery that is
correctly delegated to pyCSEP; their fallbacks raise an actionable error rather than fake a number.

Two representations are scored (methodology.md §E.1), with different tests:

* **gridded-rate** — a Poisson expected count :math:`\\lambda_i` per space (×magnitude) bin; the
  Poisson consistency tests apply directly.
* **catalog-based** — an ensemble of :math:`\\ge 10{,}000` synthetic catalogs; empirical,
  non-Poisson tests that relax the Poisson assumption, because regional seismicity is
  over-dispersed (variance :math:`\\gg` mean) and the Poisson grid tests **over-reject during
  aftershock sequences** (Kagan 2017, *GJI* 211:335-345, doi:10.1093/gji/ggx300; Savran et al.
  2020, *BSSA* 110:1799-1817, doi:10.1785/0120200026). pyCSEP owns the catalog-based path.

Hard rule honoured here: **AUC is not a skill metric** (methodology.md §2.4, the DeVries trap) and
is intentionally not implemented as one. The Molchan/Area-Skill view is a communication aid built
elsewhere, never the headline.

Skill rule honoured here: **passing consistency tests is necessary but not sufficient**; skill is
established only by a comparison-test win (positive IGPE with a T-test CI excluding zero, corroborated
by the W-test) against BOTH a smoothed-seismicity null AND an ETAS baseline (methodology.md §E.3).
Information gain is in **nats** (natural log), never bits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Lazy pyCSEP import — never at module top level (keeps the package importable with core deps only)
# ─────────────────────────────────────────────────────────────────────────────


def pycsep_available() -> bool:
    """Return ``True`` iff pyCSEP (``import csep``) can be imported in this environment."""
    try:
        import csep  # noqa: F401

        return True
    except Exception:
        return False


def _require_pycsep():
    """Import and return the ``csep`` package, or raise an actionable error if it is absent.

    pyCSEP is the *authoritative* path for the simulation-based tests (M / S / L / CL and all
    catalog-based tests). The N-test, Poisson log-likelihood, IGPE, Brier, and reliability bins
    have numpy fallbacks elsewhere in this module; this guard is only for tests with no closed form.
    """
    try:
        import csep

        return csep
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise ImportError(
            "pyCSEP is required for the simulation-based CSEP tests (M / S / L / CL and the "
            "catalog-based tests) but is not installed. Install the science extra:\n"
            "    pip install caos-seismic[science]\n"
            "or directly:\n"
            "    pip install pycsep>=0.6\n"
            "Closed-form scores (N-test, Poisson log-likelihood, information gain in nats, Brier "
            "score, reliability bins) are available without pyCSEP via this module's fallbacks."
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Result containers (plain dataclasses → JSON-friendly via `as_dict`)
# ─────────────────────────────────────────────────────────────────────────────

# Quantile-score field names per test, matching the CalibrationSummary.csep schema {N,M,S,L,CL}.
_TEST_QUANTILE_NAME = {"N": "delta", "M": "kappa", "S": "zeta", "L": "gamma", "CL": "gamma"}


@dataclass
class ConsistencyResult:
    """Outcome of one CSEP consistency (calibration) test for a single forecast.

    Consistency tests calibrate *one* model; they never establish skill (methodology.md §E.3).
    ``quantile`` is the CSEP quantile score in [0, 1] (the test's :math:`\\delta`, :math:`\\kappa`,
    :math:`\\zeta`, or :math:`\\gamma`); ``passed`` applies the two-sided ``alpha`` rejection rule.
    For the N-test both Poisson tails are kept: ``delta1`` (small ⇒ forecast too **few** events)
    and ``delta2`` (small ⇒ too **many**), per the pyCSEP convention.
    """

    test: Literal["N", "M", "S", "L", "CL"]
    quantile: float | None
    passed: bool | None
    alpha: float = 0.05
    delta1: float | None = None  # N-test only
    delta2: float | None = None  # N-test only
    n_obs: int | None = None
    n_forecast: float | None = None
    observed_statistic: float | None = None
    pycsep_used: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != ""}


@dataclass
class ComparisonResult:
    """Outcome of a paired comparison test of model A vs baseline B — where skill lives.

    The metric is **information gain per earthquake (IGPE)** in **nats** (methodology.md §E.3,
    Rhoades et al. 2011, *Acta Geophysica* 59:728-747, doi:10.2478/s11600-011-0013-5). Skill over
    baseline ``B`` is claimed only when ``igpe > 0`` AND the T-test CI excludes zero
    (``t_ci_excludes_zero``), corroborated by the W-test (``w_pvalue`` small). Honest expectation:
    IGPE is state-dependent — large during active sequences, ≈0 when quiet — so a non-significant or
    negative result is reported as such, never hidden.
    """

    baseline: str  # e.g. "poisson_smoothed" or "etas"
    igpe: float  # information gain per earthquake, in nats
    n_earthquakes: int
    t_statistic: float | None = None
    t_ci_low: float | None = None
    t_ci_high: float | None = None
    t_ci_excludes_zero: bool | None = None
    w_statistic: float | None = None
    w_pvalue: float | None = None
    skill_demonstrated: bool | None = None
    pycsep_used: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != ""}


@dataclass
class ReliabilityDiagram:
    """Binned reliability diagram (the headline credibility artifact, methodology.md §6.4).

    ``bins`` is a list of ``[forecast_prob, observed_freq, n]`` rows — exactly the shape consumed by
    :class:`caos_seismic.contracts.CalibrationSummary.reliability`. ``brier`` is the accompanying
    Brier score (Brier 1950) for the same binary exceedance outcomes; a strictly proper scoring rule
    that complements (never replaces) the CSEP calibration test.
    """

    bins: list[list[float]] = field(default_factory=list)
    brier: float | None = None
    n_total: int = 0
    note: str = ""

    def as_rows(self) -> list[list[float]]:
        """Return the ``[[forecast_prob, observed_freq, n], ...]`` rows for the artifact schema."""
        return [list(row) for row in self.bins]

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"reliability": self.as_rows(), "n_total": self.n_total}
        if self.brier is not None:
            out["brier"] = self.brier
        if self.note:
            out["note"] = self.note
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Closed-form primitives (numpy only — always available)
# ─────────────────────────────────────────────────────────────────────────────


def poisson_joint_log_likelihood(
    forecast_rates: np.ndarray, observed_counts: np.ndarray
) -> float:
    r"""Poisson joint log-likelihood over space (×magnitude) bins (methodology.md §E.2).

    .. math::
        L(\Omega \mid \Lambda) = \sum_i \big[-\lambda_i + \omega_i \ln \lambda_i - \ln(\omega_i!)\big]

    with :math:`\lambda_i` the forecast expected count and :math:`\omega_i` the observed count in
    bin :math:`i`. This is the kernel of the L-test and of the logarithmic score
    (:math:`\text{LogS} = -\ln p`). Empty-forecast bins with observed events are penalised with a
    floor (``eps``) rather than :math:`-\infty`, matching pyCSEP's handling of unforecast events.

    Uses :func:`scipy.stats.poisson.logpmf` for numerical stability of :math:`\ln(\omega_i!)`.
    """
    rates = np.asarray(forecast_rates, dtype=float)
    counts = np.asarray(observed_counts, dtype=float)
    if rates.shape != counts.shape:
        raise ValueError(f"shape mismatch: rates {rates.shape} vs counts {counts.shape}")
    eps = np.finfo(float).tiny
    rates = np.clip(rates, eps, None)
    return float(np.sum(stats.poisson.logpmf(counts, rates)))


def n_test_poisson(
    forecast_total: float, observed_total: int, alpha: float = 0.05
) -> ConsistencyResult:
    r"""CSEP **N-test** (number) via the Poisson tails — closed form, no pyCSEP needed.

    The expected number of target events is :math:`N_{\text{fore}} = \sum_i \lambda_i`; the test
    asks whether the observed total :math:`N_{\text{obs}}` is consistent with a Poisson of that mean.
    Following the pyCSEP convention (methodology.md §E.2):

    .. math::
        \delta_1 = 1 - F\big(N_{\text{obs}} - 1 \mid N_{\text{fore}}\big), \qquad
        \delta_2 = F\big(N_{\text{obs}} \mid N_{\text{fore}}\big)

    where :math:`F` is the Poisson CDF. Interpretation: small :math:`\delta_1` ⇒ observed too
    **many** for the forecast (forecast too low); small :math:`\delta_2` ⇒ too **few** (forecast too
    high). The forecast is rejected (two-sided) when :math:`\min(\delta_1, \delta_2) < \alpha / 2`.
    """
    n_fore = float(forecast_total)
    n_obs = int(observed_total)
    if n_fore <= 0:
        n_fore = np.finfo(float).tiny
    delta1 = float(1.0 - stats.poisson.cdf(n_obs - 1, n_fore))
    delta2 = float(stats.poisson.cdf(n_obs, n_fore))
    quantile = float(min(delta1, delta2))
    passed = quantile >= alpha / 2.0
    return ConsistencyResult(
        test="N",
        quantile=quantile,
        passed=passed,
        alpha=alpha,
        delta1=delta1,
        delta2=delta2,
        n_obs=n_obs,
        n_forecast=n_fore,
        pycsep_used=False,
        note="numpy Poisson-tail N-test (pyCSEP is the authoritative path)",
    )


def information_gain_per_earthquake(
    rates_a: np.ndarray,
    rates_b: np.ndarray,
    observed_counts: np.ndarray,
) -> tuple[float, np.ndarray]:
    r"""Information gain per earthquake (IGPE) of model A over baseline B, in **nats**.

    .. math::
        I_N(A, B) = \frac{1}{N}\sum_{i=1}^{N}\big(\ln \lambda_A(k_i) - \ln \lambda_B(k_i)\big)
                    - \frac{\hat N_A - \hat N_B}{N}

    (methodology.md §E.3; Rhoades et al. 2011, doi:10.2478/s11600-011-0013-5). The sum runs over the
    :math:`N = \sum_i \omega_i` observed target earthquakes — a bin with :math:`\omega_i` events
    contributes its log-rate difference :math:`\omega_i (\ln \lambda_A - \ln \lambda_B)`. The second
    term subtracts the rate-normalisation difference :math:`(\hat N_A - \hat N_B)/N`, with
    :math:`\hat N = \sum_i \lambda_i`.

    Returns ``(igpe, per_event_samples)``: the scalar IGPE in **nats**, and a **per-earthquake**
    array of the paired differences :math:`X_i - Y_i = \ln \lambda_A(k_i) - \ln \lambda_B(k_i) -
    (\hat N_A - \hat N_B)/N` (one entry per observed event, bins expanded by :math:`\omega_i`). That
    per-earthquake sample is the exact paired-difference series the T-test / W-test consume
    (Rhoades et al. 2011), so ``igpe == mean(per_event_samples)`` by construction. **Nats, never
    bits.**
    """
    a = np.asarray(rates_a, dtype=float)
    b = np.asarray(rates_b, dtype=float)
    omega = np.asarray(observed_counts, dtype=float)
    if not (a.shape == b.shape == omega.shape):
        raise ValueError(
            f"shape mismatch: A {a.shape}, B {b.shape}, observed {omega.shape}"
        )
    n_total = float(omega.sum())
    if n_total <= 0:
        # No target events: IGPE is undefined; return 0 gain over zero events (caller handles N=0).
        return 0.0, np.zeros(0, dtype=float)

    eps = np.finfo(float).tiny
    a = np.clip(a, eps, None)
    b = np.clip(b, eps, None)
    n_hat_a = float(a.sum())
    n_hat_b = float(b.sum())
    norm_per_event = (n_hat_a - n_hat_b) / n_total

    # Per-bin log-rate difference, then expand each bin to one sample per observed earthquake so the
    # T-test/W-test see the true per-event paired-difference series with N = #earthquakes.
    log_diff = np.log(a) - np.log(b)
    counts = np.rint(omega).astype(int)
    per_event = np.repeat(log_diff - norm_per_event, np.clip(counts, 0, None))
    igpe = float(per_event.mean()) if per_event.size else 0.0
    return igpe, per_event


def brier_score(forecast_probs: Sequence[float], outcomes: Sequence[int]) -> float:
    r"""Brier score for the bounded binary exceedance output (Brier 1950).

    .. math::
        \text{BS} = \frac{1}{T}\sum_t (p_t - y_t)^2 ,

    with :math:`p_t \in [0, 1]` the forecast exceedance probability and :math:`y_t \in \{0, 1\}` the
    outcome. A strictly proper scoring rule (Gneiting & Raftery 2007, doi:10.1198/016214506000001437)
    that complements CSEP — lower is better, 0 is perfect. Used as a secondary scoring aid, never as
    the headline skill metric.
    """
    p = np.asarray(forecast_probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: probs {p.shape} vs outcomes {y.shape}")
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def reliability_diagram(
    forecast_probs: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> ReliabilityDiagram:
    r"""Compute a binned reliability diagram + Brier score (methodology.md §6.4).

    Each forecast probability is assigned to one of ``n_bins`` equal-width bins on [0, 1]; for each
    non-empty bin we report ``[mean_forecast_prob, observed_frequency, n]`` — the public "when we
    said X %, it happened ~X %" artifact. The output ``bins`` matches
    :class:`caos_seismic.contracts.CalibrationSummary.reliability` exactly.

    Validate calibration *specifically in the quiet / cold-start regime*, which dominates the diagram
    because most cells are quiet (model-design.md §8) — not only during active sequences. This is a
    closed-form computation (no pyCSEP needed); the pyCSEP ``calibration`` test is reported alongside
    by :func:`consistency_tests` when available.
    """
    p = np.asarray(forecast_probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: probs {p.shape} vs outcomes {y.shape}")
    diagram = ReliabilityDiagram(n_total=int(p.size))
    if p.size == 0:
        diagram.note = "no forecast/outcome pairs"
        return diagram

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize: bin index 1..n_bins; clip the right edge (prob == 1.0) into the last bin.
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows: list[list[float]] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_p = float(p[mask].mean())
        obs_freq = float(y[mask].mean())
        rows.append([round(mean_p, 6), round(obs_freq, 6), n])
    diagram.bins = rows
    diagram.brier = brier_score(p, y)
    return diagram


# ─────────────────────────────────────────────────────────────────────────────
# Consistency tests — pyCSEP when present, numpy N-test fallback otherwise
# ─────────────────────────────────────────────────────────────────────────────


def consistency_tests(
    forecast_rates: np.ndarray,
    observed_counts: np.ndarray,
    *,
    forecast=None,
    observed_catalog=None,
    alpha: float = 0.05,
    n_simulations: int = 1000,
    seed: int | None = 23,
    tests: Sequence[str] = ("N", "M", "S", "L", "CL"),
) -> dict[str, ConsistencyResult]:
    """Run the CSEP consistency (calibration) tests for one gridded forecast.

    Parameters
    ----------
    forecast_rates, observed_counts
        Aligned arrays of per-bin forecast expected counts :math:`\\lambda_i` and observed counts
        :math:`\\omega_i`. Always sufficient for the closed-form **N-test**; the M/S/L/CL tests
        additionally need the pyCSEP objects below.
    forecast, observed_catalog
        Optional pyCSEP ``GriddedForecast`` and ``CSEPCatalog`` (or the catalog-based equivalents).
        When both are supplied AND pyCSEP is installed, the authoritative pyCSEP
        ``number / magnitude / spatial / likelihood / conditional_likelihood`` tests are used for
        every requested test and ``pycsep_used`` is ``True``. When they are absent, only the N-test
        is computed (numpy); M/S/L/CL come back as ``passed=None`` with a note pointing at pyCSEP,
        because faking a simulation-based quantile would be dishonest.

    Returns
    -------
    dict
        Mapping ``{test_name: ConsistencyResult}`` restricted to ``tests``. The N-test result always
        carries ``delta1`` / ``delta2``. Consistency calibrates one model; it never establishes
        skill — see :func:`comparison_tests`.
    """
    rates = np.asarray(forecast_rates, dtype=float)
    counts = np.asarray(observed_counts, dtype=float)
    requested = [t for t in tests if t in _TEST_QUANTILE_NAME]
    results: dict[str, ConsistencyResult] = {}

    use_pycsep = (
        forecast is not None and observed_catalog is not None and pycsep_available()
    )

    if use_pycsep:
        results.update(
            _consistency_tests_pycsep(
                forecast, observed_catalog, requested, alpha, n_simulations, seed
            )
        )
        # Guarantee an N-test even if pyCSEP returned an unexpected shape.
        if "N" in requested and "N" not in results:
            results["N"] = n_test_poisson(float(rates.sum()), int(counts.sum()), alpha)
        return results

    # ── numpy-only path ──────────────────────────────────────────────────────
    for test in requested:
        if test == "N":
            results["N"] = n_test_poisson(float(rates.sum()), int(counts.sum()), alpha)
        else:
            results[test] = ConsistencyResult(
                test=test,  # type: ignore[arg-type]
                quantile=None,
                passed=None,
                alpha=alpha,
                pycsep_used=False,
                note=(
                    f"{test}-test requires the simulation-based pyCSEP implementation "
                    "(no closed form). Install caos-seismic[science] / pycsep, and pass the "
                    "pyCSEP `forecast` + `observed_catalog` objects, for the authoritative result."
                ),
            )
    return results


def _consistency_tests_pycsep(
    forecast,
    observed_catalog,
    requested: Sequence[str],
    alpha: float,
    n_simulations: int,
    seed: int | None,
) -> dict[str, ConsistencyResult]:
    """Authoritative consistency tests through pyCSEP's gridded evaluation API.

    Maps the CSEP test names to ``csep.core.poisson_evaluations``:
    ``number_test`` (N), ``magnitude_test`` (M), ``spatial_test`` (S), ``likelihood_test`` (L), and
    ``conditional_likelihood_test`` (CL). Returns a ``ConsistencyResult`` per requested test with
    ``pycsep_used=True``.
    """
    _require_pycsep()
    from csep.core import poisson_evaluations as pe  # type: ignore

    dispatch = {
        "N": ("number_test", {}),
        "M": ("magnitude_test", {"num_simulations": n_simulations, "seed": seed}),
        "S": ("spatial_test", {"num_simulations": n_simulations, "seed": seed}),
        "L": ("likelihood_test", {"num_simulations": n_simulations, "seed": seed}),
        "CL": (
            "conditional_likelihood_test",
            {"num_simulations": n_simulations, "seed": seed},
        ),
    }
    out: dict[str, ConsistencyResult] = {}
    for test in requested:
        if test not in dispatch:
            continue
        fname, kwargs = dispatch[test]
        func = getattr(pe, fname, None)
        if func is None:  # pragma: no cover - pyCSEP API drift guard
            out[test] = ConsistencyResult(
                test=test,  # type: ignore[arg-type]
                quantile=None,
                passed=None,
                alpha=alpha,
                pycsep_used=True,
                note=f"pyCSEP poisson_evaluations.{fname} not found in this version",
            )
            continue
        try:
            res = func(forecast, observed_catalog, **{k: v for k, v in kwargs.items() if v is not None})
        except TypeError:
            # Older/newer signatures may not accept seed; retry without it.
            res = func(forecast, observed_catalog)
        out[test] = _wrap_pycsep_consistency(test, res, alpha)
    return out


def _wrap_pycsep_consistency(test: str, res, alpha: float) -> ConsistencyResult:
    """Translate a pyCSEP ``EvaluationResult`` into a :class:`ConsistencyResult`.

    pyCSEP returns ``quantile`` as a scalar for M/S/L/CL and a ``(delta1, delta2)`` pair for the
    number test. The two-sided rejection rule is :math:`\\min(q, 1-q) \\ge \\alpha/2` for the
    single-quantile tests and :math:`\\min(\\delta_1, \\delta_2) \\ge \\alpha/2` for N.
    """
    quantile = getattr(res, "quantile", None)
    observed = getattr(res, "observed_statistic", None)
    if test == "N" and isinstance(quantile, (tuple, list)) and len(quantile) == 2:
        delta1, delta2 = float(quantile[0]), float(quantile[1])
        q = min(delta1, delta2)
        return ConsistencyResult(
            test="N",
            quantile=float(q),
            passed=bool(q >= alpha / 2.0),
            alpha=alpha,
            delta1=delta1,
            delta2=delta2,
            observed_statistic=_to_float(observed),
            pycsep_used=True,
            note="pyCSEP poisson number_test",
        )
    q = _to_float(quantile)
    passed = None if q is None else bool(min(q, 1.0 - q) >= alpha / 2.0)
    return ConsistencyResult(
        test=test,  # type: ignore[arg-type]
        quantile=q,
        passed=passed,
        alpha=alpha,
        observed_statistic=_to_float(observed),
        pycsep_used=True,
        note=f"pyCSEP poisson {test}-test",
    )


def _to_float(value) -> float | None:
    """Coerce a scalar (possibly a 1-element array / tuple) to ``float``; ``None`` stays ``None``."""
    if value is None:
        return None
    if isinstance(value, (tuple, list, np.ndarray)):
        arr = np.asarray(value, dtype=float).ravel()
        return float(arr[0]) if arr.size else None
    return float(value)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison tests — the only place skill is established
# ─────────────────────────────────────────────────────────────────────────────


def _paired_t_test_igpe(per_event_terms: np.ndarray, alpha: float) -> dict[str, Any]:
    r"""Paired T-test on the per-earthquake information-gain terms (Rhoades et al. 2011).

    Given the per-bin IGPE contributions :math:`X_i - Y_i` (already weighted by the observed count
    and net of the rate-normalisation term, as returned by
    :func:`information_gain_per_earthquake`), the statistic is

    .. math::
        T = \frac{I_N(A, B)}{s / \sqrt{N}} \sim t_{N-1}, \qquad
        s^2 = \frac{1}{N-1}\sum (X_i - Y_i)^2 - \frac{1}{N^2 - N}\Big[\sum (X_i - Y_i)\Big]^2 .

    The two-sided :math:`(1-\alpha)` CI on the mean IGPE is returned; skill requires that CI to
    exclude zero on the positive side.
    """
    x = np.asarray(per_event_terms, dtype=float)
    # Expand bin contributions to one entry per earthquake is unnecessary: the variance formula
    # operates on the per-bin differences scaled to per-event already. We treat each nonzero
    # contribution as the paired difference sample. N is the number of observed earthquakes.
    diffs = x[x != 0.0] if x.size else x
    n = int(round(float(np.sum(np.where(x != 0.0, 1.0, 0.0)))))  # count of contributing bins
    # Use the actual earthquake count where available via the caller; here approximate by samples.
    samples = diffs
    n_samples = samples.size
    if n_samples < 2:
        return {
            "t_statistic": None,
            "t_ci_low": None,
            "t_ci_high": None,
            "t_ci_excludes_zero": None,
            "note": "fewer than 2 contributing samples — T-test undefined",
        }
    mean_diff = float(samples.mean())
    # Rhoades variance (unbiased), guarded against tiny negative round-off.
    s2 = float(samples.var(ddof=1))
    s = np.sqrt(max(s2, 0.0))
    if s == 0.0:
        return {
            "t_statistic": None,
            "t_ci_low": mean_diff,
            "t_ci_high": mean_diff,
            "t_ci_excludes_zero": bool(mean_diff != 0.0),
            "note": "zero variance across samples",
        }
    se = s / np.sqrt(n_samples)
    t_stat = mean_diff / se
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n_samples - 1))
    ci_low = mean_diff - tcrit * se
    ci_high = mean_diff + tcrit * se
    return {
        "t_statistic": float(t_stat),
        "t_ci_low": float(ci_low),
        "t_ci_high": float(ci_high),
        "t_ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
        "note": "",
    }


def _w_test_igpe(
    log_rates_a: np.ndarray, log_rates_b: np.ndarray, observed_counts: np.ndarray
) -> dict[str, Any]:
    r"""Wilcoxon signed-rank (W-) test companion to the paired T-test (methodology.md §E.3).

    Operates on the per-earthquake log-rate differences :math:`\ln \lambda_A(k_i) - \ln \lambda_B(k_i)`
    (one sample per observed earthquake, expanding each bin by its observed count). A small p-value
    with a positive median corroborates a comparison-test win; the non-parametric W-test guards
    against the T-test's normality assumption when the per-event gains are skewed.
    """
    a = np.clip(np.asarray(log_rates_a, dtype=float), np.finfo(float).tiny, None)
    b = np.clip(np.asarray(log_rates_b, dtype=float), np.finfo(float).tiny, None)
    omega = np.asarray(observed_counts, dtype=float)
    diff_per_bin = np.log(a) - np.log(b)
    # Expand to one sample per observed earthquake (integer counts assumed for the target catalog).
    counts = np.rint(omega).astype(int)
    samples = np.repeat(diff_per_bin, np.clip(counts, 0, None))
    samples = samples[samples != 0.0]
    if samples.size < 1:
        return {"w_statistic": None, "w_pvalue": None}
    try:
        res = stats.wilcoxon(samples, zero_method="wilcox", alternative="two-sided")
        return {"w_statistic": float(res.statistic), "w_pvalue": float(res.pvalue)}
    except ValueError:
        return {"w_statistic": None, "w_pvalue": None}


def comparison_tests(
    rates_model: np.ndarray,
    rates_baseline: np.ndarray,
    observed_counts: np.ndarray,
    *,
    baseline_name: str,
    alpha: float = 0.05,
    forecast_model=None,
    forecast_baseline=None,
    observed_catalog=None,
) -> ComparisonResult:
    r"""Paired comparison test of the model vs ONE baseline — IGPE in nats + T-test + W-test.

    This is the **only** place forecasting skill is established (methodology.md §E.3). It computes
    information gain per earthquake :math:`I_N(\text{model}, \text{baseline})` in **nats**, the paired
    **T-test** statistic/CI (Rhoades et al. 2011), and the non-parametric **W-test**. Skill over the
    baseline is flagged (``skill_demonstrated``) only when ``igpe > 0`` and the T-test CI excludes
    zero (positive side); the W-test p-value corroborates.

    Call once per baseline. The mandatory baselines are a smoothed-seismicity/Poisson null **and**
    an ETAS model — the model must beat **both** to claim skill. Pass ``forecast_model`` /
    ``forecast_baseline`` / ``observed_catalog`` pyCSEP objects to route the T/W tests through
    pyCSEP's ``poisson_evaluations.paired_t_test`` / ``w_test`` when installed (``pycsep_used=True``);
    otherwise the numpy implementation here is used.
    """
    a = np.asarray(rates_model, dtype=float)
    b = np.asarray(rates_baseline, dtype=float)
    omega = np.asarray(observed_counts, dtype=float)
    n_eq = int(round(float(omega.sum())))

    use_pycsep = (
        forecast_model is not None
        and forecast_baseline is not None
        and observed_catalog is not None
        and pycsep_available()
    )
    if use_pycsep:
        return _comparison_tests_pycsep(
            forecast_model, forecast_baseline, observed_catalog, baseline_name, n_eq, alpha
        )

    if n_eq == 0:
        return ComparisonResult(
            baseline=baseline_name,
            igpe=0.0,
            n_earthquakes=0,
            skill_demonstrated=False,
            pycsep_used=False,
            note="no observed target earthquakes in window — IGPE undefined (quiet period)",
        )

    igpe, per_bin = information_gain_per_earthquake(a, b, omega)
    t = _paired_t_test_igpe(per_bin, alpha)
    w = _w_test_igpe(a, b, omega)
    skill = bool(igpe > 0.0 and t.get("t_ci_excludes_zero") and (t.get("t_ci_low") or 0.0) > 0.0)
    return ComparisonResult(
        baseline=baseline_name,
        igpe=float(igpe),
        n_earthquakes=n_eq,
        t_statistic=t["t_statistic"],
        t_ci_low=t["t_ci_low"],
        t_ci_high=t["t_ci_high"],
        t_ci_excludes_zero=t["t_ci_excludes_zero"],
        w_statistic=w["w_statistic"],
        w_pvalue=w["w_pvalue"],
        skill_demonstrated=skill,
        pycsep_used=False,
        note=t.get("note", "") or "numpy IGPE/T/W (pyCSEP is the authoritative path)",
    )


def _comparison_tests_pycsep(
    forecast_model,
    forecast_baseline,
    observed_catalog,
    baseline_name: str,
    n_eq: int,
    alpha: float,
) -> ComparisonResult:
    """Authoritative paired-T + W comparison through pyCSEP's ``poisson_evaluations``."""
    _require_pycsep()
    from csep.core import poisson_evaluations as pe  # type: ignore

    t_res = pe.paired_t_test(forecast_model, forecast_baseline, observed_catalog, alpha=alpha)
    igpe = _to_float(getattr(t_res, "observed_statistic", None))
    ci = getattr(t_res, "test_distribution", None)  # pyCSEP packs the CI here as [low, high]
    ci_low = ci_high = None
    if isinstance(ci, (list, tuple, np.ndarray)) and len(ci) >= 2:
        ci_low, ci_high = float(ci[0]), float(ci[1])
    excludes = None if ci_low is None else bool(ci_low > 0.0 or ci_high < 0.0)

    w_stat = w_p = None
    try:
        w_res = pe.w_test(forecast_model, forecast_baseline, observed_catalog)
        w_p = _to_float(getattr(w_res, "quantile", None))
        w_stat = _to_float(getattr(w_res, "observed_statistic", None))
    except Exception:  # pragma: no cover - W-test optional in some pyCSEP builds
        pass

    skill = bool((igpe or 0.0) > 0.0 and excludes and (ci_low or 0.0) > 0.0)
    return ComparisonResult(
        baseline=baseline_name,
        igpe=float(igpe) if igpe is not None else 0.0,
        n_earthquakes=n_eq,
        t_ci_low=ci_low,
        t_ci_high=ci_high,
        t_ci_excludes_zero=excludes,
        w_statistic=w_stat,
        w_pvalue=w_p,
        skill_demonstrated=skill,
        pycsep_used=True,
        note="pyCSEP paired_t_test + w_test",
    )

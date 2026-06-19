"""E13 — the negative-binomial (over-dispersion-honest) N-test.

The gridded-Poisson N-test assumes Var[N]=E[N] and so over-rejects clustered (ETAS branching) counts
during sequences. `n_test_negbinom` scores the total against a negative-binomial null with the same
over-dispersion the forecast product already uses. These tests pin the three properties that make it a
correct, honest evaluation fix (not a way to launder a real rate miscalibration):

1. r → ∞ recovers the Poisson test exactly.
2. For moderate over-dispersion, a vigorous-but-plausible sequence the Poisson test wrongly rejects is
   accepted.
3. An extreme rate UNDER-forecast (our measured benchmark: 64 forecast vs 248 observed) STILL fails — a
   dispersion fix must not whitewash a genuine rate bias.
"""

from __future__ import annotations

from caos_seismic.eval.csep import n_test_negbinom, n_test_poisson


def test_negbinom_recovers_poisson_in_large_r_limit():
    """As the dispersion r → ∞ the NB N-test quantile converges to the Poisson one."""
    pois = n_test_poisson(50.0, 60)
    nb = n_test_negbinom(50.0, 60, dispersion=1e6)
    assert abs((nb.quantile or 0) - (pois.quantile or 0)) < 1e-3


def test_negbinom_accepts_moderate_overdispersion_that_poisson_rejects():
    """A clustered count the Poisson test rejects is accepted under realistic over-dispersion."""
    # forecast 20, observed 38: ~4 sigma for Poisson (rejected), well within an over-dispersed NB.
    pois = n_test_poisson(20.0, 38)
    nb = n_test_negbinom(20.0, 38, dispersion=4.0)
    assert pois.passed is False
    assert nb.passed is True
    assert (nb.quantile or 0) > (pois.quantile or 0)


def test_negbinom_still_fails_extreme_rate_underforecast():
    """The measured benchmark gap (ETAS 64.2 vs 248 observed) is a RATE bias, not dispersion — it must
    still fail under the configured r=4 (a dispersion fix cannot whitewash a 3.9x rate miss)."""
    nb = n_test_negbinom(64.2, 248, dispersion=4.0)
    assert nb.passed is False
    # and the failure is the "too many observed" tail (forecast too low)
    assert (nb.delta1 or 1.0) < nb.alpha / 2.0


def test_negbinom_degenerate_dispersion_falls_back_to_poisson():
    """Non-positive / non-finite dispersion falls back to the Poisson test (defensive)."""
    pois = n_test_poisson(30.0, 30)
    nb = n_test_negbinom(30.0, 30, dispersion=0.0)
    assert nb.passed == pois.passed
    assert abs((nb.quantile or 0) - (pois.quantile or 0)) < 1e-9

"""Shared numerical primitives for the forecaster family (Gutenberg-Richter, geometry, Poisson).

These are the small, well-cited building blocks reused by the smoothed-seismicity null, ETAS, and
Reasenberg-Jones. Keeping them here avoids three subtly-different copies of the same equation and
makes the governing references unambiguous. Core deps only (numpy); no heavy imports.
"""

from __future__ import annotations

import numpy as np

#: Mean Earth radius (km), IUGG.
EARTH_RADIUS_KM = 6371.0088
#: Kilometres per degree of arc on a great circle (= R * pi / 180).
DEG2KM = EARTH_RADIUS_KM * np.pi / 180.0
#: ln(10), reused for beta = b * ln10 and 10^x = exp(x * ln10).
LN10 = float(np.log(10.0))


def haversine_km(
    lat0: float, lon0: float, lat: np.ndarray | float, lon: np.ndarray | float
) -> np.ndarray:
    """Great-circle distance(s) in km from one point ``(lat0, lon0)`` to one or many ``(lat, lon)``.

    Standard haversine formula; vectorized over the second argument. Used for the adaptive-kernel
    bandwidths and the ETAS spatial kernel (epicentral distance). Inputs in degrees.
    """
    lat0r, lon0r = np.radians(lat0), np.radians(lon0)
    latr, lonr = np.radians(lat), np.radians(lon)
    dlat = latr - lat0r
    dlon = lonr - lon0r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0r) * np.cos(latr) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bvalue_aki_utsu(
    mags: np.ndarray, mc: float, delta_m: float = 0.1
) -> tuple[float, float]:
    """Binning-corrected Aki-Utsu maximum-likelihood Gutenberg-Richter ``b`` and its 1σ error.

    .. math::
        \\hat b = \\frac{\\log_{10} e}{\\bar m - (M_c - \\Delta M / 2)}

    with the Tinti-Mulargia (1987) binning correction ``-ΔM/2`` on the completeness threshold. The
    standard error follows Shi & Bolt (1982).

    Parameters
    ----------
    mags:
        Magnitudes (Mw-equivalent) of events at or above ``mc``.
    mc:
        Magnitude of completeness.
    delta_m:
        Magnitude bin width ``ΔM`` (catalog rounding; 0.1 by CSEP convention).

    Returns
    -------
    (b, b_err):
        Estimate and its standard error. Never hard-coded to 1.0 — this is the estimator the
        completeness config mandates (``aki_utsu_mle``).

    References
    ----------
    Aki, K. (1965), Bull. Earthq. Res. Inst. 43, 237-239.
    Tinti, S. & Mulargia, F. (1987), BSSA 77(6), 2125-2134.
    Shi, Y. & Bolt, B. A. (1982), BSSA 72(5), 1677-1687 (standard error).
    """
    m = np.asarray(mags, dtype=float)
    m = m[m >= mc - 1e-9]
    n = m.size
    if n < 2:
        raise ValueError(f"need >= 2 events at/above Mc={mc} to estimate b (got {n})")
    mean_m = float(np.mean(m))
    denom = mean_m - (mc - delta_m / 2.0)
    if denom <= 0:
        raise ValueError(
            f"mean magnitude ({mean_m:.3f}) not above corrected Mc ({mc - delta_m / 2.0:.3f}); "
            "cannot estimate b — check Mc"
        )
    b = float(np.log10(np.e) / denom)
    var_m = float(np.sum((m - mean_m) ** 2) / (n * (n - 1)))
    b_err = float(2.30 * b**2 * np.sqrt(var_m))  # Shi & Bolt
    return b, b_err


def gr_exceedance_fraction(
    m_threshold: float, b: float, mc: float, m_max: float | None = None
) -> float:
    """Gutenberg-Richter tail fraction ``Φ(M*) = P(M >= M* | M >= Mc)`` for one event.

    The methodology's magnitude term is :math:`\\Phi(M^*) = 10^{-b (M^* - M_c)}`. When a finite
    ``m_max`` is given the distribution is the **bounded** (truncated) Gutenberg-Richter, so the
    tail above ``m_max`` is removed and the public probability cannot be inflated by an unbounded
    magnitude integral::

        Φ(M*) = (10^{-b(M*-Mc)} - 10^{-b(m_max-Mc)}) / (1 - 10^{-b(m_max-Mc)})

    For ``M* <= Mc`` the fraction is 1 (every complete event qualifies); for ``M* >= m_max`` it is 0.

    References
    ----------
    Methodology §1.1, §1.10 (bounded GR; ``m_max`` bounds the exceedance integral per region).
    """
    if m_threshold <= mc:
        return 1.0
    unbounded = 10.0 ** (-b * (m_threshold - mc))
    if m_max is None:
        return float(np.clip(unbounded, 0.0, 1.0))
    if m_threshold >= m_max:
        return 0.0
    tail_max = 10.0 ** (-b * (m_max - mc))
    frac = (unbounded - tail_max) / (1.0 - tail_max)
    return float(np.clip(frac, 0.0, 1.0))


def poisson_p_at_least_one(n_expected: float) -> float:
    """Public exceedance probability ``P(>=1 event) = 1 - e^{-N}`` from an expected count ``N``.

    This formula NEVER changes (methodology §1.10); only the quality of the rate feeding ``N``
    improves. ``N`` is clipped to be non-negative.
    """
    return float(1.0 - np.exp(-max(n_expected, 0.0)))


def sample_truncated_gr(
    rng: np.random.Generator, size: int, b: float, mc: float, m_max: float
) -> np.ndarray:
    """Draw ``size`` magnitudes from the **bounded** Gutenberg-Richter law on ``[Mc, m_max]``.

    Inverse-CDF sampling of the truncated exponential ``f(m) = β e^{-β(m-Mc)} / (1 - e^{-β(m_max-Mc)})``
    with ``β = b ln 10``. Used by the ETAS catalog simulator to mark each synthetic event with a
    magnitude consistent with the fitted ``b``. Deterministic given ``rng``.
    """
    beta = b * LN10
    u = rng.random(size)
    span = 1.0 - np.exp(-beta * (m_max - mc))
    return mc - np.log1p(-u * span) / beta

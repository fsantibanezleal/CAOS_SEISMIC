"""Magnitude of completeness ``Mc`` and the Gutenberg–Richter ``b``-value.

Implements §1.1 of the methodology synthesis (Gutenberg–Richter, Aki–Utsu MLE) and §3 of the
data-and-pipelines synthesis (the ordered ``Mc(x,y,t)`` step). Core deps only (numpy / pandas /
scipy) so this is importable on the ComCat spine without any heavy geophysics stack.

What this module computes
-------------------------
* **Frequency–magnitude distribution (FMD)** — the binned incremental and cumulative counts
  underlying every ``Mc`` and ``b`` estimate.
* **Maximum-curvature ``Mc`` (MAXC)** — the magnitude of the maximum of the *non-cumulative* FMD,
  plus a configurable correction (default ``+0.2``).
  .. warning::
     The ``+0.2`` MAXC correction was calibrated on **California** (Wiemer & Wyss 2000) and is
     **not established as universal**. Re-validate it per region (GFT / EMR cross-check + FMD
     inspection) and take the conservative value. The correction is a config knob
     (``configs/completeness.yaml: mc.maxc_correction``), never a literal constant in the science.
* **Goodness-of-fit ``Mc`` (GFT)** — Wiemer & Wyss (2000) goodness-of-fit cross-check: the lowest
  ``Mc`` whose modelled FMD explains ≥ ``target_R`` % of the observed event count (90 %/95 % levels).
* **Rolling space–time ``Mc``** — re-estimation on a moving time window (and optionally per spatial
  cell), because a single global ``Mc`` injects fake non-stationarity (synthesis §3, step 1).
* **Aki–Utsu binning-corrected ``b``-value MLE** with Shi & Bolt (1982) uncertainty.
  ``b`` is **always estimated, never hard-coded to 1.0** (methodology §1.1).

Governing equations
--------------------
Gutenberg–Richter (cumulative)::

    log10 N(>= M) = a - b M,            M >= Mc

Aki–Utsu maximum-likelihood ``b`` with the Utsu / Tinti–Mulargia binning correction::

    b_hat = log10(e) / ( mean_m - (Mc - dM/2) )

(``mean_m`` is the mean magnitude of events ``>= Mc``; ``dM`` is the magnitude bin width.) The
estimator is strongly biased if ``Mc`` is mis-estimated, so ``Mc`` and ``b`` are re-estimated on a
rolling window and their uncertainty propagated (Shi & Bolt 1982)::

    sigma_b = 2.30 * b_hat^2 * sqrt( sum_i (m_i - mean_m)^2 / (n (n - 1)) )

References
----------
Aki, K. (1965). *Bull. Earthq. Res. Inst.* 43, 237–239 (MLE).
Utsu, T. (1965). *Geophys. Bull. Hokkaido Univ.* 13, 99–103.
Tinti, S. & Mulargia, F. (1987). *BSSA* 77(6), 2125–2134 (binning correction).
Shi, Y. & Bolt, B. A. (1982). *BSSA* 72(5), 1677–1687 (b-value uncertainty).
Wiemer, S. & Wyss, M. (2000). *BSSA* 90(4), 859–869, doi:10.1785/0119990114 (MAXC, GFT).
Woessner, J. & Wiemer, S. (2005). *BSSA* 95(2), 684–698, doi:10.1785/0120040007 (EMR, uncertainty).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# log10(e) — the constant numerator of the Aki–Utsu estimator. Spelled out so the equation in the
# docstring is literally the code below; never a magic 0.4343.
_LOG10_E = math.log10(math.e)


# ─────────────────────────────────────────────────────────────────────────────
# Result value objects
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class McEstimate:
    """Magnitude of completeness estimate with its method and cross-checks.

    Attributes
    ----------
    mc:
        The adopted magnitude of completeness (already includes the MAXC correction when relevant).
    method:
        Primary estimator that produced :attr:`mc` (``"maxc"`` or ``"gft"``).
    maxc_raw:
        Raw maximum-curvature magnitude *before* the ``+correction`` (diagnostic).
    correction:
        The MAXC correction that was added (California-tuned default; re-validate per region).
    gft_mc, gft_level:
        Goodness-of-fit cross-check ``Mc`` and the confidence level (% of events explained) at which
        it was attained (``None`` if no level reached). A large ``|maxc - gft|`` is a red flag.
    n_above:
        Number of events at or above the adopted :attr:`mc`.
    """

    mc: float
    method: str
    maxc_raw: float
    correction: float
    gft_mc: float | None = None
    gft_level: float | None = None
    n_above: int = 0


@dataclass(frozen=True)
class BValueEstimate:
    """Aki–Utsu binning-corrected ``b``-value with Shi & Bolt (1982) uncertainty.

    Attributes
    ----------
    b:
        Maximum-likelihood Gutenberg–Richter slope. **Never hard-coded** — if you read ``1.0`` here
        it was *estimated* to be ~1, not assumed.
    b_uncertainty:
        Shi & Bolt (1982) 1-sigma standard error of ``b``.
    a:
        Gutenberg–Richter ``a`` (productivity), normalised to the cumulative form at ``Mc``.
    mc:
        Completeness used for the fit.
    mean_mag:
        Mean magnitude of events ``>= Mc`` (the sufficient statistic of the estimator).
    n:
        Number of events ``>= Mc`` used.
    dm:
        Magnitude bin width assumed for the binning correction.
    beta:
        ``beta = b * ln(10)`` — the exponential-tail rate used by ETAS stability gates (§1.3).
    """

    b: float
    b_uncertainty: float
    a: float
    mc: float
    mean_mag: float
    n: int
    dm: float

    @property
    def beta(self) -> float:
        """``beta = b ln 10`` — the magnitude-density rate; ETAS requires ``alpha < beta``."""
        return self.b * math.log(10.0)


# ─────────────────────────────────────────────────────────────────────────────
# Frequency–magnitude distribution
# ─────────────────────────────────────────────────────────────────────────────


def fmd(
    magnitudes: np.ndarray | pd.Series,
    dm: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binned frequency–magnitude distribution.

    Parameters
    ----------
    magnitudes:
        1-D array of magnitudes (already homogenized to Mw upstream; this function is agnostic).
    dm:
        Magnitude bin width (``configs/grid.yaml: fit.mag_bin``, typically 0.1).

    Returns
    -------
    centers, incremental, cumulative:
        ``centers`` are bin-centre magnitudes; ``incremental[k]`` is the count of events in bin ``k``
        (the *non-cumulative* FMD whose peak defines MAXC); ``cumulative[k] = N(>= centers[k])`` is
        the survival count used to fit Gutenberg–Richter. Empty input yields three empty arrays.
    """
    mags = np.asarray(magnitudes, dtype=float)
    mags = mags[np.isfinite(mags)]
    if mags.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty.copy(), empty.copy()

    # Bin to a regular grid snapped to multiples of dm so bin centres are stable across windows.
    m_min = math.floor(mags.min() / dm) * dm
    m_max = math.ceil(mags.max() / dm) * dm
    # Guard the degenerate single-bin case.
    n_bins = max(1, int(round((m_max - m_min) / dm)))
    edges = m_min + dm * np.arange(n_bins + 1)
    centers = edges[:-1] + dm / 2.0

    incremental, _ = np.histogram(mags, bins=edges)
    incremental = incremental.astype(float)
    # Cumulative survival count N(>= center): reverse-cumsum of the incremental histogram.
    cumulative = np.cumsum(incremental[::-1])[::-1].astype(float)
    return centers, incremental, cumulative


# ─────────────────────────────────────────────────────────────────────────────
# Maximum-curvature (MAXC) Mc
# ─────────────────────────────────────────────────────────────────────────────


def maxc_mc(
    magnitudes: np.ndarray | pd.Series,
    dm: float = 0.1,
    correction: float = 0.2,
) -> tuple[float, float]:
    """Maximum-curvature magnitude of completeness with the configurable correction.

    MAXC takes ``Mc`` as the magnitude of the maximum of the *non-cumulative* FMD — the most
    populated magnitude bin — which marks the roll-off below which detection becomes incomplete
    (Wiemer & Wyss 2000). A positive ``correction`` (default ``+0.2``) compensates for MAXC's known
    tendency to *under*-estimate ``Mc`` for curved/gradual FMDs.

    .. warning::
       The ``+0.2`` value is **California-tuned**; re-validate per region and take the conservative
       value. Pass the region's configured ``mc.maxc_correction`` here — do not assume universality.

    Returns
    -------
    mc, maxc_raw:
        Corrected ``Mc`` and the raw maximum-curvature magnitude before the correction.
        Returns ``(nan, nan)`` if there are no finite magnitudes.
    """
    centers, incremental, _ = fmd(magnitudes, dm=dm)
    if centers.size == 0 or incremental.sum() == 0:
        return float("nan"), float("nan")
    # argmax returns the first (lowest-magnitude) peak on ties — the conservative choice for Mc.
    maxc_raw = float(centers[int(np.argmax(incremental))])
    return maxc_raw + float(correction), maxc_raw


# ─────────────────────────────────────────────────────────────────────────────
# Goodness-of-fit (GFT) cross-check
# ─────────────────────────────────────────────────────────────────────────────


def gft_mc(
    magnitudes: np.ndarray | pd.Series,
    dm: float = 0.1,
    levels: tuple[float, ...] = (90.0, 95.0),
) -> tuple[float | None, float | None]:
    """Goodness-of-fit test (GFT) magnitude of completeness — the Wiemer & Wyss (2000) cross-check.

    For each candidate ``Mc`` (each FMD bin centre), the synthetic GR model is built from the
    Aki–Utsu ``b`` and the observed ``a`` *at that cutoff*, and the normalised absolute residual
    between observed and synthetic *cumulative* counts is measured::

        R(Mc) = 100 * (1 - sum_i |B_i - S_i| / sum_i B_i)

    where ``B_i`` are observed and ``S_i`` modelled cumulative counts for magnitudes ``>= Mc``.
    ``R`` is the percentage of the observed event count the GR model explains. The adopted GFT ``Mc``
    is the **lowest** cutoff reaching the highest available confidence level (90 % preferred, else
    95 %, per Wiemer & Wyss). Returns ``(None, None)`` when no level is reached (a flag that the FMD
    is ill-behaved and ``Mc`` should be set conservatively from MAXC).

    Returns
    -------
    mc, level:
        The GFT completeness and the confidence level (%) at which it was attained.
    """
    centers, _, cumulative = fmd(magnitudes, dm=dm)
    if centers.size == 0:
        return None, None

    levels_sorted = tuple(sorted(levels))  # try the most demanding (highest %) reachable level
    best_per_level: dict[float, float] = {}

    for k, mc_candidate in enumerate(centers):
        n_above = cumulative[k]
        if n_above < 2:  # need at least a couple of events to fit anything
            continue
        sub = centers[k:]
        obs_cum = cumulative[k:]
        # b at this cutoff via the binning-corrected Aki–Utsu estimator (mean over events >= Mc).
        # Reconstruct the mean magnitude from the incremental counts inside the survival window.
        inc = np.diff(np.concatenate([obs_cum, [0.0]]) * -1.0)  # incremental from cumulative tail
        inc = np.where(inc < 0, 0.0, inc)
        total = inc.sum()
        if total <= 0:
            continue
        mean_m = float(np.sum(sub * inc) / total)
        denom = mean_m - (float(mc_candidate) - dm / 2.0)
        if denom <= 0:
            continue
        b = _LOG10_E / denom
        # Synthetic cumulative GR counts anchored so S(Mc) == N(>= Mc): S_i = N0 * 10^{-b (M_i - Mc)}.
        synth_cum = n_above * np.power(10.0, -b * (sub - float(mc_candidate)))
        denom_b = float(obs_cum.sum())
        if denom_b <= 0:
            continue
        residual = float(np.sum(np.abs(obs_cum - synth_cum)) / denom_b)
        r = 100.0 * (1.0 - residual)
        for lvl in levels_sorted:
            if r >= lvl and lvl not in best_per_level:
                best_per_level[lvl] = float(mc_candidate)

    for lvl in levels_sorted:  # prefer the most demanding level that was actually reached
        if lvl in best_per_level:
            return best_per_level[lvl], lvl
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Combined Mc estimate (MAXC primary + GFT cross-check)
# ─────────────────────────────────────────────────────────────────────────────


def mc_estimate(
    magnitudes: np.ndarray | pd.Series,
    dm: float = 0.1,
    correction: float = 0.2,
    gft_levels: tuple[float, ...] = (90.0, 95.0),
    min_events: int = 50,
    regional_default: float | None = None,
) -> McEstimate:
    """Adopt ``Mc`` from MAXC (primary) with a GFT cross-check, falling back when data are thin.

    Mirrors ``configs/completeness.yaml``: ``primary: maxc``, ``maxc_correction``, ``cross_check``
    includes ``gft``, ``min_events``, and ``regional_default``. When fewer than ``min_events`` finite
    magnitudes are available the estimate falls back to ``regional_default`` (if provided) rather than
    trusting a MAXC peak from a handful of events.

    Parameters mirror :func:`maxc_mc` and :func:`gft_mc`; see ``configs/completeness.yaml`` for the
    region-tuned values. The returned :class:`McEstimate` carries both the MAXC and GFT results so the
    caller can flag disagreement (and the manifest can record it).
    """
    mags = np.asarray(magnitudes, dtype=float)
    mags = mags[np.isfinite(mags)]

    g_mc, g_level = gft_mc(mags, dm=dm, levels=gft_levels)

    if mags.size < min_events and regional_default is not None:
        # Too few events to trust MAXC: adopt the conservative regional floor, but still report the
        # raw MAXC peak (if any) for diagnostics.
        _, maxc_raw = maxc_mc(mags, dm=dm, correction=correction)
        n_above = int(np.count_nonzero(mags >= regional_default))
        return McEstimate(
            mc=float(regional_default),
            method="regional_default",
            maxc_raw=maxc_raw,
            correction=correction,
            gft_mc=g_mc,
            gft_level=g_level,
            n_above=n_above,
        )

    mc, maxc_raw = maxc_mc(mags, dm=dm, correction=correction)
    if not math.isfinite(mc):
        # No usable FMD: fall back to the regional default if available, else NaN.
        mc = float(regional_default) if regional_default is not None else float("nan")
        method = "regional_default" if regional_default is not None else "maxc"
    else:
        method = "maxc"
    n_above = int(np.count_nonzero(mags >= mc)) if math.isfinite(mc) else 0
    return McEstimate(
        mc=mc,
        method=method,
        maxc_raw=maxc_raw,
        correction=correction,
        gft_mc=g_mc,
        gft_level=g_level,
        n_above=n_above,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rolling space–time Mc
# ─────────────────────────────────────────────────────────────────────────────


def rolling_mc(
    catalog: pd.DataFrame,
    window_days: float = 365.0,
    step_days: float | None = None,
    dm: float = 0.1,
    correction: float = 0.2,
    min_events: int = 50,
    regional_default: float | None = None,
    mag_col: str = "mw",
    time_col: str = "time",
) -> pd.DataFrame:
    """Rolling-time ``Mc(t)`` over a moving window — exposes ``Mc`` non-stationarity (synthesis §3).

    A single global ``Mc`` injects fake non-stationarity into the GR tail and every downstream rate.
    This re-estimates ``Mc`` (and the raw MAXC peak + event count) on a window of ``window_days``,
    advancing by ``step_days`` (default = ``window_days`` for non-overlapping windows; pass a smaller
    step for a smooth rolling curve). The window is **right-labelled at its end time**, so each row's
    ``Mc`` uses only events ``<= window_end`` within the trailing window — leakage-safe for the
    forecast clock when consumed causally.

    Parameters
    ----------
    catalog:
        Event DataFrame with at least ``time`` (UTC datetime) and ``mw`` columns (the homogenized
        magnitude; configurable via ``mag_col``/``time_col``).
    window_days, step_days:
        Trailing window length and advance step, in days.
    dm, correction, min_events, regional_default:
        Passed to :func:`mc_estimate`.

    Returns
    -------
    DataFrame indexed 0..n-1 with columns ``window_start``, ``window_end``, ``mc``, ``maxc_raw``,
    ``n_events``, ``method``. Empty catalog → empty frame with those columns.
    """
    cols = ["window_start", "window_end", "mc", "maxc_raw", "n_events", "method"]
    if catalog.empty or mag_col not in catalog or time_col not in catalog:
        return pd.DataFrame(columns=cols)

    times = pd.to_datetime(catalog[time_col], utc=True)
    order = np.argsort(times.values)
    times = times.iloc[order].reset_index(drop=True)
    mags = pd.to_numeric(catalog[mag_col].iloc[order], errors="coerce").reset_index(drop=True)

    t0, t1 = times.iloc[0], times.iloc[-1]
    win = pd.Timedelta(days=window_days)
    step = pd.Timedelta(days=step_days if step_days is not None else window_days)

    rows: list[dict] = []
    window_end = t0 + win
    # Advance the right edge until it passes the last event (inclusive of a final partial window).
    while window_end <= t1 + step:
        window_start = window_end - win
        mask = (times > window_start) & (times <= window_end)
        sub = mags[mask.values].to_numpy()
        est = mc_estimate(
            sub,
            dm=dm,
            correction=correction,
            min_events=min_events,
            regional_default=regional_default,
        )
        rows.append(
            {
                "window_start": window_start,
                "window_end": window_end,
                "mc": est.mc,
                "maxc_raw": est.maxc_raw,
                "n_events": int(sub.size),
                "method": est.method,
            }
        )
        window_end = window_end + step

    return pd.DataFrame(rows, columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# Aki–Utsu binning-corrected b-value MLE
# ─────────────────────────────────────────────────────────────────────────────


def aki_utsu_b_value(
    magnitudes: np.ndarray | pd.Series,
    mc: float,
    dm: float = 0.1,
) -> BValueEstimate:
    """Aki–Utsu maximum-likelihood ``b``-value with the Tinti–Mulargia binning correction.

    Implements methodology §1.1::

        b_hat = log10(e) / ( mean_m - (Mc - dM/2) )

    where ``mean_m`` is the mean magnitude of events ``>= Mc`` and ``dM`` the bin width. The
    ``-dM/2`` term is the binning correction (Utsu 1965; Tinti & Mulargia 1987): with magnitudes
    rounded to bins of width ``dM`` the unbinned MLE ``b = log10(e) / (mean_m - Mc)`` is biased, and
    shifting the reference down by half a bin removes it. The Shi & Bolt (1982) standard error::

        sigma_b = 2.30 * b^2 * sqrt( sum_i (m_i - mean_m)^2 / (n (n - 1)) )

    propagates into the forecast (``b`` uncertainty is a real component of the published bounds, §E.7).

    ``b`` is **never hard-coded** — it is the MLE of the data above ``Mc``. A mis-estimated ``Mc``
    biases ``b`` strongly, which is why ``Mc`` is re-estimated on a rolling window (:func:`rolling_mc`)
    and its uncertainty carried alongside.

    Parameters
    ----------
    magnitudes:
        Magnitudes (homogenized to Mw upstream). Events below ``mc`` are dropped internally.
    mc:
        Magnitude of completeness (e.g. from :func:`mc_estimate`). Must be finite.
    dm:
        Magnitude bin width (``configs/grid.yaml: fit.mag_bin``).

    Returns
    -------
    BValueEstimate

    Raises
    ------
    ValueError
        If ``mc`` is not finite, or if fewer than 2 events lie ``>= mc`` (the MLE is undefined), or if
        the denominator ``mean_m - (mc - dm/2)`` is non-positive (degenerate FMD — typically ``mc``
        set too high).
    """
    if not math.isfinite(mc):
        raise ValueError("aki_utsu_b_value requires a finite Mc")
    mags = np.asarray(magnitudes, dtype=float)
    mags = mags[np.isfinite(mags)]
    # Tolerance of half a bin so events numerically at the cutoff are kept.
    above = mags[mags >= mc - dm / 2.0]
    n = int(above.size)
    if n < 2:
        raise ValueError(f"aki_utsu_b_value needs >= 2 events above Mc={mc}, got {n}")

    mean_m = float(above.mean())
    denom = mean_m - (float(mc) - dm / 2.0)
    if denom <= 0:
        raise ValueError(
            "degenerate FMD: mean magnitude not above (Mc - dM/2); Mc is likely set too high "
            f"(mean_m={mean_m:.3f}, Mc={mc}, dM={dm})"
        )

    b = _LOG10_E / denom
    # Shi & Bolt (1982) 1-sigma uncertainty on b.
    var_m = float(np.sum((above - mean_m) ** 2) / (n * (n - 1)))
    sigma_b = 2.30 * b * b * math.sqrt(var_m)
    # Gutenberg–Richter 'a' normalised to the cumulative form at Mc: log10 N(>= Mc) = a - b*Mc.
    a = math.log10(n) + b * float(mc)

    return BValueEstimate(
        b=b,
        b_uncertainty=sigma_b,
        a=a,
        mc=float(mc),
        mean_mag=mean_m,
        n=n,
        dm=dm,
    )

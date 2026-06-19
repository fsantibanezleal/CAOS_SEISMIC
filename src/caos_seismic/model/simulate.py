"""ETAS branching-process forward simulation — the over-dispersion-honest count distribution (E13).

The deterministic ETAS forecast integrates the conditional intensity with the parent set **frozen at the
issue time**: ``N_fore = ∫ μ + Σ_parents k(m_j)[G(age_j+H) − G(age_j)]``. That counts only the background
plus the FIRST generation of offspring from already-observed events. The *realised* count also includes
**within-window secondary triggering** — an offspring early in the window triggers its own offspring
before the window closes — and the realised total is over-dispersed (a branching process has
``Var[N] ≫ E[N]``). The Poisson N-test assumes ``Var[N] = E[N]`` and therefore over-rejects vigorous-but-
plausible sequences (Werner 2010; Kagan 2017; Savran 2020).

This module forward-simulates the full ETAS cascade to produce the honest distribution of the in-window
``M ≥ mc`` count. Locations are irrelevant to the *count*, so only the temporal/productivity kernels are
simulated (cheap). The mean of the simulated counts exceeds ``N_fore`` by exactly the secondary-cascade
contribution; the spread is the over-dispersion. Comparing the observed total to this distribution is the
catalog-based N-test (pyCSEP's number test on simulated catalogs) — it separates three causes of an
apparent under-forecast: (a) secondary cascade the frozen intensity omits, (b) over-dispersion, and
(c) genuine rate bias that even the heavy-tailed simulation cannot reach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .etas import omori_utsu_cumulative, utsu_productivity


def _omori_cumulative(tau: np.ndarray, c: float, p: float) -> np.ndarray:
    """G(τ) = 1 − (1 + τ/c)^{−(p−1)} — the fraction of Omori-Utsu offspring with elapsed time ≤ τ."""
    return omori_utsu_cumulative(tau, c, p)


def _sample_omori_elapsed(
    tau_lo: np.ndarray, tau_hi: np.ndarray, c: float, p: float, rng: np.random.Generator
) -> np.ndarray:
    """Inverse-CDF sample of the Omori-Utsu elapsed time τ on ``[tau_lo, tau_hi)`` (per-offspring).

    The cumulative is ``G(τ) = 1 − (1 + τ/c)^{−(p−1)}``; inverting ``u = G(τ)`` gives
    ``τ = c[(1 − u)^{−1/(p−1)} − 1]``. Sampling ``u`` uniformly in ``[G(tau_lo), G(tau_hi))`` draws from the
    correctly truncated Omori law. ``p = 1`` would be logarithmic; the fit bounds keep ``p > 1``.
    """
    g_lo = _omori_cumulative(tau_lo, c, p)
    g_hi = _omori_cumulative(tau_hi, c, p)
    u = g_lo + (g_hi - g_lo) * rng.random(tau_lo.shape)
    exponent = -1.0 / (p - 1.0)
    return c * (np.power(1.0 - u, exponent) - 1.0)


@dataclass
class SimulatedCounts:
    """Result of the branching simulation."""

    counts: np.ndarray  # (n_sims,) total in-window M>=mc events per simulation
    n_sims: int
    forecast_first_gen: float  # the deterministic frozen-intensity expectation (background + first gen)
    simulated_mean: float  # E[N] incl. secondary cascade
    simulated_var: float
    fano: float  # Var/mean — the over-dispersion factor (1 = Poisson)


def simulate_window_counts(
    *,
    K: float,
    alpha: float,
    c: float,
    p: float,
    m0: float,
    mc: float,
    beta: float,
    bg_expected: float,
    parent_ages: np.ndarray,
    parent_mags: np.ndarray,
    horizon_days: float,
    n_sims: int,
    rng: np.random.Generator,
    max_events_per_sim: int = 200_000,
) -> SimulatedCounts:
    """Forward-simulate the ETAS branching count over ``[0, H)`` (issue time = 0); return per-sim totals.

    Pooled across all simulations: events carry a simulation index so every generation is one vectorized
    Poisson + Omori draw, and the per-sim totals are a final ``bincount``. Subcritical fits (branching
    ratio < 1, enforced by the ETAS stability gate) terminate in a few generations; a hard event cap is a
    runaway backstop only.

    Parameters mirror the fitted ETAS: productivity ``k(m) = K e^{α(m−m0)}``, Omori-Utsu ``(c, p)``,
    Gutenberg-Richter ``β`` (so ``M − mc ~ Exp(β)``), and ``bg_expected`` background immigrants over the
    window (``Σ`` of the smoothed background's expected counts). ``parent_ages`` are the conditioning
    events' ages in days before issue (``≥ 0``); ``parent_mags`` their magnitudes.
    """
    H = float(horizon_days)
    sim_counts = np.zeros(int(n_sims), dtype=np.int64)

    def _draw_mags(n: int) -> np.ndarray:
        return mc + rng.exponential(1.0 / beta, size=n)

    # ── generation 0: background immigrants (uniform in the window) ─────────────────────────────────
    n_bg = rng.poisson(bg_expected, size=n_sims)
    bg_sim = np.repeat(np.arange(n_sims), n_bg)
    bg_birth = rng.random(bg_sim.size) * H

    # ── generation 0: first offspring of the REAL (pre-issue) parents ───────────────────────────────
    ages = np.asarray(parent_ages, dtype=float)
    pm = np.asarray(parent_mags, dtype=float)
    # per-parent expected first-gen offspring in the window = k(m)[G(age+H) − G(age)] — matches the
    # frozen-intensity forecast term exactly.
    kt = utsu_productivity(pm, K, alpha, m0) * (
        _omori_cumulative(ages + H, c, p) - _omori_cumulative(ages, c, p)
    )
    kt = np.clip(kt, 0.0, None)
    forecast_first_gen = float(bg_expected + kt.sum())

    # draw first-gen offspring counts for every (sim, parent), then expand to an event list
    par_draws = rng.poisson(np.tile(kt, (n_sims, 1)))  # (n_sims, n_parents)
    nz_sim, nz_par = np.nonzero(par_draws)
    reps = par_draws[nz_sim, nz_par]
    fg_sim = np.repeat(nz_sim, reps)
    fg_parent = np.repeat(nz_par, reps)
    # offspring elapsed time τ ∈ [age, age+H); birth in window = τ − age
    if fg_parent.size:
        a = ages[fg_parent]
        tau = _sample_omori_elapsed(a, a + H, c, p, rng)
        fg_birth = np.clip(tau - a, 0.0, H - 1e-9)
    else:
        fg_birth = np.empty(0, dtype=float)

    # current generation = background ∪ first-gen offspring
    cur_sim = np.concatenate([bg_sim, fg_sim])
    cur_birth = np.concatenate([bg_birth, fg_birth])
    cur_mag = _draw_mags(cur_sim.size)

    total_events = 0
    while cur_sim.size:
        np.add.at(sim_counts, cur_sim, 1)
        total_events += cur_sim.size
        if total_events > max_events_per_sim * n_sims:
            break  # runaway backstop (should never trip for a subcritical fit)
        # each current event (born at s, mag m) triggers offspring in [s, H): mean = k(m)·G(H − s)
        remaining = H - cur_birth
        mean_off = utsu_productivity(cur_mag, K, alpha, m0) * _omori_cumulative(remaining, c, p)
        mean_off = np.clip(mean_off, 0.0, None)
        n_off = rng.poisson(mean_off)
        if not n_off.any():
            break
        parent_of = np.repeat(np.arange(cur_sim.size), n_off)
        # offspring elapsed τ ∈ [0, remaining); birth = parent_birth + τ
        rem = remaining[parent_of]
        tau = _sample_omori_elapsed(np.zeros_like(rem), rem, c, p, rng)
        nxt_birth = np.clip(cur_birth[parent_of] + tau, 0.0, H - 1e-9)
        nxt_sim = cur_sim[parent_of]
        cur_sim, cur_birth = nxt_sim, nxt_birth
        cur_mag = _draw_mags(cur_sim.size)

    counts = sim_counts.astype(float)
    mean = float(counts.mean()) if counts.size else 0.0
    var = float(counts.var()) if counts.size else 0.0
    return SimulatedCounts(
        counts=counts,
        n_sims=int(n_sims),
        forecast_first_gen=forecast_first_gen,
        simulated_mean=mean,
        simulated_var=var,
        fano=(var / mean if mean > 0 else 1.0),
    )


def catalog_based_n_test(
    simulated_counts: np.ndarray, observed_total: int, alpha: float = 0.05
) -> dict:
    """Catalog-based N-test: where does the observed total fall in the simulated count distribution?

    Two-sided quantile (the pyCSEP number-test convention): ``δ1 = P(N_sim ≥ N_obs)`` (small ⇒ forecast
    too few), ``δ2 = P(N_sim ≤ N_obs)`` (small ⇒ too many); rejected when ``min(δ1, δ2) < α/2``. Unlike the
    Poisson/NB analytic tests, the null here is the **empirical** branching distribution (correct
    over-dispersion AND the secondary cascade), so it neither over-rejects vigorous sequences nor whitewashes
    a genuine rate miss.
    """
    s = np.asarray(simulated_counts, dtype=float)
    n = int(observed_total)
    if s.size == 0:
        return {"test": "N_catalog", "quantile": None, "passed": None}
    delta1 = float(np.mean(s >= n))
    delta2 = float(np.mean(s <= n))
    quantile = float(min(delta1, delta2))
    return {
        "test": "N_catalog",
        "quantile": quantile,
        "passed": bool(quantile >= alpha / 2.0),
        "alpha": alpha,
        "delta1": delta1,
        "delta2": delta2,
        "n_obs": n,
        "sim_mean": float(s.mean()),
        "sim_median": float(np.median(s)),
        "sim_p05": float(np.percentile(s, 5)),
        "sim_p95": float(np.percentile(s, 95)),
        "n_sims": int(s.size),
    }

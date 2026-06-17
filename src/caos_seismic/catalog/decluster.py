"""Stage C (declustering) — Gardner–Knopoff windows + Zaliapin–Ben-Zion nearest-neighbour proximity.

This module implements the **DUAL-CATALOG RULE** — the single most common pipeline mistake, made
explicit (``configs/declustering.yaml``; methodology §E.6; data-and-pipelines §3 step 3):

* the **declustered** catalog (independent mainshocks only) feeds **ONLY** the stationary
  Poisson / smoothed-seismicity background ``μ(x,y)`` and the Poisson-baseline calibration;
* the **FULL, un-declustered** catalog feeds the **conditional / ETAS** model, because
  aftershock/foreshock **triggering is the predictable signal** — declustering the conditional
  input throws the signal away;
* **scoring** is on the **non-declustered** catalog, because the product deliberately forecasts
  clustering.

Two complementary methods are provided, as the config specifies:

1. **Gardner–Knopoff (1974) windowing** — transparent space/time windows around each potential
   mainshock; events inside a larger event's window are flagged as dependent. Coefficients are the
   **OpenQuake hmtk** parameterization carried in ``configs/declustering.yaml``::

       L(M) = 10^(0.1238·M + 0.983) km                                 (space window)
       T(M) = 10^(0.032·M + 2.7389) d   for M ≥ 6.5,
              10^(0.5409·M − 0.547) d   otherwise                      (time window)

   This yields the declustered **background** catalog.

2. **Zaliapin–Ben-Zion nearest-neighbour proximity** (Baiesi–Paczuski metric) — for each event ``j``
   its strongest parent ``i`` minimizes the rescaled proximity::

       η_ij = t_ij · (r_ij)^{d_f} · 10^{−b·m_i}                        (t_ij in years, r_ij in km, d_f fractal dim)
       T_j  = t_ij · 10^{−q·b·m_i},   R_j = (r_ij)^{d_f} · 10^{−(1−q)·b·m_i},   η_j = T_j · R_j   (q ≈ 0.5)

   ``log10 η`` is **bimodal** — a clustered mode (small η, triggered) and a background mode (large η).
   We expose ``η/T/R`` (and ``log10`` of each) as **ML features** *and* derive **cluster labels** from
   a threshold on ``log10 η`` (the valley between the two modes), the principled labeler the synthesis
   asks for. This is a cross-check on Gardner–Knopoff and the feature source for the conditional model.

Core deps only (``numpy``/``pandas``/``scipy``) — declustering runs on the ComCat spine without any
heavy geophysics stack.

References
----------
* Gardner, J. K. & Knopoff, L. (1974). *BSSA* 64(5), 1363–1367 (windows).
* van Stiphout, T., Zhuang, J. & Marsan, D. (2012). CORSSA (GK coefficient tables; OpenQuake hmtk).
* Baiesi, M. & Paczuski, M. (2004). *Phys. Rev. E* 69, 066106 (nearest-neighbour proximity metric).
* Zaliapin, I. & Ben-Zion, Y. (2013/2020). *JGR Solid Earth*, doi:10.1029/2018JB017120
  (η/T/R decomposition, bimodal log η, cluster identification).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contracts import validate_catalog
from ..model._common import haversine_km

logger = logging.getLogger(__name__)

#: Seconds per (Julian) year — η times are expressed in years per Zaliapin–Ben-Zion convention.
_SECONDS_PER_YEAR = 365.25 * 86400.0
#: Days per year, for the Gardner–Knopoff time windows (expressed in days).
_DAYS_PER_YEAR = 365.25


# ─────────────────────────────────────────────────────────────────────────────
# The dual-catalog rule, documented loudly
# ─────────────────────────────────────────────────────────────────────────────

DUAL_CATALOG_DOC = """DUAL-CATALOG RULE (the most common pipeline mistake, made explicit):

  • background μ(x,y):  DECLUSTERED catalog (Gardner–Knopoff independent mainshocks only)
                        → smoothed-seismicity null + Poisson-baseline calibration ONLY.
  • conditional/ETAS:   FULL, UN-declustered catalog. Triggering IS the predictable signal;
                        declustering the conditional model's input destroys it.
  • scoring:            NON-declustered catalog (we deliberately forecast clustering).

Use `dual_catalog()` to get both views from one input and never cross the wires.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Gardner–Knopoff windows (OpenQuake hmtk coefficients)
# ─────────────────────────────────────────────────────────────────────────────


def gardner_knopoff_windows(
    magnitudes: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Gardner–Knopoff (1974) space/time windows from the OpenQuake hmtk coefficients.

    Vectorized over an array of magnitudes. Returns ``(L_km, T_days)``:

    * space window ``L(M) = 10^(0.1238·M + 0.983)`` km;
    * time window  ``T(M) = 10^(0.032·M + 2.7389)`` days for ``M ≥ 6.5``,
      else ``10^(0.5409·M − 0.547)`` days.

    These are exactly the coefficients recorded in ``configs/declustering.yaml`` (and used by the
    OpenQuake hazard modeller's toolkit). The piecewise time window is the GK feature that gives large
    mainshocks the much longer aftershock window their sequences need.
    """
    m = np.asarray(magnitudes, dtype=float)
    l_km = np.power(10.0, 0.1238 * m + 0.983)
    t_days = np.where(
        m >= 6.5,
        np.power(10.0, 0.032 * m + 2.7389),
        np.power(10.0, 0.5409 * m - 0.547),
    )
    return l_km, t_days


@dataclass
class DeclusterResult:
    """Output of a declustering pass.

    Attributes
    ----------
    labels:
        Per-event integer cluster id aligned to the **input row order**. ``-1`` marks an independent
        event (its own cluster of one / a mainshock that triggered nothing). Each cluster's mainshock
        keeps its own positive id; dependent events carry the id of the cluster they belong to.
    is_mainshock:
        Boolean per event (input order): ``True`` for the events that survive into the **declustered
        background** catalog (independent events + each cluster's largest event).
    method:
        ``"gardner_knopoff"`` or ``"zaliapin_ben_zion"``.
    stats:
        Summary counts for the ``mc_decluster`` manifest (n events, n clusters, n background, fraction).
    """

    labels: np.ndarray
    is_mainshock: np.ndarray
    method: str
    stats: dict = field(default_factory=dict)


def gardner_knopoff(
    catalog: pd.DataFrame,
    *,
    mag_col: str = "mw",
    time_col: str = "time",
) -> DeclusterResult:
    """Gardner–Knopoff windowing declustering → independent-mainshock (background) catalog.

    Algorithm (the classic GK sweep): process events from **largest to smallest** magnitude. For the
    current event, open its space–time window ``(L(M), T(M))``; every *not-yet-assigned* event that
    lies within ``L`` km **and** within ``T`` days **after** the mainshock (and, symmetrically, the
    short foreshock window before it) is flagged as dependent and assigned to this cluster. Events
    never captured by any larger event's window are independent.

    The largest event of each cluster (and every independent event) is a **mainshock** and survives
    into the declustered background; all flagged dependents are removed. Returns a
    :class:`DeclusterResult` whose ``labels``/``is_mainshock`` are aligned to the **input row order**.

    Notes
    -----
    * Magnitudes must be the homogenized ``mw`` (per the hygiene order: homogenize *before*
      decluster). NaN magnitudes are treated as independent (cannot anchor or be captured by a window).
    * The window is applied causally in time *magnitude*-first, which is the standard GK definition; it
      is deterministic and order-independent given the magnitude ranking.
    """
    validate_catalog(catalog)
    n = len(catalog)
    labels = np.full(n, -1, dtype=int)
    is_main = np.ones(n, dtype=bool)
    if n == 0:
        return DeclusterResult(labels, is_main, "gardner_knopoff", _gk_stats(labels, is_main))

    df = catalog.reset_index(drop=False).rename(columns={"index": "_orig"})
    df = df.copy()
    df["time"] = pd.to_datetime(df[time_col], utc=True)
    mag = pd.to_numeric(df[mag_col], errors="coerce").to_numpy()
    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)
    # Time in days as float since the catalog start (monotone; sign carries fore/aftershock).
    t0 = df["time"].min()
    t_days = (df["time"] - t0).dt.total_seconds().to_numpy() / 86400.0
    orig = df["_orig"].to_numpy()

    l_km, t_win = gardner_knopoff_windows(np.where(np.isfinite(mag), mag, -np.inf))

    assigned = np.zeros(n, dtype=bool)
    # Process from largest to smallest magnitude (NaN/-inf last; they can only be independent).
    order = np.argsort(-np.where(np.isfinite(mag), mag, -np.inf))
    next_cluster = 0
    for k in order:
        if assigned[k] or not np.isfinite(mag[k]):
            # NaN-magnitude rows: independent singletons (cannot open a window).
            if not assigned[k]:
                assigned[k] = True
            continue
        # Open this event's window and capture unassigned neighbours within space + time.
        dist = haversine_km(lat[k], lon[k], lat, lon)
        dt = np.abs(t_days - t_days[k])
        within = (dist <= l_km[k]) & (dt <= t_win[k]) & (~assigned)
        within[k] = True  # the mainshock belongs to its own cluster
        idx = np.where(within)[0]
        cluster_id = next_cluster
        next_cluster += 1
        for j in idx:
            assigned[j] = True
            labels[int(orig[j])] = cluster_id
        # Within this cluster the current event k is the largest (we sweep mag-descending), so all
        # other members are dependents and drop out of the background.
        for j in idx:
            is_main[int(orig[j])] = (j == k)
        # A singleton cluster is an independent event: keep it, and mark label -1 for clarity.
        if idx.size == 1:
            labels[int(orig[k])] = -1

    return DeclusterResult(labels, is_main, "gardner_knopoff", _gk_stats(labels, is_main))


def _gk_stats(labels: np.ndarray, is_main: np.ndarray) -> dict:
    n = int(labels.size)
    n_bg = int(is_main.sum())
    clustered = labels[labels >= 0]
    n_clusters = int(np.unique(clustered).size) if clustered.size else 0
    return {
        "n_events": n,
        "n_background": n_bg,
        "n_dependent": n - n_bg,
        "n_clusters": n_clusters,
        "background_fraction": (n_bg / n) if n else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Zaliapin–Ben-Zion nearest-neighbour proximity (η / T / R) + cluster labels
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NearestNeighborResult:
    """Zaliapin–Ben-Zion nearest-neighbour proximity features + derived cluster labels.

    All arrays are aligned to the **input row order**. The first event (no past parent) has NaN
    proximities and is treated as background.

    Attributes
    ----------
    parent:
        Index (input row order) of each event's nearest-neighbour **parent** (the past event ``i``
        minimizing ``η_ij``); ``-1`` for events with no eligible parent.
    eta, T, R:
        The proximity ``η_j`` and its time/space factors ``T_j`` / ``R_j`` (Zaliapin decomposition,
        ``q``). These are the **ML features** (also exposed as ``log10`` via :meth:`as_frame`).
    log_eta:
        ``log10 η_j`` — the bimodal quantity whose valley separates clustered from background.
    is_background:
        Boolean per event: ``True`` where ``log10 η`` is in the **background** mode (above the
        threshold), i.e. the declustered background; ``False`` for clustered (triggered) events.
    threshold:
        The ``log10 η`` separating threshold actually used (data-driven valley, or the supplied value).
    df_fractal, q, b:
        The parameters used (recorded in the manifest).
    """

    parent: np.ndarray
    eta: np.ndarray
    T: np.ndarray
    R: np.ndarray
    log_eta: np.ndarray
    is_background: np.ndarray
    threshold: float
    df_fractal: float
    q: float
    b: float

    def as_frame(self) -> pd.DataFrame:
        """Feature frame (input order) with ``eta/T/R``, their ``log10``, ``parent`` and label."""
        with np.errstate(divide="ignore", invalid="ignore"):
            log_T = np.log10(self.T)
            log_R = np.log10(self.R)
        return pd.DataFrame(
            {
                "zbz_parent": self.parent,
                "zbz_eta": self.eta,
                "zbz_T": self.T,
                "zbz_R": self.R,
                "zbz_log_eta": self.log_eta,
                "zbz_log_T": log_T,
                "zbz_log_R": log_R,
                "zbz_is_background": self.is_background,
            }
        )


def zaliapin_ben_zion(
    catalog: pd.DataFrame,
    *,
    b: float,
    df_fractal: float = 1.6,
    q: float = 0.5,
    threshold: float | None = None,
    mag_col: str = "mw",
    time_col: str = "time",
) -> NearestNeighborResult:
    """Zaliapin–Ben-Zion nearest-neighbour proximity (η/T/R) and cluster labels.

    For each event ``j`` we find the **past** event ``i`` (``t_i < t_j``) that minimizes the
    Baiesi–Paczuski rescaled proximity::

        η_ij = t_ij · (r_ij)^{d_f} · 10^{−b·m_i}

    with ``t_ij`` the inter-event time **in years**, ``r_ij`` the epicentral distance **in km**,
    ``d_f`` the (region-tunable) fractal dimension, ``m_i`` the *parent* magnitude, and ``b`` the
    Gutenberg–Richter slope (estimated upstream — never hard-coded). The minimizing ``i`` is the
    nearest-neighbour **parent**, and ``η_j = min_i η_ij`` decomposes into

        T_j = t_ij · 10^{−q·b·m_i},   R_j = (r_ij)^{d_f} · 10^{−(1−q)·b·m_i},   η_j = T_j · R_j

    (``q ≈ 0.5`` splits the magnitude weight between the time and space factors). ``log10 η`` is
    **bimodal**: a clustered mode (small η) and a background mode (large η). The separating
    ``threshold`` on ``log10 η`` is, by default, found data-driven as the **valley** (1-D minimum of a
    Gaussian-KDE of ``log10 η``) between the two modes; pass an explicit value to pin it. Events with
    ``log10 η`` **above** the threshold are labelled **background**; below it, **clustered**.

    Parameters
    ----------
    catalog:
        Event DataFrame (homogenized ``mw``, UTC ``time``, lat/lon). Must be the **full** catalog.
    b:
        Gutenberg–Richter ``b`` (e.g. from :func:`caos_seismic.catalog.aki_utsu_b_value`).
    df_fractal, q:
        Fractal dimension ``d_f`` and the time/space split ``q`` (``configs/declustering.yaml``).
    threshold:
        Optional fixed ``log10 η`` separation; if ``None`` the KDE valley is used (falling back to the
        midpoint between the two largest KDE peaks, or the median, when a clean valley is not found).

    Returns
    -------
    NearestNeighborResult
        With ``η/T/R``, ``log10 η``, the nearest-neighbour ``parent`` index, the ``is_background``
        labels, and the threshold actually used — all aligned to the input row order.

    Notes
    -----
    ``O(N²)`` in the catalog size (each event scans its predecessors), which is fine for regional daily
    catalogs (10³–10⁵). The features feed the conditional model; the labels are a principled
    cross-check on Gardner–Knopoff. Per the dual-catalog rule, the **labels** never decluster the
    conditional model's input — they are features/diagnostics.
    """
    validate_catalog(catalog)
    n = len(catalog)
    parent = np.full(n, -1, dtype=int)
    eta = np.full(n, np.nan, dtype=float)
    T = np.full(n, np.nan, dtype=float)
    R = np.full(n, np.nan, dtype=float)
    if n == 0:
        return NearestNeighborResult(
            parent, eta, T, R, np.full(0, np.nan), np.ones(0, dtype=bool),
            float("nan"), df_fractal, q, b,
        )

    df = catalog.copy()
    df["time"] = pd.to_datetime(df[time_col], utc=True)
    # Elapsed seconds since the first event as a plain float (tz-aware columns are object-dtype under
    # ``to_numpy()``, so subtract via the pandas Series and convert the timedelta to seconds).
    elapsed_s = (df["time"] - df["time"].min()).dt.total_seconds().to_numpy()
    # Sort by time so each event's predecessors are a contiguous prefix; remember original order.
    order = np.argsort(elapsed_s, kind="mergesort")

    t_year = elapsed_s[order] / _SECONDS_PER_YEAR
    lat = df["latitude"].to_numpy(dtype=float)[order]
    lon = df["longitude"].to_numpy(dtype=float)[order]
    mag = pd.to_numeric(df[mag_col], errors="coerce").to_numpy()[order]

    eta_s = np.full(n, np.nan, dtype=float)
    T_s = np.full(n, np.nan, dtype=float)
    R_s = np.full(n, np.nan, dtype=float)
    parent_s = np.full(n, -1, dtype=int)

    for j in range(1, n):
        # Candidate parents: all earlier events with a finite magnitude.
        mi = mag[:j]
        valid = np.isfinite(mi)
        if not valid.any():
            continue
        t_ij = t_year[j] - t_year[:j]  # years, > 0 for past events
        t_ij = np.where(t_ij <= 0, np.nan, t_ij)
        r_ij = haversine_km(lat[j], lon[j], lat[:j], lon[:j])  # km
        # Distance can be exactly 0 for co-located events → r^d_f = 0 → η = 0. Floor r to a tiny
        # positive value so a true zero-distance pair does not collapse η to a degenerate 0.
        r_ij = np.where(r_ij <= 0, 1e-3, r_ij)

        with np.errstate(invalid="ignore"):
            eta_ij = t_ij * np.power(r_ij, df_fractal) * np.power(10.0, -b * mi)
        eta_ij = np.where(valid, eta_ij, np.nan)
        if not np.isfinite(eta_ij).any():
            continue
        i_star = int(np.nanargmin(eta_ij))
        parent_s[j] = i_star
        eta_s[j] = float(eta_ij[i_star])
        # Decompose η at the chosen parent.
        T_s[j] = float(t_ij[i_star] * np.power(10.0, -q * b * mi[i_star]))
        R_s[j] = float(np.power(r_ij[i_star], df_fractal) * np.power(10.0, -(1.0 - q) * b * mi[i_star]))

    # Map back to input order; translate sorted parent indices to original indices.
    for j_sorted in range(n):
        j_orig = order[j_sorted]
        p_sorted = parent_s[j_sorted]
        parent[j_orig] = order[p_sorted] if p_sorted >= 0 else -1
        eta[j_orig] = eta_s[j_sorted]
        T[j_orig] = T_s[j_sorted]
        R[j_orig] = R_s[j_sorted]

    with np.errstate(divide="ignore", invalid="ignore"):
        log_eta = np.log10(eta)

    thr = threshold if threshold is not None else _bimodal_threshold(log_eta)
    # Background = large η (above threshold) OR no parent (the first event / isolated events).
    is_background = (log_eta >= thr) | ~np.isfinite(log_eta)

    return NearestNeighborResult(
        parent=parent,
        eta=eta,
        T=T,
        R=R,
        log_eta=log_eta,
        is_background=is_background,
        threshold=float(thr),
        df_fractal=float(df_fractal),
        q=float(q),
        b=float(b),
    )


def _bimodal_threshold(log_eta: np.ndarray) -> float:
    """Find the ``log10 η`` valley separating the clustered and background modes.

    Builds a Gaussian-KDE of the finite ``log10 η`` values, samples it on a dense grid, and returns the
    location of the deepest **interior local minimum** between the two highest peaks (the classic
    Zaliapin–Ben-Zion valley). Falls back to the midpoint of the two largest peaks, and finally to the
    median, when the distribution is not cleanly bimodal (so a threshold is always returned).
    """
    x = log_eta[np.isfinite(log_eta)]
    if x.size < 10:
        return float(np.median(x)) if x.size else 0.0
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(x)
        grid = np.linspace(float(x.min()), float(x.max()), 512)
        dens = kde(grid)
    except Exception:  # pragma: no cover - scipy edge cases
        return float(np.median(x))

    # Local maxima (peaks) and minima (valleys) on the interior of the grid.
    is_peak = (dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:])
    is_valley = (dens[1:-1] < dens[:-2]) & (dens[1:-1] < dens[2:])
    peak_idx = np.where(is_peak)[0] + 1
    valley_idx = np.where(is_valley)[0] + 1

    if peak_idx.size >= 2:
        # The two strongest peaks define the modes; the valley between them is the threshold.
        top2 = peak_idx[np.argsort(dens[peak_idx])[-2:]]
        lo, hi = int(min(top2)), int(max(top2))
        between = valley_idx[(valley_idx > lo) & (valley_idx < hi)]
        if between.size:
            deepest = between[np.argmin(dens[between])]
            return float(grid[deepest])
        return float(0.5 * (grid[lo] + grid[hi]))
    return float(np.median(x))


# ─────────────────────────────────────────────────────────────────────────────
# The dual-catalog assembler — the rule, enforced in one call
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DualCatalog:
    """The two catalog views the rule mandates, plus the ZBZ feature/label table.

    Attributes
    ----------
    full:
        The **un-declustered** catalog (the input, validated) — feeds the conditional/ETAS model and
        is the catalog the forecast is **scored** against.
    background:
        The **declustered** catalog (Gardner–Knopoff independent mainshocks) — feeds the
        smoothed-seismicity null and the Poisson baseline **only**.
    gk:
        The :class:`DeclusterResult` from Gardner–Knopoff (labels + mainshock mask + stats).
    nnd:
        The :class:`NearestNeighborResult` from Zaliapin–Ben-Zion (η/T/R features + cluster labels).
    """

    full: pd.DataFrame
    background: pd.DataFrame
    gk: DeclusterResult
    nnd: NearestNeighborResult | None = None

    def features(self) -> pd.DataFrame:
        """The full catalog with the ZBZ η/T/R feature columns joined (input order preserved)."""
        if self.nnd is None:
            return self.full.copy()
        feat = self.nnd.as_frame()
        feat.index = self.full.index
        return pd.concat([self.full.copy(), feat], axis=1)

    def manifest_stats(self) -> dict:
        """Decluster provenance for the ``mc_decluster`` manifest (both methods' summaries)."""
        out = {"gardner_knopoff": self.gk.stats}
        if self.nnd is not None:
            n = int(self.nnd.is_background.size)
            n_bg = int(self.nnd.is_background.sum())
            out["zaliapin_ben_zion"] = {
                "n_events": n,
                "n_background": n_bg,
                "n_clustered": n - n_bg,
                "log_eta_threshold": self.nnd.threshold,
                "df_fractal": self.nnd.df_fractal,
                "q": self.nnd.q,
                "b": self.nnd.b,
            }
        return out


def dual_catalog(
    catalog: pd.DataFrame,
    *,
    b: float | None = None,
    df_fractal: float = 1.6,
    q: float = 0.5,
    zbz_threshold: float | None = None,
    compute_nnd: bool = True,
    mag_col: str = "mw",
    time_col: str = "time",
) -> DualCatalog:
    """Build BOTH catalog views from one input — the dual-catalog rule, enforced.

    Returns a :class:`DualCatalog` with:

    * ``full`` — the un-declustered catalog (validated copy) for the **conditional/ETAS** model and
      for **scoring**;
    * ``background`` — the **Gardner–Knopoff** declustered (independent-mainshock) catalog for the
      **smoothed-seismicity background and Poisson baseline only**;
    * ``gk`` / ``nnd`` — the Gardner–Knopoff result and (optionally) the Zaliapin–Ben-Zion η/T/R
      features + cluster labels.

    The function never feeds the declustered catalog to a conditional model — that is the caller's
    contract, made hard to get wrong by handing back both views clearly labelled (see
    :data:`DUAL_CATALOG_DOC`).

    Parameters
    ----------
    catalog:
        The full, cleaned, Mw-homogenized, below-Mc-cut catalog.
    b:
        Gutenberg–Richter ``b`` for the ZBZ proximity weighting. If ``None`` and the ZBZ pass is
        requested, ``b`` is estimated from the catalog with the binning-corrected Aki–Utsu MLE
        (never hard-coded) using the catalog's minimum ``mw`` as a conservative completeness proxy.
    df_fractal, q, zbz_threshold:
        Zaliapin–Ben-Zion parameters (see ``configs/declustering.yaml``).
    compute_nnd:
        Skip the (``O(N²)``) ZBZ pass when only the GK background is needed.
    """
    validate_catalog(catalog)
    full = catalog.reset_index(drop=True).copy()

    gk = gardner_knopoff(full, mag_col=mag_col, time_col=time_col)
    background = full.loc[gk.is_mainshock].reset_index(drop=True).copy()
    validate_catalog(background)

    nnd: NearestNeighborResult | None = None
    if compute_nnd and len(full) > 1:
        b_used = b
        if b_used is None:
            b_used = _estimate_b(full, mag_col=mag_col)
        nnd = zaliapin_ben_zion(
            full,
            b=b_used,
            df_fractal=df_fractal,
            q=q,
            threshold=zbz_threshold,
            mag_col=mag_col,
            time_col=time_col,
        )

    return DualCatalog(full=full, background=background, gk=gk, nnd=nnd)


def _estimate_b(catalog: pd.DataFrame, *, mag_col: str = "mw") -> float:
    """Estimate ``b`` (binning-corrected Aki–Utsu MLE) for the ZBZ weighting; default 1.0 on failure.

    Uses the catalog's minimum finite ``mw`` as a conservative completeness proxy (the production
    pipeline passes the per-region rolling ``Mc`` explicitly). ``b`` is **estimated**, never assumed —
    the 1.0 fallback is only for a degenerate catalog too small to fit, and is logged.
    """
    from .completeness import aki_utsu_b_value

    mags = pd.to_numeric(catalog[mag_col], errors="coerce").to_numpy()
    mags = mags[np.isfinite(mags)]
    if mags.size < 2:
        logger.warning("ZBZ: catalog too small to estimate b; falling back to b=1.0")
        return 1.0
    mc = float(np.min(mags))
    try:
        return aki_utsu_b_value(mags, mc, dm=0.1).b
    except ValueError as exc:
        logger.warning("ZBZ: b estimation failed (%s); falling back to b=1.0", exc)
        return 1.0

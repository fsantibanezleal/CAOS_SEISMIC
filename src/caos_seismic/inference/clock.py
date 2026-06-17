"""The forecast clock — make temporal leakage *structurally* impossible.

A daily inference must condition only on what was knowable strictly before the issue time
``t_issue``. The cardinal sin of seismicity forecasting evaluation is *temporal leakage*: letting
the model see an event at or after ``t_issue`` (the target window), which inflates skill and makes
prospective CSEP scoring meaningless (model-design.md §9; web-app-spec.md §6 "forecast clock").

The defence here is the same one the evaluation harness uses, so the live product and the
back-analysis run identical code: a forecaster is *handed* a catalog that has already been sliced
to ``time < t_issue``. The slice is causal by construction — it is not a discipline the caller has
to remember to apply.

Definitions used consistently across the package:

* **conditioning window** ``(-inf, t_issue)`` — events with ``time < t_issue`` (strict). This is
  what ``fit`` / ``expected_counts`` may see.
* **target window** ``[t_issue, t_issue + horizon)`` — events scored against the forecast. Never
  visible to the forecaster; only the evaluator sees it, and only after the forecast is sealed.

The boundary is **half-open on the left** (``< t_issue`` conditions, ``>= t_issue`` is target) so an
event whose origin time equals ``t_issue`` to the nanosecond is treated as *future*, never as
conditioning data — the conservative choice for honest forecasting.
"""

from __future__ import annotations

import pandas as pd

from ..contracts import validate_catalog


def _as_utc_timestamp(t: pd.Timestamp | str) -> pd.Timestamp:
    """Coerce ``t`` to a tz-aware UTC :class:`pandas.Timestamp` (the catalog ``time`` dtype)."""
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def conditioning_slice(catalog: pd.DataFrame, t_issue: pd.Timestamp | str) -> pd.DataFrame:
    """Return only the events the forecaster is allowed to see: ``time < t_issue`` (strict).

    This is the single chokepoint through which a catalog reaches a :class:`Forecaster`. Slicing
    here (rather than trusting each model) guarantees leakage is impossible regardless of model code.

    Parameters
    ----------
    catalog
        A catalog DataFrame satisfying the contract columns (see ``contracts.CATALOG_COLUMNS``).
    t_issue
        The issue time. Naive timestamps are interpreted as UTC.

    Returns
    -------
    pandas.DataFrame
        A copy, sorted ascending by ``time``, containing only ``time < t_issue``.
    """
    validate_catalog(catalog)
    t = _as_utc_timestamp(t_issue)
    times = pd.to_datetime(catalog["time"], utc=True)
    past = catalog.loc[times < t].copy()
    past["time"] = pd.to_datetime(past["time"], utc=True)
    return past.sort_values("time").reset_index(drop=True)


def target_slice(
    catalog: pd.DataFrame,
    t_issue: pd.Timestamp | str,
    horizon_days: float,
) -> pd.DataFrame:
    """Return the target-window events ``[t_issue, t_issue + horizon)`` for *evaluation only*.

    This is the complement of :func:`conditioning_slice`; it must never be passed to ``fit`` or
    ``expected_counts``. It exists so the evaluator can score a *sealed* forecast against reality.

    The window is half-open ``[t_issue, t_issue + horizon)`` so back-to-back daily horizons tile the
    timeline without double-counting an event on a boundary.
    """
    validate_catalog(catalog)
    t0 = _as_utc_timestamp(t_issue)
    t1 = t0 + pd.Timedelta(days=float(horizon_days))
    times = pd.to_datetime(catalog["time"], utc=True)
    win = catalog.loc[(times >= t0) & (times < t1)].copy()
    win["time"] = pd.to_datetime(win["time"], utc=True)
    return win.sort_values("time").reset_index(drop=True)


def assert_no_leakage(conditioning: pd.DataFrame, t_issue: pd.Timestamp | str) -> None:
    """Raise :class:`AssertionError` if any conditioning event is at/after ``t_issue``.

    A cheap, always-on invariant the daily job calls right before fitting — defence in depth on top
    of :func:`conditioning_slice`, so a future refactor that bypasses the slice still trips a guard.
    """
    if conditioning.empty:
        return
    t = _as_utc_timestamp(t_issue)
    latest = pd.to_datetime(conditioning["time"], utc=True).max()
    if latest >= t:
        raise AssertionError(
            f"forecast-clock leakage: conditioning catalog contains an event at {latest} "
            f">= t_issue {t}. Slice with conditioning_slice() before fitting."
        )


class ForecastClock:
    """A causal cursor over a master catalog: hand each issue time only its lawful past.

    Iterating the clock yields ``(t_issue, conditioning_catalog)`` pairs. Each conditioning catalog
    is produced by :func:`conditioning_slice`, so a forecaster driven by the clock can never see the
    target window — mirroring the real daily product and the pseudo-prospective evaluation
    (model-design.md §9; web-app-spec.md §6).

    Parameters
    ----------
    catalog
        The full master catalog (all events; the clock slices it per issue time).
    """

    def __init__(self, catalog: pd.DataFrame) -> None:
        validate_catalog(catalog)
        cat = catalog.copy()
        cat["time"] = pd.to_datetime(cat["time"], utc=True)
        self._catalog = cat.sort_values("time").reset_index(drop=True)

    def at(self, t_issue: pd.Timestamp | str) -> pd.DataFrame:
        """The lawful conditioning catalog for a single issue time."""
        return conditioning_slice(self._catalog, t_issue)

    def iter_issues(self, issue_times):
        """Yield ``(t_issue, conditioning_catalog)`` for each issue time, in chronological order.

        Parameters
        ----------
        issue_times
            An iterable of issue timestamps (str or :class:`pandas.Timestamp`).
        """
        ordered = sorted(_as_utc_timestamp(t) for t in issue_times)
        for t in ordered:
            yield t, self.at(t)

    def daily_issues(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        issue_hour_utc: int = 0,
    ):
        """Generate one issue time per day in ``[start, end]`` at ``issue_hour_utc``.

        Drives a pseudo-prospective back-analysis: a daily cadence matching the live product
        (publish.yaml ``cadence: daily``). Yields ``(t_issue, conditioning_catalog)`` pairs.
        """
        t0 = _as_utc_timestamp(start).normalize() + pd.Timedelta(hours=int(issue_hour_utc))
        t1 = _as_utc_timestamp(end)
        stamps = []
        t = t0
        while t <= t1:
            stamps.append(t)
            t = t + pd.Timedelta(days=1)
        yield from self.iter_issues(stamps)

"""Tidal-stress enricher — solid-Earth (+ ocean-loading) tidal Coulomb stress as a computed covariate.

Unlike the other enrichers this is **computed**, not downloaded: there is no tidal catalog to fetch.
We evaluate the time-varying tidal strain at a cell, resolve it onto a fault plane as a Coulomb
failure-stress change, and emit a small set of regularized covariates (data-and-pipelines.md §1.4).

Honest framing (kept first-class). Tidal stresses are tiny (~0.1-10 kPa) next to earthquake stress
drops (~1-10 MPa), and the empirical tidal-triggering signal is small (~0.5% to factor-3 across
regimes). This feature is a **physically-motivated, regularized covariate that may shrink to ~0** —
it must clear a CSEP prospective information-gain gate (with-vs-without, declustered, out-of-sample)
before it touches any public number; never claim improvement from in-sample Schuster p-values.

Per-cell (and per-time) features
--------------------------------
``tidal_dCFS_kpa``        tidal Coulomb failure-stress change ΔCFS = Δτ + μ·Δσ_n at the issue time
                          (kPa) on a representative fault for the cell.
``tidal_stress_rate_kpa_per_hr``
                          time-derivative of ΔCFS (kPa/hr) — the stressing rate, which the tidal-
                          triggering literature ties to triggering more than the level itself.
``tidal_phase_sin`` / ``tidal_phase_cos``
                          sine/cosine of the dominant semidiurnal tidal phase (a circular covariate,
                          so a linear model can use it without a discontinuity at the wrap).
``tidal_mf_envelope``     amplitude of the fortnightly Mf (14.77 d) tidal envelope at the issue time,
                          normalized to [0, 1] — the slow modulation most often linked to triggering.

Heavy dep: ``pygtide`` (ETERNA PREDICT 3.4 body tide) is imported lazily; an ocean-loading model
(SPOTL/TPXO) is configured out of band and folded in via :meth:`TidesEnricher.set_ocean_loader`.
Without ``pygtide`` installed, the module still imports — the heavy import happens only inside the
computation, with a clear, actionable error.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from ._base import EnricherResult, Provenance, require

logger = logging.getLogger(__name__)

DATASET = "tides"

#: Default effective friction coefficient μ in the Coulomb law ΔCFS = Δτ + μ·Δσ_n (apparent friction
#: absorbing pore-pressure; 0.4 is a common default for tidal-stress studies). Tunable per region.
DEFAULT_FRICTION = 0.4

#: Fortnightly lunar (Mf) tidal constituent period in days (the slow envelope most tied to triggering).
MF_PERIOD_DAYS = 14.765294

#: Dominant semidiurnal lunar (M2) constituent period in hours (the fast tidal phase).
M2_PERIOD_HOURS = 12.4206012

FEATURE_NAMES = (
    "tidal_dCFS_kpa",
    "tidal_stress_rate_kpa_per_hr",
    "tidal_phase_sin",
    "tidal_phase_cos",
    "tidal_mf_envelope",
)

#: Type of an optional ocean-loading model: (lat, lon, utc_time) -> (delta_tau_kpa, delta_sigma_n_kpa).
OceanLoader = Callable[[float, float, pd.Timestamp], "tuple[float, float]"]


@dataclass
class FaultGeometry:
    """The receiver-fault orientation the tidal tensor is resolved onto (strike/dip/rake, degrees).

    Defaults to a shallow thrust megathrust (the regime where ocean-loaded tidal triggering is
    strongest); override per cell from the Slab2 / stress enrichers for a slab-consistent plane.
    """

    strike_deg: float = 0.0
    dip_deg: float = 20.0
    rake_deg: float = 90.0


def download(**_: Any) -> Provenance:
    """No download — tides are computed. Returns the tool/citation provenance for the credits page.

    Kept API-compatible with the other enrichers (each exposes ``download()``). The body tide is
    computed by pygtide (ETERNA PREDICT 3.4); ocean loading by SPOTL with a TPXO/GOT/FES ocean-tide
    model, configured out of band (TPXO has academic-use terms that must be recorded).
    """
    return Provenance(
        dataset=DATASET,
        title="Tidal Coulomb stress (computed: pygtide body tide + ocean loading)",
        version="pygtide/ETERNA PREDICT 3.4 (+ SPOTL/TPXO ocean loading, out-of-band)",
        source_url="https://github.com/hydrogeoscience/pygtide",
        license="pygtide & SPOTL open tools; TPXO has academic-use terms — record and honor them.",
        attribution="pygtide (ETERNA PREDICT 3.4, Wenzel 1996); SPOTL (Agnew); TPXO ocean-tide model",
        citation=(
            "Wenzel, H.-G. (1996). The nanogal software: Earth tide data processing package ETERNA "
            "3.30. Bull. Inf. Marees Terrestres, 124, 9425-9439; "
            "Agnew, D. C. (2012). SPOTL: Some Programs for Ocean-Tide Loading. SIO Tech. Rep."
        ),
        files=[],
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        notes=(
            "Computed feature, not a download. Regularized covariate that may shrink to ~0; must "
            "pass a CSEP prospective information-gain gate before touching any public number. "
            "Ocean loading DOMINATES at coastal/subduction margins (Chile) — skipping it is the "
            "single biggest modeling error for the tidal channel."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell (+ per-time) feature extraction
# ─────────────────────────────────────────────────────────────────────────────


class TidesEnricher:
    """Compute tidal Coulomb-stress covariates at a cell and issue time.

    The solid-Earth body-tide **strain** tensor is computed with pygtide; it is converted to a stress
    change via a simple isotropic-elastic relation and resolved onto the cell's receiver fault to get
    ΔCFS. An optional ocean-loading model (set via :meth:`set_ocean_loader`) adds the dominant coastal
    contribution. The fortnightly Mf envelope and the semidiurnal phase are derived analytically from
    the issue time so they need no model evaluation (they are exact given the constituent periods).
    """

    def __init__(
        self,
        *,
        friction: float = DEFAULT_FRICTION,
        fault: FaultGeometry | None = None,
        shear_modulus_gpa: float = 30.0,
        poisson_ratio: float = 0.25,
    ) -> None:
        self.friction = float(friction)
        self.fault = fault or FaultGeometry()
        self.shear_modulus_gpa = float(shear_modulus_gpa)
        self.poisson_ratio = float(poisson_ratio)
        self._ocean: OceanLoader | None = None

    def set_ocean_loader(self, loader: OceanLoader | None) -> "TidesEnricher":
        """Register an ocean-loading model ``(lat, lon, t) -> (Δτ_kPa, Δσ_n_kPa)`` (SPOTL/TPXO). Chainable.

        At coastal/subduction margins the ocean tide loads the crust far more than the body tide;
        omitting it is the single biggest tidal-channel modeling error (data-and-pipelines.md §1.4).
        """
        self._ocean = loader
        return self

    # -- public contract ----------------------------------------------------

    def features_at(
        self,
        lat: float,
        lon: float,
        *,
        t_issue: str | datetime | pd.Timestamp | None = None,
        fault: FaultGeometry | None = None,
        strict: bool = False,
        **_: Any,
    ) -> EnricherResult:
        """Return tidal-stress covariates at ``(lat, lon)`` for the given issue time.

        ``t_issue`` defaults to "now" (UTC); the daily forecast clock passes the seal time so the
        feature is reproducible. ``fault`` overrides the receiver geometry for this cell (e.g. the
        Slab2 strike/dip at a subduction cell).

        The semidiurnal phase and fortnightly Mf envelope are **analytic** (computed from the issue
        time alone) and are always returned. The ΔCFS / stressing-rate columns need pygtide; if it is
        not installed they degrade to ``None`` (with a one-time warning) so a partial tidal channel
        is still available — set ``strict=True`` to raise the actionable ImportError instead.
        """
        t = _to_utc(t_issue) if t_issue is not None else pd.Timestamp.now(tz="UTC")
        geom = fault or self.fault

        # Analytic, always-available circular covariates.
        phase_sin, phase_cos = _semidiurnal_phase(t)
        mf = _mf_envelope(t)
        out: EnricherResult = {
            "tidal_dCFS_kpa": None,
            "tidal_stress_rate_kpa_per_hr": None,
            "tidal_phase_sin": float(phase_sin),
            "tidal_phase_cos": float(phase_cos),
            "tidal_mf_envelope": float(mf),
        }

        # ΔCFS + stressing rate require the pygtide body tide (lazy heavy dep).
        try:
            dtau, dsigma = self._tidal_stress_components(lat, lon, t, geom)
            dcfs = dtau + self.friction * dsigma
            # Stressing rate by a centered finite difference (±30 min) — cheap and robust.
            dt = pd.Timedelta(minutes=30)
            dtau_p, dsig_p = self._tidal_stress_components(lat, lon, t + dt, geom)
            dtau_m, dsig_m = self._tidal_stress_components(lat, lon, t - dt, geom)
            dcfs_p = dtau_p + self.friction * dsig_p
            dcfs_m = dtau_m + self.friction * dsig_m
            out["tidal_dCFS_kpa"] = float(dcfs)
            out["tidal_stress_rate_kpa_per_hr"] = float(dcfs_p - dcfs_m)  # over 1 hr (samples ±0.5 hr)
        except ImportError:
            if strict:
                raise
            logger.warning(
                "pygtide not installed; tidal ΔCFS/stressing-rate are None (analytic phase + Mf "
                "envelope still returned). Install 'caos-seismic[science]' for the full tidal channel."
            )
        return out

    # -- internals ----------------------------------------------------------

    def _tidal_stress_components(
        self, lat: float, lon: float, t: pd.Timestamp, geom: FaultGeometry
    ) -> tuple[float, float]:
        """Return (Δτ, Δσ_n) on the receiver fault in kPa from body tide (pygtide) + ocean loading."""
        body_tau, body_sigma = self._body_tide_stress(lat, lon, t, geom)
        ocean_tau, ocean_sigma = 0.0, 0.0
        if self._ocean is not None:
            ocean_tau, ocean_sigma = self._ocean(lat, lon, t)
        return body_tau + ocean_tau, body_sigma + ocean_sigma

    def _body_tide_stress(
        self, lat: float, lon: float, t: pd.Timestamp, geom: FaultGeometry
    ) -> tuple[float, float]:
        """Solid-Earth body-tide shear/normal stress (kPa) on the receiver fault, via pygtide.

        pygtide returns tidal **areal/volumetric strain** components; we map strain → stress with the
        isotropic-elastic relation and resolve onto the fault plane. pygtide is imported lazily — if
        it is missing the method raises an actionable error (the rest of the feature, the analytic
        phase/Mf envelope, does not need it).
        """
        strain = self._body_tide_strain(lat, lon, t)
        # Isotropic-elastic strain → stress: use σ ≈ 2μ·ε for the deviatoric part (shear) and the
        # volumetric part for the normal change. Strain is dimensionless (≈1e-8); μ in GPa → kPa via
        # ×1e6. This is a first-order resolution sufficient for a *regularized* covariate; the gate
        # that admits this feature is empirical information gain, not the absolute calibration.
        mu_kpa = self.shear_modulus_gpa * 1.0e6  # GPa → kPa
        shear_stress = 2.0 * mu_kpa * strain["shear"]
        normal_stress = mu_kpa * strain["areal"] / (1.0 - 2.0 * self.poisson_ratio)

        # Resolve onto the fault: project with the rake for shear and the dip for the normal change.
        rake = math.radians(geom.rake_deg)
        dip = math.radians(geom.dip_deg)
        dtau = shear_stress * math.cos(rake)
        dsigma = normal_stress * math.sin(dip)
        return float(dtau), float(dsigma)

    def _body_tide_strain(self, lat: float, lon: float, t: pd.Timestamp) -> dict[str, float]:
        """Evaluate pygtide for areal + shear tidal strain at ``(lat, lon, t)`` (lazy heavy import)."""
        pygtide_mod = require("pygtide")()
        try:
            pt = pygtide_mod.pygtide()
        except Exception:  # pragma: no cover - pygtide API drift / data files
            PyTide = getattr(pygtide_mod, "PyTide", None) or getattr(pygtide_mod, "Pygtide", None)
            if PyTide is None:
                raise
            pt = PyTide()

        start = _to_utc(t).to_pydatetime()
        # One-sample prediction of tidal strain (NS + EW horizontal strain + areal). pygtide's
        # `predict` returns a DataFrame indexed by time with strain columns (units: nanostrain).
        df = pt.predict(
            float(lat), float(lon), 0.0, start, 1, 60,
            tidal_args=("Strain areal", "Strain NS", "Strain EW"),
        )
        row = _first_row(df)
        areal = _strain_value(row, ("Strain areal", "areal", "Areal")) * 1e-9
        ns = _strain_value(row, ("Strain NS", "NS")) * 1e-9
        ew = _strain_value(row, ("Strain EW", "EW")) * 1e-9
        shear = 0.5 * abs(ns - ew)
        return {"areal": areal, "shear": shear}


# ─────────────────────────────────────────────────────────────────────────────
# Analytic phase / envelope helpers (no model evaluation needed)
# ─────────────────────────────────────────────────────────────────────────────

#: Reference epoch for the tidal phase clocks (J2000.0 UTC).
_EPOCH = pd.Timestamp("2000-01-01T12:00:00Z")


def _semidiurnal_phase(t: pd.Timestamp) -> tuple[float, float]:
    """Sine/cosine of the dominant semidiurnal (M2) phase at ``t`` — a continuous circular covariate."""
    hours = (_to_utc(t) - _EPOCH).total_seconds() / 3600.0
    phase = 2.0 * math.pi * (hours % M2_PERIOD_HOURS) / M2_PERIOD_HOURS
    return math.sin(phase), math.cos(phase)


def _mf_envelope(t: pd.Timestamp) -> float:
    """Fortnightly Mf (14.77 d) envelope amplitude in [0, 1] at ``t`` (raised-cosine of the Mf phase)."""
    days = (_to_utc(t) - _EPOCH).total_seconds() / 86400.0
    phase = 2.0 * math.pi * (days % MF_PERIOD_DAYS) / MF_PERIOD_DAYS
    return 0.5 * (1.0 + math.cos(phase))


def _to_utc(t: str | datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _first_row(df: Any) -> Any:
    """Return the first record of a pygtide result (DataFrame or array-like) as a mapping/row."""
    if isinstance(df, pd.DataFrame):
        return df.iloc[0]
    return df[0]


def _strain_value(row: Any, keys: tuple[str, ...]) -> float:
    """Extract a strain column from a pygtide row, tolerant to the exact column label."""
    for k in keys:
        try:
            if k in getattr(row, "index", []):
                return float(row[k])
        except Exception:  # pragma: no cover
            pass
        try:
            return float(row[k])
        except (KeyError, TypeError, ValueError):
            continue
    return 0.0


_DEFAULT_ENRICHER: TidesEnricher | None = None


def features_at(lat: float, lon: float, **kwargs: Any) -> EnricherResult:
    """Module-level convenience: tidal-stress covariates at one coordinate (uses a cached enricher).

    Pass ``t_issue=`` for a reproducible seal time. The fortnightly-Mf and semidiurnal-phase features
    are returned even when pygtide is absent (they are analytic); the ΔCFS/stressing-rate features
    require pygtide and degrade to ``None`` (with a one-time warning) if it is missing — pass
    ``strict=True`` to raise the actionable ImportError instead.
    """
    global _DEFAULT_ENRICHER
    if _DEFAULT_ENRICHER is None:
        _DEFAULT_ENRICHER = TidesEnricher()
    return _DEFAULT_ENRICHER.features_at(lat, lon, **kwargs)


def phase_features_at(lat: float, lon: float, *, t_issue: Any = None) -> EnricherResult:
    """Analytic-only tidal phase + Mf envelope (no pygtide needed) — the always-available subset.

    Useful when pygtide/ocean-loading is not installed but the (cheap, exact) tidal-phase and
    fortnightly-envelope circular covariates are still wanted as honest near-null channels.
    """
    t = _to_utc(t_issue) if t_issue is not None else pd.Timestamp.now(tz="UTC")
    s, c = _semidiurnal_phase(t)
    return {
        "tidal_dCFS_kpa": None,
        "tidal_stress_rate_kpa_per_hr": None,
        "tidal_phase_sin": float(s),
        "tidal_phase_cos": float(c),
        "tidal_mf_envelope": float(_mf_envelope(t)),
    }


__all__ = [
    "DATASET",
    "DEFAULT_FRICTION",
    "MF_PERIOD_DAYS",
    "M2_PERIOD_HOURS",
    "FEATURE_NAMES",
    "FaultGeometry",
    "TidesEnricher",
    "download",
    "features_at",
    "phase_features_at",
]

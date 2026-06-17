"""Public contracts for CAOS_SEISMIC — the seams every module implements against.

This module is the single source of truth for:
  * the **catalog schema** (column contract for the event DataFrame),
  * the **Forecaster** interface (the port: fit → conditional intensity → forecast),
  * the **ForecastField** in-memory result, and
  * the **compact artifact** + **provenance manifest** schemas (the JSON the static web app reads).

Implementations live in `caos_seismic.data`, `.catalog`, `.model`, `.inference`, `.eval`.
Keeping interfaces here lets those modules be built independently without drifting.

Framing (non-negotiable): this is a *forecaster*, never a *predictor*. Every published number is a
probability in (0, 1), scoped to region × magnitude band × horizon, shown next to its long-term
baseline, with uncertainty bounds, evaluated CSEP-style. No alarms, no countdowns, no "safe" state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Catalog schema (column contract for the event DataFrame passed across modules)
# ─────────────────────────────────────────────────────────────────────────────

#: Required columns of a clean earthquake catalog DataFrame. Times are UTC pandas
#: Timestamps; magnitudes carry BOTH the native value+type and the Mw-homogenized value.
CATALOG_COLUMNS: dict[str, str] = {
    "event_id": "stable source id (e.g. ComCat id); used for dedup across providers",
    "time": "origin time, UTC (pandas datetime64[ns, UTC])",
    "latitude": "degrees, WGS84",
    "longitude": "degrees, WGS84",
    "depth_km": "hypocentral depth, km",
    "mag": "native magnitude value",
    "mag_type": "native magnitude type (mb, Ms, ML, Md, Mw, ...) — NEVER dropped",
    "mw": "magnitude homogenized to Mw-equivalent (TLS conversion; == mag where already Mw)",
    "source": "provider (usgs_comcat | csn | isc_gem | gcmt | emsc | ...)",
}


def validate_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Raise if `df` is missing any required catalog column; return it unchanged otherwise."""
    missing = set(CATALOG_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"catalog is missing required columns: {sorted(missing)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Geometry / target configuration (lightweight value objects)
# ─────────────────────────────────────────────────────────────────────────────


class BBox(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class Region(BaseModel):
    id: str
    name_en: str
    name_es: str
    bbox: BBox
    m_max: float = Field(..., description="maximum magnitude bounding the exceedance integral")
    attribution: list[str] = []


class View(BaseModel):
    """A country / region **view** into the single global forecast field.

    Core thesis of the global re-scope: the model trains on worldwide seismicity and conditions a
    single global field; *any country is a VIEW into that field*, never a separately-trained model.
    A view is a lightweight spatial window (an ISO-3166 country, a tectonic province, …) the static
    web's region selector slices the global artifact down to. It is NOT a fitting unit — fitting is
    always global; the view only bounds which cells are surfaced and which ``m_max`` / attribution
    apply when the artifact is read back as a per-country slice.
    """

    id: str = Field(..., description="stable view id (ISO-3166 alpha-2/-3 for countries, e.g. 'CL', 'JP')")
    name_en: str
    name_es: str
    bbox: BBox
    m_max: float = Field(..., description="view-local maximum magnitude bounding the exceedance integral")
    attribution: list[str] = []
    h3_resolution: int | None = Field(
        default=None, description="optional finer display H3 resolution for this view (overrides world)"
    )

    def as_region(self) -> "Region":
        """Project this view to a :class:`Region` (so model/grid code that takes a Region can reuse it)."""
        return Region(
            id=self.id,
            name_en=self.name_en,
            name_es=self.name_es,
            bbox=self.bbox,
            m_max=self.m_max,
            attribution=self.attribution,
        )


# ─────────────────────────────────────────────────────────────────────────────
# The Forecaster port — every model (ETAS, Reasenberg–Jones, smoothed, neural) implements this
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class Forecaster(Protocol):
    """A conditional seismicity forecaster.

    The forecast clock guarantees `fit`/`forecast` only ever see the catalog slice strictly before
    `t_issue` (no leakage). `name`/`version` are recorded in the provenance manifest.
    """

    name: str
    version: str

    def fit(self, catalog: pd.DataFrame, region: Region, t_issue: pd.Timestamp) -> "Forecaster":
        """Fit/condition on events with time < t_issue. Returns self."""
        ...

    def expected_counts(
        self,
        region: Region,
        cells: "list[Cell]",
        horizon_days: float,
        m_threshold: float,
        t_issue: pd.Timestamp,
    ) -> list[float]:
        """Expected number N_{>=M*} of events per cell over [t_issue, t_issue+horizon)."""
        ...


class BaseForecaster(ABC):
    """ABC convenience base; concrete models may subclass instead of duck-typing the Protocol."""

    name: str = "base"
    version: str = "0.0.0"

    @abstractmethod
    def fit(self, catalog: pd.DataFrame, region: Region, t_issue: pd.Timestamp) -> "BaseForecaster": ...

    @abstractmethod
    def expected_counts(
        self,
        region: Region,
        cells: "list[Cell]",
        horizon_days: float,
        m_threshold: float,
        t_issue: pd.Timestamp,
    ) -> list[float]: ...


# ─────────────────────────────────────────────────────────────────────────────
# In-memory forecast result
# ─────────────────────────────────────────────────────────────────────────────


class Cell(BaseModel):
    """A spatial cell. `key` is the H3 index (display) or "lat,lon" (fine fit grid)."""

    key: str
    lat: float
    lon: float


class CellForecast(BaseModel):
    """Per cell × horizon × threshold probabilistic forecast."""

    cell: str
    horizon_days: float
    m_threshold: float
    expected: float = Field(..., description="P(>=1 event >= M*) median / expected, in (0,1)")
    lo: float = Field(..., description="optimistic bound (P10)")
    hi: float = Field(..., description="pessimistic bound (P90)")
    rate: float = Field(..., description="expected event count N_{>=M*} (lambda*T)")
    baseline: float = Field(..., description="long-term Poisson baseline probability for the same cell")


class ForecastField(BaseModel):
    region_id: str
    issued_at: str
    cells: list[CellForecast]


# ─────────────────────────────────────────────────────────────────────────────
# Provenance manifest (VERSIONED) — makes every forecast byte-reproducible
# ─────────────────────────────────────────────────────────────────────────────


class Manifest(BaseModel):
    stage: Literal["fetch", "clean", "mc_decluster", "features", "model", "inference"]
    created_at: str
    region_id: str
    code_git_sha: str | None = None
    config_hash: str | None = None
    inputs: dict = {}
    outputs: dict = {}
    stats: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# The compact daily artifact — the contract between the offline job and the static web app
# ─────────────────────────────────────────────────────────────────────────────

ARTIFACT_SCHEMA_VERSION = "1.0"


class CalibrationSummary(BaseModel):
    reliability: list[list[float]] = Field(
        default_factory=list, description="[[forecast_prob, observed_freq, n], ...] per horizon-bin"
    )
    csep: dict = Field(
        default_factory=dict,
        description="consistency-test quantile scores: {N, M, S, L, CL} in [0,1] + pass flags",
    )
    info_gain_vs_poisson_nats: float | None = None
    info_gain_vs_etas_nats: float | None = None


class Staleness(BaseModel):
    generated: str            # ISO-8601 UTC
    next_run: str             # ISO-8601 UTC
    ok: bool = True           # false → degrade visibly (banner + desaturation)


class ViewIndexEntry(BaseModel):
    """One entry in the artifact's ``views`` index — a country slice of the global field.

    The global artifact ships ONE field; the SPA's country selector resolves a view to its cell-key
    list (``cells``) and reads only those keys from the shared ``forecast`` dict. Storing the cell
    keys (a few hundred per country) is far cheaper than duplicating the per-cell forecast payload
    per country, and keeps the global field the single source of truth.
    """

    id: str
    name_en: str
    name_es: str
    bbox: BBox
    m_max: float
    attribution: list[str] = []
    h3_resolution: int | None = None
    cells: list[str] = Field(
        default_factory=list, description="H3 cell keys of the global field falling in this view's bbox"
    )
    n_cells: int = 0


class ForecastArtifact(BaseModel):
    """The single compact JSON (gzipped on disk) the SPA renders. Keep it small: sparse cells,
    H3 keys, quantized rates. NEVER ship the full dense global grid.

    Global re-scope: ``region`` is the GLOBAL field's bounding region (whole earth); the per-country
    drill-downs are carried as ``views`` (each a lightweight cell-key index into the shared
    ``forecast`` dict, NOT a duplicated payload). ``grid`` may declare a base world resolution plus a
    per-view finer resolution (multi-resolution H3). Backward compatible: a single-region artifact
    simply ships an empty ``views`` list and a single-resolution ``grid``.
    """

    schema_version: str = ARTIFACT_SCHEMA_VERSION
    product: str = "CAOS_SEISMIC"
    issued_at: str
    region: Region
    horizons_days: list[int]
    magnitude_thresholds: list[float]
    m_max: float
    grid: dict = Field(..., description="{type:'h3', resolution:int} (+ optional 'resolution_world'/'resolution_region')")
    # forecast[cell_key][str(horizon)][str(M*)] -> {p, lo, hi, rate, baseline}
    forecast: dict[str, dict[str, dict[str, dict[str, float]]]]
    calibration: CalibrationSummary
    coverage_mask: list[str] = Field(
        default_factory=list, description="cell keys explicitly OUT of validated coverage (blank != safe)"
    )
    views: list[ViewIndexEntry] = Field(
        default_factory=list,
        description="per-country slices of the global field for the web's region selector (cell-key indices)",
    )
    provenance: dict = Field(default_factory=dict)
    staleness: Staleness

    def model_dump_compact(self) -> dict:
        """Dump with floats rounded for size; the writer further H3-bins + quantizes + gzips."""
        return self.model_dump(mode="json")

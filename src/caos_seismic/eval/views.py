"""Country VIEWs into the global field — the unit the global back-analysis scores over.

The re-scoped thesis: **global context conditions short-term local forecasts**. The model trains on
worldwide seismicity and conditions a single global field; *any country is a VIEW into that field*,
not a model of its own. The evaluation scores the same global model through a set of country windows
(bounding boxes) plus the global window, so a reviewer can ask not only "is it calibrated?" but "is
it calibrated *everywhere*, or does it over-fit the loud subduction margins?".

This module is a **thin, config-driven adapter** over the canonical view registry in
``configs/views.yaml`` (loaded by :mod:`caos_seismic.config`). It deliberately does NOT redefine any
geometry: bounding boxes, ``m_max`` and attribution all come from that single source of truth so the
back-analysis can never drift from the views the daily product and the web's country selector use.
What this module adds is the **pre-registered seismicity-class partition** used by the high-vs-low
bias comparison — each registry entry carries ``seismicity_class`` (``high`` = active plate boundary,
``low`` = stable interior / slow-deforming) and ``plate_setting``, surfaced here as a typed
:class:`CountryView`.

The set scored by the global driver defaults to ``default_backanalysis_views`` in the registry: a
tectonically diverse high-seismicity set (Chile, Japan, California, New Zealand) plus a
low-seismicity control set (Central/Eastern US, Western Europe, Eastern Australia), chosen *before*
scoring (evaluation-plan §2) so the high-vs-low partition is fixed, not a post-hoc split.

Imports only core deps + the package config/contracts — safe to import anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..config import (
    default_backanalysis_view_ids,
    load_region,
    load_views,
    view_metadata,
)
from ..contracts import Region

SeismicityClass = Literal["high", "low"]


@dataclass(frozen=True)
class CountryView:
    """One country window into the global field, with its pre-registered seismicity class.

    Attributes
    ----------
    region:
        The :class:`Region` (id, names, bbox, ``m_max``, attribution) scored by the forecast clock —
        projected from the registry :class:`~caos_seismic.contracts.View` via ``as_region()``.
    seismicity_class:
        ``"high"`` (active plate boundary) or ``"low"`` (stable interior / slow-deforming), read from
        ``configs/views.yaml``. Fixed *before* scoring so the high-vs-low bias comparison is a
        pre-registered partition.
    plate_setting:
        Short human-readable tectonic descriptor (subduction / transform / intraplate / ...), from the
        registry; shown in the web app and recorded in provenance so the partition is auditable.
    """

    region: Region
    seismicity_class: SeismicityClass
    plate_setting: str


def _country_view(view_id: str) -> CountryView:
    """Build a :class:`CountryView` for one registry view id (region + class + tectonic setting)."""
    view = load_views([view_id])[0]
    meta = view_metadata(view_id)
    cls: SeismicityClass = "low" if meta["seismicity_class"] == "low" else "high"
    return CountryView(region=view.as_region(), seismicity_class=cls, plate_setting=meta["plate_setting"])


def _global_view() -> CountryView:
    """The whole-Earth GLOBAL view, from ``configs/region.global.yaml``.

    The global field is productivity-dominated by active margins, so it is tagged ``high`` for the
    partition; it is appended *after* the country views in :func:`all_views` so a reader sees the
    per-country results before the pooled global number.
    """
    return CountryView(
        region=load_region("global"),
        seismicity_class="high",
        plate_setting="whole Earth (all tectonic settings)",
    )


#: The whole-Earth view, exposed for callers that want only the global window.
GLOBAL_VIEW: CountryView = _global_view()


def country_views(view_ids: list[str] | None = None) -> list[CountryView]:
    """The pre-registered country views (excluding GLOBAL), in registry order.

    ``view_ids`` overrides the default back-analysis set (``default_backanalysis_views`` in
    ``configs/views.yaml``).
    """
    ids = view_ids if view_ids is not None else default_backanalysis_view_ids()
    return [_country_view(vid) for vid in ids]


def high_seismicity_views(view_ids: list[str] | None = None) -> list[CountryView]:
    """The pre-registered HIGH-seismicity country views (active plate boundaries)."""
    return [v for v in country_views(view_ids) if v.seismicity_class == "high"]


def low_seismicity_views(view_ids: list[str] | None = None) -> list[CountryView]:
    """The pre-registered LOW-seismicity country views (stable interiors / slow-deforming)."""
    return [v for v in country_views(view_ids) if v.seismicity_class == "low"]


def all_views(include_global: bool = True, view_ids: list[str] | None = None) -> list[CountryView]:
    """All views the global back-analysis scores: every country view, optionally the GLOBAL view.

    The GLOBAL view is appended last so a reader sees per-country results before the pooled global
    number (which is dominated by the high-seismicity margins — the exact bias the high-vs-low
    comparison quantifies).
    """
    views = country_views(view_ids)
    if include_global:
        views.append(_global_view())
    return views


def view_by_id(view_id: str) -> CountryView | None:
    """Return the :class:`CountryView` with ``view_id`` (including ``"global"``), or ``None``."""
    if view_id == "global":
        return _global_view()
    try:
        return _country_view(view_id)
    except (KeyError, IndexError, FileNotFoundError):
        return None

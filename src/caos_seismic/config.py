"""Typed loading of the YAML configs in `configs/`.

Configs are the versioned source of truth for the region, grid, completeness, declustering, ETAS, the
forecast/output settings, and publishing. Raw data, features, and weights are NEVER versioned — they are
rebuildable from these configs + manifests + code.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .contracts import BBox, Region, View

# Repo root = three parents up from this file (src/caos_seismic/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

#: The default region the global pipeline operates on (whole-Earth field).
DEFAULT_REGION = "global"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=None)
def load_region(region_id: str = DEFAULT_REGION) -> Region:
    """Load `configs/region.<id>.yaml` into a typed `Region` (default: the global field)."""
    raw = _load_yaml(CONFIG_DIR / f"region.{region_id}.yaml")
    name = raw.get("name", {})
    bbox = raw["bbox"]
    return Region(
        id=raw["id"],
        name_en=name.get("en", raw["id"]),
        name_es=name.get("es", raw["id"]),
        bbox=BBox(**bbox),
        m_max=float(raw["m_max"]),
        attribution=list(raw.get("attribution", [])),
    )


@lru_cache(maxsize=None)
def _load_views_raw() -> dict[str, Any]:
    """Load the raw `configs/views.yaml` registry (returns `{}` if it does not exist)."""
    path = CONFIG_DIR / "views.yaml"
    if not path.exists():
        return {}
    return _load_yaml(path)


def _view_from_raw(view_id: str, raw: dict[str, Any]) -> View:
    """Build a typed :class:`View` from one entry of the views registry."""
    name = raw.get("name", {})
    h3_res = raw.get("h3_resolution")
    return View(
        id=view_id,
        name_en=name.get("en", view_id),
        name_es=name.get("es", view_id),
        bbox=BBox(**raw["bbox"]),
        m_max=float(raw["m_max"]),
        attribution=list(raw.get("attribution", [])),
        h3_resolution=int(h3_res) if h3_res is not None else None,
    )


def load_views(view_ids: list[str] | None = None) -> list[View]:
    """Load the configured country/region :class:`View` objects (slices of the global field).

    Parameters
    ----------
    view_ids:
        Explicit ids to load (in order). When ``None`` the ``default_views`` list from
        ``configs/views.yaml`` is used; an unknown id raises ``KeyError`` with the available set.
    """
    raw = _load_views_raw()
    registry: dict[str, Any] = raw.get("views", {}) or {}
    if view_ids is None:
        view_ids = list(raw.get("default_views", list(registry.keys())))
    out: list[View] = []
    for vid in view_ids:
        if vid not in registry:
            raise KeyError(
                f"unknown view {vid!r}; configured views are {sorted(registry)} "
                f"(edit configs/views.yaml to add a country)"
            )
        out.append(_view_from_raw(vid, registry[vid]))
    return out


def list_view_ids() -> list[str]:
    """All configured view ids (for the CLI `views` command)."""
    return sorted((_load_views_raw().get("views", {}) or {}).keys())


def default_view_ids() -> list[str]:
    """The view ids inference materializes by default (the `default_views` list)."""
    raw = _load_views_raw()
    return list(raw.get("default_views", list((raw.get("views", {}) or {}).keys())))


def default_backanalysis_view_ids() -> list[str]:
    """The pre-registered view ids the global back-analysis scores over (``default_backanalysis_views``).

    Falls back to ``default_views`` (then all configured views) when the back-analysis set is not
    declared, so a partially-edited registry still runs.
    """
    raw = _load_views_raw()
    if raw.get("default_backanalysis_views"):
        return list(raw["default_backanalysis_views"])
    return default_view_ids()


def view_metadata(view_id: str) -> dict[str, Any]:
    """Return the back-analysis metadata for a view: ``seismicity_class`` + ``plate_setting``.

    These tag the pre-registered HIGH/LOW partition for the bias comparison (they live in
    ``configs/views.yaml`` alongside each view but are not part of the lightweight :class:`View`
    contract). Unknown views / missing keys degrade to ``seismicity_class='high'`` (the conservative
    default — an unclassified loud region is assumed to be in the dominant class) and an empty
    ``plate_setting``.
    """
    registry = (_load_views_raw().get("views", {}) or {})
    entry = registry.get(view_id, {}) if isinstance(registry, dict) else {}
    cls = str(entry.get("seismicity_class", "high")).lower()
    if cls not in ("high", "low"):
        cls = "high"
    return {"seismicity_class": cls, "plate_setting": str(entry.get("plate_setting", ""))}


@lru_cache(maxsize=None)
def load(name: str) -> dict[str, Any]:
    """Load any `configs/<name>.yaml` as a dict (grid, completeness, declustering, etas, forecast, publish)."""
    return _load_yaml(CONFIG_DIR / f"{name}.yaml")


def config_hash(*names: str) -> str:
    """Stable short hash of one or more configs, for provenance manifests."""
    payload = {n: load(n) for n in names}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]

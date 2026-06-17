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

from .contracts import BBox, Region

# Repo root = three parents up from this file (src/caos_seismic/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=None)
def load_region(region_id: str = "chile") -> Region:
    """Load `configs/region.<id>.yaml` into a typed `Region`."""
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
def load(name: str) -> dict[str, Any]:
    """Load any `configs/<name>.yaml` as a dict (grid, completeness, declustering, etas, forecast, publish)."""
    return _load_yaml(CONFIG_DIR / f"{name}.yaml")


def config_hash(*names: str) -> str:
    """Stable short hash of one or more configs, for provenance manifests."""
    payload = {n: load(n) for n in names}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]

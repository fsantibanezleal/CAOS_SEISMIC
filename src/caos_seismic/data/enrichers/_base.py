"""Shared scaffolding for the global geophysical enrichers.

Every enricher in this subpackage is a *global* static-context loader: it downloads a worldwide
dataset once (cached to the gitignored ``data/enrichers/`` store), then answers
``features_at(lat, lon)`` for any cell on Earth. The core thesis of CAOS_SEISMIC is that the global
context conditions short-term *local* forecasts — so the enrichers are deliberately global, and any
region is just a spatial *view* into the same global covariate field.

This module holds the small, dependency-light pieces every enricher reuses:

* :data:`ENRICHER_CACHE` — the gitignored cache root (``data/enrichers/``), with a per-dataset
  subdirectory helper. Raw downloads and parsed caches are *never* versioned (rebuildable from the
  provenance record + code), exactly like the catalog stores.
* :class:`Provenance` — the license/citation/URL record every ``download()`` returns, so the public
  credits page is never forgotten (data-and-pipelines.md §9).
* :func:`http_download` — a polite, resumable-enough file GET reusing the catalog fetch
  ``User-Agent`` and a small retry/backoff (no obspy; ``requests`` only).
* :data:`EnricherResult` typing — the per-cell feature mapping ``features_at`` returns.

Heavy geospatial dependencies (``geopandas``/``shapely``/``netCDF4``/``xarray``/``pygtide``) are
**never** imported here; each enricher imports them lazily inside its own functions with a clear,
actionable error if the science extra is missing. This keeps ``import caos_seismic`` on the core
deps alone.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import requests

from ...config import REPO_ROOT

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Cache root (gitignored — rebuildable from provenance + code)
# ─────────────────────────────────────────────────────────────────────────────

#: Root of the gitignored enricher cache. Mirrors the ``data/raw`` / ``data/features`` convention:
#: raw downloaded grids/shapefiles and any parsed caches live here and are NEVER committed.
ENRICHER_CACHE = REPO_ROOT / "data" / "enrichers"

#: Polite contact User-Agent env var (shared with the catalog fetch layer). Never a secret.
ENV_USER_AGENT = "CAOS_SEISMIC_USER_AGENT"
_FALLBACK_USER_AGENT = "CAOS_SEISMIC/0.1 (+https://github.com/fsantibanezleal/CAOS_SEISMIC)"

#: A per-cell feature mapping: feature name -> value (float, or None where the cell is outside the
#: dataset's footprint, e.g. a slab grid only covers subduction margins).
EnricherResult = dict[str, float | None]


def cache_dir(dataset: str) -> Path:
    """Return (creating) the cache subdirectory for one dataset (``data/enrichers/<dataset>/``)."""
    d = ENRICHER_CACHE / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_agent() -> str:
    """Polite contact ``User-Agent`` from the environment, or a safe public fallback."""
    return os.environ.get(ENV_USER_AGENT, "").strip() or _FALLBACK_USER_AGENT


# ─────────────────────────────────────────────────────────────────────────────
# Provenance — license + citation + source URLs (for the public credits page)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Provenance:
    """The license/citation/source record an enricher's ``download()`` returns.

    These obligations are tracked so the public app's credits page is complete and the share-alike
    licenses (e.g. GEM faults CC-BY-SA 4.0) are never silently stripped (data-and-pipelines.md §9).
    """

    dataset: str
    title: str
    version: str | None
    source_url: str
    license: str
    attribution: str
    citation: str
    files: list[str] = field(default_factory=list)
    retrieved_at: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-able dict for embedding in a manifest / the artifact provenance block."""
        return {
            "dataset": self.dataset,
            "title": self.title,
            "version": self.version,
            "source_url": self.source_url,
            "license": self.license,
            "attribution": self.attribution,
            "citation": self.citation,
            "files": list(self.files),
            "retrieved_at": self.retrieved_at,
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Polite HTTP download (requests only; no obspy)
# ─────────────────────────────────────────────────────────────────────────────


def http_download(
    url: str,
    dest: Path,
    *,
    session: requests.Session | None = None,
    max_retries: int = 4,
    timeout_s: float = 600.0,
    chunk_bytes: int = 1 << 20,
    overwrite: bool = False,
    _sleep=time.sleep,
) -> Path:
    """Stream-download ``url`` to ``dest`` with a polite User-Agent and exponential backoff.

    Skips the download when ``dest`` already exists and ``overwrite`` is false (the cache is
    content-addressable by the source URL's filename, so an existing file is the cached copy).
    Writes to a ``.part`` sidecar and atomically renames on success so a half-written file is never
    mistaken for a complete cache entry. Raises :class:`RuntimeError` after exhausting retries.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        logger.info("cache hit: %s (skip download)", dest)
        return dest

    sess = session or requests
    headers = {"User-Agent": user_agent()}
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with sess.get(url, headers=headers, timeout=timeout_s, stream=True) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"download failed ({resp.status_code}) for {url}")
                total = 0
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_bytes):
                        if chunk:
                            fh.write(chunk)
                            total += len(chunk)
            tmp.replace(dest)
            logger.info("downloaded %s → %s (%d bytes)", url, dest, total)
            return dest
        except (requests.RequestException, RuntimeError) as exc:
            last_exc = exc
            tmp.unlink(missing_ok=True)
            logger.warning("download error (%s); retry %d/%d", exc, attempt + 1, max_retries)
            if attempt < max_retries:
                _sleep(min(60.0, 1.0 * (2.0**attempt)))
                continue
            raise RuntimeError(f"failed to download {url} after {max_retries} retries: {exc}") from exc
    raise RuntimeError(f"unreachable: exhausted retries for {url}: {last_exc}")  # pragma: no cover


# ─────────────────────────────────────────────────────────────────────────────
# Lazy-import guards for the heavy geospatial science deps
# ─────────────────────────────────────────────────────────────────────────────


def require(module_hint: str, *, extra: str = "science"):
    """Build a lazy-importer that raises an actionable error if the science extra is missing.

    Usage inside an enricher function::

        xr = require("xarray")()
        gpd = require("geopandas")()

    The returned callable imports the module on first call and returns it; on ``ImportError`` it
    raises with the exact ``pip install`` line. Never imported at module top level — that would put
    the heavy dep on the core import path, which the contracts forbid.
    """
    top = module_hint.split(".")[0]

    def _import():
        try:
            mod = __import__(module_hint, fromlist=["__name__"])
        except ImportError as exc:  # pragma: no cover - exercised only when the dep is absent
            raise ImportError(
                f"'{top}' is required for this enricher but is not installed. It is a heavy "
                f"geospatial dependency kept off the core import path. Install the science extra:\n"
                f"    pip install 'caos-seismic[{extra}]'   (or: pip install {top})"
            ) from exc
        return mod

    return _import


def features_for_cells(enricher: Any, cells: list, **kwargs: Any) -> "list[EnricherResult]":
    """Apply an enricher's ``features_at`` across a list of :class:`~caos_seismic.contracts.Cell`.

    A tiny convenience used by :func:`caos_seismic.catalog.features.build_context_features` so each
    enricher only has to implement the scalar ``features_at(lat, lon)`` contract. ``cells`` may be
    ``Cell`` objects (``.lat``/``.lon``) or plain ``(lat, lon)`` tuples.
    """
    out: list[EnricherResult] = []
    for c in cells:
        lat = getattr(c, "lat", None)
        lon = getattr(c, "lon", None)
        if lat is None or lon is None:
            lat, lon = c  # (lat, lon) tuple
        out.append(enricher.features_at(float(lat), float(lon), **kwargs))
    return out


def empty_like(feature_names: "list[str] | Mapping[str, Any]") -> EnricherResult:
    """Return an all-``None`` feature mapping for the given feature names (out-of-footprint cells)."""
    names = feature_names.keys() if isinstance(feature_names, Mapping) else feature_names
    return {name: None for name in names}

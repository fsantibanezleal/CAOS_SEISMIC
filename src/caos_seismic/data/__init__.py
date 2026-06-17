"""Data layer — catalog acquisition and cleaning (stages A and B of the pipeline DAG).

This subpackage implements the **fetch** and **clean** stages defined in
``docs/data-and-pipelines.md`` §1–§3:

* :mod:`caos_seismic.data.fetch` — real catalog acquisition. The *spine* is USGS ComCat
  pulled over raw FDSN with ``requests`` alone (no ObsPy): ``/count`` first, tile around the
  20,000-event/request cap, ``updatedafter`` incremental deltas, polite retry/backoff. Optional
  helpers for CSN/ISC/EMSC (via ObsPy FDSN) and ISC-GEM/GCMT downloads lazily import the heavy
  deps and raise a clear, actionable error if they are absent.
* :mod:`caos_seismic.data.clean` — dedupe across providers by preferred id, magnitude
  homogenization to **Mw** via a total-least-squares (orthogonal) regression anchored on the
  ISC-GEM/GCMT overlap, and validation against the catalog column contract.

Both stages write their data into the gitignored ``data/`` store (Parquet) and a provenance
:class:`~caos_seismic.contracts.Manifest` into ``manifests/``. The data itself is never
committed; the manifests are.
"""

from __future__ import annotations

from .clean import (
    CleanResult,
    TLSFit,
    build_mw_anchor,
    clean_catalog,
    dedupe_events,
    fit_conversions,
    homogenize_to_mw,
    load_clean_catalog,
    normalize_mag_type,
    save_clean_catalog,
    tls_regression,
)
from .fetch import (
    DEFAULT_COMCAT_BASE,
    ComCatError,
    fetch_comcat,
    fetch_comcat_count,
    fetch_region_comcat,
    run_fetch,
)

__all__ = [
    "DEFAULT_COMCAT_BASE",
    "ComCatError",
    "fetch_comcat",
    "fetch_comcat_count",
    "fetch_region_comcat",
    "run_fetch",
    "CleanResult",
    "TLSFit",
    "build_mw_anchor",
    "clean_catalog",
    "dedupe_events",
    "fit_conversions",
    "homogenize_to_mw",
    "load_clean_catalog",
    "save_clean_catalog",
    "normalize_mag_type",
    "tls_regression",
]

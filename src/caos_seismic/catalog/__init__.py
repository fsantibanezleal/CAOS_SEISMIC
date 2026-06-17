"""Catalog hygiene — magnitude of completeness, the Gutenberg–Richter b-value, and declustering.

This subpackage implements stage (C) of the pipeline DAG (``Mc + DECLUSTERING``) described in the
data-and-pipelines synthesis. It depends only on the *core* deps (``numpy``, ``pandas``, ``scipy``)
so the catalog spine works without any heavy geophysics stack installed.

Two public concerns live here:

* :mod:`caos_seismic.catalog.completeness` — the magnitude of completeness ``Mc`` (maximum-curvature
  with the configurable California-tuned ``+0.2`` correction and a goodness-of-fit cross-check), its
  rolling space–time variant, and the Aki–Utsu binning-corrected maximum-likelihood ``b``-value with
  Shi & Bolt uncertainty. **``b`` is always estimated, never hard-coded.**
* :mod:`caos_seismic.catalog.decluster` — Gardner–Knopoff windowing (OpenQuake hmtk coefficients) for
  the *declustered background* catalog, and Zaliapin–Ben-Zion nearest-neighbour proximity
  (``eta``/``T``/``R``) used **both** as ML features and as principled cluster labels.

THE DUAL-CATALOG RULE (the most common pipeline mistake, made explicit in :func:`dual_catalog`):

* the **declustered** catalog feeds ONLY the stationary Poisson / smoothed-seismicity background;
* the **full, un-declustered** catalog feeds the conditional / ETAS model — triggering *is* the
  predictable signal;
* scoring is on the **non-declustered** catalog, because the product deliberately forecasts clustering.

References (canonical, public):
  Aki (1965), BERI 43, 237–239; Tinti & Mulargia (1987), BSSA 77(6), 2125–2134;
  Shi & Bolt (1982), BSSA 72(5), 1677–1687; Wiemer & Wyss (2000), BSSA 90(4), 859–869;
  Woessner & Wiemer (2005), BSSA 95(2), 684–698; Gardner & Knopoff (1974), BSSA 64(5), 1363–1367;
  Zaliapin & Ben-Zion (2013/2020), JGR Solid Earth (doi:10.1029/2018JB017120).
"""

from __future__ import annotations

from .completeness import (
    BValueEstimate,
    McEstimate,
    aki_utsu_b_value,
    fmd,
    gft_mc,
    maxc_mc,
    mc_estimate,
    rolling_mc,
)
from .decluster import (
    DUAL_CATALOG_DOC,
    DeclusterResult,
    DualCatalog,
    NearestNeighborResult,
    dual_catalog,
    gardner_knopoff,
    gardner_knopoff_windows,
    zaliapin_ben_zion,
)

__all__ = [
    # completeness
    "BValueEstimate",
    "McEstimate",
    "aki_utsu_b_value",
    "fmd",
    "gft_mc",
    "maxc_mc",
    "mc_estimate",
    "rolling_mc",
    # declustering
    "DUAL_CATALOG_DOC",
    "DeclusterResult",
    "DualCatalog",
    "NearestNeighborResult",
    "dual_catalog",
    "gardner_knopoff",
    "gardner_knopoff_windows",
    "zaliapin_ben_zion",
]

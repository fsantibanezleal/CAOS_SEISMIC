"""Covariate-field providers — wire the real geophysical enrichers behind the context-TPP's
``CovariateFieldProvider`` seam (``model/context_tpp.CovariateFieldProvider``).

A provider is a ``(region, t_issue) -> CovariateField`` callable. The context-conditioned neural TPP
ingests that gridded multi-channel field as the CNN's "image". When no provider is wired the model runs
on a zeros field (catalog/seismicity context only, honestly flagged). This module assembles the field
from the cached enricher outputs — starting with **GNSS strain rate** (the covariate with the strongest
established time-independent forecasting value; Bird et al. 2010, GEAR1 Strader et al. 2018), which the
research identifies as the one external channel with demonstrated GLOBAL (not regional) prospective skill.

The provider grids each channel at a deliberately COARSE pitch (default 1° for the whole-Earth field):
context varies on a tectonic scale, the CNN down-samples anyway, and a 0.25° global grid (~10⁶ cells)
would make the per-cell enricher evaluation needlessly expensive. The model's ``vector_at`` maps any
fit cell to the nearest covariate cell, so the coarse field is sampled correctly at inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import Region
from ..model.context_tpp import CovariateField


def make_strain_provider(cell_deg: float = 1.0):
    """Return a provider that fills the ``gnss_strain_rate`` channel from the cached NGL MIDAS field.

    Requires the MIDAS table to be cached (``data.enrichers.gnss.download()``); raises a clear error
    otherwise. The other geophysical channels stay zero-filled + flagged missing — the field is honest
    about what is and is not wired. Channel values are standardized per-channel at training time, so the
    raw nanostrain/yr units (and the occasional noisy-station outlier) are absorbed by the model.
    """
    from .enrichers import gnss

    enricher = gnss.GnssEnricher()  # lazy-loads the cached MIDAS combined table

    def provider(region: Region, t_issue: pd.Timestamp) -> CovariateField:
        field = CovariateField.zeros(region, cell_deg=cell_deg)
        if "gnss_strain_rate" not in field.channels:
            return field
        ch = field.channels.index("gnss_strain_rate")
        data = field.data.copy()
        n_filled = 0
        for i in range(field.n_lat):
            lat = field.lat0 + i * field.cell_deg
            for j in range(field.n_lon):
                lon = field.lon0 + j * field.cell_deg
                res = enricher.features_at(lat, lon)
                val = res.get("gnss_strain_rate_nstrain_yr")
                if val is not None and np.isfinite(val):
                    data[ch, i, j] = float(val)
                    n_filled += 1
        # The channel is "present" only if it actually carried signal somewhere.
        missing = field.missing_channels
        if n_filled > 0:
            missing = tuple(c for c in field.missing_channels if c != "gnss_strain_rate")
        return CovariateField(
            lat0=field.lat0,
            lon0=field.lon0,
            cell_deg=field.cell_deg,
            data=data,
            channels=field.channels,
            missing_channels=missing,
        )

    return provider

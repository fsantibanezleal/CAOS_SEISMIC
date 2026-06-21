"""Context-conditioned spatio-temporal neural temporal point process — the **gated challenger**.

This module implements the thesis model of CAOS_SEISMIC: a neural conditional intensity that learns
*global context conditions short-term local forecasts*. The catalog is a realization of a marked
spatio-temporal point process; this model estimates its conditional intensity
``lambda*(t, x, y, m | H_t)`` with a **Hawkes inductive bias** (additive background + summed
self-excitation, in the FERN spirit) whose fixed ETAS kernels are replaced by small neural networks,
**plus** a **CNN spatial-context encoder** that ingests the gridded GLOBAL covariate field
(slab geometry, faults, plate boundaries, GNSS strain, stress, tides, smoothed seismicity) as a
multi-channel "image" and produces a per-cell context embedding that conditions the intensity.

Thesis (carried in code): the model trains on **worldwide** seismicity + global covariate fields; any
country is a *view into a global field*. The CNN is the spatial encoder of that context — **not** a
standalone aftershock classifier. The DeVries et al. (2018, *Nature* 560:632-634) standalone-CNN
aftershock approach is the **refuted lesson** here: a 2-parameter logistic regression matched it
(Mignan & Broccardo 2019, *Nature* 575:E1-E3, doi:10.1038/s41586-019-1582-8); per-cell AUC on
spatially-correlated pixels from a few mainshocks is the wrong target. We use the CNN only to *encode
context that conditions a proper point-process intensity*, and we score with the point-process
log-likelihood and CSEP, never AUC.

Architecture (model-design §2 + research/03-ml-approaches §9)::

    lambda(t, x, y | H_t) = mu_theta(x, y, C) + sum_{i: t_i < t} kappa_phi(m_i, C_i)
                            * g_psi(t - t_i) * f_eta(r_ij, m_i, C_i)

where ``C`` / ``C_i`` is the per-cell context embedding produced by the CNN from the covariate field.

* ``mu_theta``  — background MLP head over (smoothed-seismicity log-rate, context embedding); learns a
  conditioned tectonic background (geodetic strain has established time-independent value, model-design
  §6.2).
* ``kappa_phi`` — productivity MLP (Utsu-like: monotone-ish in magnitude via a softplus head) modulated
  by the parent-cell context embedding (learned, context-dependent productivity).
* ``g_psi``     — a normalized Omori-like temporal kernel parameterized by a small MLP over log-elapsed
  time (a flexible monotone-decaying density, integrating to ~1 by Monte-Carlo normalization on the
  unit window). Keeps the additive-Hawkes survival term tractable.
* ``f_eta``     — a spatial kernel whose magnitude-dependent scale ``zeta`` is *modulated* by the
  context embedding, letting the model learn **anisotropy aligned with fault/slab structure** without
  hand-coding geometry (the proven FERN lever — Zlydenko et al. 2023, *Sci. Reports* 13,
  doi:10.1038/s41598-023-38033-9).
* **Explicit magnitude head** — a learnable Gutenberg-Richter ``b`` (and a context-conditioned
  correction) so the model produces a real conditional magnitude distribution. Most NPPs omit this; it
  is a real gap flagged by EarthquakeNPP (Stockman, Lawson & Werner, TMLR 2026, arXiv:2410.08226).

Honest grounding (docstring-as-contract): **as of 2026, no neural point process has robustly beaten a
well-fit ETAS in prospective CSEP testing** (EarthquakeNPP: five NPPs, none beat ETAS). FERN's modest
4-12% IGPE gain came from ingesting sub-Mc events and learning anisotropy, *not* deep-learning magic,
and FERN was never CSEP-tested and shipped no uncertainty. This module is **R&D measured honestly**: it
is a *gated challenger* that reaches the public field **only if** it (a) beats ETAS in our prospective
CSEP harness (positive IGPE, paired-T CI excluding zero) **and** (b) calibrates (reliability diagram).
Until then it stays behind the feature flag and ETAS carries the region.

Importability: ``torch`` is imported **lazily** inside functions/methods, never at module top level, so
``import caos_seismic.model.context_tpp`` works with only the core deps (numpy/pandas). Constructing the
forecaster, building the covariate field, and reading checkpoints' metadata all work without torch; only
``fit``/``expected_counts``/``train`` require it and raise a clear error if it is missing.

References
----------
Ogata, Y. (1998), *Ann. Inst. Statist. Math.* 50(2), 379-402, doi:10.1023/A:1003403601725.
Zlydenko, O. et al. (2023), *Sci. Reports* 13, doi:10.1038/s41598-023-38033-9 (FERN encoder).
Dascher-Cousineau, K. et al. (2023), *GRL* 50, e2023GL103909, doi:10.1029/2023GL103909 (RECAST).
Stockman, S., Lawson, D. & Werner, M. J. (2026), *TMLR*, arXiv:2410.08226 (EarthquakeNPP benchmark).
DeVries, P. M. R. et al. (2018), *Nature* 560, 632-634, doi:10.1038/s41586-018-0438-y (refuted CNN).
Mignan, A. & Broccardo, M. (2019), *Nature* 575, E1-E3, doi:10.1038/s41586-019-1582-8 (the rebuttal).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from ..contracts import BaseForecaster, Cell, Region, validate_catalog
from ._common import (
    DEG2KM,
    LN10,
    bvalue_aki_utsu,
    gr_exceedance_fraction,
    haversine_km,
    poisson_p_at_least_one,
)
from .smoothed import SmoothedSeismicityForecaster

if TYPE_CHECKING:  # pragma: no cover - typing only; never imported at runtime top level
    import torch
    from torch import Tensor, nn

logger = logging.getLogger(__name__)


def _gr_exceedance_fraction_vec(
    m_threshold: float, b: np.ndarray, mc: float, m_max: float | None
) -> np.ndarray:
    """Vectorized bounded Gutenberg-Richter tail fraction Φ(M*) over an array of per-cell ``b`` values.

    Element-for-element identical to :func:`_common.gr_exceedance_fraction` (scalar ``m_threshold``,
    ``mc``, ``m_max``; ``b`` an array) — only the per-cell Python call is lifted into numpy so the
    magnitude term can be applied to the whole forecast field at once.
    """
    b = np.asarray(b, dtype=np.float64)
    if m_threshold <= mc:
        return np.ones_like(b)
    unbounded = np.power(10.0, -b * (m_threshold - mc))
    if m_max is None:
        return np.clip(unbounded, 0.0, 1.0)
    if m_threshold >= m_max:
        return np.zeros_like(b)
    tail_max = np.power(10.0, -b * (m_max - mc))
    frac = (unbounded - tail_max) / (1.0 - tail_max)
    return np.clip(frac, 0.0, 1.0)


#: Default checkpoint directory — OUTSIDE git (the repo .gitignore excludes data/ and weights). Neural
#: weights are never versioned; they are rebuildable from configs + the global catalog + this code.
DEFAULT_CHECKPOINT_DIR = "data/weights"

#: The covariate channels the CNN context encoder ingests, in a fixed, documented order. Each is a
#: gridded global field resampled to the model grid. Missing channels are zero-filled and flagged in
#: the manifest (the model degrades to catalog-only context, never silently). Order is the channel axis
#: of the CNN input tensor and MUST stay stable across train/inference (it is part of the contract).
COVARIATE_CHANNELS: tuple[str, ...] = (
    "smoothed_seismicity_lograte",  # log10 long-term smoothed-seismicity background (the dominant lever)
    "slab_depth_km",                # Slab2 subduction interface depth (Hayes et al. 2018)
    "slab_dip_deg",                 # Slab2 interface dip
    "dist_to_fault_km",             # distance to nearest GEM Active Fault
    "plate_boundary_type",          # Bird 2003 PB2002 boundary class (one-hot-collapsed scalar)
    "gnss_strain_rate",             # Nevada Geodetic Lab MIDAS second invariant of strain rate
    "stress_orientation",          # focal-mechanism-derived stress proxy (second tier)
    "tidal_cfs_amplitude",          # rate-and-state tidal Delta-CFS amplitude (regularized covariate)
)


# ─────────────────────────────────────────────────────────────────────────────
# Covariate field — the gridded GLOBAL context the CNN ingests (core-deps only)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CovariateField:
    """A multi-channel gridded covariate field over a lat/lon box — the CNN's "image" input.

    This is the in-memory representation of the GLOBAL context resampled to a regular grid. It is pure
    numpy (no torch, no geopandas) so it can be constructed, cached, sliced and asserted on with only
    the core deps; the heavy producers (Slab2 netCDF, GEM faults shapefile, MIDAS strain, pygtide
    tides) live in :mod:`caos_seismic.data` and write the channel arrays here. When a producer is
    unavailable the channel is zero-filled and recorded in :attr:`missing_channels` — the model then
    conditions on catalog-derived context only, and the manifest shows the degradation honestly.

    Attributes
    ----------
    lat0, lon0, cell_deg:
        South-west grid origin (cell-centre of ``[0, 0]``) and the regular pitch in degrees.
    data:
        Array ``(n_channels, n_lat, n_lon)`` in :data:`COVARIATE_CHANNELS` order (channel-first, the
        PyTorch ``NCHW`` convention). Values are per-channel standardized at training time.
    channels:
        Channel names aligned to ``data``'s first axis (defaults to :data:`COVARIATE_CHANNELS`).
    missing_channels:
        Channels that were zero-filled because their producer was unavailable (flagged in provenance).
    """

    lat0: float
    lon0: float
    cell_deg: float
    data: np.ndarray
    channels: tuple[str, ...] = COVARIATE_CHANNELS
    missing_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"covariate data must be (C, H, W); got shape {self.data.shape}")
        if self.data.shape[0] != len(self.channels):
            raise ValueError(
                f"channel count mismatch: data has {self.data.shape[0]} channels but "
                f"{len(self.channels)} names were given"
            )

    @property
    def n_lat(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_lon(self) -> int:
        return int(self.data.shape[2])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    def cell_index(self, lat: float, lon: float) -> tuple[int, int]:
        """Map ``(lat, lon)`` to the nearest grid cell index ``(i, j)``, clamped to the grid bounds."""
        i = int(round((lat - self.lat0) / self.cell_deg))
        j = int(round((lon - self.lon0) / self.cell_deg))
        i = min(max(i, 0), self.n_lat - 1)
        j = min(max(j, 0), self.n_lon - 1)
        return i, j

    def vector_at(self, lat: float, lon: float) -> np.ndarray:
        """Return the length-``C`` covariate vector at the cell containing ``(lat, lon)``."""
        i, j = self.cell_index(lat, lon)
        return self.data[:, i, j].astype(float)

    @classmethod
    def zeros(
        cls,
        region: Region,
        cell_deg: float,
        *,
        channels: tuple[str, ...] = COVARIATE_CHANNELS,
    ) -> "CovariateField":
        """Build an all-zero field over the region bbox — the honest 'no covariates available' state.

        Used when no geophysical producers are wired yet (the package ships before Slab2/faults/GNSS
        are fetched). Every channel is marked missing so the manifest and the UI coverage mask show the
        model is running catalog-context-only, never pretending it had the global field.
        """
        bb = region.bbox
        n_lat = max(int(np.ceil((bb.lat_max - bb.lat_min) / cell_deg)), 1)
        n_lon = max(int(np.ceil((bb.lon_max - bb.lon_min) / cell_deg)), 1)
        data = np.zeros((len(channels), n_lat, n_lon), dtype=np.float32)
        return cls(
            lat0=bb.lat_min + cell_deg / 2.0,
            lon0=bb.lon_min + cell_deg / 2.0,
            cell_deg=cell_deg,
            data=data,
            channels=channels,
            missing_channels=tuple(channels),
        )

    def with_smoothed_seismicity(
        self, background: SmoothedSeismicityForecaster
    ) -> "CovariateField":
        """Fill the ``smoothed_seismicity_lograte`` channel from a fitted smoothed-seismicity field.

        The long-term smoothed-seismicity log-rate is the single most informative covariate (it *is*
        the spatial prior the whole field floors to), so even with no external geophysics the CNN
        always has this real channel. Returns a new field with the channel populated and removed from
        :attr:`missing_channels`. A no-op (returns ``self``) if the channel is absent.
        """
        if "smoothed_seismicity_lograte" not in self.channels:
            return self
        ch = self.channels.index("smoothed_seismicity_lograte")
        out = self.data.copy()
        for i in range(self.n_lat):
            lat = self.lat0 + i * self.cell_deg
            for j in range(self.n_lon):
                lon = self.lon0 + j * self.cell_deg
                rate = background.background_rate_density(lat, lon)  # events / day / km^2
                out[ch, i, j] = float(np.log10(max(rate, 1e-12)))
        new_missing = tuple(c for c in self.missing_channels if c != "smoothed_seismicity_lograte")
        return CovariateField(
            lat0=self.lat0,
            lon0=self.lon0,
            cell_deg=self.cell_deg,
            data=out,
            channels=self.channels,
            missing_channels=new_missing,
        )

    def standardized(self) -> tuple["CovariateField", np.ndarray, np.ndarray]:
        """Return a per-channel standardized copy plus the ``(mean, std)`` used (for inference reuse).

        CNNs train far better on zero-mean/unit-variance channels; the same stats must be reapplied at
        inference, so they are returned for persistence alongside the checkpoint. Missing (all-zero)
        channels keep ``std=1`` so they map to zeros, not NaNs.
        """
        mean = self.data.mean(axis=(1, 2), keepdims=True)
        std = self.data.std(axis=(1, 2), keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        norm = (self.data - mean) / std
        out = CovariateField(
            lat0=self.lat0,
            lon0=self.lon0,
            cell_deg=self.cell_deg,
            data=norm.astype(np.float32),
            channels=self.channels,
            missing_channels=self.missing_channels,
        )
        return out, mean.squeeze().astype(float), std.squeeze().astype(float)


#: A provider that yields the covariate field for a region/issue time. The daily job wires the real
#: geophysical loaders behind this seam; tests and the cold-start path pass a zeros/smoothed-only field.
CovariateFieldProvider = Callable[[Region, pd.Timestamp], CovariateField]


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ContextTPPConfig:
    """Hyperparameters for the context-conditioned neural TPP (configs/neural.yaml when present).

    Deliberately small: the binding resource is **covariate availability and effective sample size**,
    not network capacity (research/03-ml-approaches §7.3). Over-parameterizing a model with few
    effective independent sequences is exactly the DeVries failure mode, so the defaults keep the
    encoder and kernel MLPs compact and lean on the Hawkes inductive bias.
    """

    # CNN spatial-context encoder.
    context_dim: int = 16          # per-cell context embedding width
    cnn_channels: tuple[int, ...] = (16, 32)  # conv widths
    cnn_kernel: int = 3
    patch_radius: int = 4          # half-size of the local covariate patch fed per cell (cells)

    # Kernel MLPs (Hawkes-inductive-bias heads).
    kernel_hidden: int = 32
    m0: float = 3.5                # reference magnitude for productivity/scale exponents

    # Temporal kernel normalization (Monte-Carlo on the unit window).
    temporal_mc_samples: int = 256

    # Training.
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4   # AdamW L2 (keeps the over-parameterized net honest)
    epochs: int = 40
    batch_events: int = 2048       # events per NLL minibatch (history-causal)
    grad_clip: float = 5.0
    seed: int = 0
    max_parents_per_event: int = 256  # truncate the triggering history for O(N * P) cost

    # Numerical floors.
    eps: float = 1.0e-8


# ─────────────────────────────────────────────────────────────────────────────
# Torch module factory (built lazily so the package imports without torch)
# ─────────────────────────────────────────────────────────────────────────────


def _import_torch():
    """Import torch lazily with an actionable error. Called by every method that needs it.

    Keeping this the single import site means the package top level never touches torch, so
    ``import caos_seismic`` and ``import caos_seismic.model.context_tpp`` both succeed with core deps
    only (the hard rule). The neural challenger is optional R&D; torch is an optional dependency.
    """
    try:
        import torch  # noqa: F401

        return torch
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without torch
        raise ModuleNotFoundError(
            "The neural challenger (context_tpp) requires PyTorch. Install the optional 'torch' "
            "dependency (`pip install torch`) to fit or run it. The ETAS core needs no torch and "
            "carries the public field until this gated challenger beats it in CSEP."
        ) from exc


def _build_network(cfg: ContextTPPConfig, n_channels: int):
    """Construct the torch ``nn.Module`` implementing the context-conditioned Hawkes intensity.

    Built inside a function (not at import) so the heavy ``nn`` symbols are only resolved when torch is
    present. Returns an instance of the locally-defined ``_ContextTPPNet`` (a closure over the imported
    ``nn``), whose ``forward`` is documented inline. Separating "math/orchestration" (this module's
    top level, numpy) from "the torch graph" (here) is what keeps the import lazy.
    """
    torch = _import_torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class _ContextEncoderCNN(nn.Module):
        """CNN that maps a local multi-channel covariate patch to a per-cell context embedding.

        This is the **spatial encoder of context**, not a forecaster. Input is an ``NCHW`` patch
        ``(B, n_channels, P, P)`` centred on the target cell; output is a length-``context_dim``
        embedding ``C``. Small by design (two conv blocks + global pooling + linear) — capacity here
        invites the DeVries overfitting failure, and the inductive bias should come from the Hawkes
        skeleton, not a deep CNN.
        """

        def __init__(self) -> None:
            super().__init__()
            convs: list[nn.Module] = []
            in_ch = n_channels
            for out_ch in cfg.cnn_channels:
                convs.append(nn.Conv2d(in_ch, out_ch, cfg.cnn_kernel, padding=cfg.cnn_kernel // 2))
                convs.append(nn.ReLU(inplace=True))
                in_ch = out_ch
            self.conv = nn.Sequential(*convs)
            self.head = nn.Linear(in_ch, cfg.context_dim)

        def forward(self, patch: "Tensor") -> "Tensor":
            h = self.conv(patch)                       # (B, C_last, P, P)
            h = h.mean(dim=(2, 3))                      # global average pool -> (B, C_last)
            return self.head(h)                         # (B, context_dim)

    class _MLP(nn.Module):
        """A two-layer MLP head (softplus output optional) used for the neural kernels."""

        def __init__(self, in_dim: int, out_dim: int, *, softplus_out: bool) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, cfg.kernel_hidden),
                nn.Tanh(),
                nn.Linear(cfg.kernel_hidden, out_dim),
            )
            self.softplus_out = softplus_out

        def forward(self, x: "Tensor") -> "Tensor":
            y = self.net(x)
            return F.softplus(y) if self.softplus_out else y

    class _ContextTPPNet(nn.Module):
        """The full context-conditioned Hawkes intensity as a single differentiable module.

        Sub-modules:

        * ``encoder``   — :class:`_ContextEncoderCNN`, the CNN context encoder.
        * ``mu_head``   — background MLP over (context, smoothed-log-rate) -> softplus rate density.
        * ``kappa_head``— productivity MLP over (magnitude, parent context) -> softplus offspring count.
        * ``g_head``    — temporal-kernel MLP over a log-elapsed-time feature -> unnormalized density.
        * ``zeta_head`` — spatial-scale MLP over (magnitude, parent context) -> softplus scale (deg).
        * ``log_b``     — learnable scalar; ``b = softplus(log_b)`` is the global Gutenberg-Richter b,
          with a small per-cell context correction ``b_ctx`` (explicit magnitude modelling).

        The intensity is assembled by :meth:`intensity_at_events` (for the NLL log-term) and the
        compensator by :meth:`integrated_intensity` (Monte-Carlo over the window); both are documented
        where they are defined in the Python forecaster, which calls these heads.
        """

        def __init__(self) -> None:
            super().__init__()
            self.encoder = _ContextEncoderCNN()
            self.mu_head = _MLP(cfg.context_dim + 1, 1, softplus_out=True)
            self.kappa_head = _MLP(cfg.context_dim + 1, 1, softplus_out=True)
            self.g_head = _MLP(1, 1, softplus_out=True)
            self.zeta_head = _MLP(cfg.context_dim + 1, 1, softplus_out=True)
            self.b_head = _MLP(cfg.context_dim, 1, softplus_out=False)
            self.log_b = nn.Parameter(torch.tensor(0.0))  # softplus(0)=~0.69; data pulls it to ~b ln-scale

        # -- kernel pieces (all operate on torch tensors) ------------------------
        def encode(self, patches: "Tensor") -> "Tensor":
            return self.encoder(patches)

        def background(self, ctx: "Tensor", smoothed_lograte: "Tensor") -> "Tensor":
            """mu(x,y,C) — softplus background rate density (events/day/deg^2), per cell."""
            x = torch.cat([ctx, smoothed_lograte.unsqueeze(-1)], dim=-1)
            return self.mu_head(x).squeeze(-1)

        def productivity(self, mag: "Tensor", ctx: "Tensor") -> "Tensor":
            """kappa(m, C) — softplus expected direct offspring of a parent (context-modulated Utsu)."""
            x = torch.cat([(mag - cfg.m0).unsqueeze(-1), ctx], dim=-1)
            return self.kappa_head(x).squeeze(-1)

        def temporal_density(self, dt_days: "Tensor") -> "Tensor":
            """g(dt) — unnormalized neural temporal kernel (normalized to a density by Monte-Carlo).

            Feature is ``log(dt + c0)`` (an Omori-like log-time coordinate) so the MLP can recover a
            power-law decay while staying a flexible monotone-ish density. The normalization to unit
            mass over the forecast window is handled by the forecaster (it draws ``temporal_mc_samples``
            uniform times, evaluates this head, and divides), keeping ``g`` a proper density.
            """
            feat = torch.log(dt_days.clamp_min(cfg.eps) + 0.01).unsqueeze(-1)
            return self.g_head(feat).squeeze(-1)

        def spatial_scale(self, mag: "Tensor", ctx: "Tensor") -> "Tensor":
            """zeta(m, C) — softplus magnitude-and-context-dependent spatial scale (deg).

            The context modulation is the learned-anisotropy lever (FERN): zeta grows with magnitude
            and is reshaped by the slab/fault/strain context around the parent cell.
            """
            x = torch.cat([(mag - cfg.m0).unsqueeze(-1), ctx], dim=-1)
            return self.zeta_head(x).squeeze(-1) + cfg.eps

        def b_value(self, ctx: "Tensor") -> "Tensor":
            """Conditional Gutenberg-Richter b(C) = softplus(log_b) + small context correction (>0)."""
            base = F.softplus(self.log_b)
            corr = 0.1 * torch.tanh(self.b_head(ctx).squeeze(-1))  # bounded +-0.1 correction
            return (base + corr).clamp_min(0.2)

    return _ContextTPPNet()


# ─────────────────────────────────────────────────────────────────────────────
# The forecaster (Forecaster contract; numpy-side orchestration, torch only inside methods)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ContextTPPForecaster(BaseForecaster):
    """Context-conditioned spatio-temporal neural TPP — the gated neural challenger to ETAS.

    Implements the :class:`~caos_seismic.contracts.Forecaster` port (``fit`` /
    ``expected_counts``) so the inference driver and the CSEP harness treat it interchangeably with
    ETAS, R-J and the smoothed null. It is **never the default**: the daily job constructs it only
    behind a feature flag, and :func:`train` records whether it cleared the ETAS gate. The Hawkes
    skeleton mirrors :class:`~caos_seismic.model.etas.ETASForecaster` exactly (additive background +
    summed self-excitation, bounded-GR magnitude tail, ``P = 1 - e^{-N}`` exceedance), so a senior
    seismologist can read the two side by side; only the kernels are neural.

    Parameters
    ----------
    config:
        :class:`ContextTPPConfig` hyperparameters.
    covariate_provider:
        Callable returning the :class:`CovariateField` for a region/issue time. If ``None``, a
        smoothed-seismicity-only field is built at ``fit`` time (the honest cold-start context — the
        model still gets the dominant covariate channel). Wire the real geophysical loaders here for a
        full global-context run.
    background:
        Pre-fit :class:`SmoothedSeismicityForecaster` for ``mu`` cold-start floor + the smoothed-rate
        covariate channel; built on the (declustered) fit catalog if ``None``.
    mc, b_value:
        Completeness and a *prior* Gutenberg-Richter ``b`` (the model learns a conditional ``b`` but
        initializes/regularizes toward this). Estimated on the fit catalog if ``None``.
    checkpoint_dir:
        Where to write weights (outside git). ``None`` keeps weights in memory only.
    device:
        ``"cuda"``/``"cpu"``/``None`` (auto: GPU if available).
    """

    name: str = "context_tpp"
    version: str = "0.1.0"

    config: ContextTPPConfig = field(default_factory=ContextTPPConfig)
    covariate_provider: CovariateFieldProvider | None = None
    background: SmoothedSeismicityForecaster | None = None
    mc: float | None = None
    b_value: float | None = None
    checkpoint_dir: str | None = DEFAULT_CHECKPOINT_DIR
    device: str | None = None

    # ── fitted state ──────────────────────────────────────────────────────────
    _net: Any = field(default=None, repr=False)             # the torch module (Any to avoid the import)
    _field: CovariateField | None = field(default=None, repr=False)
    _field_mean: np.ndarray | None = field(default=None, repr=False)
    _field_std: np.ndarray | None = field(default=None, repr=False)
    _t_issue: pd.Timestamp | None = field(default=None, repr=False)
    _region: Region | None = field(default=None, repr=False)
    _mc: float = field(default=0.0, repr=False)
    _b_prior: float = field(default=1.0, repr=False)
    _rate_cal: float = field(default=1.0, repr=False)  # absolute-rate calibration constant (fit-time)
    _ev_t: np.ndarray | None = field(default=None, repr=False)   # parent ages (days before t_issue, >=0)
    _ev_lat: np.ndarray | None = field(default=None, repr=False)
    _ev_lon: np.ndarray | None = field(default=None, repr=False)
    _ev_m: np.ndarray | None = field(default=None, repr=False)
    _device: str = field(default="cpu", repr=False)
    _history: dict = field(default_factory=dict, repr=False)
    params_used: dict = field(default_factory=dict, repr=False)

    # ── Forecaster.fit (the real training loop) ───────────────────────────────
    def fit(
        self, catalog: pd.DataFrame, region: Region, t_issue: pd.Timestamp
    ) -> "ContextTPPForecaster":
        """Train the neural intensity by point-process NLL on the catalog before ``t_issue``.

        The forecast clock guarantees only events with ``time < t_issue`` are seen (a leakage backstop
        is applied here too). Steps:

        1. Slice + sort the lawful past; estimate ``Mc`` / prior ``b``.
        2. Build (or fetch) the :class:`CovariateField`, fill the smoothed-seismicity channel, and
           standardize it (stats persisted for inference).
        3. Cache the parent-event arrays (ages in days, lat/lon/mw) — the same representation ETAS uses.
        4. Build the torch network and run the AdamW training loop minimizing the negative
           point-process log-likelihood ``-(sum_i ln lambda_i - ∫∫∫ lambda)`` with the compensator
           approximated by Monte-Carlo over the observation window (the integral term is what makes
           this a *probabilistic* model, not a regressor — research/03-ml-approaches §1).
        5. Checkpoint the weights + covariate stats outside git.

        Returns ``self``. Requires torch (raises a clear error otherwise); the ETAS core does not.
        """
        torch = _import_torch()
        validate_catalog(catalog)
        cfg = self.config

        df = catalog.loc[catalog["time"] < t_issue].copy()
        if df.empty:
            raise ValueError("no events strictly before t_issue to train the neural TPP")
        df = df.sort_values("time")
        self._t_issue = pd.Timestamp(t_issue)
        self._region = region

        # 1) Completeness + prior b.
        self._mc = float(self.mc) if self.mc is not None else float(df["mw"].min())
        complete = df.loc[df["mw"] >= self._mc - 1e-9]
        if len(complete) < 5:
            raise ValueError(
                f"need >= 5 events at/above Mc={self._mc} to train the neural TPP (got {len(complete)})"
            )
        if self.b_value is not None:
            self._b_prior = float(self.b_value)
        else:
            self._b_prior, _ = bvalue_aki_utsu(complete["mw"].to_numpy(), self._mc, delta_m=0.1)

        # 2) Background + covariate field (smoothed-seismicity channel always populated).
        if self.background is None:
            self.background = SmoothedSeismicityForecaster(
                b_value=self._b_prior, mc=self._mc
            ).fit(catalog, region, t_issue)
        elif self.background._ev_lat is None:
            self.background.fit(catalog, region, t_issue)

        field_raw = (
            self.covariate_provider(region, self._t_issue)
            if self.covariate_provider is not None
            else CovariateField.zeros(region, cell_deg=self.config_grid_deg())
        )
        field_raw = field_raw.with_smoothed_seismicity(self.background)
        self._field, self._field_mean, self._field_std = field_raw.standardized()

        # 3) Parent arrays (ages in days before t_issue, >= 0) — same representation as ETAS.
        t_days = (self._t_issue - complete["time"]).dt.total_seconds().to_numpy() / 86400.0
        self._ev_t = np.clip(t_days, 0.0, None)
        self._ev_lat = complete["latitude"].to_numpy(dtype=float)
        self._ev_lon = complete["longitude"].to_numpy(dtype=float)
        self._ev_m = complete["mw"].to_numpy(dtype=float)
        train_days = float(max(self._ev_t.max(), 1.0))

        # 4) Build + train.
        self._device = self._resolve_device()
        self._net = _build_network(cfg, self._field.n_channels).to(self._device)
        self._history = self._train_loop(train_days)

        # 4b) Absolute-rate calibration. The training compensator constrains the background only at event
        # locations (it approximates ∫mu dA by the event-mean), so the forecast-grid integral in
        # expected_counts is unconstrained in event-free cells and mis-normalizes (~hundreds×). Anchor the
        # absolute level post-hoc with a single multiplicative constant that matches the integrated daily
        # forecast rate to the training rate (events ≥ Mc per day). Leakage-free (training data only); a
        # uniform scale preserves the conditional SHAPE the NLL fitted.
        self._rate_cal = 1.0
        try:
            self._rate_cal = self._fit_rate_calibration(region, train_days, complete)
        except Exception:  # pragma: no cover - calibration is best-effort; falls back to 1.0
            self._rate_cal = 1.0

        # 5) Checkpoint outside git.
        ckpt = self._save_checkpoint(region)
        self.params_used = {
            "mc": self._mc,
            "b_prior": self._b_prior,
            "n_parents": int(self._ev_t.size),
            "train_days": train_days,
            "device": self._device,
            "context_dim": cfg.context_dim,
            "covariate_channels": list(self._field.channels),
            "missing_covariate_channels": list(self._field.missing_channels),
            "final_nll": self._history.get("final_nll"),
            "epochs": cfg.epochs,
            "checkpoint": ckpt,
            "honest_note": (
                "neural challenger; NOT default; must beat ETAS in prospective CSEP + calibrate "
                "before it reaches the public field (EarthquakeNPP 2026: no NPP has, as of 2026)."
            ),
        }
        return self

    def recondition(self, catalog: pd.DataFrame, t_issue: pd.Timestamp) -> "ContextTPPForecaster":
        """Advance the conditioning to a new ``t_issue`` WITHOUT re-training (the cadenced path).

        Keeps the trained net weights, the covariate field (the geophysical context is time-independent
        over a refit cadence), the Mc/b prior, and the ``rate_cal`` calibration; refreshes only the
        triggering CONDITIONING — which events are parents and their ages. Mirrors
        :meth:`ETASForecaster.recondition`, so a pseudo-prospective back-analysis can advance the neural
        day-to-day (fit weekly, recondition daily) instead of paying the ~74-min retrain at every issue.
        Leakage-safe: only events strictly before ``t_issue`` are admitted. Requires a prior :meth:`fit`.
        """
        if self._net is None:
            raise RuntimeError("recondition() requires a prior fit()")
        validate_catalog(catalog)
        self._t_issue = pd.Timestamp(t_issue)
        df = catalog.loc[catalog["time"] < self._t_issue]
        complete = df.loc[df["mw"] >= self._mc - 1e-9].sort_values("time")
        if complete.empty:
            self._ev_t = np.empty(0, dtype=float)
            self._ev_lat = np.empty(0, dtype=float)
            self._ev_lon = np.empty(0, dtype=float)
            self._ev_m = np.empty(0, dtype=float)
        else:
            t_days = (self._t_issue - complete["time"]).dt.total_seconds().to_numpy() / 86400.0
            self._ev_t = np.clip(t_days, 0.0, None)
            self._ev_lat = complete["latitude"].to_numpy(dtype=float)
            self._ev_lon = complete["longitude"].to_numpy(dtype=float)
            self._ev_m = complete["mw"].to_numpy(dtype=float)
        self._ctx_parents_cache = None  # parents changed → recompute their context embeddings lazily
        return self

    def config_grid_deg(self) -> float:
        """Covariate-grid pitch (deg). Coarser than the 0.1° fit grid keeps the CNN patch cheap.

        The CNN ingests *context*, which varies on a tectonic (≈0.25°) scale, not the 0.1° fit
        resolution; using a coarser context grid keeps the patch tensor small and the encoder honest.
        """
        return 0.25

    # ── the training loop (real AdamW point-process NLL) ──────────────────────
    def _train_loop(self, train_days: float) -> dict:
        """Minimize the negative point-process log-likelihood with AdamW (GPU if available).

        Loss = ``-(sum_i ln lambda(t_i, x_i, y_i) - ∫_0^T ∫_A lambda)``:

        * **log term** — for each event ``j`` the intensity is the conditioned background at its cell
          plus the summed neural triggering from its (truncated) causal parents ``i < j``. Computed in
          minibatches of events for memory.
        * **compensator** — Monte-Carlo over the window: the background mass is ``mu_total * T``; the
          triggering mass is ``sum_i kappa(m_i, C_i) * G_i`` where ``G_i`` is the (normalized) temporal
          kernel mass over the parent's in-window age — exactly the closed-form ETAS compensator with
          the neural kernels substituted. Because ``g`` is normalized to a unit-mass density on the
          window, ``G_i`` is the temporal CDF mass, mirroring ETAS's ``omori_utsu_cumulative``.

        A small prior pulls the learned ``b`` toward the Aki-Utsu prior (explicit magnitude modelling
        with a sane init). Returns a history dict (per-epoch NLL) for the manifest.
        """
        torch = _import_torch()

        net = self._net
        opt = torch.optim.AdamW(
            net.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        gen = torch.Generator(device="cpu").manual_seed(self.config.seed)

        n = int(self._ev_t.size)
        # Precompute per-event context patches + smoothed-log-rate once (static covariates).
        patches = self._event_patches(self._ev_lat, self._ev_lon)          # (n, C, P, P) np
        smoothed_lograte = self._event_smoothed_lograte(self._ev_lat, self._ev_lon)  # (n,) np
        patches_t = torch.as_tensor(patches, dtype=torch.float32, device=self._device)
        smoothed_t = torch.as_tensor(smoothed_lograte, dtype=torch.float32, device=self._device)
        ages_t = torch.as_tensor(self._ev_t, dtype=torch.float32, device=self._device)
        mags_t = torch.as_tensor(self._ev_m, dtype=torch.float32, device=self._device)
        lat_t = torch.as_tensor(self._ev_lat, dtype=torch.float32, device=self._device)
        lon_t = torch.as_tensor(self._ev_lon, dtype=torch.float32, device=self._device)

        history: list[float] = []
        net.train()
        for epoch in range(self.config.epochs):
            opt.zero_grad()
            ctx_all = net.encode(patches_t)                                  # (n, context_dim)
            mu_all = net.background(ctx_all, smoothed_t)                     # (n,) events/day/deg^2

            # Log term: intensity at each event from its causal parents (minibatched over events).
            log_term = self._log_intensity_term(
                net, ctx_all, mu_all, ages_t, mags_t, lat_t, lon_t, gen
            )

            # Compensator (Monte-Carlo window integral).
            comp = self._compensator_term(net, ctx_all, mu_all, ages_t, mags_t, train_days)

            # Magnitude prior: pull global b toward the Aki-Utsu estimate (weak L2 on the b-init).
            import torch.nn.functional as F

            b_pen = (F.softplus(net.log_b) - self._b_prior * LN10) ** 2

            nll = -(log_term - comp) / max(n, 1) + 1e-3 * b_pen
            nll.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.config.grad_clip)
            opt.step()
            history.append(float(nll.detach().cpu()))
            if epoch % max(self.config.epochs // 5, 1) == 0:
                logger.info("context_tpp epoch %d/%d NLL=%.4f", epoch, self.config.epochs, history[-1])

        net.eval()
        return {"nll_per_epoch": history, "final_nll": history[-1] if history else None}

    def _log_intensity_term(
        self, net, ctx_all, mu_all, ages_t, mags_t, lat_t, lon_t, gen
    ):
        """Sum of ``ln lambda(t_j, x_j, y_j)`` over a sampled minibatch of events (the NLL log term).

        For event ``j`` only strictly-older parents (``age_i > age_j``) contribute (Hawkes causality).
        We cap the parent set at ``max_parents_per_event`` nearest-in-time parents for an ``O(B * P)``
        cost. ``lambda_j = mu_j + sum_i kappa(m_i,C_i) g(dt_ij) f(r_ij, m_i, C_i)`` with the neural
        kernels. Returns a scalar tensor (sum over the minibatch); minibatching keeps memory bounded
        while remaining an unbiased estimate of the full log term scaled by ``n / batch``.
        """
        torch = _import_torch()
        n = int(ages_t.shape[0])
        batch = min(self.config.batch_events, n)
        idx = torch.randperm(n, generator=gen)[:batch].to(self._device)

        total = torch.zeros((), device=self._device)
        P = self.config.max_parents_per_event
        ages_np = self._ev_t
        for jj in idx.tolist():
            age_j = float(ages_np[jj])
            # Parents strictly older than j (larger age). Pick the P closest in time for cost control.
            older = np.where(self._ev_t > age_j + 1e-12)[0]
            mu_j = mu_all[jj]
            if older.size == 0:
                total = total + torch.log(mu_j.clamp_min(self.config.eps))
                continue
            if older.size > P:
                # nearest-in-time parents dominate the triggering sum (Omori decay).
                order = np.argsort(self._ev_t[older] - age_j)[:P]
                older = older[order]
            pidx = torch.as_tensor(older, dtype=torch.long, device=self._device)
            dt = (ages_t[pidx] - ages_t[jj]).clamp_min(self.config.eps)       # days since each parent
            kappa = net.productivity(mags_t[pidx], ctx_all[pidx])
            g = net.temporal_density(dt)
            zeta = net.spatial_scale(mags_t[pidx], ctx_all[pidx])
            r_km = haversine_km(
                float(self._ev_lat[jj]),
                float(self._ev_lon[jj]),
                self._ev_lat[older],
                self._ev_lon[older],
            )
            r_deg = torch.as_tensor(r_km / DEG2KM, dtype=torch.float32, device=self._device)
            # inverse-power spatial density at the learned scale (q-> use 1.5 shape; scale is learned).
            f = (0.5 / (np.pi * zeta * zeta)) * torch.pow(1.0 + (r_deg * r_deg) / (zeta * zeta), -1.5)
            lam_j = mu_j + torch.sum(kappa * g * f)
            total = total + torch.log(lam_j.clamp_min(self.config.eps))
        # Scale the minibatch sum up to the full-catalog log term (unbiased).
        return total * (n / max(batch, 1))

    def _compensator_term(self, net, ctx_all, mu_all, ages_t, mags_t, train_days):
        """``∫_0^T ∫_A lambda`` — background mass + summed neural-triggering mass over the window.

        Background mass: the region-integrated background rate times ``T``. We approximate the spatial
        integral of ``mu`` by the mean per-event background density times the region area in deg^2 (the
        events sample the region where the rate is non-negligible). Triggering mass: each parent's
        neural productivity ``kappa(m_i, C_i)`` times the temporal-kernel mass ``G_i`` over its
        in-window age, with ``g`` normalized to unit mass on the window by Monte-Carlo — so ``G_i`` is
        the fraction of the parent's offspring expected inside the window (the ETAS compensator shape).
        """
        torch = _import_torch()
        T = float(train_days)

        # Temporal-kernel unit-mass normalizer on [0, T] (Monte-Carlo): Z = mean_t g(t) * T.
        gen = torch.Generator(device="cpu").manual_seed(self.config.seed + 1)
        ts = torch.rand(self.config.temporal_mc_samples, generator=gen).to(self._device) * T
        gz = net.temporal_density(ts).mean() * T + self.config.eps           # ~∫_0^T g dt

        # Background mass: mean background density * region area (deg^2) * T.
        area_deg2 = self._region_area_deg2()
        bg_mass = mu_all.mean() * area_deg2 * T

        # Triggering mass: kappa(m_i,C_i) * (mass of g over the parent's in-window age) / Z.
        # A parent of age a_i (days before t_issue) had elapsed time a_i available inside the window.
        ages_clamped = ages_t.clamp(0.0, T)
        g_cdf = torch.stack([
            net.temporal_density(torch.linspace(self.config.eps, float(a) + self.config.eps, 16,
                                                device=self._device)).mean() * float(a)
            for a in ages_clamped.detach().cpu().tolist()
        ]) if ages_clamped.numel() else torch.zeros((), device=self._device)
        kappa = net.productivity(mags_t, ctx_all)
        trig_mass = torch.sum(kappa * (g_cdf / gz))
        return bg_mass + trig_mass

    def _fit_rate_calibration(
        self, region: Region, train_days: float, conditioning_catalog: pd.DataFrame
    ) -> float:
        """Single multiplicative constant that ties the integrated forecast rate to the training rate.

        Forecast a one-day window at ``Mc`` with the RAW (``rate_cal = 1``) integration over the SAME grid
        the forecast / gate uses (multi-resolution ``build_global_fit_cells`` globally; the fine grid for a
        bounded region — so the inferred per-cell area is identical and the constant transfers), sum it,
        and return ``(training events ≥ Mc per day) / (raw daily forecast)``. Anchors the absolute level
        the NLL left unconstrained while preserving the conditional shape; uses only training data
        (leakage-free). Returns ``1.0`` if the grid or the raw forecast is degenerate.
        """
        from ..config import load
        from ..inference.daily import build_fit_cells, build_global_fit_cells

        grid_cfg = load("grid")
        if region.id == "global":
            cells = build_global_fit_cells(region, grid_cfg, conditioning_catalog)
        else:
            cells = build_fit_cells(region, grid_cfg)
        if not cells:
            return 1.0
        self._rate_cal = 1.0  # the call below must be RAW
        raw_daily = float(np.sum(self.expected_counts(region, cells, 1.0, self._mc, self._t_issue)))
        if not np.isfinite(raw_daily) or raw_daily <= 0.0:
            return 1.0
        target_daily = float(self._ev_t.size) / max(float(train_days), 1.0)  # training events≥Mc per day
        cal = target_daily / raw_daily
        return float(cal) if np.isfinite(cal) and cal > 0.0 else 1.0

    # ── Forecaster.expected_counts ────────────────────────────────────────────
    def expected_counts(
        self,
        region: Region,
        cells: list[Cell],
        horizon_days: float,
        m_threshold: float,
        t_issue: pd.Timestamp,
    ) -> list[float]:
        """Expected count ``N_{>=M*}`` per cell over ``[t_issue, t_issue + horizon_days)``.

        Mirrors :meth:`ETASForecaster.expected_counts`: integrate the (neural) conditional intensity
        over the horizon and cell area, then multiply by the bounded Gutenberg-Richter exceedance
        fraction with the model's *conditional* ``b`` at each cell. The public probability is the
        unchanged ``P(>=1) = 1 - e^{-N}`` (model-design §3.2). Light time quadrature (the background is
        time-flat; only the neural Omori-like decay of the triggering term varies in time).
        """
        torch = _import_torch()
        self._require_fit()
        net = self._net
        net.eval()

        lats = np.array([c.lat for c in cells], dtype=float)
        lons = np.array([c.lon for c in cells], dtype=float)
        cell_area_deg2 = self._cell_area_deg2(cells)

        with torch.no_grad():
            patches = self._event_patches(lats, lons)
            slr = self._event_smoothed_lograte(lats, lons)
            patches_t = torch.as_tensor(patches, dtype=torch.float32, device=self._device)
            slr_t = torch.as_tensor(slr, dtype=torch.float32, device=self._device)
            ctx_cells = net.encode(patches_t)                                # (n_cells, context_dim)
            mu_cells = net.background(ctx_cells, slr_t).cpu().numpy()         # events/day/deg^2
            b_cells = net.b_value(ctx_cells).cpu().numpy()                    # conditional b per cell

            # Horizon-integrated neural triggering at EVERY cell, vectorized. Exact reformulation of
            # the old per-cell midpoint quadrature: the productivity (kappa) and spatial scale (zeta)
            # depend only on the parent events, and the temporal kernel g only on the step — so all
            # three are computed once over the parents and the 119k-cell × 12-step Python loop (~1.4M
            # net calls, ~50 min) collapses to a chunked distance matrix (seconds). See _triggering_field.
            steps = 12
            edges = np.linspace(0.0, float(horizon_days), steps + 1)
            mids = 0.5 * (edges[:-1] + edges[1:])
            dts = np.diff(edges)
            trig_cells = self._triggering_field(net, lats, lons, mids, dts)   # (n_cells,)

        # Conditional intensity integral per cell: time-flat background + the triggering integral.
        lam_integral = np.asarray(mu_cells, dtype=np.float64).reshape(-1) * float(horizon_days)
        lam_integral = lam_integral + trig_cells
        # bounded-GR tail with each cell's CONDITIONAL b (b_cells is beta = b·ln10 scale -> /LN10).
        b_eff = np.asarray(b_cells, dtype=np.float64).reshape(-1) / LN10
        mag_frac = _gr_exceedance_fraction_vec(m_threshold, b_eff, float(self._mc), region.m_max)
        n_exp = np.maximum(lam_integral * float(cell_area_deg2) * mag_frac, 0.0)
        rate_cal = float(getattr(self, "_rate_cal", 1.0) or 1.0)
        if rate_cal != 1.0:
            n_exp = n_exp * rate_cal
        return n_exp.tolist()

    def forecast_probabilities(
        self,
        region: Region,
        cells: list[Cell],
        horizon_days: float,
        m_threshold: float,
        t_issue: pd.Timestamp,
    ) -> list[float]:
        """Convenience: per-cell ``P(>=1 event >= M*) = 1 - e^{-N}`` (the public exceedance formula)."""
        return [
            poisson_p_at_least_one(n)
            for n in self.expected_counts(region, cells, horizon_days, m_threshold, t_issue)
        ]

    def _triggering_intensity(self, net, lat: float, lon: float, t_days_ahead: float) -> float:
        """Summed neural triggering intensity at ``(lat, lon, t_issue + t_days_ahead)`` (events/day/deg^2).

        Sum over all parent events of ``kappa(m_i, C_i) * g(dt_i) * f(r_i, m_i, C_i)`` with the neural
        kernels, where ``dt_i`` is the parent's pre-issue age plus ``t_days_ahead`` and ``C_i`` is the
        parent-cell context embedding. Background is added by the caller (it is time-flat). Evaluated
        under ``torch.no_grad`` by the caller.
        """
        torch = _import_torch()
        if self._ev_t is None or self._ev_t.size == 0:
            return 0.0
        # parent context embeddings (cache on first use).
        ctx_parents = self._parent_context(net)
        mags_t = torch.as_tensor(self._ev_m, dtype=torch.float32, device=self._device)
        dt = torch.as_tensor(self._ev_t + float(t_days_ahead), dtype=torch.float32, device=self._device)
        kappa = net.productivity(mags_t, ctx_parents)
        g = net.temporal_density(dt)
        zeta = net.spatial_scale(mags_t, ctx_parents)
        r_km = haversine_km(lat, lon, self._ev_lat, self._ev_lon)
        r_deg = torch.as_tensor(r_km / DEG2KM, dtype=torch.float32, device=self._device)
        f = (0.5 / (np.pi * zeta * zeta)) * torch.pow(1.0 + (r_deg * r_deg) / (zeta * zeta), -1.5)
        return float(torch.sum(kappa * g * f).cpu())

    def _triggering_field(
        self, net, lats: np.ndarray, lons: np.ndarray, mids: np.ndarray, dts: np.ndarray
    ) -> np.ndarray:
        """Horizon-integrated neural triggering at every cell — the vectorized form of looping
        :meth:`_triggering_intensity` over cells × time steps. **Numerically identical**, ~10³× faster.

        The per-cell quadrature is ``Σ_k dt_k · Σ_p κ_p · g(age_p + s_k) · f(r_{cell,p}, ζ_p)``. The
        productivity ``κ_p`` and spatial scale ``ζ_p`` depend only on the parent event (not the target
        cell or the step), and the temporal kernel ``g`` depends only on the step — so we evaluate the
        net ONCE over the parents, integrate the temporal kernel per parent
        (``G_p = Σ_k dt_k · g(age_p + s_k)``), and the cell loop collapses to ``f @ (κ·G)`` over a
        chunked great-circle distance matrix. Swapping the two sums is exact because ``f`` and ``κ`` do
        not depend on the step. Returns ``(n_cells,)`` events/day/deg² · day (the time-integrated rate).
        """
        torch = _import_torch()
        n_cells = len(lats)
        if self._ev_t is None or self._ev_t.size == 0:
            return np.zeros(n_cells, dtype=np.float64)
        ctx_parents = self._parent_context(net)
        mags_t = torch.as_tensor(self._ev_m, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            kappa = net.productivity(mags_t, ctx_parents).cpu().numpy().astype(np.float64)  # (P,)
            zeta = net.spatial_scale(mags_t, ctx_parents).cpu().numpy().astype(np.float64)  # (P,)
            # Time-integrate the temporal kernel per parent: G_p = Σ_k dt_k · g(age_p + s_k).
            g_int = np.zeros(self._ev_t.size, dtype=np.float64)
            for s, w in zip(mids, dts):
                dt = torch.as_tensor(self._ev_t + float(s), dtype=torch.float32, device=self._device)
                g_int += float(w) * net.temporal_density(dt).cpu().numpy().astype(np.float64)
        # Per-parent constants in float32 — the (chunk × P) kernel is summed to a per-cell scalar, so
        # single precision is ample and HALVES the peak memory vs float64. (P can be tens of thousands.)
        amp = (kappa * g_int).astype(np.float32)             # (P,) per-parent amplitude κ_p·G_p
        inv_zeta2 = (1.0 / (zeta * zeta)).astype(np.float32)  # (P,)
        norm = (0.5 / (np.pi * zeta * zeta)).astype(np.float32)  # (P,) 2-D Cauchy-like normaliser
        ev_lat = np.asarray(self._ev_lat, dtype=np.float32)
        ev_lon = np.asarray(self._ev_lon, dtype=np.float32)
        out = np.zeros(n_cells, dtype=np.float64)
        inv_deg = np.float32(1.0 / DEG2KM)
        # Small chunk so the transient (chunk × P) float32 buffers stay ~tens of MB even at P~35k
        # (the old 1024-cell × float64 chunk peaked near a gigabyte and OOM'd on large global catalogs);
        # intermediates are freed each chunk so memory cannot accumulate across cells or scoring windows.
        chunk = 256
        for a in range(0, n_cells, chunk):
            cl = np.asarray(lats[a : a + chunk], dtype=np.float32)[:, None]   # (c,1)
            clo = np.asarray(lons[a : a + chunk], dtype=np.float32)[:, None]  # (c,1)
            r_km = haversine_km(cl, clo, ev_lat, ev_lon).astype(np.float32)   # (c, P) f32
            r2_deg = (r_km * inv_deg) ** 2                                    # (c, P)
            f = norm[None, :] * np.power(1.0 + r2_deg * inv_zeta2[None, :], np.float32(-1.5))
            out[a : a + chunk] = (f @ amp).astype(np.float64)
            del cl, clo, r_km, r2_deg, f
        return out

    # ── context / patch helpers ───────────────────────────────────────────────
    def _parent_context(self, net):
        """Encode (and cache) the per-parent context embeddings — used by the triggering sum."""
        torch = _import_torch()
        if getattr(self, "_ctx_parents_cache", None) is not None:
            return self._ctx_parents_cache
        patches = self._event_patches(self._ev_lat, self._ev_lon)
        patches_t = torch.as_tensor(patches, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            ctx = net.encode(patches_t)
        self._ctx_parents_cache = ctx
        return ctx

    def _event_patches(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Extract a ``(N, C, P, P)`` stack of local covariate patches centred on each ``(lat, lon)``.

        ``P = 2 * patch_radius + 1``. Out-of-grid borders are edge-padded (replicate) so coastal/edge
        cells still get a full patch. This is the multi-channel "image" the CNN context encoder
        ingests — the spatial-context input that conditions the intensity.
        """
        fld = self._field
        assert fld is not None
        rad = self.config.patch_radius
        size = 2 * rad + 1
        C = fld.n_channels
        n = len(lats)
        out = np.zeros((n, C, size, size), dtype=np.float32)
        padded = np.pad(
            fld.data, ((0, 0), (rad, rad), (rad, rad)), mode="edge"
        )  # (C, H+2r, W+2r)
        for k in range(n):
            i, j = fld.cell_index(float(lats[k]), float(lons[k]))
            out[k] = padded[:, i : i + size, j : j + size]
        return out

    def _event_smoothed_lograte(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """The ``smoothed_seismicity_lograte`` channel value at each point (the background conditioner).

        Read straight from the standardized covariate field so it matches the CNN's channel scaling;
        falls back to the live smoothed-seismicity field if the channel is absent.
        """
        fld = self._field
        assert fld is not None
        if "smoothed_seismicity_lograte" in fld.channels:
            ch = fld.channels.index("smoothed_seismicity_lograte")
            return np.array(
                [fld.data[ch, *fld.cell_index(float(la), float(lo))] for la, lo in zip(lats, lons)],
                dtype=np.float32,
            )
        if self.background is not None:
            return np.array(
                [np.log10(max(self.background.background_rate_density(la, lo), 1e-12))
                 for la, lo in zip(lats, lons)],
                dtype=np.float32,
            )
        return np.zeros(len(lats), dtype=np.float32)

    # ── geometry helpers (shared shape with ETAS) ─────────────────────────────
    def _cell_area_deg2(self, cells: list[Cell]) -> float:
        if len(cells) >= 2:
            lats = np.array([c.lat for c in cells])
            pitch = np.median(np.diff(np.unique(np.round(lats, 4))))
            if not np.isfinite(pitch) or pitch <= 0:
                pitch = 0.1
        else:
            pitch = 0.1
        mean_lat = float(np.mean([c.lat for c in cells])) if cells else 0.0
        return float(pitch * pitch * np.cos(np.radians(mean_lat)))

    def _region_area_deg2(self) -> float:
        bb = self._region.bbox
        mean_lat = 0.5 * (bb.lat_min + bb.lat_max)
        return float((bb.lat_max - bb.lat_min) * (bb.lon_max - bb.lon_min) * np.cos(np.radians(mean_lat)))

    def _resolve_device(self) -> str:
        torch = _import_torch()
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ── checkpointing (outside git) ───────────────────────────────────────────
    def _save_checkpoint(self, region: Region) -> str | None:
        """Persist weights + covariate stats to ``checkpoint_dir`` (outside git). Returns the path.

        Weights are never versioned (the repo .gitignore excludes ``data/``/weights); they are
        rebuildable from configs + the global catalog + this code. The covariate standardization stats
        and channel order are saved alongside so inference reapplies the exact transform.
        """
        if self.checkpoint_dir is None or self._net is None:
            return None
        torch = _import_torch()
        from ..config import REPO_ROOT

        out_dir = Path(REPO_ROOT) / self.checkpoint_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = self._t_issue.strftime("%Y%m%dT%H%M%SZ") if self._t_issue else "manual"
        path = out_dir / f"context_tpp_{region.id}_{stamp}.pt"
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "config": self.config.__dict__,
                "mc": self._mc,
                "b_prior": self._b_prior,
                "channels": list(self._field.channels) if self._field else [],
                "field_mean": self._field_mean.tolist() if self._field_mean is not None else None,
                "field_std": self._field_std.tolist() if self._field_std is not None else None,
                "version": self.version,
            },
            path,
        )
        # A tiny side-car JSON for human/manifest inspection without loading torch.
        meta = path.with_suffix(".json")
        meta.write_text(
            json.dumps(
                {
                    "model": self.name,
                    "version": self.version,
                    "region": region.id,
                    "t_issue": stamp,
                    "mc": self._mc,
                    "b_prior": self._b_prior,
                    "channels": list(self._field.channels) if self._field else [],
                    "missing_channels": list(self._field.missing_channels) if self._field else [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(path)

    def _require_fit(self) -> None:
        if self._net is None or self._ev_t is None:
            raise RuntimeError("ContextTPPForecaster.fit() must be called before use")


# ─────────────────────────────────────────────────────────────────────────────
# train(...) entrypoint — callable from model/train.py and cli.py
# ─────────────────────────────────────────────────────────────────────────────


def train(
    *,
    region: Region | str = "global",
    catalog: pd.DataFrame | None = None,
    t_issue: pd.Timestamp | str | None = None,
    config: ContextTPPConfig | None = None,
    covariate_provider: CovariateFieldProvider | None = None,
    holdout_days: float = 30.0,
    gate_against_etas: bool = True,
) -> dict[str, Any]:
    """Train the context-conditioned neural TPP and **gate it against ETAS** in a CSEP check.

    This is the entrypoint ``model/train.py`` and ``cli.py`` call for the neural challenger. It trains
    on the global catalog before ``t_issue`` and then runs the *gate*: the challenger may be promoted
    toward the public field **only if** it shows positive information gain over ETAS on a held-out tail
    window (the real test is the prospective back-analysis harness; this is the fast in-loop gate).

    Parameters
    ----------
    region:
        Region or id. Default ``"global"`` — the thesis trains worldwide; any country is a view.
    catalog:
        In-memory cleaned global catalog (skips the store load; used by tests/offline).
    t_issue:
        Forecast-clock cutoff. Models see only ``time < t_issue``; the gate scores
        ``[t_issue, t_issue + holdout_days)``. Defaults to ``holdout_days`` before the last event.
    config:
        :class:`ContextTPPConfig` overrides.
    covariate_provider:
        Loader for the gridded global covariate field (Slab2/faults/plates/GNSS/stress/tides). If
        ``None`` the model trains on smoothed-seismicity context only (honest cold-start).
    holdout_days:
        Length of the held-out tail scored by the gate.
    gate_against_etas:
        Run the ETAS comparison gate (default on). When off, only the challenger is trained (for
        weight warm-up / debugging) and ``gate_passed`` is ``None``.

    Returns
    -------
    dict
        Training history + the gate outcome (challenger vs ETAS IGPE in nats, N-test, ``gate_passed``,
        and the honest note that no NPP has beaten ETAS prospectively as of 2026).

    Notes
    -----
    Requires torch. The ETAS core does not — if torch is missing this raises a clear error and the
    public field keeps using ETAS. Skill is established only by the prospective back-analysis
    (:mod:`caos_seismic.eval.backanalysis`), never by this in-loop gate alone.
    """
    from ..config import load_region
    from ..inference.clock import conditioning_slice, target_slice
    from .etas import ETASForecaster, ETASStabilityError

    reg = load_region(region) if isinstance(region, str) else region
    cfg = config or ContextTPPConfig()

    if catalog is None:
        from ..data.clean import load_clean_catalog

        catalog = load_clean_catalog(reg)
    validate_catalog(catalog)
    full = catalog.sort_values("time").reset_index(drop=True)
    if len(full) < 10:
        raise ValueError(
            f"catalog for region '{reg.id}' has too few events ({len(full)}) to train the neural TPP"
        )

    # Forecast-clock split.
    if t_issue is None:
        last = pd.to_datetime(full["time"], utc=True).max()
        cutoff = last - pd.Timedelta(days=float(holdout_days))
    else:
        cutoff = pd.Timestamp(t_issue)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")

    past = conditioning_slice(full, cutoff)
    if len(past) < 5:
        raise ValueError(
            f"only {len(past)} events before the cutoff {cutoff.isoformat()}; move the cutoff later"
        )

    challenger = ContextTPPForecaster(config=cfg, covariate_provider=covariate_provider)
    challenger.fit(past, reg, cutoff)

    result: dict[str, Any] = {
        "region": reg.id,
        "model": challenger.name,
        "version": challenger.version,
        "t_issue": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_train": int(len(past)),
        "history": challenger._history,
        "params_used": challenger.params_used,
        "gate_passed": None,
        "honest_note": (
            "Neural challenger trained. As of 2026 no neural point process has robustly beaten a "
            "well-fit ETAS in prospective CSEP (EarthquakeNPP 2026). This is R&D measured honestly; "
            "the challenger reaches the public field only via eval.backanalysis, not this in-loop gate."
        ),
    }

    if not gate_against_etas:
        return result

    # In-loop gate: challenger vs ETAS information gain on the held-out tail.
    from ..eval import csep
    from ..inference.daily import build_fit_cells, build_global_fit_cells
    from ..config import load

    # Score on the GLOBAL coarse multi-resolution grid for the whole-Earth window: the dense 0.1° world
    # grid is ~6.5M cells (never materialized). A spatially-bounded region uses its fine grid.
    grid_cfg = load("grid")
    is_global = reg.id == "global"
    cells = build_global_fit_cells(reg, grid_cfg, past) if is_global else build_fit_cells(reg, grid_cfg)
    if not cells:
        result["gate_note"] = "empty fit grid; gate skipped"
        return result

    m_star = float(min(load("forecast").get("magnitude_thresholds", [5.0])))
    horizon = float(holdout_days)

    # ETAS reference (the floor to beat). At global scope use the regime-TILED ETAS — a single monolithic
    # ETAS over the worldwide 10^5-event catalog is O(N^2) AND physically wrong (subduction != stable
    # interior); a bounded region uses one ETAS. If it cannot fit, the challenger has no honest baseline.
    etas_ok = False
    lam_etas = None
    try:
        if is_global:
            from .tiled import TiledForecaster

            etas = TiledForecaster(
                m0=challenger._mc, mc=challenger._mc, b_value=challenger._b_prior
            )
        else:
            etas = ETASForecaster(mc=challenger._mc, b_value=challenger._b_prior)
        etas.fit(past, reg, cutoff)
        lam_etas = np.asarray(etas.expected_counts(reg, cells, horizon, m_star, cutoff), float)
        etas_ok = True
    except (ETASStabilityError, ValueError) as exc:
        result["gate_note"] = f"ETAS reference unavailable ({exc}); gate inconclusive"

    lam_chal = np.asarray(challenger.expected_counts(reg, cells, horizon, m_star, cutoff), float)

    target = target_slice(full, cutoff, horizon)
    observed = target.loc[pd.to_numeric(target["mw"], errors="coerce") >= m_star - 1e-9]
    omega = _bin_counts_to_cells(observed, cells)
    n_obs = int(len(observed))

    n_test_chal = csep.n_test_poisson(float(lam_chal.sum()), n_obs)
    result["n_test_challenger"] = {"passed": bool(n_test_chal.passed), "quantile": n_test_chal.quantile}
    result["n_forecast_challenger"] = round(float(lam_chal.sum()), 6)
    result["n_observed_holdout"] = n_obs

    if etas_ok and lam_etas is not None:
        igpe, _ = csep.information_gain_per_earthquake(lam_chal, lam_etas, omega)
        result["igpe_vs_etas_nats"] = round(float(igpe), 6)
        result["n_forecast_etas"] = round(float(lam_etas.sum()), 6)
        # Gate: positive IGPE over ETAS AND a passing N-test. (The CI/T-test lives in back-analysis.)
        result["gate_passed"] = bool(igpe > 0.0 and n_test_chal.passed) if n_obs > 0 else None

        # SHAPE-only gate. The trained NLL fixes the intensity SHAPE but not its absolute LEVEL: the
        # forecast-grid integration in expected_counts is mis-normalized vs the (Monte-Carlo) training
        # compensator, so `igpe_vs_etas_nats` is dominated by the rate-normalization term, not skill.
        # Renormalize the challenger field to the ETAS total before scoring — this isolates the
        # spatial/temporal CONTEXT contribution (does the field place events BETTER than ETAS, given the
        # same count?) from the separate, still-open absolute-rate calibration problem. THIS is the number
        # that answers "does the (geodetic) context improve the forecast shape".
        chal_sum = float(lam_chal.sum())
        if chal_sum > 0.0:
            lam_chal_shape = lam_chal * (float(lam_etas.sum()) / chal_sum)
            igpe_shape, _ = csep.information_gain_per_earthquake(lam_chal_shape, lam_etas, omega)
            result["igpe_vs_etas_shape_nats"] = round(float(igpe_shape), 6)
            result["gate_passed_shape"] = bool(igpe_shape > 0.0) if n_obs > 0 else None
            result["calibration_note"] = (
                "igpe_vs_etas_nats reflects the open absolute-rate calibration bug (the neural "
                "over-/under-integrates); igpe_vs_etas_shape_nats renormalizes the field to the ETAS "
                "total and measures the context's spatial/temporal contribution net of that bug."
            )

    return result


def _bin_counts_to_cells(observed: pd.DataFrame, cells: list[Cell]) -> np.ndarray:
    """Count observed target events into the nearest fit cell (aligned to ``cells``) for the IGPE gate."""
    omega = np.zeros(len(cells), dtype=float)
    if observed.empty or not cells:
        return omega
    lats = np.array([c.lat for c in cells])
    lons = np.array([c.lon for c in cells])
    for _, ev in observed.iterrows():
        d2 = (lats - float(ev["latitude"])) ** 2 + (lons - float(ev["longitude"])) ** 2
        omega[int(np.argmin(d2))] += 1.0
    return omega

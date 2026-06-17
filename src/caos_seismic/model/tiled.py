"""Tiled, regime-aware forecaster — fit ETAS (or any per-tile Forecaster) per tile, aggregate globally.

This is the adapter that makes the conditional models tractable + meaningful at **global** scope while
keeping the :class:`~caos_seismic.contracts.Forecaster` contract intact. Instead of one global ETAS
(an ``O(N^2)`` triggering sum over a ``10^5``-``10^6``-event worldwide catalog, *and* the physically
wrong assumption that a subduction megathrust shares parameters with a stable interior), it:

1. partitions the target region into spatial **tiles** (:func:`caos_seismic.model.regime.iterate_tiles`,
   each with a halo so triggering is edge-correct);
2. assigns each tile its **dominant tectonic regime** and uses that regime's prior
   (:func:`~caos_seismic.model.regime.regime_prior`) to regularize a thin tile toward the worldwide
   behaviour of its regime (empirical-Bayes pooling / "borrow strength spatially", model-design §8);
3. **fits a per-tile** ETAS (and its smoothed-seismicity background) on only that tile's halo events,
   enforcing **both** ETAS stability gates per tile (``alpha < beta`` and branching ratio ``n < 1``);
   if a tile's ETAS fails a gate or is too thin, that tile **falls back** to its smoothed-seismicity
   null (never silently publishing a supercritical intensity);
4. answers :meth:`expected_counts` for any global cell by routing the cell to the tile that **owns**
   it (its centre lies in that tile's interior) and evaluating that tile's fitted model — then the
   per-tile answers concatenate into one **global field** over the requested cells.

The aggregate stays the calibrated **reference** the neural challenger must beat: it is exactly the
same ETAS mathematics, just fit locally and stitched, so the global IGPE comparison is apples-to-apples.

Only core deps are needed at import time. The per-tile ETAS lazily imports SciPy for its MLE (as
:mod:`caos_seismic.model.etas` already does); nothing heavy is imported here at module top level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import BaseForecaster, Cell, Region, validate_catalog
from ._common import poisson_p_at_least_one
from .etas import ETASForecaster, ETASStabilityError
from .regime import (
    Tile,
    TectonicRegime,
    dominant_regime,
    iterate_tiles,
    regime_prior,
)
from .smoothed import SmoothedSeismicityForecaster

logger = logging.getLogger(__name__)


@dataclass
class TileFit:
    """The fitted state of one tile: its regime, the model that carries it, and why (for the manifest)."""

    tile: Tile
    regime: TectonicRegime
    model: BaseForecaster              # the forecaster that answers cells for this tile (ETAS or null)
    is_etas: bool                      # True if ETAS fit cleanly; False → smoothed-null fallback
    n_events: int                      # events in the tile halo used to fit
    rejection: str | None = None       # why ETAS was rejected (stability gate / thin), if it was
    branching_ratio: float | None = None
    is_fit: bool = True                # False if even the smoothed null could not fit (degenerate tile)


@dataclass
class TiledForecaster(BaseForecaster):
    """Regime-aware, tile-fit ETAS aggregated into a global conditional field.

    Implements the :class:`~caos_seismic.contracts.Forecaster` port: :meth:`fit` conditions every
    tile on its lawful-past halo events; :meth:`expected_counts` routes each requested cell to the
    tile that owns it and returns that tile's expected count. The result is a single global field with
    the **same** per-cell semantics (``N_{>=M*}`` over ``[t_issue, t_issue+horizon)``) as the
    monolithic :class:`~caos_seismic.model.etas.ETASForecaster`, so it slots straight into the daily
    inference driver and the CSEP harness.

    Parameters
    ----------
    tile_deg, halo_deg:
        Tile geometry (see :func:`caos_seismic.model.regime.iterate_tiles`). ``tile_deg`` bounds the
        per-fit catalog size; ``halo_deg`` (≈ aftershock-zone radius) keeps triggering continuous
        across tile edges.
    m0:
        Reference magnitude for the per-tile ETAS productivity/spatial exponents.
    mc, b_value:
        Optional fixed completeness / Gutenberg-Richter ``b`` passed to every tile. ``None`` → each
        tile estimates them on its own halo (genuinely local, never hard-coded).
    require_alpha_lt_beta, reject_supercritical:
        The two ETAS stability gates, enforced **per tile** (a tile that violates either falls back to
        its smoothed null rather than poisoning the global field).
    min_events_for_etas:
        A tile with fewer than this many halo events skips the ETAS MLE and uses its smoothed null
        directly (a thin tile cannot support a seven-parameter fit; the regime prior + null carry it).
    use_regime_priors:
        When True (default), each tile's smoothed-null neighbour count is taken from its dominant
        regime's prior (dense interface zones sharpen, sparse interiors broaden); ETAS still MLE-fits
        but starts from a regime-appropriate point.
    root:
        Repo-root override for the regime enrichers (tests); defaults to the package
        :data:`~caos_seismic.config.REPO_ROOT`.
    """

    name: str = "tiled_etas"
    version: str = "0.1.0"

    tile_deg: float = 10.0
    halo_deg: float = 1.0
    m0: float = 3.5
    mc: float | None = None
    b_value: float | None = None
    require_alpha_lt_beta: bool = True
    reject_supercritical: bool = True
    min_events_for_etas: int = 30
    use_regime_priors: bool = True
    root: Path | None = None

    # ── Fitted state ─────────────────────────────────────────────────────────
    _tiles: list[TileFit] = field(default_factory=list, repr=False)
    _region: Region | None = field(default=None, repr=False)
    _t_issue: pd.Timestamp | None = field(default=None, repr=False)
    params_used: dict = field(default_factory=dict, repr=False)

    # ── Forecaster.fit ───────────────────────────────────────────────────────
    def fit(self, catalog: pd.DataFrame, region: Region, t_issue: pd.Timestamp) -> "TiledForecaster":
        """Fit one model per tile on its halo events; aggregate into the global field.

        For each tile: slice the lawful past to the tile halo, assign the dominant regime, fit the
        per-tile smoothed-seismicity null (always — it is the fallback and the ETAS background), then
        attempt the per-tile ETAS MLE with both stability gates. A gate failure or a thin tile keeps
        the smoothed null for that tile. Tiles with no halo events are skipped (they contribute a zero
        field, floored later to the region background by the inference driver).
        """
        validate_catalog(catalog)
        self._region = region
        self._t_issue = pd.Timestamp(t_issue)

        df = catalog.loc[catalog["time"] < self._t_issue].copy()
        if df.empty:
            raise ValueError("no events strictly before t_issue to fit the tiled forecaster")
        df = df.sort_values("time")

        lat = df["latitude"].to_numpy(dtype=float)
        lon = df["longitude"].to_numpy(dtype=float)
        depth = (
            pd.to_numeric(df["depth_km"], errors="coerce").to_numpy(dtype=float)
            if "depth_km" in df.columns
            else None
        )

        self._tiles = []
        n_etas, n_null, n_skipped = 0, 0, 0
        for tile in iterate_tiles(region, tile_deg=self.tile_deg, halo_deg=self.halo_deg):
            halo_df = self._slice_to_halo(df, lat, lon, tile)
            if halo_df.empty:
                n_skipped += 1
                continue
            regime = dominant_regime(
                tile, halo_df["latitude"].to_numpy(float), halo_df["longitude"].to_numpy(float),
                halo_df["depth_km"].to_numpy(float) if "depth_km" in halo_df.columns else None,
                root=self.root,
            )
            tile_fit = self._fit_one_tile(tile, halo_df, region, regime)
            self._tiles.append(tile_fit)
            if tile_fit.is_etas:
                n_etas += 1
            else:
                n_null += 1

        if not self._tiles:
            raise ValueError("no non-empty tiles to fit — catalog has no events in the region halo")

        self.params_used = {
            "tiling": {"tile_deg": self.tile_deg, "halo_deg": self.halo_deg},
            "n_tiles_fit": len(self._tiles),
            "n_tiles_etas": n_etas,
            "n_tiles_null_fallback": n_null,
            "n_tiles_empty_skipped": n_skipped,
            "n_tiles_unfit_degenerate": sum(1 for tf in self._tiles if not tf.is_fit),
            "gates": {
                "require_alpha_lt_beta": self.require_alpha_lt_beta,
                "reject_supercritical": self.reject_supercritical,
            },
            "regimes": {
                tf.tile.id: {
                    "regime": tf.regime.value,
                    "model": tf.model.name,
                    "is_etas": tf.is_etas,
                    "n_events": tf.n_events,
                    "branching_ratio": tf.branching_ratio,
                    "rejection": tf.rejection,
                }
                for tf in self._tiles
            },
        }
        return self

    def _slice_to_halo(
        self, df: pd.DataFrame, lat: np.ndarray, lon: np.ndarray, tile: Tile
    ) -> pd.DataFrame:
        """Events (rows of ``df``) whose epicentres fall inside ``tile``'s fitting halo.

        Longitudes are compared on both the [-180,180] and the halo's possibly-overflowing frame so a
        tile that straddles the antimeridian still collects its events.
        """
        b = tile.halo
        lon180 = ((lon + 180.0) % 360.0) - 180.0
        in_lat = (lat >= b.lat_min) & (lat <= b.lat_max)
        # Accept either the raw longitude or its wrapped form inside the halo bounds.
        in_lon = ((lon >= b.lon_min) & (lon <= b.lon_max)) | (
            (lon180 >= b.lon_min) & (lon180 <= b.lon_max)
        )
        return df.loc[in_lat & in_lon].copy()

    def _fit_one_tile(
        self, tile: Tile, halo_df: pd.DataFrame, region: Region, regime: TectonicRegime
    ) -> TileFit:
        """Fit the smoothed null + (if data allow) ETAS for one tile; return the carrying model.

        The smoothed-seismicity background is always fit (it is both the ETAS ``mu(x,y)`` and the
        fallback). The regime prior sets the null's adaptive-kernel neighbour count and the ETAS
        productivity/scaling start. ETAS is attempted only if the tile clears ``min_events_for_etas``;
        a stability-gate violation or a degenerate fit keeps the null for the tile. The owning
        ``tile`` is carried in the returned :class:`TileFit` so :meth:`expected_counts` can route
        cells to it by interior ownership.
        """
        prior = regime_prior(regime)
        n_events = int(len(halo_df))

        # Build a tile from the halo bbox so the per-tile models see a region scoped to the tile (its
        # m_max bounds the GR exceedance integral exactly as the parent region does).
        tile_region = Region(
            id=f"{region.id}:{regime.value}",
            name_en=region.name_en,
            name_es=region.name_es,
            bbox=region.bbox,  # cells are routed by the parent grid; the bbox is only used for m_max scope
            m_max=region.m_max,
            attribution=region.attribution,
        )

        n_neighbors = prior.n_neighbors if self.use_regime_priors else 6
        null = SmoothedSeismicityForecaster(
            n_neighbors=n_neighbors, b_value=self.b_value, mc=self.mc
        )
        try:
            null.fit(halo_df, tile_region, self._t_issue)
        except ValueError as exc:
            # A tile too thin even for the null: keep it but mark it; expected_counts returns ~0 there.
            logger.debug("tile null fit failed (%s); tile contributes no rate", exc)
            return TileFit(
                tile=tile, regime=regime, model=null,
                is_etas=False, n_events=n_events, rejection=f"null-fit-failed: {exc}",
                is_fit=False,
            )

        # ETAS only when the tile can support a seven-parameter fit.
        if n_events < self.min_events_for_etas:
            return TileFit(
                tile=tile, regime=regime, model=null,
                is_etas=False, n_events=n_events,
                rejection=f"thin tile ({n_events} < {self.min_events_for_etas}); null carries it",
            )

        etas = ETASForecaster(
            m0=self.m0,
            mc=self.mc,
            b_value=self.b_value,
            require_alpha_lt_beta=self.require_alpha_lt_beta,
            reject_supercritical=self.reject_supercritical,
            background=SmoothedSeismicityForecaster(
                n_neighbors=n_neighbors, b_value=self.b_value, mc=self.mc
            ),
            regime=regime.value,
            regime_prior=prior if self.use_regime_priors else None,
            # Cheapest per-tile MLE: a single regime-prior-seeded start (the prior is informative, and
            # there are hundreds of tiles). The neighbour cutoffs (etas.max_parent_*) keep each fit O(N·k).
            n_restarts=0,
        )
        try:
            etas.fit(halo_df, tile_region, self._t_issue)
            return TileFit(
                tile=tile, regime=regime, model=etas,
                is_etas=True, n_events=n_events,
                branching_ratio=float(etas.params_used.get("branching_ratio", float("nan"))),
            )
        except (ETASStabilityError, ValueError) as exc:
            logger.info("tile ETAS rejected (%s); smoothed null carries this tile", exc)
            return TileFit(
                tile=tile, regime=regime, model=null,
                is_etas=False, n_events=n_events, rejection=str(exc),
            )

    # ── Forecaster.expected_counts ─────────────────────────────────────────────
    def expected_counts(
        self,
        region: Region,
        cells: list[Cell],
        horizon_days: float,
        m_threshold: float,
        t_issue: pd.Timestamp,
    ) -> list[float]:
        """Expected count ``N_{>=M*}`` per cell, routing each cell to the tile that owns it.

        Each requested cell is assigned to the tile whose **interior** contains its centre (cells on a
        tile boundary go to the nearest tile centre as a tiebreak). All cells owned by one tile are
        evaluated in a single call to that tile's fitted model, so the per-tile expected counts
        concatenate into the global field over ``cells`` (order preserved). Cells that land in no
        fitted tile (e.g. an empty-skipped tile) get ``0.0`` — the inference driver floors those to
        the smoothed-seismicity background (cold-start rule, model-design §8).
        """
        self._require_fit()
        if not self._tiles:
            return [0.0] * len(cells)

        # Group cell indices by the owning tile.
        owner: dict[int, list[int]] = {}
        unowned: list[int] = []
        tile_lookup = self._tile_centres()
        for idx, cell in enumerate(cells):
            t_i = self._owning_tile_index(cell, tile_lookup)
            if t_i is None:
                unowned.append(idx)
            else:
                owner.setdefault(t_i, []).append(idx)

        out = [0.0] * len(cells)
        for t_i, cell_indices in owner.items():
            tf = self._tiles[t_i]
            sub_cells = [cells[i] for i in cell_indices]
            counts = tf.model.expected_counts(
                region, sub_cells, horizon_days, m_threshold, t_issue
            )
            for i, n in zip(cell_indices, counts):
                out[i] = float(max(n, 0.0))
        # `unowned` stays 0.0 — floored to background downstream.
        return out

    def forecast_probabilities(
        self,
        region: Region,
        cells: list[Cell],
        horizon_days: float,
        m_threshold: float,
        t_issue: pd.Timestamp,
    ) -> list[float]:
        """Per-cell ``P(>=1 event >= M*) = 1 - e^{-N}`` (the public exceedance formula), globally."""
        return [
            poisson_p_at_least_one(n)
            for n in self.expected_counts(region, cells, horizon_days, m_threshold, t_issue)
        ]

    # ── tile routing ────────────────────────────────────────────────────────
    def _tile_centres(self) -> np.ndarray:
        """``(n_tiles, 2)`` array of interior-bbox centres, for the nearest-tile boundary tiebreak."""
        centres = []
        for tf in self._tiles:
            b = tf.tile.interior
            centres.append(((b.lat_min + b.lat_max) / 2.0, (b.lon_min + b.lon_max) / 2.0))
        return np.asarray(centres, dtype=float)

    def _owning_tile_index(self, cell: Cell, centres: np.ndarray) -> int | None:
        """Index of the fit tile that owns ``cell`` (interior contains it; nearest-centre tiebreak).

        Tiles whose model could not be fit (a degenerate tile where even the smoothed null failed) are
        never returned — their cells fall through to ``unowned`` and stay ``0.0``, to be floored to the
        region background by the inference driver, rather than evaluating an unfit model.
        """
        for i, tf in enumerate(self._tiles):
            if not tf.is_fit:
                continue
            if tf.tile.interior.lat_min <= cell.lat < tf.tile.interior.lat_max and (
                tf.tile.interior.lon_min <= cell.lon < tf.tile.interior.lon_max
            ):
                return i
        # Boundary / out-of-interior cell: snap to the nearest *fit* interior centre (so edge cells of
        # the region bbox still resolve to a fitted tile rather than being dropped).
        fit_idx = [i for i, tf in enumerate(self._tiles) if tf.is_fit]
        if not fit_idx:
            return None
        sub = centres[fit_idx]
        d2 = (sub[:, 0] - cell.lat) ** 2 + (sub[:, 1] - cell.lon) ** 2
        return int(fit_idx[int(np.argmin(d2))])

    def _require_fit(self) -> None:
        if self._region is None or self._t_issue is None:
            raise RuntimeError("TiledForecaster.fit() must be called before use")

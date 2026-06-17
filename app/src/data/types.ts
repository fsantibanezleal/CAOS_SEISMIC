/**
 * CAOS_SEISMIC — TypeScript types mirroring the Python compact-artifact contracts.
 *
 * These types are the front-end half of the contract defined in
 * `src/caos_seismic/contracts.py`. They MUST stay byte-compatible with the JSON the
 * offline daily job writes (`ForecastArtifact.model_dump_compact()` →
 * `results/forecast-YYYY-MM-DD.json(.gz)`), so the static SPA can render it with zero
 * backend compute.
 *
 * Honest-framing invariant carried into the type layer (see web-app-spec.md §3, §7.3):
 * every published number is a CONDITIONAL PROBABILITY in (0, 1) — scoped to
 * region × magnitude threshold × horizon — shown next to its long-term `baseline`,
 * with `lo`/`hi` uncertainty bounds, evaluated CSEP-style. Never an alarm, never a
 * prediction, never a "safe" state. The shape below makes those companions mandatory
 * (every cell carries `baseline`, `lo`, `hi`).
 *
 * Source of truth: src/caos_seismic/contracts.py (ARTIFACT_SCHEMA_VERSION, ForecastArtifact,
 * CalibrationSummary, Staleness, Region, BBox). Do not diverge column / field names.
 */

/** Mirrors `contracts.ARTIFACT_SCHEMA_VERSION` (string, e.g. "1.0"). */
export const ARTIFACT_SCHEMA_VERSION = '1.0' as const;

// ─────────────────────────────────────────────────────────────────────────────
// Geometry / region (mirror BBox + Region)
// ─────────────────────────────────────────────────────────────────────────────

/** Geographic bounding box, WGS84 degrees. Mirrors `contracts.BBox`. */
export interface BBox {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

/** A forecast region. Mirrors `contracts.Region`. */
export interface Region {
  id: string;
  name_en: string;
  name_es: string;
  bbox: BBox;
  /** maximum magnitude bounding the exceedance integral (e.g. Chile 9.5). */
  m_max: number;
  /** attribution strings to display on any public surface using this region's data. */
  attribution: string[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-cell × horizon × threshold forecast value
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The leaf value of the sparse forecast tree:
 * `forecast[cell_key][String(horizon_days)][String(m_threshold)]`.
 *
 * Mirrors the dict written from `contracts.CellForecast`:
 *   p        ← CellForecast.expected   (P(>=1 event >= M*), in (0,1) — the public scalar)
 *   lo       ← CellForecast.lo         (optimistic bound, P10)
 *   hi       ← CellForecast.hi         (pessimistic bound, P90)
 *   rate     ← CellForecast.rate       (expected event count N_{>=M*} = lambda * T)
 *   baseline ← CellForecast.baseline   (long-term Poisson baseline probability, same cell)
 *
 * The compact writer renames `expected` → `p` for size; everything else is verbatim.
 */
export interface CellValue {
  /** P(>=1 event >= M*) median / expected, in (0,1). The headline probability. */
  p: number;
  /** Optimistic bound (P10). */
  lo: number;
  /** Pessimistic bound (P90). */
  hi: number;
  /** Expected event count N_{>=M*} (lambda * T) for the cell/horizon/threshold. On compacted
   *  (production) artifacts this is decoded by the client from the quantized `q` code via the
   *  artifact `rate_legend` at load time (see `decodeRates`). */
  rate: number;
  /** Long-term Poisson baseline probability for the same cell — the honesty companion. */
  baseline: number;
  /** Quantized log-uint16 rate code AS SHIPPED (publish-stage compaction). Decoded into `rate` at
   *  load time; absent on uncompacted (sample) artifacts. */
  q?: number;
}

/**
 * Sparse, three-level forecast tree.
 *
 * Keys are STRINGS (JSON object keys): the horizon and threshold levels are the
 * numeric values stringified (`"1"`, `"2"`, `"7"` for horizons; `"5.0"`, `"6.0"`,
 * `"7.0"` for thresholds), exactly as Python's `str(int)` / `str(float)` emits them.
 * Only cells inside the validated coverage footprint and above the rate floor are
 * present — every absent cell is implicit long-term baseline, NOT "safe".
 *
 * forecast[cellKey][String(horizonDays)][String(mThreshold)] -> CellValue
 */
export type ForecastTree = Record<string, Record<string, Record<string, CellValue>>>;

// ─────────────────────────────────────────────────────────────────────────────
// Calibration summary (mirror CalibrationSummary)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * One reliability-diagram point: `[forecast_prob, observed_freq, n]` per horizon-bin.
 * Tuple order matches `CalibrationSummary.reliability` exactly.
 */
export type ReliabilityPoint = [forecastProb: number, observedFreq: number, n: number];

/**
 * pyCSEP consistency-test summary. Free-form dict on the Python side
 * (`CalibrationSummary.csep`); typed here as the documented quantile scores in [0,1]
 * plus per-test pass flags. Additional keys are tolerated (index signature).
 *
 * N = number test, M = magnitude, S = spatial, L = likelihood, CL = conditional likelihood.
 * The traffic-light triad (green/amber/red) is reserved for MODEL QUALITY only — never
 * for earthquake danger (web-app-spec.md §7.3).
 */
export interface CsepScores {
  /** Quantile scores in [0,1] per consistency test (when available). */
  N?: number;
  M?: number;
  S?: number;
  L?: number;
  CL?: number;
  /** Per-test pass flags (within CSEP consistency vs rejected). */
  pass?: Partial<Record<'N' | 'M' | 'S' | 'L' | 'CL', boolean>>;
  [key: string]: unknown;
}

/** Mirrors `contracts.CalibrationSummary`. */
export interface CalibrationSummary {
  /** `[[forecast_prob, observed_freq, n], ...]` per horizon-bin. May be empty. */
  reliability: ReliabilityPoint[];
  /** Consistency-test quantile scores {N,M,S,L,CL} + pass flags. May be empty. */
  csep: CsepScores;
  /** Information gain per earthquake vs the Poisson null, in NATS (not bits). */
  info_gain_vs_poisson_nats: number | null;
  /** Information gain per earthquake vs the ETAS reference, in NATS (not bits). */
  info_gain_vs_etas_nats: number | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Staleness (mirror Staleness)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Last-run / next-run indicator. Mirrors `contracts.Staleness`.
 * `ok === false` → the UI MUST degrade visibly (banner + desaturation/hatch);
 * a stale or failed artifact is worse than honestly saying "unavailable".
 */
export interface Staleness {
  /** ISO-8601 UTC timestamp the artifact was generated. */
  generated: string;
  /** ISO-8601 UTC timestamp of the next scheduled run. */
  next_run: string;
  /** false → degrade visibly. */
  ok: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid descriptor
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Spatial-discretization descriptor. Mirrors `ForecastArtifact.grid`
 * ("{type:'h3', resolution:int}"). `type` is open-ended (`'h3'` today; `'latlon'`
 * possible for a fine fit grid), `resolution` is the H3 resolution for display cells.
 *
 * Global field (multi-resolution H3): the writer (`inference/artifact.serialize_artifact`) also
 * emits `resolution_world` (the coarse world overview every surfaced cell is at least as coarse as)
 * and `resolution_region` (the per-country drill-down resolution; views may override it). `resolution`
 * is the base display resolution (== `resolution_world` for the global field). All optional so a
 * single-region artifact (one resolution) still validates.
 */
export interface GridDescriptor {
  type: 'h3' | 'latlon' | string;
  resolution: number;
  /** Coarse world-overview H3 resolution (global field). */
  resolution_world?: number;
  /** Per-country / per-region drill-down H3 resolution (global field). */
  resolution_region?: number;
  [key: string]: unknown;
}

// ─────────────────────────────────────────────────────────────────────────────
// View index (mirror ViewIndexEntry) — a country slice of the single global field
// ─────────────────────────────────────────────────────────────────────────────

/**
 * One entry in the artifact's `views` index. Mirrors `contracts.ViewIndexEntry`.
 *
 * Core thesis of the global re-scope (web-app-spec.md §7.1, contracts.View): the model conditions
 * ONE global field; any country is a VIEW into that field, never a separately-trained model. The
 * global artifact ships one `forecast` dict; the SPA's country selector resolves a view to its
 * `cells` (H3 cell keys of the shared field falling in this view's bbox) and reads only those keys.
 * Storing the cell keys (a few hundred per country) is far cheaper than duplicating the per-cell
 * forecast payload per country.
 */
export interface ViewIndexEntry {
  /** Stable view id — ISO-3166 alpha-2/-3 for countries (e.g. "CL", "JP", "US-CA"). */
  id: string;
  name_en: string;
  name_es: string;
  bbox: BBox;
  /** View-local maximum magnitude bounding the exceedance integral (e.g. Chile 9.5, UK 7.0). */
  m_max: number;
  /** Attribution required on any public surface that surfaces this view. */
  attribution: string[];
  /** Optional finer display H3 resolution for this view (overrides the world resolution). */
  h3_resolution?: number | null;
  /** H3 cell keys of the global field falling in this view's bbox (the slice index). */
  cells: string[];
  /** Convenience count (== `cells.length`). */
  n_cells: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// The compact daily artifact (mirror ForecastArtifact)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The single compact JSON the SPA renders. Mirrors `contracts.ForecastArtifact`
 * after `model_dump_compact()` (the writer further H3-bins, quantizes, and gzips).
 *
 * Global re-scope (contracts.ForecastArtifact, web-app-spec.md §7.1): `region` is the GLOBAL
 * field's bounding region (the whole Earth); the per-country drill-downs are carried as `views`
 * (each a lightweight cell-key index into the shared `forecast` dict, NOT a duplicated payload).
 * `grid` may declare a base world resolution plus per-view finer resolutions (multi-resolution H3).
 * Backward compatible: a single-region artifact ships an empty `views` list and a single-resolution
 * `grid`.
 *
 * Field order and names match the Python model. Keep it small: sparse cells, H3 keys.
 */
export interface ForecastArtifact {
  /** == ARTIFACT_SCHEMA_VERSION; bump on any breaking shape change. */
  schema_version: string;
  /** Constant product tag, "CAOS_SEISMIC". */
  product: string;
  /** ISO-8601 UTC issue time (the sealed forecast-clock instant). */
  issued_at: string;
  /** The GLOBAL field's bounding region (whole Earth for the global artifact). */
  region: Region;
  /** Horizons in days, e.g. [1, 2, 7]. Numeric here; STRING keys inside `forecast`. */
  horizons_days: number[];
  /** Magnitude thresholds M*, e.g. [5.0, 6.0, 7.0]. STRING keys inside `forecast`. */
  magnitude_thresholds: number[];
  /** Maximum magnitude bounding the exceedance integral (== region.m_max). */
  m_max: number;
  /** Grid descriptor, e.g. { type: 'h3', resolution: 3, resolution_world: 3, resolution_region: 5 }. */
  grid: GridDescriptor;
  /** Sparse forecast[cell][horizon][threshold] -> {p, lo, hi, rate, baseline}. */
  forecast: ForecastTree;
  /** CSEP / reliability summary — the product's central credibility artifact. */
  calibration: CalibrationSummary;
  /** Cell keys explicitly OUT of validated coverage (hatch in the UI; blank != safe). */
  coverage_mask: string[];
  /**
   * Per-country slices of the global field for the web's country selector (cell-key indices into
   * the shared `forecast` dict). Empty for a single-region artifact.
   */
  views?: ViewIndexEntry[];
  /** Free-form provenance (catalog versions, Mc version, model version, config hash, ...). */
  provenance: Record<string, unknown>;
  /** Generated / next-run / ok staleness indicator. */
  staleness: Staleness;
  /** Quantization legend for the compacted per-cell `q` rate codes (absent on uncompacted artifacts). */
  rate_legend?: RateLegend;
  /** Publish-stage compaction metadata (H3 cell counts, floors). Informational. */
  compaction?: Record<string, unknown>;
}

/**
 * Log-domain uint16 quantization legend for the compacted per-cell `q` rate codes — the publish
 * stage stores the expected-count rate as a 16-bit code to keep the world payload small. Decode:
 * `rate = q <= 0 ? 0 : rate_min * (rate_max / rate_min) ** ((q - 1) / (levels - 1))`.
 */
export interface RateLegend {
  /** Quantization scheme tag, e.g. "log_uint16". */
  kind: string;
  rate_min: number;
  rate_max: number;
  levels: number;
  note?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// index.json — the latest-pointer + rolling-calibration manifest (web-app-spec.md §8.3)
// ─────────────────────────────────────────────────────────────────────────────

/** One country view sliceable out of a (global) artifact — the region-selector menu entry. */
export interface IndexViewEntry {
  id: string;
  name_en: string;
  n_cells: number;
}

/**
 * One forecast entry in the index — both the `latest` pointer and every element of the rolling
 * `forecasts` history share this shape. Mirrors the publish stage's index entry written by
 * `inference/artifact.py: update_index` (data-and-pipelines.md §4 (G), §7).
 */
export interface ForecastIndexEntry {
  /** ISO-8601 UTC issue time, e.g. "2026-06-17T00:00:00Z". */
  issued_at: string;
  /** Region id this artifact is for ("global" or a focused region). */
  region: string;
  /** Relative path under the static data host, e.g. "forecast-global-2026-06-17.json.gz". */
  file: string;
  horizons_days: number[];
  magnitude_thresholds: number[];
  /** Compressed payload size on disk. */
  size_bytes?: number;
  /** Whether the artifact passed the staleness gate when it was published. */
  staleness_ok?: boolean;
  /** ISO-8601 UTC time the artifact's underlying data was generated. */
  generated?: string;
  /** ISO-8601 UTC time the next scheduled run is expected. */
  next_run?: string;
  /** The country views available as slices of this field (the SPA's region selector). */
  views?: IndexViewEntry[];
}

/**
 * `data/index.json` — the latest pointer + rolling history + CSEP calibration the client reads
 * first. Mirrors the publish stage's `results/index.json` (data-and-pipelines.md §4 (G), §7).
 * `latest` is the most recent artifact entry; `forecasts` is the rolling list (oldest→newest) for
 * the "forecast from {past date}" selector; `latest_by_region` is the per-region newest entry.
 */
export interface ForecastIndex {
  /** == ARTIFACT_SCHEMA_VERSION of the artifacts this index points at. */
  schema_version: string;
  product: string;
  /** ISO-8601 UTC time this index was written. */
  updated_at: string;
  /** The most recent artifact entry (the default Monitoring view points here). */
  latest: ForecastIndexEntry;
  /** Most recent entry per region id. */
  latest_by_region?: Record<string, ForecastIndexEntry>;
  /** Rolling history of daily artifacts, oldest→newest (bounded window). */
  forecasts: ForecastIndexEntry[];
  /** Optional rolling-window CSEP calibration snapshot for the always-on badge. */
  calibration?: CalibrationSummary;
}

// ─────────────────────────────────────────────────────────────────────────────
// Selector helper types (used by client.ts selectors)
// ─────────────────────────────────────────────────────────────────────────────

/** The three uncertainty bounds (the honesty triad, web-app-spec.md §7.3). */
export type Bound = 'lo' | 'expected' | 'hi';

/** Map `Bound` to the `CellValue` field it reads. */
export const BOUND_FIELD: Record<Bound, keyof CellValue> = {
  lo: 'lo',
  expected: 'p',
  hi: 'hi',
};

/**
 * The id the country selector uses for the default WORLD view — the whole global field, NOT a slice.
 * It is not present in `artifact.views`; selecting it surfaces every cell. Any other value is a
 * `ViewIndexEntry.id` (a country slice).
 */
export const WORLD_VIEW_ID = 'world' as const;

/** A flattened per-cell selection for one (horizon, threshold, bound) slice. */
export interface CellSelection {
  /** Cell key (H3 index for display, or "lat,lon" for a fine fit grid). */
  cell: string;
  /** The chosen bound's probability value, in (0,1). */
  value: number;
  /** P(>=1 event >= M*) median / expected. */
  p: number;
  lo: number;
  hi: number;
  /** Expected event count N_{>=M*}. */
  rate: number;
  /** Long-term Poisson baseline probability (the honesty companion). */
  baseline: number;
  /** rate / baseline ratio surrogate at the probability level: p / baseline (NaN-safe). */
  ratioToBaseline: number;
}

/**
 * CAOS_SEISMIC — static data client.
 *
 * The SPA is STATIC-FIRST, stateless, read-only (web-app-spec.md §8.3): there is NO
 * backend call here. The client reads two static assets from the data host:
 *   1. `<base>/data/index.json`            — the latest pointer + rolling history
 *   2. `<base>/data/forecast-<date>.json`  — the compact daily artifact (optionally `.gz`)
 *
 * Both are produced offline by the daily git-as-data publish job
 * (data-and-pipelines.md §4 (G)). Until real artifacts exist, the bundled SAMPLE
 * artifact in `public/data/` is served by the same code path — the mock lives ONLY at
 * the data boundary, behind this real interface; swapping in real artifacts requires no
 * client change.
 *
 * Decoding: artifacts may be served either as plain JSON (host applies
 * Content-Encoding: gzip transparently) OR as a raw `.json.gz` blob (static hosts that
 * do not negotiate gzip for `.gz` files). This client handles BOTH — if the bytes look
 * gzip-compressed (magic 0x1f 0x8b) it inflates them via DecompressionStream before
 * parsing, otherwise it parses the text directly.
 *
 * Selectors expose the artifact by horizon, by threshold, and by bound (lo / expected /
 * hi — the honesty triad), and compute the mandatory ratio-to-baseline companion. They
 * never invent values: a missing cell is implicit long-term baseline, not "safe".
 */

import {
  BOUND_FIELD,
  WORLD_VIEW_ID,
  type BBox,
  type Bound,
  type CalibrationSummary,
  type CellSelection,
  type CellValue,
  type ForecastArtifact,
  type ForecastIndex,
  type Staleness,
  type ViewIndexEntry,
} from './types';

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

export interface ClientConfig {
  /**
   * Base URL under which `data/index.json` and `data/forecast-*.json(.gz)` live.
   * Defaults to the app's deploy base (Vite `import.meta.env.BASE_URL`) so the client
   * works under a sub-path (e.g. GitHub Pages `/CAOS_SEISMIC/`). No trailing slash needed.
   */
  baseUrl?: string;
  /** Sub-directory holding the static data files. Defaults to "data". */
  dataDir?: string;
  /** Optional `fetch` override (tests / SSR). Defaults to the global `fetch`. */
  fetchImpl?: typeof fetch;
}

/** Resolve the Vite base path without hard-importing `import.meta` in non-Vite test envs. */
function defaultBaseUrl(): string {
  try {
    // `import.meta.env.BASE_URL` is replaced at build time by Vite; guard for non-Vite.
    const env = (import.meta as unknown as { env?: { BASE_URL?: string } }).env;
    if (env?.BASE_URL) return env.BASE_URL;
  } catch {
    /* import.meta not available (e.g. CommonJS test runner) — fall through */
  }
  return '/';
}

function joinUrl(base: string, ...parts: string[]): string {
  const trimmedBase = base.replace(/\/+$/, '');
  const tail = parts
    .map((p) => p.replace(/^\/+|\/+$/g, ''))
    .filter(Boolean)
    .join('/');
  return tail ? `${trimmedBase}/${tail}` : trimmedBase;
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-level fetch + gzip-aware decode
// ─────────────────────────────────────────────────────────────────────────────

/** gzip member magic bytes (RFC 1952): 0x1f 0x8b. */
function looksGzipped(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

/**
 * Inflate a gzip-compressed buffer using the platform DecompressionStream
 * (Chrome 80+/Firefox 113+/Safari 16.4+). Throws a clear error if unavailable so the
 * caller can surface "your browser cannot decode the compressed forecast" rather than a
 * cryptic parse failure.
 */
async function gunzip(buffer: ArrayBuffer): Promise<string> {
  const DS = (globalThis as { DecompressionStream?: typeof DecompressionStream }).DecompressionStream;
  if (!DS) {
    throw new Error(
      'This browser cannot decode a raw .gz forecast (DecompressionStream unavailable). ' +
        'Serve the artifact as plain .json with Content-Encoding: gzip, or use a modern browser.',
    );
  }
  const stream = new Blob([buffer]).stream().pipeThrough(new DS('gzip'));
  return await new Response(stream).text();
}

/**
 * Fetch a static JSON asset, transparently inflating it if the response body is a raw
 * gzip blob. Returns the parsed object typed as `T`.
 */
async function fetchJson<T>(url: string, fetchImpl: typeof fetch): Promise<T> {
  const res = await fetchImpl(url, { method: 'GET', cache: 'no-cache' });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }

  const isGzExt = /\.gz($|\?)/.test(url);
  // If the host already decoded gzip (Content-Encoding), res.json() works directly.
  // We only need manual inflation when the *body* is still a raw gzip member.
  const buffer = await res.arrayBuffer();
  const head = new Uint8Array(buffer.slice(0, 2));

  let text: string;
  if ((isGzExt || looksGzipped(head)) && looksGzipped(head)) {
    text = await gunzip(buffer);
  } else {
    text = new TextDecoder('utf-8').decode(buffer);
  }
  return JSON.parse(text) as T;
}

// ─────────────────────────────────────────────────────────────────────────────
// The client
// ─────────────────────────────────────────────────────────────────────────────

export class ForecastClient {
  private readonly baseUrl: string;
  private readonly dataDir: string;
  private readonly fetchImpl: typeof fetch;
  private indexCache: ForecastIndex | null = null;
  private readonly artifactCache = new Map<string, ForecastArtifact>();

  constructor(config: ClientConfig = {}) {
    this.baseUrl = config.baseUrl ?? defaultBaseUrl();
    this.dataDir = config.dataDir ?? 'data';
    const f = config.fetchImpl ?? (globalThis.fetch as typeof fetch | undefined);
    if (!f) {
      throw new Error('No fetch implementation available; pass config.fetchImpl.');
    }
    // Bind to globalThis to avoid "Illegal invocation" on some runtimes.
    this.fetchImpl = config.fetchImpl ? config.fetchImpl : f.bind(globalThis);
  }

  /** URL for the index manifest. */
  private indexUrl(): string {
    return joinUrl(this.baseUrl, this.dataDir, 'index.json');
  }

  /** URL for a named artifact file (already includes any `.gz`). */
  private artifactUrl(file: string): string {
    return joinUrl(this.baseUrl, this.dataDir, file);
  }

  /**
   * Load `data/index.json` (the latest pointer + rolling history). Cached after the
   * first successful load; pass `force` to refetch.
   */
  async loadIndex(force = false): Promise<ForecastIndex> {
    if (this.indexCache && !force) return this.indexCache;
    const index = await fetchJson<ForecastIndex>(this.indexUrl(), this.fetchImpl);
    this.indexCache = index;
    return index;
  }

  /**
   * Load the LATEST artifact named by the index. This is the default Monitoring view.
   */
  async loadLatest(): Promise<ForecastArtifact> {
    const index = await this.loadIndex();
    if (!index.latest) {
      throw new Error('index.json has no `latest` artifact pointer.');
    }
    return this.loadArtifactFile(index.latest);
  }

  /**
   * Load the artifact for a specific date ("YYYY-MM-DD") from the rolling history —
   * the "forecast from {past date}" selector. Falls back to the conventional filename
   * `forecast-<date>.json` if the date is not present in the index history.
   */
  async loadByDate(date: string): Promise<ForecastArtifact> {
    const index = await this.loadIndex();
    const entry =
      index.history.find((e) => e.date === date) ??
      (index.latest && index.latest.includes(date) ? { date, file: index.latest } : undefined);
    const file = entry?.file ?? `forecast-${date}.json`;
    return this.loadArtifactFile(file);
  }

  /** Load a specific artifact file by (relative) name, with caching. */
  async loadArtifactFile(file: string): Promise<ForecastArtifact> {
    const cached = this.artifactCache.get(file);
    if (cached) return cached;
    const artifact = await fetchJson<ForecastArtifact>(this.artifactUrl(file), this.fetchImpl);
    this.artifactCache.set(file, artifact);
    return artifact;
  }

  /** Clear in-memory caches (e.g. after a known publish). */
  clearCache(): void {
    this.indexCache = null;
    this.artifactCache.clear();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure typed selectors over a loaded artifact (no I/O)
// ─────────────────────────────────────────────────────────────────────────────

/** String key for a numeric horizon as Python's `str(int)` emits it ("1", "2", "7"). */
export function horizonKey(horizonDays: number): string {
  return String(horizonDays);
}

/**
 * String key for a numeric magnitude threshold as Python's `str(float)` emits it.
 * Python prints `5.0` (one decimal) for the values in `configs/forecast.yaml`
 * ([5.0, 6.0, 7.0]); we normalize a JS number to that representation so lookups hit.
 */
export function thresholdKey(mThreshold: number): string {
  // Python `str(5.0)` -> "5.0"; integers carried as floats keep one decimal.
  return Number.isInteger(mThreshold) ? `${mThreshold}.0` : String(mThreshold);
}

/**
 * Read one leaf `CellValue` for a (cell, horizon, threshold), or `undefined` if the cell
 * is absent (implicit long-term baseline — NOT "safe"; the UI must render "no forecast").
 */
export function getCellValue(
  artifact: ForecastArtifact,
  cell: string,
  horizonDays: number,
  mThreshold: number,
): CellValue | undefined {
  return artifact.forecast[cell]?.[horizonKey(horizonDays)]?.[thresholdKey(mThreshold)];
}

/**
 * Project the sparse artifact to a flat per-cell selection for one
 * (horizon, threshold, bound) slice — the array the deck.gl H3 layer / no-map summary
 * consumes. Cells in `coverage_mask` are excluded (they render as an explicit hatch,
 * handled separately by `getCoverageMask`).
 *
 * Global re-scope (web-app-spec.md §7.1): pass `restrictToCells` to surface only one country VIEW's
 * slice of the single global field (the cell-key index from `ViewIndexEntry.cells`). Omit it (or pass
 * `null`/the WORLD view) to surface the whole global field — the default world probability field.
 * The forecast dict is the same one global field in both cases; the view only narrows which cells are
 * returned, never re-computes anything.
 */
export function selectField(
  artifact: ForecastArtifact,
  horizonDays: number,
  mThreshold: number,
  bound: Bound = 'expected',
  restrictToCells?: Iterable<string> | null,
): CellSelection[] {
  const hKey = horizonKey(horizonDays);
  const tKey = thresholdKey(mThreshold);
  const field = BOUND_FIELD[bound];
  const masked = new Set(artifact.coverage_mask);
  const restrict = restrictToCells ? new Set(restrictToCells) : null;
  const out: CellSelection[] = [];

  for (const [cell, byHorizon] of Object.entries(artifact.forecast)) {
    if (masked.has(cell)) continue;
    if (restrict && !restrict.has(cell)) continue;
    const v = byHorizon[hKey]?.[tKey];
    if (!v) continue;
    const value = v[field] as number;
    out.push({
      cell,
      value,
      p: v.p,
      lo: v.lo,
      hi: v.hi,
      rate: v.rate,
      baseline: v.baseline,
      // Mandatory honesty companion (web-app-spec.md §7.3): ratio of forecast to
      // baseline probability. Guard a zero/near-zero baseline -> Infinity is reported
      // honestly (a large ratio on a near-zero baseline still reads "still unlikely"
      // via the absolute `p`).
      ratioToBaseline: v.baseline > 0 ? v.p / v.baseline : Number.POSITIVE_INFINITY,
    });
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Country VIEW helpers (slices of the single global field) — web-app-spec.md §7.1
// ─────────────────────────────────────────────────────────────────────────────

/** The configured country views (slices of the global field). Empty for a single-region artifact. */
export function getViews(artifact: ForecastArtifact): ViewIndexEntry[] {
  return artifact.views ?? [];
}

/** Whether this artifact carries country views (i.e. it is the global, multi-country field). */
export function isGlobalArtifact(artifact: ForecastArtifact): boolean {
  return getViews(artifact).length > 0 || artifact.region.id === 'global';
}

/** Find a country view by id, or `undefined` (e.g. when the WORLD view is selected). */
export function findView(artifact: ForecastArtifact, viewId: string): ViewIndexEntry | undefined {
  if (viewId === WORLD_VIEW_ID) return undefined;
  return getViews(artifact).find((v) => v.id === viewId);
}

/**
 * The flat per-cell field for the currently-selected country view (or the whole world when `viewId`
 * is the WORLD view / unknown). Restricts to the view's cell-key index — the cheap slice of the one
 * global field, never a re-fit.
 */
export function selectViewField(
  artifact: ForecastArtifact,
  viewId: string,
  horizonDays: number,
  mThreshold: number,
  bound: Bound = 'expected',
): CellSelection[] {
  const view = findView(artifact, viewId);
  return selectField(artifact, horizonDays, mThreshold, bound, view ? view.cells : null);
}

/** Centre `[lon, lat]` of a bbox — the map's initial centre for a view (or the world). */
export function bboxCenter(bbox: BBox): [number, number] {
  return [(bbox.lon_min + bbox.lon_max) / 2, (bbox.lat_min + bbox.lat_max) / 2];
}

/**
 * A rough MapLibre zoom level that frames a bbox (degrees of span → zoom). The world view sits near
 * zoom 1; a country view zooms in. Heuristic only — the map is interactive after the initial frame.
 */
export function bboxZoom(bbox: BBox): number {
  const spanLat = Math.abs(bbox.lat_max - bbox.lat_min);
  const spanLon = Math.abs(bbox.lon_max - bbox.lon_min);
  const span = Math.max(spanLat, spanLon, 1e-3);
  // ~360° span → zoom 0; halving the span adds ~1 zoom level. Clamp to a sane window.
  const zoom = Math.log2(360 / span);
  return Math.min(7, Math.max(1, Math.round(zoom * 10) / 10));
}

/** All distinct horizons present in the artifact metadata (sorted ascending). */
export function availableHorizons(artifact: ForecastArtifact): number[] {
  return [...artifact.horizons_days].sort((a, b) => a - b);
}

/** All distinct magnitude thresholds present in the artifact metadata (sorted ascending). */
export function availableThresholds(artifact: ForecastArtifact): number[] {
  return [...artifact.magnitude_thresholds].sort((a, b) => a - b);
}

/** The cells explicitly OUT of validated coverage (render as a hatch; blank != safe). */
export function getCoverageMask(artifact: ForecastArtifact): string[] {
  return artifact.coverage_mask;
}

/** The CSEP / reliability calibration summary (drives the always-on credibility badge). */
export function getCalibration(artifact: ForecastArtifact): CalibrationSummary {
  return artifact.calibration;
}

/** The staleness indicator; `ok === false` -> the UI must degrade visibly. */
export function getStaleness(artifact: ForecastArtifact): Staleness {
  return artifact.staleness;
}

/**
 * Convenience: peak probability across all included cells for one (horizon, threshold,
 * bound) slice, for the header summary / sort. Returns 0 for an empty slice.
 */
export function peakProbability(
  artifact: ForecastArtifact,
  horizonDays: number,
  mThreshold: number,
  bound: Bound = 'expected',
): number {
  const field = selectField(artifact, horizonDays, mThreshold, bound);
  return field.reduce((max, c) => (c.value > max ? c.value : max), 0);
}

/** A ready-to-use default client bound to the app's deploy base. */
export const forecastClient = new ForecastClient();

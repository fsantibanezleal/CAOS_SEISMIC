/**
 * CAOS_SEISMIC — GLOBAL back-analysis (context-contribution) types + loader.
 *
 * This is the front-end half of the THESIS measurement produced offline by
 * `src/caos_seismic/eval/global_backanalysis.py` (`_write_global_summary`). The re-scoped product is
 * a single GLOBAL context-conditioned forecaster; any country is a VIEW into that one global field
 * (web-app-spec.md §7.1). The global back-analysis scores the same global model through every
 * pre-registered country view AND a global view, then reduces the per-view ledgers to the two
 * headline numbers the whole re-scoping exists to report:
 *
 *  1. **Context gain over catalog-only ETAS** (`context_gain`) — per view × horizon and pooled
 *     globally, in NATS. ETAS already reproduces Omori/Utsu clustering, so a positive, significant
 *     gain over it is NOT "I predicted aftershocks" — it quantifies how much the GLOBAL context
 *     (worldwide seismicity + complementary covariates) adds to the LOCAL short-term forecast. When
 *     the context channel has not yet landed (enricher stack feature-flagged off), the gain is ~0 by
 *     construction and `context_channel_active` says so honestly — it is never faked positive.
 *
 *  2. **High-vs-low-seismicity bias** (`high_vs_low_bias`) — the same skill/calibration metrics
 *     computed separately over the HIGH-seismicity views (active plate boundaries) and the
 *     LOW-seismicity views (stable interiors), with their gap. A single pooled global number is
 *     dominated by the loud subduction margins; this partition asks the adversarial question — does
 *     the model only look good because it over-fits high-seismicity zones?
 *
 * Same static-first, read-only contract as the forecast artifact and the per-region back-analysis:
 * the page renders a bundled SAMPLE (`public/data/backanalysis-global-sample.json`, written by
 * `app/scripts/gen_global_backanalysis_sample.py`) until the real `results/backanalysis-global-*.json`
 * replaces it byte-for-byte — the mock lives ONLY at the data boundary, behind this real interface.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Per-view ledger (mirrors `_view_block` + BackAnalysisResult.per_horizon)
// ─────────────────────────────────────────────────────────────────────────────

/** One reliability-diagram point `[forecast_prob, observed_freq, n]` (matches the artifact). */
export type ReliabilityPoint = [forecastProb: number, observedFreq: number, n: number];

/**
 * One horizon's scored scalars for a view (mirrors `eval/backanalysis._reduce_per_horizon`). All
 * scalars may be `null` when a horizon could not be scored (kept, never dropped — honest reporting).
 */
export interface ViewHorizon {
  horizon_days: number;
  n_scored: number;
  /** Fraction of issue days whose N-test was within CSEP consistency, in [0,1]. */
  n_test_pass_rate: number | null;
  /** Mean information gain per earthquake over the null (smoothed-seismicity Poisson), in NATS. */
  mean_igpe_vs_null_nats: number | null;
  /** Mean information gain over catalog-only ETAS, in NATS — the CONTEXT contribution. */
  mean_context_gain_vs_etas_nats: number | null;
  /** Whether the context channel (enricher stack) was active for this view/horizon. */
  context_channel_active: boolean;
  /** Whether the mean context gain is strictly positive. */
  context_gain_positive: boolean;
  /** Mean Brier score for the binary exceedance outcome (lower is better). */
  brier: number | null;
}

/** One country / global view's block (mirrors `_view_block`). */
export interface ViewBlock {
  view_id: string;
  name_en: string;
  /** "high" = active plate boundary; "low" = stable interior (the pre-registered partition). */
  seismicity_class: 'high' | 'low' | string;
  /** Short tectonic descriptor (subduction / transform / intraplate / ...). */
  plate_setting: string;
  /** Effective fit-grid cell size in degrees (`null` ⇒ configured fine grid; a number ⇒ coarsened). */
  fit_cell_deg: number | null;
  n_issue_days?: number;
  n_scored_days?: number;
  n_failed_days?: number;
  per_horizon: ViewHorizon[];
  reliability?: ReliabilityPoint[];
  /** A view that could not be scored at all carries an `error` and an empty `per_horizon`. */
  error?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Context-gain reduction (mirrors `_reduce_context_gain`)
// ─────────────────────────────────────────────────────────────────────────────

/** Per-view × horizon context-gain entry. */
export interface ContextGainByHorizon {
  mean_context_gain_vs_etas_nats: number | null;
  context_channel_active: boolean;
  context_gain_positive: boolean;
  n_scored: number;
}

/** Per-view context-gain block (`by_horizon` keyed by the stringified horizon, e.g. "1"). */
export interface ContextGainPerView {
  seismicity_class: 'high' | 'low' | string;
  by_horizon: Record<string, ContextGainByHorizon>;
}

/** Pooled (across-view, scored-day-weighted) context gain for one horizon. */
export interface ContextGainPooled {
  mean_context_gain_vs_etas_nats: number | null;
  context_gain_positive: boolean;
  n_scored_weight: number;
}

/** The THESIS headline reduction (mirrors `_reduce_context_gain`). */
export interface ContextGain {
  definition: string;
  /** False ⇒ the context channel has NOT landed yet; the gain is ~0 by construction, reported so. */
  context_channel_active: boolean;
  channel_note: string;
  per_view: Record<string, ContextGainPerView>;
  /** Pooled gain keyed by stringified horizon, e.g. { "1": {...}, "2": {...}, "7": {...} }. */
  pooled: Record<string, ContextGainPooled>;
}

// ─────────────────────────────────────────────────────────────────────────────
// High-vs-low bias reduction (mirrors `_reduce_high_low_bias`)
// ─────────────────────────────────────────────────────────────────────────────

/** The four bias scalars pooled per seismicity class per horizon. */
export interface BiasClassHorizon {
  n_scored: number;
  n_test_pass_rate: number | null;
  mean_igpe_vs_null_nats: number | null;
  mean_context_gain_vs_etas_nats: number | null;
  brier: number | null;
}

/** One seismicity class (high / low): its member views + per-horizon pooled scalars. */
export interface BiasClass {
  view_ids: string[];
  by_horizon: Record<string, BiasClassHorizon>;
}

/** The gap (high − low) for one scalar field at one horizon. */
export interface BiasGapField {
  high: number | null;
  low: number | null;
  gap_high_minus_low: number | null;
  /** The advantage of HIGH over LOW in the good direction (Brier is lower-is-better). */
  high_better_by?: number;
}

/** The HIGH-vs-LOW bias reduction (mirrors `_reduce_high_low_bias`). */
export interface HighVsLowBias {
  definition: string;
  /** Per field: "higher_better" | "lower_better" (Brier is the only lower-better one). */
  field_directions: Record<string, 'higher_better' | 'lower_better' | string>;
  per_class: Record<'high' | 'low' | string, BiasClass>;
  /** Gap keyed by stringified horizon → field name → gap entry. */
  gap: Record<string, Record<string, BiasGapField>>;
}

// ─────────────────────────────────────────────────────────────────────────────
// The committed global back-analysis summary (mirrors `_write_global_summary`)
// ─────────────────────────────────────────────────────────────────────────────

export interface GlobalBackAnalysis {
  product: string;
  kind: 'backanalysis_global' | string;
  generated_at: string;
  period: { start: string; end: string };
  issue_cadence: string;
  horizons_days: number[];
  magnitude_thresholds: number[];
  reliability_threshold: number;
  /** The view ids scored (country views first, then the global view). */
  views: string[];
  per_view: ViewBlock[];
  /** THESIS headline: context gain over catalog-only ETAS, per view × horizon and pooled. */
  context_gain: ContextGain;
  /** Adversarial check: does the model only look good on high-seismicity margins? */
  high_vs_low_bias: HighVsLowBias;
  pycsep_used: boolean;
  framing: string;
  /** True ⇒ render a prominent "illustrative SAMPLE" banner (not real CSEP scores). */
  sample?: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Loader (static-first, read-only — same contract as the forecast client)
// ─────────────────────────────────────────────────────────────────────────────

/** Default location of the committed global back-analysis summary under the static data host. */
const DEFAULT_FILE = 'backanalysis-global-sample.json';

export interface LoadGlobalBackAnalysisOptions {
  /** Base URL; defaults to the Vite deploy base. */
  baseUrl?: string;
  /** Data sub-dir; defaults to "data". */
  dataDir?: string;
  /** Filename; defaults to the bundled sample until real results land. */
  file?: string;
  fetchImpl?: typeof fetch;
}

function resolveBase(): string {
  try {
    const env = (import.meta as unknown as { env?: { BASE_URL?: string } }).env;
    if (env?.BASE_URL) return env.BASE_URL;
  } catch {
    /* non-Vite env */
  }
  return '/';
}

/**
 * Fetch the committed global back-analysis summary. The Context-contribution page renders the SAMPLE
 * until the real `eval/global_backanalysis` output replaces it byte-for-byte (no page change needed).
 */
export async function loadGlobalBackAnalysis(
  opts: LoadGlobalBackAnalysisOptions = {},
): Promise<GlobalBackAnalysis> {
  const base = (opts.baseUrl ?? resolveBase()).replace(/\/+$/, '');
  const dir = (opts.dataDir ?? 'data').replace(/^\/+|\/+$/g, '');
  const file = (opts.file ?? DEFAULT_FILE).replace(/^\/+/, '');
  const url = `${base}/${dir}/${file}`;
  const f = opts.fetchImpl ?? (globalThis.fetch as typeof fetch);
  const res = await f(url, { method: 'GET', cache: 'no-cache' });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as GlobalBackAnalysis;
}

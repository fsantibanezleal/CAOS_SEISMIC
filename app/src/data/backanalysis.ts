/**
 * CAOS_SEISMIC — Back-analysis result types + loader.
 *
 * The Back-analysis page (web-app-spec.md §6, evaluation-plan.md §7/§9) renders the
 * retrospective, pseudo-prospective CSEP results across regions × periods × horizons. Those
 * results are produced offline by the evaluation job (pyCSEP) and committed as a single
 * static JSON (`data/back-analysis.json`) — the same static-first, read-only contract as the
 * daily forecast artifact.
 *
 * This module mirrors that JSON's shape and provides a tiny loader. Until real pyCSEP results
 * exist, a bundled SAMPLE (`public/data/back-analysis-sample.json`, written by
 * `app/scripts/gen_backanalysis_sample.py`) is served by the same code path — the mock lives
 * ONLY at the data boundary. The sample is honest: it includes cells where the model does NOT
 * beat ETAS (a published failure), as required by the multiple-testing discipline.
 *
 * The report follows evaluation-plan.md §7's table:
 *   Consistency: N-test (δ1, δ2), M-test (κ), S-test (ζ), L/CL-test (γ) — gridded AND catalog.
 *   Comparison:  IGPE (nats) vs smoothed-seismicity AND vs ETAS, with T-test CI + W-test p.
 *   Calibration: reliability diagram + pyCSEP calibration test.
 *   Communication / scoring rules: Area Skill Score, Brier, Log score, CRPS.
 */

/** A single consistency-test outcome: a quantile score in [0,1] and a pass flag. */
export interface ConsistencyTest {
  /** Quantile score in [0,1] (within the consistency band ⇒ pass). */
  quantile: number;
  /** Whether the model is within CSEP consistency for this test. */
  pass: boolean;
  /** Optional second quantile (the N-test reports δ1 too-few AND δ2 too-many). */
  quantile2?: number;
}

/** The N/M/S/L/CL consistency block for one forecast representation (gridded or catalog). */
export interface ConsistencyBlock {
  /** N-test: `quantile` = δ1 (too few), `quantile2` = δ2 (too many). */
  N: ConsistencyTest;
  /** M-test (magnitude / GR shape; quantile κ). */
  M: ConsistencyTest;
  /** S-test (spatial; quantile ζ). */
  S: ConsistencyTest;
  /** L-test (joint likelihood; quantile γ). */
  L: ConsistencyTest;
  /** CL-test (conditional likelihood; preferred over raw L). */
  CL: ConsistencyTest;
}

/** A comparison-test outcome vs ONE baseline (smoothed-seismicity or ETAS). */
export interface ComparisonTest {
  /** Baseline label (e.g. "ETAS", "smoothed-seismicity"). */
  baseline: "etas" | "smoothed" | string;
  /** Information Gain Per Earthquake, in NATS (not bits). Positive ⇒ model better. */
  igpe_nats: number;
  /** T-test 95% confidence interval on IGPE, [lo, hi]. Skill claimed only if lo > 0. */
  t_ci: [number, number];
  /** Wilcoxon signed-rank companion p-value. */
  w_pvalue: number;
  /** Whether skill is established (IGPE>0, T-CI excludes 0, W corroborates). */
  skill: boolean;
}

/** One reliability-diagram point `[forecast_prob, observed_freq, n]` (matches the artifact). */
export type ReliabilityPoint = [forecastProb: number, observedFreq: number, n: number];

/** Secondary proper-scoring / communication metrics (evaluation-plan §6.3). */
export interface ScoringMetrics {
  /** Brier score for the bounded binary exceedance output (lower is better). */
  brier: number;
  /** Logarithmic score (mean −log p(y); lower is better). */
  log_score: number;
  /** Continuous Ranked Probability Score for the full predictive distribution. */
  crps: number;
  /** Area Skill Score from the Molchan diagram (1 = perfect, 0.5 = random). */
  area_skill_score: number;
}

/** One region × horizon report cell (the unit of the §7 table). */
export interface BackAnalysisCell {
  /** Horizon in days (1, 2, 7). */
  horizon_days: number;
  /** Consistency tests, gridded-rate representation (Poisson CSEP tests). */
  consistency_gridded: ConsistencyBlock;
  /** Consistency tests, catalog-based representation (over-dispersion-honest). */
  consistency_catalog: ConsistencyBlock;
  /** Comparison vs each mandatory baseline (smoothed-seismicity AND ETAS). */
  comparison: ComparisonTest[];
  /** Reliability-diagram points for this region × horizon. */
  reliability: ReliabilityPoint[];
  /** Secondary scoring metrics. */
  scoring: ScoringMetrics;
  /** Expected-vs-observed event counts over the test split (for the time-series summary). */
  expected_vs_observed: {
    /** Period label, "YYYY-MM" or similar. */
    period: string;
    expected: number;
    observed: number;
  }[];
  /**
   * Whether a Poisson grid test over-rejected during a sequence and is therefore paired with
   * its catalog-based result (evaluation-plan §1). Honest annotation, not a model fault.
   */
  poisson_over_rejected?: boolean;
}

/** All horizons for one region, plus region metadata. */
export interface BackAnalysisRegion {
  /** Region id (matches the forecast artifact's region ids: chile, california, japan, …). */
  region_id: string;
  name_en: string;
  name_es: string;
  /** Authoritative catalog source string (CSN, ANSS/ComCat, JMA, GeoNet, INGV). */
  catalog: string;
  /** Why this region is included (tectonic-diversity rationale). */
  rationale_en: string;
  rationale_es: string;
  /** Learning / testing period bounds (ISO dates), pre-registered. */
  train_period: [string, string];
  test_period: [string, string];
  /** Uniform target M_min applied to the model AND every baseline. */
  m_min: number;
  /** Per region × horizon report cells. */
  cells: BackAnalysisCell[];
}

/** The committed back-analysis report (static, read-only). */
export interface BackAnalysisReport {
  schema_version: string;
  product: string;
  /** ISO-8601 UTC time this report was produced. */
  generated_at: string;
  /** pyCSEP version used (reviewers can dispute the model, not the test code). */
  tooling: { pycsep_version: string; note_en: string; note_es: string };
  /** Whether these are SAMPLE numbers (true ⇒ render a prominent "illustrative" banner). */
  sample: boolean;
  /** The diverse regions evaluated (≥4 to prevent cherry-picking). */
  regions: BackAnalysisRegion[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Loader (static-first, read-only — same contract as the forecast client)
// ─────────────────────────────────────────────────────────────────────────────

/** Default location of the committed report under the static data host. */
const DEFAULT_FILE = "back-analysis-sample.json";

export interface LoadOptions {
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
  return "/";
}

/**
 * Fetch the committed back-analysis report. The page renders the SAMPLE until the real
 * pyCSEP output replaces it byte-for-byte (no page change required).
 */
export async function loadBackAnalysis(opts: LoadOptions = {}): Promise<BackAnalysisReport> {
  const base = (opts.baseUrl ?? resolveBase()).replace(/\/+$/, "");
  const dir = (opts.dataDir ?? "data").replace(/^\/+|\/+$/g, "");
  const file = (opts.file ?? DEFAULT_FILE).replace(/^\/+/, "");
  const url = `${base}/${dir}/${file}`;
  const f = opts.fetchImpl ?? (globalThis.fetch as typeof fetch);
  const res = await f(url, { method: "GET", cache: "no-cache" });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as BackAnalysisReport;
}

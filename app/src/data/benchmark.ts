/**
 * Loader + types for the committed multi-model benchmark (`results/benchmark.json`, written by the
 * benchmark job). A single-window, leakage-free IGPE-vs-the-Poisson-null comparison of every model
 * (ETAS, the ensemble, Reasenberg–Jones, the null) per view — the "show every model's performance,
 * even the ones we don't ship" record. Same static-first read-only contract as the forecast client.
 */

/** One model's scores in the benchmark window. */
export interface BenchmarkModelScore {
  /** Information gain per earthquake over the Poisson null, in NATS (the model's skill). */
  igpe_vs_null_nats: number;
  /** Expected count Σλ over the window. */
  n_forecast: number;
  n_test_passed: boolean;
  n_test_quantile: number;
}

/** One view's per-model benchmark block. */
export interface BenchmarkView {
  view: string;
  seismicity_class?: 'high' | 'low' | string;
  n_cells?: number;
  n_observed: number;
  /** Keyed by model id: "etas", "ensemble", "reasenberg_jones", "smoothed_null". */
  models: Record<string, BenchmarkModelScore>;
  error?: string;
}

/** The committed benchmark summary. */
export interface ModelBenchmark {
  product: string;
  kind: 'model_benchmark' | string;
  horizon_days: number;
  m_threshold: number;
  method: string;
  framing: string;
  per_view: BenchmarkView[];
}

const DEFAULT_FILE = 'benchmark.json';

export interface LoadBenchmarkOptions {
  baseUrl?: string;
  dataDir?: string;
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

/** The display order + labels of the models in the benchmark table. */
export const BENCHMARK_MODELS: { id: string; label: string }[] = [
  { id: 'etas', label: 'ETAS (tiled)' },
  { id: 'ensemble', label: 'Ensemble' },
  { id: 'reasenberg_jones', label: 'Reasenberg–Jones' },
  { id: 'smoothed_null', label: 'Smoothed null' },
];

export async function loadBenchmark(opts: LoadBenchmarkOptions = {}): Promise<ModelBenchmark> {
  const base = (opts.baseUrl ?? resolveBase()).replace(/\/+$/, '');
  const dir = (opts.dataDir ?? 'data').replace(/^\/+|\/+$/g, '');
  const file = (opts.file ?? DEFAULT_FILE).replace(/^\/+/, '');
  const url = `${base}/${dir}/${file}`;
  const f = opts.fetchImpl ?? (globalThis.fetch as typeof fetch);
  const res = await f(url, { method: 'GET', cache: 'no-cache' });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as ModelBenchmark;
}

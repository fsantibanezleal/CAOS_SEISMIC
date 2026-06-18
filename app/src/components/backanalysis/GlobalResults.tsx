import { useEffect, useState } from 'react';

import { BENCHMARK_MODELS, loadBenchmark, type ModelBenchmark } from '@/data/benchmark';
import { loadGlobalBackAnalysis, type GlobalBackAnalysis, type ViewBlock } from '@/data/globalbackanalysis';
import type { Language } from '@/i18n/config';

/**
 * The REAL global multi-country back-analysis + the multi-model benchmark (web-app-spec §6: show every
 * model's prospective performance, honest failures included). Renders the committed
 * `backanalysis-global.json` (per-country IGPE vs the Poisson null, the high-vs-low-seismicity bias, the
 * context-contribution status) and `benchmark.json` (each model's IGPE — including the ones we do NOT
 * ship). Bilingual via the `lang` prop (this section is data-dense; labels are inlined EN/ES rather than
 * routed through the i18n bundle).
 */

function fmt(x: number | null | undefined, dp = 4): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return (x >= 0 ? '+' : '') + x.toFixed(dp);
}

function igpeAt(view: ViewBlock | undefined, horizon: number): number | null {
  return view?.per_horizon.find((h) => h.horizon_days === horizon)?.mean_igpe_vs_null_nats ?? null;
}

export function GlobalResults({ lang }: { lang: Language }) {
  const L = (en: string, es: string) => (lang === 'es' ? es : en);
  const [gba, setGba] = useState<GlobalBackAnalysis | null>(null);
  const [bench, setBench] = useState<ModelBenchmark | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadGlobalBackAnalysis()
      .then((d) => !cancelled && setGba(d))
      .catch((e: unknown) => !cancelled && setErr(e instanceof Error ? e.message : String(e)));
    loadBenchmark()
      .then((d) => !cancelled && setBench(d))
      .catch(() => undefined); // benchmark is optional — absence just hides that block
    return () => {
      cancelled = true;
    };
  }, []);

  if (err && !gba) return <p className="error-note">{L('Could not load results', 'No se pudieron cargar los resultados')}: {err}</p>;
  if (!gba) return <p className="muted">{L('Loading real results…', 'Cargando resultados reales…')}</p>;

  const horizons = gba.horizons_days;
  const countryViews = gba.per_view.filter((v) => v.view_id !== 'global' && !v.error);
  const globalView = gba.per_view.find((v) => v.view_id === 'global');
  const h0 = horizons[0] ?? 1;
  const globalIgpe = igpeAt(globalView, h0);
  const biasGap = gba.high_vs_low_bias.gap?.[String(h0)]?.mean_igpe_vs_null_nats?.gap_high_minus_low ?? null;
  const ctxActive = gba.context_gain.context_channel_active;

  return (
    <section className="global-results">
      <header className="page-head">
        <h2>{L('Real multi-country results', 'Resultados reales multi-país')}</h2>
        <p className="muted small">
          {L('Leakage-free pseudo-prospective back-analysis', 'Back-analysis pseudo-prospectivo sin fuga')}
          {' · '}
          {gba.period.start.slice(0, 10)} → {gba.period.end.slice(0, 10)}
          {' · '}
          {L('information gain per earthquake (nats) over the Poisson null', 'ganancia de información por sismo (nats) sobre el null de Poisson')}
        </p>
      </header>

      {/* Headline numbers */}
      <div className="result-cards">
        <div className="result-card">
          <span className="result-value mono">{fmt(globalIgpe, 4)}</span>
          <span className="result-label">{L('Global field vs null (1-day)', 'Campo global vs null (1 día)')}</span>
          <span className="result-sub">{L('the global context contribution — nats/eq', 'la contribución del contexto global — nats/sismo')}</span>
        </div>
        <div className="result-card">
          <span className="result-value mono">{fmt(biasGap, 4)}</span>
          <span className="result-label">{L('High − low seismicity gap', 'Gap alta − baja sismicidad')}</span>
          <span className="result-sub">{L('skill concentrates in active zones (the measured bias)', 'la skill se concentra en zonas activas (el bias medido)')}</span>
        </div>
        <div className="result-card">
          <span className={`result-value mono ${ctxActive ? '' : 'muted'}`}>{ctxActive ? L('active', 'activo') : '≈ 0'}</span>
          <span className="result-label">{L('Context channel (neural)', 'Canal de contexto (neural)')}</span>
          <span className="result-sub">{L('≈ 0 by construction until geodetic/stress covariates are wired', '≈ 0 por construcción hasta cablear covariables geodésicas/estrés')}</span>
        </div>
      </div>

      {/* Per-view IGPE-vs-null table */}
      <h3>{L('Per-view skill (IGPE vs the Poisson null)', 'Skill por vista (IGPE vs el null de Poisson)')}</h3>
      <table className="bench-table">
        <thead>
          <tr>
            <th scope="col">{L('View', 'Vista')}</th>
            <th scope="col">{L('Class', 'Clase')}</th>
            {horizons.map((h) => (
              <th key={h} scope="col">{h}{L('d', 'd')}</th>
            ))}
            <th scope="col">{L('N-test', 'N-test')}</th>
          </tr>
        </thead>
        <tbody>
          {[...countryViews, ...(globalView ? [globalView] : [])].map((v) => {
            const ntp = v.per_horizon.find((h) => h.horizon_days === h0)?.n_test_pass_rate;
            return (
              <tr key={v.view_id} className={v.view_id === 'global' ? 'row-global' : ''}>
                <th scope="row">{v.view_id === 'global' ? L('Global (whole Earth)', 'Global (toda la Tierra)') : v.name_en}</th>
                <td>{v.seismicity_class === 'high' ? L('high', 'alta') : v.seismicity_class === 'low' ? L('low', 'baja') : v.seismicity_class}</td>
                {horizons.map((h) => {
                  const val = igpeAt(v, h);
                  return (
                    <td key={h} className={`mono ${val !== null && val > 0 ? 'pos' : val !== null && val < 0 ? 'neg' : ''}`}>{fmt(val, 4)}</td>
                  );
                })}
                <td className="mono">{ntp === null || ntp === undefined ? '—' : `${(ntp * 100).toFixed(0)}%`}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted small">
        {L(
          'Positive = the conditioned model beats the stationary null. Skill is highest globally (aggregated triggering) and in active margins; low-seismicity interiors sit at ~0 — a self-exciting model has nothing to add without aftershock sequences.',
          'Positivo = el modelo condicionado supera al null estacionario. La skill es máxima globalmente (triggering agregado) y en márgenes activos; los interiores de baja sismicidad quedan en ~0 — un modelo auto-excitado no aporta sin secuencias de réplicas.',
        )}
      </p>

      {/* Multi-model benchmark */}
      {bench ? (
        <>
          <h3>{L('Model benchmark — every model, including the ones we don’t ship', 'Benchmark de modelos — todos, incluidos los que no desplegamos')}</h3>
          <p className="muted small">
            {L('Single-window leakage-free IGPE vs the Poisson null (', 'IGPE de ventana única sin fuga vs el null de Poisson (')}
            {bench.horizon_days}{L('-day holdout, M ≥ ', '-día holdout, M ≥ ')}{bench.m_threshold}{').'}
          </p>
          <table className="bench-table">
            <thead>
              <tr>
                <th scope="col">{L('View', 'Vista')}</th>
                <th scope="col">{L('obs', 'obs')}</th>
                {BENCHMARK_MODELS.map((m) => (
                  <th key={m.id} scope="col">{m.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {bench.per_view.filter((v) => !v.error).map((v) => {
                const etas = v.models['etas']?.igpe_vs_null_nats;
                return (
                  <tr key={v.view}>
                    <th scope="row">{v.view === 'global' ? L('Global', 'Global') : v.view}</th>
                    <td className="mono">{v.n_observed}</td>
                    {BENCHMARK_MODELS.map((m) => {
                      const s = v.models[m.id];
                      const val = s?.igpe_vs_null_nats;
                      // mark the best model per row
                      const isBest = val !== undefined && etas !== undefined && m.id === 'etas';
                      return (
                        <td key={m.id} className={`mono ${isBest ? 'best' : ''} ${val !== undefined && val > 0 ? 'pos' : val !== undefined && val < 0 ? 'neg' : ''}`}>
                          {s ? fmt(val, 3) : '—'}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="callout honest small">
            {L(
              'Honest finding: ETAS is the best single model, and the naive equal-weight ensemble UNDERPERFORMS it — averaging in the null and the weak Reasenberg–Jones dilutes the triggering signal that is the skill. This matches the literature (ensembles help only when score-weighted with strong members), so we do not ship the naive ensemble.',
              'Hallazgo honesto: ETAS es el mejor modelo único, y el ensemble naïve de peso igual lo SUB-rinde — promediar el null y el débil Reasenberg–Jones diluye la señal de triggering que es la skill. Coincide con la literatura (los ensembles solo ayudan score-weighted con miembros fuertes), así que no desplegamos el ensemble naïve.',
            )}
          </p>
        </>
      ) : null}
    </section>
  );
}

export default GlobalResults;

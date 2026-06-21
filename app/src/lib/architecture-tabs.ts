/**
 * Architecture / "How it works" modal content (ADR-0058) — CAOS_SEISMIC.
 *
 * Five tabs, each pairing ONE hand-authored, theme-aware SVG (public/svg/tech/) with a compact bilingual
 * explanation at COMPLETE depth. CAOS_SEISMIC is a non-shell app (its own Layout), so it implements the
 * ADR-0058 pattern directly. The SVGs use the app's CSS-variable palette tokens (zero hardcoded hex) and
 * are fetched + inlined by ArchitectureModal so they inherit the theme.
 *
 * The diagrams + copy are the per-product content; the modal chrome is generic.
 */

export interface ArchTab {
  id: string;
  /** SVG file under public/svg/tech/. */
  svg: string;
  label: { en: string; es: string };
  /** Paragraphs (rendered as <p>), bilingual. */
  body: { en: string[]; es: string[] };
}

export const ARCH_TABS: ArchTab[] = [
  {
    id: "app",
    svg: "01-overview.svg",
    label: { en: "The app", es: "La app" },
    body: {
      en: [
        "CAOS_SEISMIC is a GLOBAL, conditional, probabilistic short-term seismic FORECASTING system — forecasts (a probability field), never deterministic predictions. It quantifies how global context conditions the short-term local forecast.",
        "Design-build flow: deep research → models in src/caos_seismic/model/ (regime-tiled ETAS + a geodetic context-neural) → leakage-free pseudo-prospective CSEP validation → bake a compact artifact under results/ → the SPA in app/ replays it → git-as-data publish → GitHub Pages.",
        "The pipeline + its honest validation ARE the product; the web app replays a validated, committed subset — it never computes a forecast in the browser.",
      ],
      es: [
        "CAOS_SEISMIC es un sistema GLOBAL de PRONÓSTICO sísmico probabilístico condicional a corto plazo — pronósticos (un campo de probabilidad), nunca predicciones deterministas. Cuantifica cómo el contexto global condiciona el pronóstico local de corto plazo.",
        "Flujo de diseño: investigación profunda → modelos en src/caos_seismic/model/ (ETAS por-régimen + un neural geodésico de contexto) → validación CSEP pseudo-prospectiva sin fuga → hornear un artefacto compacto en results/ → la SPA en app/ lo reproduce → publicación git-as-data → GitHub Pages.",
        "El pipeline + su validación honesta SON el producto; la app web reproduce un subconjunto validado y commiteado — nunca calcula un pronóstico en el navegador.",
      ],
    },
  },
  {
    id: "lanes",
    svg: "02-lanes.svg",
    label: { en: "Lanes — web / offline / compute", es: "Carriles — web / offline / cómputo" },
    body: {
      en: [
        "OFFLINE / COMPUTE (on the GPU workstation): the DAILY job `caos-seismic daily` fetches the ISC/USGS catalog, fits the regime-tiled ETAS, and bakes the 1–7 day forecast field. The WEEKLY job `caos-seismic outlook` fits the strain-conditioned neural and bakes the 30-day geodetic background field.",
        "WEB (live in the browser): there is NO live inference. The SPA is a pure REPLAY of the committed artifacts — it loads the gzip-compressed field and renders it with deck.gl + MapLibre. Zero server, zero in-browser compute.",
        "THE BRIDGE — git-as-data: each job commits only results/ to main via a robust commit-tree publish; the GitHub Pages workflow copies the fresh artifacts into the served bundle, so the live site is always current.",
      ],
      es: [
        "OFFLINE / CÓMPUTO (en la estación GPU): el job DIARIO `caos-seismic daily` baja el catálogo ISC/USGS, ajusta el ETAS por-régimen y hornea el campo de pronóstico de 1–7 días. El job SEMANAL `caos-seismic outlook` ajusta el neural condicionado por strain y hornea el campo background geodésico de 30 días.",
        "WEB (live en el navegador): NO hay inferencia live. La SPA es una REPRODUCCIÓN pura de los artefactos commiteados — carga el campo comprimido en gzip y lo renderiza con deck.gl + MapLibre. Cero servidor, cero cómputo en el navegador.",
        "EL PUENTE — git-as-data: cada job commitea solo results/ a main vía un publish robusto con commit-tree; el workflow de GitHub Pages copia los artefactos frescos al bundle servido, así el sitio en vivo siempre está al día.",
      ],
    },
  },
  {
    id: "webapp",
    svg: "03-webapp.svg",
    label: { en: "Web-app flow", es: "Flujo de la web-app" },
    body: {
      en: [
        "The SPA (Vite + React + TypeScript) ships eight pages: Introduction, The problem, Methodology, Implementation, Experiments, Back-analysis, Monitoring (the 1–7 day field), and the 30-day Outlook.",
        "The read-only data client (src/data/client.ts) is gzip-aware: it loads data/index.json then the compact forecast-<date>.json.gz, decoding the quantized log-uint16 rate codes. The field renders as a deck.gl H3 / heatmap layer over a MapLibre base; a TypeScript contract type mirrors the artifact so a schema drift fails the build.",
        "Deploy: the Pages workflow runs `npm run build`, copies results/* into dist/data, and serves the SPA + the artifacts statically; client-side routing handles the deep links (404.html = index.html).",
      ],
      es: [
        "La SPA (Vite + React + TypeScript) trae ocho páginas: Introducción, El problema, Metodología, Implementación, Experimentos, Retroanálisis, Monitoreo (el campo de 1–7 días) y el Pronóstico a 30 días.",
        "El cliente de datos de solo-lectura (src/data/client.ts) es gzip-aware: carga data/index.json y luego el compacto forecast-<fecha>.json.gz, decodificando los códigos de tasa cuantizados log-uint16. El campo se renderiza como capa H3 / heatmap de deck.gl sobre una base MapLibre; un tipo-contrato TypeScript espeja el artefacto, así un drift de esquema rompe el build.",
        "Deploy: el workflow de Pages corre `npm run build`, copia results/* a dist/data y sirve la SPA + los artefactos estáticamente; el ruteo client-side maneja los deep-links (404.html = index.html).",
      ],
    },
  },
  {
    id: "science",
    svg: "04-science.svg",
    label: { en: "The science", es: "La ciencia" },
    body: {
      en: [
        "Base model — regime-tiled ETAS (Ogata): the conditional intensity λ(x,t) = μ(x) [smoothed-seismicity background] + Σ over past events of an Omori–Utsu time kernel × Utsu productivity × a magnitude-dependent spatial kernel. Fit per tectonic-regime tile (~195 tiles), b ≈ 1.34, Mc = 5.35.",
        "Geodetic challenger — a context-neural background: a CNN encodes the local GNSS strain-rate field; its calibrated background beats ETAS at the 30-day horizon (validated +0.106 IGPE, all views positive) but NOT at 1–7 days, where ETAS triggering dominates. So ETAS serves the daily product; the neural serves the 30-day outlook.",
        "Honesty layer — the catalog-based N-test (experiment E13): an ETAS branching simulation gives the over-dispersion-honest count distribution, so an apparent under-forecast reads as over-dispersion + secondary cascade, not a model failure. Skill is measured by information gain per earthquake (IGPE, nats) vs the Poisson null AND ETAS, leakage-free.",
      ],
      es: [
        "Modelo base — ETAS por-régimen (Ogata): la intensidad condicional λ(x,t) = μ(x) [background de sismicidad suavizada] + Σ sobre eventos pasados de un kernel temporal Omori–Utsu × productividad de Utsu × un kernel espacial dependiente de la magnitud. Ajustado por tile de régimen tectónico (~195 tiles), b ≈ 1.34, Mc = 5.35.",
        "Retador geodésico — un background neural de contexto: una CNN codifica el campo local de tasa de deformación GNSS; su background calibrado le gana a ETAS al horizonte de 30 días (validado +0.106 IGPE, todas las vistas positivas) pero NO a 1–7 días, donde domina el triggering de ETAS. Así ETAS sirve el producto diario; el neural sirve el outlook de 30 días.",
        "Capa de honestidad — el N-test catalog-based (experimento E13): una simulación de ramificación ETAS da la distribución de conteo honesta a la sobre-dispersión, así un aparente sub-pronóstico se lee como sobre-dispersión + cascada secundaria, no falla del modelo. La habilidad se mide por ganancia de información por sismo (IGPE, nats) vs el null de Poisson Y ETAS, sin fuga.",
      ],
    },
  },
  {
    id: "contracts",
    svg: "05-contracts.svg",
    label: { en: "Data contracts & design", es: "Contratos de datos y diseño" },
    body: {
      en: [
        "Ingestion contract (raw → pipeline): the global catalog (ISC/USGS) is homogenized to Mw (Scordilis conversions), the space–time completeness Mc and the Gutenberg–Richter b are estimated (Aki–Utsu), and the catalog is declustered — bad/incomplete data is handled explicitly, never silently coerced.",
        "Artifact contract (pipeline → web): the forecast is a SPARSE, quantized field (log-uint16 rate codes, H3-indexed) + an index.json pointer, mirrored by a TypeScript type so any drift fails the build. The 30-day outlook ships its own field + a multi-region validation evidence file.",
        "Design discipline: cases are country VIEWS into the one global field (Chile, Japan, California, NZ, …); the honest framing is enforced everywhere (forecasts not predictions; null/negative results are recorded; numbers are sourced; the experiment register E1→E14 is the day-by-day audit trail).",
      ],
      es: [
        "Contrato de ingestión (crudo → pipeline): el catálogo global (ISC/USGS) se homogeniza a Mw (conversiones de Scordilis), se estiman la completitud espacio-temporal Mc y la b de Gutenberg–Richter (Aki–Utsu), y se descluster — los datos malos/incompletos se manejan explícitamente, nunca se coercen en silencio.",
        "Contrato de artefacto (pipeline → web): el pronóstico es un campo DISPERSO y cuantizado (códigos de tasa log-uint16, indexado por H3) + un puntero index.json, espejado por un tipo TypeScript así cualquier drift rompe el build. El outlook de 30 días trae su propio campo + un archivo de evidencia de validación multi-región.",
        "Disciplina de diseño: los casos son VISTAS de país dentro del único campo global (Chile, Japón, California, NZ, …); el framing honesto se aplica en todas partes (pronósticos no predicciones; los resultados nulos/negativos se registran; los números tienen fuente; el registro de experimentos E1→E14 es la bitácora-auditoría día a día).",
      ],
    },
  },
];

/**
 * Perceptually-uniform SEQUENTIAL colormap for the probability field
 * (web-app-spec.md §7.3). This is the load-bearing visual-honesty decision: the forecast
 * field is coloured on a viridis/magma-class ramp — NEVER a red/orange traffic-light ramp.
 * The red/amber/green triad is reserved exclusively for the calibration (model-quality) badge.
 *
 * We embed a small set of control colours sampled from the canonical viridis ramp and
 * linearly interpolate between them. Embedding the table (rather than pulling d3-scale-chromatic)
 * keeps the Monitoring bundle lean — the heavy bytes are MapLibre + deck.gl, not the colormap.
 *
 * The field value is the chosen bound's exceedance PROBABILITY in (0, 1). Probabilities span
 * orders of magnitude (typically < 1% per day; ICEF/Jordan et al. 2011), so the default
 * mapping is on a log scale between a floor and a ceiling, which is also how the numeric
 * legend is binned. A linear option is exposed for completeness.
 */

export type RGB = [number, number, number];
export type RGBA = [number, number, number, number];

/**
 * Viridis control points (t in [0,1] → sRGB). Sampled from the canonical matplotlib viridis
 * lookup table at 9 evenly spaced stops. Perceptually uniform, colour-vision-deficiency safe,
 * monotonic in lightness — the properties Schneider et al. (2022) call for.
 */
const VIRIDIS: RGB[] = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [110, 206, 88],
  [253, 231, 37],
];

/**
 * Magma control points (an alternative perceptually-uniform sequential ramp). Offered so the
 * dark-technical theme can use a ramp that reads well on a near-black base map without ever
 * being a traffic-light. NOTE: magma's high end is light-yellow, low end near-black — still
 * sequential, still NOT a danger ramp.
 */
const MAGMA: RGB[] = [
  [0, 0, 4],
  [28, 16, 68],
  [79, 18, 123],
  [129, 37, 129],
  [181, 54, 122],
  [229, 80, 100],
  [251, 135, 97],
  [254, 194, 135],
  [252, 253, 191],
];

export type RampName = "viridis" | "magma";

const RAMPS: Record<RampName, RGB[]> = { viridis: VIRIDIS, magma: MAGMA };

/** Linearly interpolate a control-point ramp at t in [0,1] → sRGB. */
export function sampleRamp(t: number, ramp: RampName = "viridis"): RGB {
  const stops = RAMPS[ramp];
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = stops[i] ?? stops[0]!;
  const b = stops[Math.min(i + 1, stops.length - 1)] ?? a;
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

export interface ScaleOptions {
  /** Lower bound of the value domain (probabilities below floor map to t=0). */
  floor?: number;
  /** Upper bound of the value domain (probabilities at/above ceil map to t=1). */
  ceil?: number;
  /** "log" (default — probabilities span orders of magnitude) or "linear". */
  mode?: "log" | "linear";
  ramp?: RampName;
  /** Alpha for the field polygons (0–255). */
  alpha?: number;
}

const DEFAULTS: Required<ScaleOptions> = {
  floor: 1e-4,
  ceil: 0.3,
  mode: "log",
  ramp: "viridis",
  alpha: 200,
};

/** Map a raw value to the normalized position t in [0,1] used by the ramp + the legend. */
export function valueToT(value: number, opts: ScaleOptions = {}): number {
  const { floor, ceil, mode } = { ...DEFAULTS, ...opts };
  if (value <= floor) return 0;
  if (value >= ceil) return 1;
  if (mode === "linear") return (value - floor) / (ceil - floor);
  const lf = Math.log(floor);
  const lc = Math.log(ceil);
  return (Math.log(value) - lf) / (lc - lf);
}

/** Map a raw probability value to an RGBA colour (deck.gl `getFillColor` shape). */
export function valueToColor(value: number, opts: ScaleOptions = {}): RGBA {
  const merged = { ...DEFAULTS, ...opts };
  const t = valueToT(value, merged);
  const [r, g, b] = sampleRamp(t, merged.ramp);
  return [r, g, b, merged.alpha];
}

/** CSS `rgb()` string for a value (used by the SVG no-map summary + the legend swatches). */
export function valueToCss(value: number, opts: ScaleOptions = {}): string {
  const [r, g, b] = valueToColor(value, opts);
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Build N legend stops across the domain (for the numeric legend). Returns each stop's
 * representative value and its CSS colour, log- or linear-spaced to match `valueToT`.
 */
export interface LegendStop {
  value: number;
  css: string;
  t: number;
}

export function legendStops(n = 6, opts: ScaleOptions = {}): LegendStop[] {
  const { floor, ceil, mode } = { ...DEFAULTS, ...opts };
  const out: LegendStop[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    let value: number;
    if (mode === "linear") {
      value = floor + (ceil - floor) * t;
    } else {
      value = Math.exp(Math.log(floor) + (Math.log(ceil) - Math.log(floor)) * t);
    }
    out.push({ value, t, css: valueToCss(value, opts) });
  }
  return out;
}

export const COLORMAP_DEFAULTS = DEFAULTS;

import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";
import { cellToLatLng } from "h3-js";

import { sampleRamp, valueToColor, valueToT, type RGB, type ScaleOptions } from "@/components/monitoring/colormap";
import type { CellSelection } from "@/data/types";

/** The two field renderings (purely visual — same data, same scale). */
export type FieldMode = "hexbins" | "heatmap";

/**
 * The world / per-country probability FIELD (web-app-spec.md §7.1, §7.2).
 *
 * Architecture exactly per spec:
 *   - MapLibre GL JS as the WebGL2 base map (vector borders/labels stay legible).
 *   - deck.gl overlay attached via `MapboxOverlay` with `interleaved: true`, so the
 *     probability surface renders INTO MapLibre's single WebGL context (labels above the
 *     field). The heavy data goes through deck.gl's `H3HexagonLayer` — NOT MapLibre vector
 *     polygons (which benchmark slow, Çabuk et al. 2025).
 *   - The hexagons are coloured on the perceptually-uniform SEQUENTIAL colormap
 *     (colormap.ts) — NEVER a red traffic-light ramp.
 *   - Cells in the coverage mask are NOT coloured by value; they render as a desaturated
 *     "out of coverage" hatch surrogate (blank ≠ safe). On a failed/stale run the whole field
 *     is desaturated by the parent (it passes `degraded`).
 *
 * This module is imported only by the lazily-loaded `MonitoringField` wrapper, so MapLibre +
 * deck.gl stay out of the text-route bundles (bundle discipline, §7.2). It is a self-contained
 * imperative MapLibre integration (no react-map-gl dependency required): we create the map,
 * add a `MapboxOverlay` control, and update deck layers when props change.
 *
 * A free demo raster/vector style is used as the base; the data host can swap in a self-hosted
 * style without any code change. No tokens, no keys — public-repo safe.
 */

/** Minimal raster base style (OSM tiles) — no API key, public-repo safe. The data host can
 * replace this with a self-hosted vector style; the overlay code is unchanged. */
const BASE_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    "osm-raster": {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [
    { id: "bg", type: "background", paint: { "background-color": "#0d1117" } },
    { id: "osm", type: "raster", source: "osm-raster", paint: { "raster-opacity": 0.55, "raster-saturation": -0.6 } },
  ],
};

export interface ProbabilityFieldMapProps {
  cells: CellSelection[];
  /** H3 keys explicitly out of validated coverage — rendered as a desaturated hatch. */
  coverageMask: string[];
  /** Initial centre [lng, lat] + zoom (region-aware; world default if absent). */
  center?: [number, number];
  zoom?: number;
  scale?: ScaleOptions;
  /** Stale/failed run → desaturate the whole field (visible degradation). */
  degraded?: boolean;
  /** Field rendering: discrete H3 hexbins (default) or a continuous KDE heatmap. Visual only —
   * both read the SAME per-cell values on the SAME perceptually-uniform colormap. */
  mode?: FieldMode;
  /** Called when a hexagon is clicked/hovered, for the drill-down panel. */
  onPick?: (cell: CellSelection | null) => void;
}

export function ProbabilityFieldMap({
  cells,
  coverageMask,
  center = [-71, -31],
  zoom = 4,
  scale,
  degraded = false,
  mode = "hexbins",
  onPick,
}: ProbabilityFieldMapProps) {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);
  const [ready, setReady] = useState(false);
  const [glError, setGlError] = useState<string | null>(null);

  const maskedSet = useMemo(() => new Set(coverageMask), [coverageMask]);

  // Coverage-mask cells as their own dataset (rendered desaturated/hatched).
  const maskCells = useMemo(
    () => coverageMask.map((h3) => ({ cell: h3 })),
    [coverageMask],
  );

  // Build deck layers from current props. Re-created on every relevant change.
  const buildLayers = useMemo(() => {
    return () => {
      const fieldCells = cells.filter((c) => !maskedSet.has(c.cell));
      const alphaScale: ScaleOptions = degraded ? { ...scale, alpha: 70 } : (scale ?? {});

      // Coverage mask: a flat desaturated grey hatch surrogate (blank != safe). Shared by both modes.
      const maskLayer = new H3HexagonLayer<{ cell: string }>({
        id: "coverage-mask",
        data: maskCells,
        pickable: false,
        filled: true,
        stroked: true,
        getHexagon: (d) => d.cell,
        getFillColor: [120, 120, 120, 60],
        getLineColor: [150, 150, 150, 120],
        lineWidthMinPixels: 0.5,
      });

      if (mode === "heatmap") {
        // Continuous KDE surface: each cell centroid weighted by its normalized value (same scale as
        // the hexbins), smoothed on the GPU. Colours come from the SAME perceptually-uniform ramp.
        const ramp = scale?.ramp ?? "viridis";
        const colorRange = [0.15, 0.3, 0.45, 0.6, 0.78, 0.95].map(
          (s) => sampleRamp(s, ramp) as RGB,
        );
        const points = fieldCells.map((c) => {
          const [lat, lng] = cellToLatLng(c.cell);
          return { position: [lng, lat] as [number, number], weight: valueToT(c.value, scale) };
        });
        const heatLayer = new HeatmapLayer<{ position: [number, number]; weight: number }>({
          id: "probability-heatmap",
          data: points,
          getPosition: (d) => d.position,
          getWeight: (d) => d.weight,
          colorRange,
          aggregation: "SUM",
          radiusPixels: 38,
          intensity: 1,
          threshold: 0.05,
          opacity: degraded ? 0.4 : 0.85,
          pickable: false,
          updateTriggers: { getWeight: [scale] },
        });
        // Invisible but pickable hexbins UNDER the heatmap, so the cell drill-down still works.
        const pickLayer = new H3HexagonLayer<CellSelection>({
          id: "heatmap-pick",
          data: fieldCells,
          pickable: true,
          filled: true,
          stroked: false,
          getHexagon: (d: CellSelection) => d.cell,
          getFillColor: [0, 0, 0, 0],
          onClick: (info) => onPick?.((info.object as CellSelection) ?? null),
        });
        return [maskLayer, pickLayer, heatLayer];
      }

      const fieldLayer = new H3HexagonLayer<CellSelection>({
        id: "probability-field",
        data: fieldCells,
        pickable: true,
        wireframe: false,
        filled: true,
        extruded: false,
        stroked: true,
        getHexagon: (d: CellSelection) => d.cell,
        getFillColor: (d: CellSelection) => valueToColor(d.value, alphaScale),
        getLineColor: [13, 17, 23, 120],
        lineWidthMinPixels: 0.5,
        onClick: (info) => onPick?.((info.object as CellSelection) ?? null),
        updateTriggers: {
          getFillColor: [degraded, scale],
        },
      });

      return [maskLayer, fieldLayer];
    };
  }, [cells, maskCells, maskedSet, scale, degraded, mode, onPick]);

  // Create the MapLibre map + deck overlay once.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    let map: maplibregl.Map;
    try {
      map = new maplibregl.Map({
        container: containerRef.current,
        style: BASE_STYLE,
        center,
        zoom,
        attributionControl: { compact: true },
      });
    } catch (e) {
      setGlError(e instanceof Error ? e.message : String(e));
      return;
    }
    mapRef.current = map;

    const overlay = new MapboxOverlay({ interleaved: true, layers: [] });
    overlayRef.current = overlay;
    map.addControl(overlay as unknown as maplibregl.IControl);
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

    map.on("load", () => setReady(true));
    map.on("error", (ev) => {
      // Tile-fetch errors are non-fatal (offline preview); a context-loss error is fatal.
      const msg = (ev as { error?: { message?: string } }).error?.message ?? "";
      if (/webgl|context/i.test(msg)) setGlError(msg);
    });

    return () => {
      map.remove();
      mapRef.current = null;
      overlayRef.current = null;
    };
    // center/zoom are only initial; intentionally omitted from deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push fresh layers whenever the field data / scale / degraded state changes.
  useEffect(() => {
    if (!ready || !overlayRef.current) return;
    overlayRef.current.setProps({ layers: buildLayers() });
  }, [ready, buildLayers]);

  if (glError) {
    // Honest no-WebGL message; the parent offers the no-map summary as the real fallback.
    return (
      <div className="map-fallback">
        <p>{t("monitoring.map.noWebgl")}</p>
        <p className="muted small">{glError}</p>
      </div>
    );
  }

  return (
    <div className="probability-field-map">
      <div ref={containerRef} className="map-canvas" aria-label={t("monitoring.map.aria")} />
      {!ready ? <div className="map-loading">{t("common.loading")}</div> : null}
    </div>
  );
}

export default ProbabilityFieldMap;

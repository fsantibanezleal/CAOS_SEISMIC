import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";

import { sampleRamp, type RGB } from "@/components/monitoring/colormap";
import type { OutlookCell } from "@/data/outlook";

/**
 * The 30-day outlook field as a continuous deck.gl HeatmapLayer over a desaturated MapLibre base — the
 * geodetic-context background expected-count surface. Lazily imported (MapLibre + deck.gl are heavy) so
 * the bundle discipline matches Monitoring. Same perceptually-uniform viridis ramp as the daily field.
 */

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

export interface OutlookFieldMapProps {
  field: OutlookCell[];
}

export function OutlookFieldMap({ field }: OutlookFieldMapProps) {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);
  const [ready, setReady] = useState(false);
  const [glError, setGlError] = useState<string | null>(null);

  // Normalize the per-cell expected count to [0,1] by the 98th percentile (robust to a single hot cell).
  const points = useMemo(() => {
    if (!field.length) return [] as { position: [number, number]; weight: number }[];
    const sorted = [...field.map((c) => c.n30)].sort((a, b) => a - b);
    const p98 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.98))] || 1;
    return field.map((c) => ({ position: [c.lon, c.lat] as [number, number], weight: Math.min(c.n30 / p98, 1) }));
  }, [field]);

  const buildLayers = useMemo(() => {
    return () => {
      const colorRange = [0.15, 0.3, 0.45, 0.6, 0.78, 0.95].map((s) => sampleRamp(s, "viridis") as RGB);
      return [
        new HeatmapLayer<{ position: [number, number]; weight: number }>({
          id: "outlook-30d-heatmap",
          data: points,
          getPosition: (d) => d.position,
          getWeight: (d) => d.weight,
          colorRange,
          aggregation: "SUM",
          radiusPixels: 34,
          intensity: 1,
          threshold: 0.05,
          opacity: 0.85,
          pickable: false,
        }),
      ];
    };
  }, [points]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    let map: maplibregl.Map;
    try {
      map = new maplibregl.Map({
        container: containerRef.current,
        style: BASE_STYLE,
        center: [10, 20],
        zoom: 1.1,
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
      const msg = (ev as { error?: { message?: string } }).error?.message ?? "";
      if (/webgl|context/i.test(msg)) setGlError(msg);
    });
    return () => {
      map.remove();
      mapRef.current = null;
      overlayRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!ready || !overlayRef.current) return;
    overlayRef.current.setProps({ layers: buildLayers() });
  }, [ready, buildLayers]);

  if (glError) {
    return (
      <div className="map-fallback">
        <p>{t("outlook.noWebgl")}</p>
        <p className="muted small">{glError}</p>
      </div>
    );
  }
  return (
    <div className="probability-field-map">
      <div ref={containerRef} className="map-canvas" aria-label={t("outlook.mapAria")} />
      {!ready ? <div className="map-loading">{t("common.loading")}</div> : null}
    </div>
  );
}

export default OutlookFieldMap;

import { Suspense, lazy } from "react";
import { useTranslation } from "react-i18next";

import type { ProbabilityFieldMapProps } from "@/components/monitoring/ProbabilityFieldMap";

/**
 * Lazy, code-split wrapper around the heavy MapLibre + deck.gl field
 * (web-app-spec.md §7.2 bundle discipline). The text routes never load these bytes; only the
 * Monitoring route, and only when the user keeps the map view (the no-map summary is the
 * zero-map-bundle default-capable alternative). `React.lazy` + a dynamic import puts MapLibre
 * and deck.gl in their own chunk (vite.config.ts already isolates them into the `maplibre` /
 * `deckgl` vendor chunks).
 */
const ProbabilityFieldMap = lazy(() => import("@/components/monitoring/ProbabilityFieldMap"));

export function MonitoringField(props: ProbabilityFieldMapProps) {
  const { t } = useTranslation();
  return (
    <Suspense fallback={<div className="map-loading map-canvas">{t("monitoring.map.loadingStack")}</div>}>
      <ProbabilityFieldMap {...props} />
    </Suspense>
  );
}

// Single source of truth for the six product routes.
//
// Both the router (router.tsx) and the header navigation (components/Layout.tsx) read this
// list, so the nav can never drift from what is actually routed. `labelKey` is the i18n key
// under `nav.*`; `path` is the URL path relative to the app base.
//
// Order matters: it is the canonical page order from web-app-spec.md §1
// (Introduction → Problem → Methodology → Implementation → Back-analysis → Monitoring).

export interface RouteDef {
  /** URL path relative to the router basename (no leading slash except the index "/"). */
  path: string;
  /** i18n key under `nav.*` for the visible label. */
  labelKey: string;
  /** Stable id (handy for keys / analytics). */
  id: string;
}

export const ROUTES: readonly RouteDef[] = [
  { id: "introduction", path: "/", labelKey: "nav.introduction" },
  { id: "problem", path: "/problem", labelKey: "nav.problem" },
  { id: "methodology", path: "/methodology", labelKey: "nav.methodology" },
  { id: "implementation", path: "/implementation", labelKey: "nav.implementation" },
  { id: "back-analysis", path: "/back-analysis", labelKey: "nav.backAnalysis" },
  { id: "monitoring", path: "/monitoring", labelKey: "nav.monitoring" },
] as const;

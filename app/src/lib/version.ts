// Build/version provenance surfaced in the footer. APP_VERSION is bumped by
// hand per release (X.XX.XXX-style); the rest are injected at build time by
// vite.config.ts `define`.

export const APP_VERSION = "0.1.000";

export const APP_BUILD_TIME: string =
  typeof __APP_BUILD_TIME__ !== "undefined" ? __APP_BUILD_TIME__ : "dev";

export const APP_COMMIT_SHA: string =
  typeof __APP_COMMIT_SHA__ !== "undefined" ? __APP_COMMIT_SHA__ : "dev";

export const APP_BRANCH: string =
  typeof __APP_BRANCH__ !== "undefined" ? __APP_BRANCH__ : "dev";

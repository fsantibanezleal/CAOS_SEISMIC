import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { execSync } from "node:child_process";

// The SPA is static: at runtime it only reads the precomputed daily forecast
// artifact (served as a static file / via a thin read-only API). There is no
// backend in this repo. The dev proxy below is only for local convenience when
// an artifact-serving endpoint happens to be running on :8108.
const apiTarget = process.env.VITE_PROXY_TARGET ?? "http://127.0.0.1:8108";

function git(cmd: string): string {
  try {
    return execSync(cmd, { stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
  } catch {
    return "unknown";
  }
}

const BUILD_TIME = new Date().toISOString();
const COMMIT_SHA = git("git rev-parse --short HEAD");
const BRANCH = git("git rev-parse --abbrev-ref HEAD");

export default defineConfig({
  plugins: [react()],
  define: {
    __APP_BUILD_TIME__: JSON.stringify(BUILD_TIME),
    __APP_COMMIT_SHA__: JSON.stringify(COMMIT_SHA),
    __APP_BRANCH__: JSON.stringify(BRANCH),
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      // Read-only artifact endpoints (see web-app-spec §8.3). Optional in dev.
      "/api": { target: apiTarget, changeOrigin: true },
    },
  },
  build: {
    target: "es2022",
    sourcemap: false,
    // Bundle ALL CSS into one stylesheet (no per-chunk CSS). The lazy map route imports
    // maplibre-gl's stylesheet; with cssCodeSplit on, Vite emits a separate `maplibre-*.css`
    // chunk and `__vitePreload` tries to preload it when the route loads — which throws
    // "Unable to preload CSS" on some hosts/networks even though the file is served (a known
    // Vite issue). One combined stylesheet eliminates that preload step entirely. Total CSS is
    // small (~125 KB raw / ~25 KB gzip), so loading it upfront is a non-issue.
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        // Keep the heavy map stack (MapLibre + deck.gl) in its own chunk so the
        // text-only routes (Intro/Problem/Methodology/Implementation) stay light.
        // The Monitoring route lazy-loads it; this just isolates the vendor bytes.
        manualChunks: (id) => {
          if (id.includes("node_modules/react-router")) return "router";
          if (id.includes("node_modules/react-dom") || id.includes("node_modules/react/"))
            return "react";
          if (id.includes("node_modules/i18next") || id.includes("node_modules/react-i18next"))
            return "i18n";
          if (id.includes("node_modules/zustand")) return "zustand";
          if (id.includes("node_modules/katex") || id.includes("node_modules/react-katex"))
            return "katex";
          if (id.includes("node_modules/maplibre-gl")) return "maplibre";
          if (id.includes("node_modules/@deck.gl") || id.includes("node_modules/h3-js"))
            return "deckgl";
          return undefined;
        },
      },
    },
  },
});

#!/usr/bin/env bash
# dev.sh — serve the static web app locally (preview the SPA; NOT a processing backend).
#
# The web app is a pure static viewer: it reads the precomputed daily forecast artifact from
# app/public/data/ (or results/ at deploy time). There is NO server-side computation. This script just
# serves the static files for local preview.
#
#   scripts/dev.sh                  # `npm run dev` if app/node_modules is present (Vite HMR),
#                                   # otherwise a plain static server over the last build / public dir.
#   scripts/dev.sh --build          # `npm run build` first, then serve app/dist statically.
#   scripts/dev.sh --static --port 8000   # force the plain static server (no Node/Vite needed).
#
# Public-safe: resolves paths relative to the repo root; binds to 127.0.0.1; no secrets.

set -euo pipefail
# shellcheck source=_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PORT=5173
STATIC=0
BUILD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port|-p)  PORT="$2"; shift 2 ;;
    --static|-s) STATIC=1; shift ;;
    --build|-b) BUILD=1; shift ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac
done

APP_DIR="${REPO_ROOT}/app"
[ -d "${APP_DIR}" ] || { err "app/ directory not found at ${APP_DIR}."; exit 1; }

# Serve a directory with a dependency-free static server (prefers the venv Python, then any python).
serve_static() {
  local dir="$1" serve_port="$2" py=""
  if [ ! -d "${dir}" ]; then
    err "nothing to serve: '${dir}' does not exist. Run with --build (needs npm) or build the SPA first."
    exit 1
  fi
  if [ -x "${VENV_DIR}/bin/python" ]; then py="${VENV_DIR}/bin/python"
  elif [ -x "${VENV_DIR}/Scripts/python.exe" ]; then py="${VENV_DIR}/Scripts/python.exe"
  elif command -v python3 >/dev/null 2>&1; then py="python3"
  elif command -v python  >/dev/null 2>&1; then py="python"
  fi
  [ -n "${py}" ] || { err "no Python for the static server. Run scripts/setup.sh, or install Node and use npm."; exit 1; }
  step "Static server: http://127.0.0.1:${serve_port}  (serving ${dir})"
  info "This is a STATIC viewer — it computes nothing. Ctrl+C to stop."
  ( cd "${dir}" && exec "${py}" -m http.server "${serve_port}" --bind 127.0.0.1 )
}

# Pick a built dist/ if present, else the public/ dir.
serve_dir() {
  if [ -d "${APP_DIR}/dist" ]; then printf '%s\n' "${APP_DIR}/dist"; else printf '%s\n' "${APP_DIR}/public"; fi
}

if [ "${BUILD}" -eq 1 ]; then
  command -v npm >/dev/null 2>&1 || { err "--build requires npm (Node.js) on PATH."; exit 1; }
  step "Building the SPA (npm install + npm run build)"
  ( cd "${APP_DIR}" && { [ -d node_modules ] || npm install; } && npm run build )
  serve_static "${APP_DIR}/dist" "${PORT}"
  exit 0
fi

if [ "${STATIC}" -eq 1 ]; then
  serve_static "$(serve_dir)" "${PORT}"
  exit 0
fi

# Default: Vite dev server (HMR) if Node is available; otherwise degrade to the static server.
if command -v npm >/dev/null 2>&1; then
  step "Vite dev server (npm run dev) on http://127.0.0.1:${PORT}"
  info "Static viewer with HMR — no processing backend. Ctrl+C to stop."
  ( cd "${APP_DIR}" \
      && { [ -d node_modules ] || { info "installing app dependencies (first run)..."; npm install; }; } \
      && npm run dev -- --port "${PORT}" --strictPort --host 127.0.0.1 )
else
  warn "npm not found — falling back to the dependency-free static server."
  serve_static "$(serve_dir)" "${PORT}"
fi

#!/usr/bin/env pwsh
# dev.ps1 - serve the static web app locally (preview the SPA; NOT a processing backend).
#
# The web app is a pure static viewer: it reads the precomputed daily forecast artifact from
# app/public/data/ (or results/ at deploy time). There is NO server-side computation. This script just
# serves the static files for local preview.
#
#   .\scripts\dev.ps1                 # `npm run dev` if app/node_modules is present (Vite HMR),
#                                     # otherwise a plain static server over the last build / public dir.
#   .\scripts\dev.ps1 -Build          # `npm run build` first, then serve app/dist statically.
#   .\scripts\dev.ps1 -Static -Port 8000   # force the plain static server (no Node/Vite needed).
#
# Public-safe: resolves paths relative to the repo root; binds to 127.0.0.1; no secrets.

[CmdletBinding()]
param(
  [int]$Port = 5173,
  [switch]$Static,   # force the dependency-free Python static server (skip Vite)
  [switch]$Build     # build the SPA first, then serve app/dist
)

. (Join-Path $PSScriptRoot '_common.ps1')

$repo = Get-RepoRoot
$appDir = Join-Path $repo 'app'
if (-not (Test-Path $appDir)) { throw "app/ directory not found at $appDir." }

$npm = Get-Command 'npm' -ErrorAction SilentlyContinue

# Helper: serve a directory with a dependency-free static server (prefers the venv Python, then any python).
function Start-StaticServer([string]$Dir, [int]$ServePort) {
  if (-not (Test-Path $Dir)) {
    throw "nothing to serve: '$Dir' does not exist. Run with -Build (needs npm) or build the SPA first."
  }
  $py = $null
  try { $py = Get-VenvPython } catch { }
  if (-not $py) {
    $cmd = Get-Command 'python' -ErrorAction SilentlyContinue
    if (-not $cmd) { $cmd = Get-Command 'python3' -ErrorAction SilentlyContinue }
    if ($cmd) { $py = $cmd.Source }
  }
  if (-not $py) { throw "no Python interpreter available for the static server. Run scripts\setup.ps1, or install Node and run with npm." }
  Write-Step "Static server: http://127.0.0.1:$ServePort  (serving $Dir)"
  Write-Info  "This is a STATIC viewer - it computes nothing. Ctrl+C to stop."
  Push-Location $Dir
  try {
    & $py '-m' 'http.server' "$ServePort" '--bind' '127.0.0.1'
  } finally {
    Pop-Location
  }
}

if ($Build) {
  if (-not $npm) { throw "-Build requires npm (Node.js) on PATH." }
  Write-Step "Building the SPA (npm install + npm run build)"
  Push-Location $appDir
  try {
    if (-not (Test-Path (Join-Path $appDir 'node_modules'))) { & npm install }
    & npm run build
    if ($LASTEXITCODE -ne 0) { throw "npm run build failed." }
  } finally { Pop-Location }
  Start-StaticServer (Join-Path $appDir 'dist') $Port
  return
}

if ($Static) {
  # Prefer a built dist/, fall back to the public/ dir (has the sample artifact + index.html assets).
  $dist = Join-Path $appDir 'dist'
  $pub = Join-Path $appDir 'public'
  $serveDir = if (Test-Path $dist) { $dist } else { $pub }
  Start-StaticServer $serveDir $Port
  return
}

# Default: Vite dev server (HMR) if Node is available; otherwise degrade to the static server.
if ($npm) {
  Write-Step "Vite dev server (npm run dev) on http://127.0.0.1:$Port"
  Write-Info  "Static viewer with HMR - no processing backend. Ctrl+C to stop."
  Push-Location $appDir
  try {
    if (-not (Test-Path (Join-Path $appDir 'node_modules'))) {
      Write-Info "installing app dependencies (first run)..."
      & npm install
    }
    # Pass Vite flags through npm. The '--' separator and flags are kept in an array so PowerShell does
    # not try to interpret the bare '--' as its own end-of-parameters token.
    $viteArgs = @('run', 'dev', '--', '--port', "$Port", '--strictPort', '--host', '127.0.0.1')
    & npm @viteArgs
  } finally { Pop-Location }
} else {
  Write-Warn2 "npm not found - falling back to the dependency-free static server."
  $dist = Join-Path $appDir 'dist'
  $pub = Join-Path $appDir 'public'
  $serveDir = if (Test-Path $dist) { $dist } else { $pub }
  Start-StaticServer $serveDir $Port
}

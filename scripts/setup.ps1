#!/usr/bin/env pwsh
# setup.ps1 - create the project virtualenv and install the Python dependencies.
#
#   .\scripts\setup.ps1            # create .venv (Python 3.12 if available) + pip install -r requirements.txt
#   .\scripts\setup.ps1 -Force     # recreate the .venv from scratch
#
# Public-safe: resolves everything relative to the repo root; no secrets, no machine-specific paths.

[CmdletBinding()]
param(
  [switch]$Force
)

. (Join-Path $PSScriptRoot '_common.ps1')

$repo = Get-RepoRoot
$venv = Join-Path $repo '.venv'

if ($Force -and (Test-Path $venv)) {
  Write-Step "Removing existing virtualenv (-Force) at $venv"
  Remove-Item -Recurse -Force $venv
}

if (-not (Test-Path $venv)) {
  $boot = Get-BootstrapPython          # e.g. @('py','-3.12') or @('python')
  Write-Step "Creating virtualenv with: $($boot -join ' ') -m venv .venv"
  & $boot[0] @($boot[1..($boot.Length - 1)]) '-m' 'venv' $venv
  if ($LASTEXITCODE -ne 0) { throw "failed to create the virtualenv." }
} else {
  Write-Info "virtualenv already present at $venv (use -Force to recreate)."
}

$py = Get-VenvPython
Write-Step "Upgrading pip / setuptools / wheel"
& $py -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "failed to upgrade pip." }

$req = Join-Path $repo 'requirements.txt'
Write-Step "Installing requirements from requirements.txt"
& $py -m pip install -r $req
if ($LASTEXITCODE -ne 0) { throw "failed to install requirements." }

# Install the package itself (editable) so `caos-seismic` / `python -m caos_seismic.cli` resolves.
Write-Step "Installing the caos-seismic package (editable)"
& $py -m pip install -e $repo
if ($LASTEXITCODE -ne 0) { throw "failed to install the package (editable)." }

Write-Step "Smoke test: caos-seismic version"
& $py -m caos_seismic.cli version
if ($LASTEXITCODE -ne 0) { throw "the package did not import cleanly after install." }

Write-Step "Setup complete."
Write-Info "Next:  .\scripts\check.ps1   then   .\scripts\fetch.ps1"

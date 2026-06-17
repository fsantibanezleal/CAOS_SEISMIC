# _common.ps1 - shared helpers for the CAOS_SEISMIC PowerShell scripts.
# Dot-sourced by setup.ps1, fetch.ps1, build-features.ps1, train.ps1, infer.ps1, daily.ps1, dev.ps1,
# check.ps1, schedule-daily.ps1. Public-safe: no secrets, no machine-specific paths (everything is
# resolved relative to the repo root).

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Repo root = parent of the scripts/ directory that contains this file.
$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$script:VenvDir = Join-Path $RepoRoot '.venv'

function Get-RepoRoot { return $script:RepoRoot }

function Write-Step([string]$Message) {
  Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Info([string]$Message) {
  Write-Host "    $Message" -ForegroundColor DarkGray
}

function Write-Warn2([string]$Message) {
  Write-Host "WARN: $Message" -ForegroundColor Yellow
}

function Write-Err2([string]$Message) {
  Write-Host "ERROR: $Message" -ForegroundColor Red
}

# Resolve the .venv Python interpreter (Windows layout). Throws if the venv is absent.
function Get-VenvPython {
  $py = Join-Path $script:VenvDir 'Scripts\python.exe'
  if (-not (Test-Path $py)) {
    throw "virtualenv not found at '$($script:VenvDir)'. Run  scripts\setup.ps1  first."
  }
  return $py
}

# Pick an interpreter to BOOTSTRAP the venv: prefer Python 3.12 via the launcher, else python/python3.
function Get-BootstrapPython {
  # Try the Windows launcher pinned to 3.12.
  $launcher = Get-Command 'py' -ErrorAction SilentlyContinue
  if ($launcher) {
    try {
      & py -3.12 --version *> $null
      if ($LASTEXITCODE -eq 0) { return @('py', '-3.12') }
    } catch { }
  }
  foreach ($name in @('python', 'python3')) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { return @($cmd.Source) }
  }
  throw "no Python interpreter found on PATH (need Python 3.12). Install it from https://www.python.org/downloads/"
}

# Invoke the package console entry point inside the venv:  caos-seismic <args...>
# We call the module form so it works even if the console-script shim is not on PATH.
function Invoke-Caos {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
  $py = Get-VenvPython
  Push-Location $script:RepoRoot
  try {
    & $py '-m' 'caos_seismic.cli' @Args
    if ($LASTEXITCODE -ne 0) { throw "caos-seismic $($Args -join ' ')  exited with code $LASTEXITCODE." }
  } finally {
    Pop-Location
  }
}

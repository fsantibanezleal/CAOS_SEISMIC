#!/usr/bin/env pwsh
# outlook.ps1 - the WEEKLY 30-day outlook job: fit the geodetic neural background -> validate at 30 d ->
# SCOPED publish, via the self-sufficient CLI (`caos-seismic outlook`).
#
# A thin wrapper over `caos-seismic outlook`, mirroring scripts\daily.ps1. The geodetic context beats ETAS
# only at the 30-day horizon (experiments E11/E14), and that background is time-flat, so this runs on a
# WEEKLY cadence (separate from the daily ETAS job, scripts\daily.ps1). It fits the strain-neural, writes
# the 30-day field + per-view validation evidence under results/, and (unless -NoPublish) performs the same
# scoped commit-tree publish the daily job uses (NEVER `git add -A`; abort on out-of-allowlist staged paths;
# push to origin/main robust to the working branch). Memory-safe on the full global catalog (E14 fix).
#
#   .\scripts\outlook.ps1                 # full job (fit + validate + publish), region global
#   .\scripts\outlook.ps1 -NoPublish      # generate only (local dry run, no commit/push)
#
# Public-safe: no secrets, no machine-specific paths. The push credential lives only in the local git
# credential store (a least-privilege deploy key / fine-grained PAT scoped to THIS repo) -- never here.

[CmdletBinding()]
param(
  [string]$Region = 'global',
  [switch]$NoPublish
)

. (Join-Path $PSScriptRoot '_common.ps1')

$caosArgs = @('outlook', '--region', $Region)
if ($NoPublish) { $caosArgs += '--no-publish' }

Write-Step "outlook  (caos-seismic $($caosArgs -join ' '))"
Invoke-Caos @caosArgs
Write-Step "outlook  done."

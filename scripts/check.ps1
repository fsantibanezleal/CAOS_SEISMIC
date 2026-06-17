#!/usr/bin/env pwsh
# check.ps1 - environment + repo + config sanity checks (no network, no science deps required).
#
#   .\scripts\check.ps1 -Region chile
#
# Thin wrapper over:  caos-seismic check --region <id>
# Exits non-zero if any hard check fails (used as a pre-flight in CI / before the daily job).

[CmdletBinding()]
param(
  [string]$Region = 'global'
)

. (Join-Path $PSScriptRoot '_common.ps1')

Write-Step "caos-seismic check --region $Region"
Invoke-Caos 'check' '--region' $Region

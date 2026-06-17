#!/usr/bin/env pwsh
# backanalysis.ps1 - pseudo-prospective CSEP back-analysis over a date range.
#
#   .\scripts\backanalysis.ps1 -Region chile -Start 2024-01-01 -End 2024-12-31
#
# The forecast clock advances day by day; each forecast sees only the catalog slice (-inf, t) and is
# scored against the catalog as it was at issue time.
#
# Thin wrapper over:  caos-seismic backanalysis --region <id> --start YYYY-MM-DD --end YYYY-MM-DD

[CmdletBinding()]
param(
  [string]$Region = 'chile',
  [Parameter(Mandatory = $true)][string]$Start,
  [Parameter(Mandatory = $true)][string]$End
)

. (Join-Path $PSScriptRoot '_common.ps1')

Write-Step "caos-seismic backanalysis --region $Region --start $Start --end $End"
Invoke-Caos 'backanalysis' '--region' $Region '--start' $Start '--end' $End

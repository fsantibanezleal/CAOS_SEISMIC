#!/usr/bin/env pwsh
# build-features.ps1 - Mc + b-value, Mw homogenization, dual-catalog declustering, feature extraction.
#
#   .\scripts\build-features.ps1 -Region chile
#
# Thin wrapper over:  caos-seismic build-features --region <id>

[CmdletBinding()]
param(
  [string]$Region = 'chile'
)

. (Join-Path $PSScriptRoot '_common.ps1')

Write-Step "caos-seismic build-features --region $Region"
Invoke-Caos 'build-features' '--region' $Region

#!/usr/bin/env pwsh
# train.ps1 - fit the stationary smoothed-seismicity null + space-time ETAS (+ R-J fallback).
#
#   .\scripts\train.ps1 -Region chile
#
# Thin wrapper over:  caos-seismic train --region <id>

[CmdletBinding()]
param(
  [string]$Region = 'chile'
)

. (Join-Path $PSScriptRoot '_common.ps1')

Write-Step "caos-seismic train --region $Region"
Invoke-Caos 'train' '--region' $Region

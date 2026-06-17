#!/usr/bin/env pwsh
# infer.ps1 - run the daily forecast clock for an issue date -> compact artifact under results/.
#
#   .\scripts\infer.ps1                                 # issue date = today (UTC)
#   .\scripts\infer.ps1 -Region chile -Issue 2026-06-16 # a specific issue date
#
# Thin wrapper over:  caos-seismic infer --region <id> [--issue YYYY-MM-DD]

[CmdletBinding()]
param(
  [string]$Region = 'chile',
  [string]$Issue
)

. (Join-Path $PSScriptRoot '_common.ps1')

$argv = @('infer', '--region', $Region)
if ($Issue) { $argv += @('--issue', $Issue) }

Write-Step "caos-seismic $($argv -join ' ')"
Invoke-Caos @argv

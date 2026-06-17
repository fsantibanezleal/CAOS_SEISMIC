#!/usr/bin/env pwsh
# fetch.ps1 - pull the recent + historical catalog (ComCat spine + regional/anchor sources).
#
#   .\scripts\fetch.ps1                         # default region (chile), configured fetch window
#   .\scripts\fetch.ps1 -Region chile -Days 30  # only the last 30 days
#
# Thin wrapper over:  caos-seismic fetch --region <id> [--days N] [--focus key]

[CmdletBinding()]
param(
  [string]$Region = 'chile',
  [int]$Days,
  [string]$Focus
)

. (Join-Path $PSScriptRoot '_common.ps1')

$argv = @('fetch', '--region', $Region)
if ($PSBoundParameters.ContainsKey('Days'))  { $argv += @('--days', "$Days") }
if ($Focus)                                  { $argv += @('--focus', $Focus) }

Write-Step "caos-seismic $($argv -join ' ')"
Invoke-Caos @argv

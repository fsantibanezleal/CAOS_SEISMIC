#!/usr/bin/env pwsh
# daily.ps1 - the PRODUCTION daily job: fetch -> infer -> SCOPED publish, via the self-sufficient CLI.
#
# A thin wrapper over `caos-seismic daily`, which performs fetch + infer (today + any MISSED prior days)
# AND the scoped git publish (stage ONLY configs/publish.yaml `git.add_allowlist` = results/ + manifests/,
# abort if anything outside the allowlist is staged, commit with the configured prefix, push) ENTIRELY in
# Python -- the single, robust source of truth. The scheduled task does NOT call this script: it runs
# scripts\job.ps1 -Job daily from the dedicated job checkout (scripts\setup-job-checkout.ps1), which syncs
# that checkout first and is the only place that publishes (docs\deploy.md section 4). Use this script for
# a local dry run.
#
#   .\scripts\daily.ps1 -NoPublish       # fetch + infer only (local dry run, no commit/push)
#   .\scripts\daily.ps1 -NoCatchUp       # only today's issue date (skip missed-day backfill)
#   .\scripts\daily.ps1                  # full job, including the publish (meant for the job checkout)
#
# Scoped-publish discipline lives in the CLI (cli.py `_publish_scoped`): NEVER `git add -A`/`.`, abort on
# any out-of-allowlist staged path. Public-safe: no secrets, no machine-specific paths. The push
# credential lives only in the local git credential store (a least-privilege deploy key / fine-grained PAT
# scoped to THIS repo) -- never here.

[CmdletBinding()]
param(
  [string]$Region = 'global',
  [switch]$NoPublish,
  [switch]$NoCatchUp
)

. (Join-Path $PSScriptRoot '_common.ps1')

$caosArgs = @('daily', '--region', $Region)
if ($NoPublish) { $caosArgs += '--no-publish' }
if ($NoCatchUp) { $caosArgs += '--no-catch-up' }

Write-Step "daily  (caos-seismic $($caosArgs -join ' '))"
Invoke-Caos @caosArgs
Write-Step "daily  done."

#!/usr/bin/env pwsh
# daily.ps1 - the PRODUCTION daily job: fetch -> infer -> SCOPED publish (commit + push).
#
# This is what the Windows Task Scheduler task (scripts\schedule-daily.ps1) and the systemd timer fire
# once per day (~03:00 local, configs/publish.yaml `schedule.time_local`).
#
# Flow:
#   1. caos-seismic fetch  --region <id>          (refresh the catalog)
#   2. caos-seismic infer  --region <id>          (today + any MISSED prior days - catch-up)
#   3. SCOPED publish: stage ONLY the configs/publish.yaml `git.add_allowlist` paths (results/,
#      manifests/), ABORT if anything outside the allowlist is staged, commit with the configured
#      prefix, push.
#
# Scoped-publish discipline (hard rules - this machine also has data/, models/, .venv/, .env):
#   * NEVER `git add -A` / `git add .` - only the explicit allowlist paths.
#   * ABORT the commit if any staged path falls outside the allowlist.
#
#   .\scripts\daily.ps1                  # full job (fetch + infer + publish), region chile
#   .\scripts\daily.ps1 -NoPublish       # fetch + infer only (local dry run, no commit/push)
#   .\scripts\daily.ps1 -NoCatchUp       # only today's issue date (skip missed-day backfill)
#
# Public-safe: no secrets, no machine-specific paths. The push credential lives only in the local git
# credential store (a least-privilege deploy key / fine-grained PAT scoped to THIS repo) - never here.

[CmdletBinding()]
param(
  [string]$Region = 'global',
  [switch]$NoPublish,
  [switch]$NoCatchUp
)

. (Join-Path $PSScriptRoot '_common.ps1')

$repo = Get-RepoRoot

#  Read the publish config (allowlist, commit prefix, remote, branch) 
function Get-PublishConfig {
  $py = Get-VenvPython
  $code = @'
import json, sys
sys.path.insert(0, "src")
from caos_seismic.config import load
cfg = load("publish")
git = cfg.get("git", {}) or {}
print(json.dumps({
    "allowlist": [str(p) for p in git.get("add_allowlist", [])],
    "prefix": str(git.get("commit_message_prefix", "data: daily forecast")),
    "remote": str(git.get("remote", "origin")),
    "branch": str(git.get("publish_branch", "main")),
}))
'@
  Push-Location $repo
  try {
    $out = & $py '-c' $code
    if ($LASTEXITCODE -ne 0) { throw "failed to read configs/publish.yaml." }
  } finally { Pop-Location }
  return ($out | ConvertFrom-Json)
}

#  Determine which issue dates to run (today + missed prior days, bounded to a week) 
function Get-IssueDates([string]$RegionId, [bool]$CatchUp) {
  $today = (Get-Date).ToUniversalTime().Date
  $dates = New-Object System.Collections.Generic.List[datetime]

  $resultsDir = Join-Path $repo 'results'
  $have = New-Object System.Collections.Generic.HashSet[string]
  if (Test-Path $resultsDir) {
    Get-ChildItem -Path $resultsDir -Filter "forecast-$RegionId-*.json*" -File -ErrorAction SilentlyContinue | ForEach-Object {
      $stem = $_.Name
      foreach ($ext in @('.json.gz', '.json')) { if ($stem.EndsWith($ext)) { $stem = $stem.Substring(0, $stem.Length - $ext.Length); break } }
      $parts = $stem -split '-'
      if ($parts.Length -ge 3) {
        $token = ($parts[($parts.Length - 3)..($parts.Length - 1)]) -join '-'
        [void]$have.Add($token)
      }
    }
  }

  if ($CatchUp) {
    for ($back = 7; $back -ge 0; $back--) {
      $d = $today.AddDays(-$back)
      if (-not $have.Contains($d.ToString('yyyy-MM-dd'))) { $dates.Add($d) }
    }
  }
  if (-not ($dates -contains $today)) { $dates.Add($today) }
  return ($dates | Sort-Object -Unique)
}

#  Scoped publish: stage ONLY the allowlist, abort on any out-of-allowlist staged path, commit, push 
function Publish-Scoped($cfg, [int]$NDates) {
  $allowlist = @($cfg.allowlist)
  if ($allowlist.Count -eq 0) { throw "publish.yaml git.add_allowlist is empty; refusing to publish." }
  foreach ($entry in $allowlist) {
    if ($entry.Trim() -in @('.', '-A', '--all', '*')) {
      throw "publish.yaml allowlist has a non-scoped entry '$entry'; refusing to publish."
    }
  }

  Push-Location $repo
  try {
    # Reset the index so a pre-existing staged change cannot ride along.
    git reset -q | Out-Null
    foreach ($entry in $allowlist) { git add -- $entry; if ($LASTEXITCODE -ne 0) { throw "git add '$entry' failed." } }

    $staged = @(git diff --cached --name-only | Where-Object { $_.Trim() })
    if ($staged.Count -eq 0) { Write-Info "nothing to commit (no new artifacts)."; return }

    # Guard: every staged path MUST be under an allowlist entry.
    $allowedPrefixes = $allowlist | ForEach-Object { $_.TrimEnd('/') }
    $offenders = @()
    foreach ($s in $staged) {
      $ok = $false
      foreach ($p in $allowedPrefixes) { if ($s -eq $p -or $s.StartsWith("$p/")) { $ok = $true; break } }
      if (-not $ok) { $offenders += $s }
    }
    if ($offenders.Count -gt 0) {
      git reset -q | Out-Null
      throw "scoped-publish guard tripped: out-of-allowlist paths were staged ($($offenders -join ', ')). Index reset; nothing committed."
    }

    $todayIso = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd')
    $suffix = "$Region $todayIso"
    if ($NDates -gt 1) { $suffix += " (+$($NDates - 1) catch-up)" }
    $message = "$($cfg.prefix): $suffix"
    git commit -q -m $message
    if ($LASTEXITCODE -ne 0) { throw "git commit failed." }
    Write-Step "committed: $message"

    git push $cfg.remote "HEAD:$($cfg.branch)"
    if ($LASTEXITCODE -ne 0) {
      Write-Warn2 "commit made locally but push failed (remote '$($cfg.remote)', branch '$($cfg.branch)'). Configure the deploy credential and retry; the commit is preserved."
      throw "push failed."
    }
    Write-Step "pushed to $($cfg.remote) $($cfg.branch)."
  } finally { Pop-Location }
}

#  Run 
$cfg = Get-PublishConfig
$catchUp = -not $NoCatchUp
$issueDates = Get-IssueDates $Region $catchUp
$isoDates = ($issueDates | ForEach-Object { $_.ToString('yyyy-MM-dd') })
Write-Step "daily  region=$Region  issue_dates=$($isoDates -join ', ')"

# 1) Fetch once - the freshest catalog covers every issue date in this batch.
Write-Step "fetch"
Invoke-Caos 'fetch' '--region' $Region

# 2) Infer for each issue date (oldest first, so results/index.json ends current).
foreach ($iso in $isoDates) {
  Write-Step "infer  issue=$iso"
  Invoke-Caos 'infer' '--region' $Region '--issue' $iso
}

if ($NoPublish) {
  Write-Step "daily  -NoPublish set; produced $($isoDates.Count) artifact(s), not committing."
  return
}

# 3) Scoped publish.
Write-Step "publish (scoped)"
Publish-Scoped $cfg $issueDates.Count
Write-Step "daily  done."

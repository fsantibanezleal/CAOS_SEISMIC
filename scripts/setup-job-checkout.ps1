#!/usr/bin/env pwsh
# setup-job-checkout.ps1 - create the dedicated JOB checkout that the scheduled tasks run from.
#
# Run it from the developer checkout. The job checkout is a git worktree of this repository with a
# DETACHED HEAD at origin/<publish branch>, outside the developer checkout and used by nothing but
# scripts\job.ps1 (docs\deploy.md section 4). This script:
#   1. fetches the publish branch and adds the worktree (default: a sibling folder named <repo>.job);
#   2. writes the marker file that the CLI requires before it syncs or publishes (.caos-seismic-job);
#   3. copies the gitignored stores the pipeline reads: every file git ignores under data\ and results\
#      (the raw and clean catalogs, the enricher caches, the neural weights and checkpoints). From then on
#      the job checkout owns its copy; the developer checkout's copy is never read by the jobs;
#   4. prints the commands that register the two scheduled tasks against the job checkout. Those need an
#      ELEVATED PowerShell (the tasks run whether you are logged on or not, at the highest run level).
#
#   .\scripts\setup-job-checkout.ps1                          # creates <parent>\<repo>.job
#   .\scripts\setup-job-checkout.ps1 -Path <dir>              # explicit location
#   .\scripts\setup-job-checkout.ps1 -RefreshData             # re-copy the data stores into an existing one
#   .\scripts\setup-job-checkout.ps1 -VenvPath <dir>          # environment the tasks will run with
#                                                             #   (default: this checkout's .venv)
#
# Public-safe: no secrets; every path is derived from this checkout or passed in.

[CmdletBinding()]
param(
  [string]$Path,
  [string]$Remote = 'origin',
  [string]$Branch = 'main',
  [string]$VenvPath,
  [switch]$RefreshData
)

. (Join-Path $PSScriptRoot '_common.ps1')

$repo = Get-RepoRoot
if (Test-JobCheckout) {
  throw "run setup-job-checkout.ps1 from the developer checkout, not from a job checkout ($repo)."
}
if (-not $Path) { $Path = Join-Path (Split-Path $repo -Parent) ((Split-Path $repo -Leaf) + '.job') }
if (-not $VenvPath) { $VenvPath = Join-Path $repo '.venv' }
$marker = Join-Path $Path $script:JobMarker

# git reports progress on stderr. When the caller redirects this script's output, Windows PowerShell 5.1 turns
# each stderr line into an error record, which the script-wide 'Stop' preference would make fatal. Judge git by
# its exit code only (the preference set here is local to the function).
function Invoke-RepoGit {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
  $ErrorActionPreference = 'Continue'
  & git -C $repo @GitArgs | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "git $($GitArgs -join ' ') failed with exit code $LASTEXITCODE." }
}
function Get-RepoGit {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
  $ErrorActionPreference = 'Continue'
  $out = @(& git -C $repo @GitArgs)
  if ($LASTEXITCODE -ne 0) { throw "git $($GitArgs -join ' ') failed with exit code $LASTEXITCODE." }
  return $out
}

if (Test-Path $Path) {
  if (-not (Test-Path $marker)) {
    throw "'$Path' exists and is not a job checkout (no $($script:JobMarker)); pass another -Path."
  }
  if (-not $RefreshData) {
    Write-Info "Job checkout already present at $Path (use -RefreshData to re-copy the data stores)."
  }
} else {
  if ($RefreshData) { throw "no job checkout at '$Path' to refresh." }
  Write-Step "Fetching $Remote/$Branch"
  Invoke-RepoGit fetch $Remote $Branch
  $base = "$(Get-RepoGit rev-parse FETCH_HEAD)".Trim()
  Write-Step "Adding the job worktree at $Path (detached at $Remote/$Branch, $($base.Substring(0, 9)))"
  Invoke-RepoGit worktree add --detach $Path $base
  $note = "Dedicated CAOS_SEISMIC job checkout: the scheduled jobs run here (scripts\job.ps1). " +
          "Do not develop or switch branches here; see docs\deploy.md section 4." + [Environment]::NewLine
  [System.IO.File]::WriteAllText($marker, $note, (New-Object System.Text.UTF8Encoding($false)))
  $RefreshData = $true   # a new job checkout always gets the data stores
}

if ($RefreshData) {
  Write-Step "Copying the gitignored data stores from $repo"
  $files = Get-RepoGit ls-files --others --ignored --exclude-standard -- data results
  foreach ($rel in $files) {
    if (-not $rel) { continue }
    $dst = Join-Path $Path $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $dst -Parent) | Out-Null
    Copy-Item -LiteralPath (Join-Path $repo $rel) -Destination $dst -Force
    Write-Info $rel
  }
  Write-Info "$($files.Count) file(s) copied."
}

Write-Step "Job checkout ready: $Path"
Write-Info "Dry run (no git write):     & '$Path\scripts\job.ps1' -Job daily -NoPublish -VenvPath '$VenvPath'"
Write-Info "Then, from an ELEVATED PowerShell, point the scheduled tasks at it:"
Write-Info "  & '$Path\scripts\schedule-daily.ps1'   -VenvPath '$VenvPath'"
Write-Info "  & '$Path\scripts\schedule-outlook.ps1' -VenvPath '$VenvPath'"

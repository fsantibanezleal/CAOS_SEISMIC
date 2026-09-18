#!/usr/bin/env pwsh
# job.ps1 - the SCHEDULED entry point. It runs ONLY in the dedicated job checkout.
#
# The job checkout is a git worktree of this repository with a DETACHED HEAD at the publish branch (main),
# outside the developer checkout and used by nothing but the scheduled tasks. scripts\setup-job-checkout.ps1
# creates it; docs\deploy.md section 4 explains it. One run takes a lock, then runs two processes:
#
#   1. `caos-seismic job-sync`: fast-forward the job checkout to origin/main, after first pushing any data
#      commit an earlier run could not push. It is its own process so that step 2 imports ONE consistent
#      version of the code.
#   2. `caos-seismic daily` (or `outlook`): compute and publish. The data commit lands on the detached HEAD
#      and that same commit is fast-forward pushed to main.
#
# The publish branch therefore has a single writer, and the branch the developer checkout is on never
# matters.
#
#   .\scripts\job.ps1 -Job daily                         # the daily task (fetch, infer, publish)
#   .\scripts\job.ps1 -Job outlook                       # the weekly task (30-day outlook, publish)
#   .\scripts\job.ps1 -Job daily -VenvPath <dir>         # run with an existing environment (CAOS_SEISMIC_VENV)
#   .\scripts\job.ps1 -Job daily -NoPublish              # compute only: no job-sync, no commit, no push
#   .\scripts\job.ps1 -SyncOnly                          # job-sync only
#   .\scripts\job.ps1 -Job daily -PublishBranch <name>   # end-to-end test against a scratch branch
#
# Every run appends to logs\job-<job>-<UTC stamp>.log (gitignored; the newest -KeepLogs files are kept). The
# lock file logs\job.lock is held open for the whole run, so the daily and the weekly job never overlap.
# The exit code is 0 on success and 1 on any failure (Task Scheduler shows it as the last run result).
# Public-safe: no secrets, no machine-specific paths.

[CmdletBinding()]
param(
  [ValidateSet('daily', 'outlook')][string]$Job = 'daily',
  [string]$Region = 'global',
  [string]$VenvPath,
  [string]$PublishBranch,
  [switch]$NoPublish,
  [switch]$NoCatchUp,
  [switch]$SyncOnly,
  [int]$LockWaitMinutes = 90,
  [int]$KeepLogs = 120
)

. (Join-Path $PSScriptRoot '_common.ps1')

$repo = Get-RepoRoot
if ($SyncOnly -and $NoPublish) { throw "-SyncOnly and -NoPublish exclude each other." }
if ($VenvPath) { $env:CAOS_SEISMIC_VENV = $VenvPath }
if ($PublishBranch) { $env:CAOS_SEISMIC_PUBLISH_BRANCH = $PublishBranch }

$logDir = Join-Path $repo 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$label = if ($SyncOnly) { 'sync' } else { $Job }
$log = Join-Path $logDir ("job-{0}-{1}.log" -f $label, (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'))
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# HEAD of this checkout, for the log. git is judged by its exit code only: under the script-wide 'Stop'
# preference, Windows PowerShell 5.1 would turn a git stderr line into a fatal error whenever the caller
# redirects this script's output (the preference set here is local to the function).
function Get-JobHead {
  $ErrorActionPreference = 'Continue'
  $h = & git -C $repo rev-parse --short HEAD
  if ($LASTEXITCODE -ne 0 -or -not $h) { return '?' }
  return "$h".Trim()
}

function Write-JobLog([string]$Message) {
  $line = '[{0}] {1}' -f (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'), $Message
  [System.IO.File]::AppendAllText($log, $line + [Environment]::NewLine, $utf8NoBom)
  Write-Host $line
}

Write-JobLog "job=$label region=$Region checkout=$repo user=$env:USERNAME"
if (-not (Test-JobCheckout)) {
  Write-JobLog "ERROR: not the dedicated job checkout (no '$($script:JobMarker)' in $repo). Create one with scripts\setup-job-checkout.ps1 (docs\deploy.md section 4); never schedule a developer checkout."
  exit 1
}
try {
  Write-JobLog ("python=" + (Get-CaosPython))
} catch {
  Write-JobLog "ERROR: $($_.Exception.Message)"
  exit 1
}
if ($PublishBranch) { Write-JobLog "publish branch override: $PublishBranch (test run)" }

# One job at a time in this checkout: an exclusive handle on the lock file, released when this process
# exits (even if it is killed), so a crash can never leave a stale lock behind.
$lockPath = Join-Path $logDir 'job.lock'
$lock = $null
$deadline = (Get-Date).AddMinutes($LockWaitMinutes)
while ($null -eq $lock) {
  try {
    $lock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
  } catch {
    if ((Get-Date) -ge $deadline) {
      Write-JobLog "ERROR: another job still holds $lockPath after $LockWaitMinutes min; giving up (the next run catches up)."
      exit 1
    }
    Start-Sleep -Seconds 30
  }
}

$exitCode = 0
try {
  $head = Get-JobHead
  Write-JobLog "lock acquired; HEAD $head"
  if (-not $NoPublish) {
    Write-JobLog 'caos-seismic job-sync'
    $code = Invoke-CaosToLog -LogFile $log -CaosArgs @('job-sync')
    if ($code -ne 0) { throw "job-sync exited with code $code" }
  }
  if (-not $SyncOnly) {
    $caosArgs = @($Job, '--region', $Region)
    if ($NoPublish) { $caosArgs += '--no-publish' }
    if ($NoCatchUp -and $Job -eq 'daily') { $caosArgs += '--no-catch-up' }
    Write-JobLog "caos-seismic $($caosArgs -join ' ')"
    $code = Invoke-CaosToLog -LogFile $log -CaosArgs $caosArgs
    if ($code -ne 0) { throw "caos-seismic $($caosArgs -join ' ') exited with code $code" }
  }
  $head = Get-JobHead
  Write-JobLog "done; HEAD $head"
} catch {
  $exitCode = 1
  Write-JobLog "ERROR: $($_.Exception.Message)"
} finally {
  $lock.Dispose()
  Get-ChildItem -Path $logDir -Filter 'job-*.log' |
    Sort-Object LastWriteTime -Descending |
    Select-Object -Skip $KeepLogs |
    Remove-Item -Force -ErrorAction SilentlyContinue
}
exit $exitCode

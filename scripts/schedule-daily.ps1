#!/usr/bin/env pwsh
# schedule-daily.ps1 - register (or remove) a Windows Task Scheduler task that runs the daily job.
#
# The task fires once per day at the local time from configs/publish.yaml (`schedule.time_local`, default
# 03:00), invoking scripts\daily.ps1. It is configured to:
#   * run whether the user is logged on or not,
#   * wake the computer to run (so a sleeping laptop still fires),
#   * start the missed run on next wake if the machine was off at the scheduled time (catch-up; the job
#     itself also backfills missed issue dates).
#
#   .\scripts\schedule-daily.ps1                 # register the task (idempotent - re-registers cleanly)
#   .\scripts\schedule-daily.ps1 -Region chile
#   .\scripts\schedule-daily.ps1 -Time 03:30     # override the time (otherwise read from publish.yaml)
#   .\scripts\schedule-daily.ps1 -Remove         # unregister the task
#
# Run from an ELEVATED PowerShell (Task Scheduler registration needs admin). Public-safe: paths are
# resolved relative to the repo root at registration time; no secrets are stored in the task.

[CmdletBinding()]
param(
  [string]$Region = 'chile',
  [string]$Time,                              # HH:mm; default: from configs/publish.yaml
  [string]$TaskName = 'CAOS_SEISMIC daily forecast',
  [switch]$Remove
)

. (Join-Path $PSScriptRoot '_common.ps1')

$repo = Get-RepoRoot

if ($Remove) {
  if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Step "Removed scheduled task '$TaskName'."
  } else {
    Write-Info "No scheduled task named '$TaskName' found."
  }
  return
}

# Resolve the daily run time: explicit -Time wins, else configs/publish.yaml schedule.time_local.
if (-not $Time) {
  $py = $null
  try { $py = Get-VenvPython } catch { }
  if ($py) {
    $code = @'
import sys
sys.path.insert(0, "src")
from caos_seismic.config import load
print((load("publish").get("schedule", {}) or {}).get("time_local", "03:00"))
'@
    Push-Location $repo
    try { $Time = (& $py '-c' $code).Trim() } finally { Pop-Location }
  }
  if (-not $Time) { $Time = '03:00' }
}

# Validate HH:mm and build a "today at that time" DateTime for the trigger.
if ($Time -notmatch '^\d{1,2}:\d{2}$') { throw "invalid -Time '$Time'; expected HH:mm (e.g. 03:00)." }
$parts = $Time.Split(':')
$hour = [int]$parts[0]; $minute = [int]$parts[1]
if ($hour -lt 0 -or $hour -gt 23 -or $minute -lt 0 -or $minute -gt 59) { throw "time '$Time' is out of range." }
$at = (Get-Date).Date.AddHours($hour).AddMinutes($minute)

# pwsh if available, else Windows PowerShell - run daily.ps1 hidden, non-interactive.
# (No null-conditional `?.` here - keep this script parseable on Windows PowerShell 5.1 too.)
$psExe = $null
$pwshCmd = Get-Command 'pwsh' -ErrorAction SilentlyContinue
if ($pwshCmd) { $psExe = $pwshCmd.Source }
if (-not $psExe) {
  $winpsCmd = Get-Command 'powershell' -ErrorAction SilentlyContinue
  if ($winpsCmd) { $psExe = $winpsCmd.Source }
}
if (-not $psExe) { throw "neither pwsh nor powershell found on PATH." }

$dailyScript = Join-Path $PSScriptRoot 'daily.ps1'
$argLine = "-NoProfile -ExecutionPolicy Bypass -NonInteractive -WindowStyle Hidden -File `"$dailyScript`" -Region $Region"

$action  = New-ScheduledTaskAction -Execute $psExe -Argument $argLine -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Daily -At $at

# Settings: wake to run, start-when-available (catch-up if the machine was off), run on battery, no time cap.
$settings = New-ScheduledTaskSettingsSet `
  -WakeToRun `
  -StartWhenAvailable `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
  -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 10) `
  -MultipleInstances IgnoreNew

# Run whether logged on or not, with highest privileges (S4U: no stored password, no interactive session).
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
  Write-Info "Task '$TaskName' exists - re-registering."
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
  -TaskName $TaskName `
  -Action $action `
  -Trigger $trigger `
  -Settings $settings `
  -Principal $principal `
  -Description "CAOS_SEISMIC: fetch + infer + scoped publish (forecasts, never predictions). Region '$Region'. Runs scripts\daily.ps1 daily at $Time local." | Out-Null

Write-Step "Registered '$TaskName' - daily at $Time local (wake-to-run, run-whether-logged-on-or-not)."
Write-Info  "Inspect:  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Info  "Run now:  Start-ScheduledTask -TaskName '$TaskName'"
Write-Info  "Remove:   .\scripts\schedule-daily.ps1 -Remove"

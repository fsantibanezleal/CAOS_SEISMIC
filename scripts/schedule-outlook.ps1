#!/usr/bin/env pwsh
# schedule-outlook.ps1 - register (or remove) a Windows Task Scheduler task that runs the WEEKLY 30-day
# outlook job (scripts\outlook.ps1).
#
# The outlook background is time-flat (driven by slow GNSS strain), so it is refit WEEKLY, not daily. The
# task fires once a week (default Sunday 04:00 local - offset from the 03:00 daily job so they never
# collide) and is configured to:
#   * run whether the user is logged on or not,
#   * wake the computer to run (so a sleeping laptop still fires),
#   * start the missed run on next wake if the machine was off at the scheduled time.
# A generous 3-hour execution limit covers the ~20-min neural fit + the 10 validation windows.
#
#   .\scripts\schedule-outlook.ps1                 # register (idempotent - re-registers cleanly)
#   .\scripts\schedule-outlook.ps1 -Day Saturday -Time 05:00
#   .\scripts\schedule-outlook.ps1 -Remove         # unregister
#
# Run from an ELEVATED PowerShell (Task Scheduler registration needs admin). Public-safe.

[CmdletBinding()]
param(
  [string]$Region = 'global',
  [ValidateSet('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')]
  [string]$Day = 'Sunday',
  [string]$Time = '04:00',                     # HH:mm local
  [string]$TaskName = 'CAOS_SEISMIC weekly outlook',
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

# Validate HH:mm and build a "today at that time" DateTime for the trigger.
if ($Time -notmatch '^\d{1,2}:\d{2}$') { throw "invalid -Time '$Time'; expected HH:mm (e.g. 04:00)." }
$parts = $Time.Split(':')
$hour = [int]$parts[0]; $minute = [int]$parts[1]
if ($hour -lt 0 -or $hour -gt 23 -or $minute -lt 0 -or $minute -gt 59) { throw "time '$Time' is out of range." }
$at = (Get-Date).Date.AddHours($hour).AddMinutes($minute)

# pwsh if available, else Windows PowerShell - run outlook.ps1 hidden, non-interactive.
$psExe = $null
$pwshCmd = Get-Command 'pwsh' -ErrorAction SilentlyContinue
if ($pwshCmd) { $psExe = $pwshCmd.Source }
if (-not $psExe) {
  $winpsCmd = Get-Command 'powershell' -ErrorAction SilentlyContinue
  if ($winpsCmd) { $psExe = $winpsCmd.Source }
}
if (-not $psExe) { throw "neither pwsh nor powershell found on PATH." }

$outlookScript = Join-Path $PSScriptRoot 'outlook.ps1'
$argLine = "-NoProfile -ExecutionPolicy Bypass -NonInteractive -WindowStyle Hidden -File `"$outlookScript`" -Region $Region"

$action  = New-ScheduledTaskAction -Execute $psExe -Argument $argLine -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Day -At $at

# Settings: wake to run, start-when-available (catch-up if the machine was off), run on battery, 3h cap.
$settings = New-ScheduledTaskSettingsSet `
  -WakeToRun `
  -StartWhenAvailable `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
  -RestartCount 1 -RestartInterval (New-TimeSpan -Minutes 30) `
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
  -Description "CAOS_SEISMIC: weekly 30-day outlook (geodetic neural background) -> validate -> scoped publish. Region '$Region'. Runs scripts\outlook.ps1 every $Day at $Time local." | Out-Null

Write-Step "Registered '$TaskName' - $Day at $Time local (wake-to-run, run-whether-logged-on-or-not)."
Write-Info  "Inspect:  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Info  "Run now:  Start-ScheduledTask -TaskName '$TaskName'"
Write-Info  "Remove:   .\scripts\schedule-outlook.ps1 -Remove"

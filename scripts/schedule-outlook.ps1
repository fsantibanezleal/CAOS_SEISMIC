#!/usr/bin/env pwsh
# schedule-outlook.ps1 - register (or remove) the Windows Task Scheduler task that runs the WEEKLY 30-day
# outlook job.
#
# Run it FROM THE DEDICATED JOB CHECKOUT (scripts\setup-job-checkout.ps1 creates it and prints this exact
# command). The task runs that checkout's scripts\job.ps1 -Job outlook, never a developer checkout
# (docs\deploy.md section 4). The outlook background is time-flat (driven by slow GNSS strain), so it is
# refit WEEKLY, not daily. The task fires once a week (default Sunday 04:00 LOCAL, also across daylight
# saving; an hour after the 03:00 daily job) and is configured to:
#   * run whether the user is logged on or not (S4U principal: no stored password),
#   * wake the computer to run (so a sleeping laptop still fires),
#   * start a missed run on next wake if the machine was off at the scheduled time,
#   * never overlap the daily job (job.ps1 holds a lock for the whole run).
# A generous 3-hour execution limit covers the ~20-min neural fit + the 10 validation windows.
#
#   .\scripts\schedule-outlook.ps1 -VenvPath <dir>              # register (idempotent - re-registers cleanly)
#   .\scripts\schedule-outlook.ps1 -Day Saturday -Time 05:00
#   .\scripts\schedule-outlook.ps1 -Remove                      # unregister
#
# -VenvPath names an existing environment to run with (e.g. the developer checkout's .venv); omit it when
# the job checkout has its own .venv. An S4U principal at the highest run level needs an ELEVATED
# PowerShell; -LogonType Interactive -RunLevel Limited registers without elevation (for tests). Public-safe.

[CmdletBinding()]
param(
  [string]$Region = 'global',
  [ValidateSet('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')]
  [string]$Day = 'Sunday',
  [string]$Time = '04:00',                     # HH:mm local
  [string]$VenvPath,
  [string]$TaskName = 'CAOS_SEISMIC weekly outlook',
  [ValidateSet('S4U', 'Interactive')][string]$LogonType = 'S4U',
  [ValidateSet('Highest', 'Limited')][string]$RunLevel = 'Highest',
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

if (-not (Test-JobCheckout)) {
  throw ("register the task from the dedicated job checkout, not from '$repo'. Run " +
         "scripts\setup-job-checkout.ps1 in the developer checkout first; it prints the command to run here.")
}
if ($VenvPath) { $env:CAOS_SEISMIC_VENV = $VenvPath }
$null = Get-CaosPython   # fails now, not on Sunday, when the environment is missing

# Validate HH:mm and build a "today at that time" DateTime for the trigger.
if ($Time -notmatch '^\d{1,2}:\d{2}$') { throw "invalid -Time '$Time'; expected HH:mm (e.g. 04:00)." }
$parts = $Time.Split(':')
$hour = [int]$parts[0]; $minute = [int]$parts[1]
if ($hour -lt 0 -or $hour -gt 23 -or $minute -lt 0 -or $minute -gt 59) { throw "time '$Time' is out of range." }
$at = (Get-Date).Date.AddHours($hour).AddMinutes($minute)

# pwsh if available, else Windows PowerShell - run job.ps1 hidden, non-interactive.
$psExe = $null
$pwshCmd = Get-Command 'pwsh' -ErrorAction SilentlyContinue
if ($pwshCmd) { $psExe = $pwshCmd.Source }
if (-not $psExe) {
  $winpsCmd = Get-Command 'powershell' -ErrorAction SilentlyContinue
  if ($winpsCmd) { $psExe = $winpsCmd.Source }
}
if (-not $psExe) { throw "neither pwsh nor powershell found on PATH." }

$jobScript = Join-Path $PSScriptRoot 'job.ps1'
$argLine = "-NoProfile -ExecutionPolicy Bypass -NonInteractive -WindowStyle Hidden -File `"$jobScript`" -Job outlook -Region $Region"
if ($VenvPath) { $argLine += " -VenvPath `"$VenvPath`"" }

$action  = New-ScheduledTaskAction -Execute $psExe -Argument $argLine -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Day -At $at
# A start boundary WITHOUT a UTC offset keeps the fire time at the local $Time across daylight saving.
$trigger.StartBoundary = $at.ToString('yyyy-MM-ddTHH:mm:ss')

# Settings: wake to run, start-when-available (catch-up if the machine was off), run on battery, 3h cap.
$settings = New-ScheduledTaskSettingsSet `
  -WakeToRun `
  -StartWhenAvailable `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
  -RestartCount 1 -RestartInterval (New-TimeSpan -Minutes 30) `
  -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType $LogonType -RunLevel $RunLevel

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
  -Description "CAOS_SEISMIC: weekly 30-day outlook (geodetic neural background), validate, publish to main from the dedicated job checkout. Region '$Region'. Runs scripts\job.ps1 -Job outlook every $Day at $Time local." | Out-Null

Write-Step "Registered '$TaskName' - $Day at $Time local, running $jobScript ($LogonType, $RunLevel)."
Write-Info  "Inspect:  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Info  "Run now:  Start-ScheduledTask -TaskName '$TaskName'"
Write-Info  "Logs:     $(Join-Path $repo 'logs')"
Write-Info  "Remove:   .\scripts\schedule-outlook.ps1 -Remove"

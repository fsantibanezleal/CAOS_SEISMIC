#!/usr/bin/env pwsh
# schedule-daily.ps1 - register (or remove) the Windows Task Scheduler task that runs the DAILY job.
#
# Run it FROM THE DEDICATED JOB CHECKOUT (scripts\setup-job-checkout.ps1 creates it and prints this exact
# command). The task runs that checkout's scripts\job.ps1 -Job daily, never a developer checkout, so the
# daily data reaches main from one place only (docs\deploy.md section 4). It fires once per day at the
# LOCAL time from configs/publish.yaml (`schedule.time_local`, default 03:00), also across daylight-saving
# changes, and is configured to:
#   * run whether the user is logged on or not (S4U principal: no stored password),
#   * wake the computer to run (so a sleeping laptop still fires),
#   * start a missed run on next wake (the job also backfills missed issue dates),
#   * never start a second instance while one runs (job.ps1 also locks against the weekly job).
#
#   .\scripts\schedule-daily.ps1 -VenvPath <dir>     # register (idempotent - re-registers cleanly)
#   .\scripts\schedule-daily.ps1 -Time 03:30         # override the time (otherwise read from publish.yaml)
#   .\scripts\schedule-daily.ps1 -Remove             # unregister the task
#
# -VenvPath names an existing environment to run with (e.g. the developer checkout's .venv); omit it when
# the job checkout has its own .venv. An S4U principal at the highest run level needs an ELEVATED
# PowerShell. -LogonType Interactive -RunLevel Limited registers a run-only-when-logged-on task without
# elevation (useful to test the task end to end). Public-safe: no secrets are stored in the task.
# -PublishBranch <name> (tests only) makes the registered task publish to a scratch branch, so a test
# task can run the real job end to end without touching main.

[CmdletBinding()]
param(
  [string]$Region = 'global',
  [string]$Time,                              # HH:mm; default: from configs/publish.yaml
  [string]$VenvPath,
  [string]$PublishBranch,                     # test only: the task publishes to this branch instead of main
  [string]$TaskName = 'CAOS_SEISMIC daily forecast',
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
$py = Get-CaosPython   # fails now, not at 03:00, when the environment is missing

# Resolve the daily run time: explicit -Time wins, else configs/publish.yaml schedule.time_local.
if (-not $Time) {
  # SINGLE-LINE python -c (a multi-line here-string is mangled when PowerShell passes it to a native
  # exe -> SyntaxError). Any failure falls through to the 03:00 default below.
  $oneLine = "import sys; sys.path.insert(0,'src'); from caos_seismic.config import load; print((load('publish').get('schedule',{}) or {}).get('time_local','03:00'))"
  Push-Location $repo
  try {
    $val = & $py '-c' $oneLine 2>$null | Select-Object -First 1
    if ($val) { $Time = "$val".Trim() }
  } catch { $Time = $null } finally { Pop-Location }
  if (-not $Time) { $Time = '03:00' }
}

# Validate HH:mm and build a "today at that time" DateTime for the trigger.
if ($Time -notmatch '^\d{1,2}:\d{2}$') { throw "invalid -Time '$Time'; expected HH:mm (e.g. 03:00)." }
$parts = $Time.Split(':')
$hour = [int]$parts[0]; $minute = [int]$parts[1]
if ($hour -lt 0 -or $hour -gt 23 -or $minute -lt 0 -or $minute -gt 59) { throw "time '$Time' is out of range." }
$at = (Get-Date).Date.AddHours($hour).AddMinutes($minute)

# pwsh if available, else Windows PowerShell - run job.ps1 hidden, non-interactive.
# (No null-conditional `?.` here - keep this script parseable on Windows PowerShell 5.1 too.)
$psExe = $null
$pwshCmd = Get-Command 'pwsh' -ErrorAction SilentlyContinue
if ($pwshCmd) { $psExe = $pwshCmd.Source }
if (-not $psExe) {
  $winpsCmd = Get-Command 'powershell' -ErrorAction SilentlyContinue
  if ($winpsCmd) { $psExe = $winpsCmd.Source }
}
if (-not $psExe) { throw "neither pwsh nor powershell found on PATH." }

$jobScript = Join-Path $PSScriptRoot 'job.ps1'
$argLine = "-NoProfile -ExecutionPolicy Bypass -NonInteractive -WindowStyle Hidden -File `"$jobScript`" -Job daily -Region $Region"
if ($VenvPath) { $argLine += " -VenvPath `"$VenvPath`"" }
if ($PublishBranch) { $argLine += " -PublishBranch `"$PublishBranch`"" }
$target = if ($PublishBranch) { "$PublishBranch (TEST TASK)" } else { 'main' }

$action  = New-ScheduledTaskAction -Execute $psExe -Argument $argLine -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Daily -At $at
# A start boundary WITHOUT a UTC offset keeps the fire time at the local $Time across daylight saving
# (with an offset, Task Scheduler pins it to UTC and it drifts by an hour twice a year).
$trigger.StartBoundary = $at.ToString('yyyy-MM-ddTHH:mm:ss')

# Settings: wake to run, start-when-available (catch-up if the machine was off), run on battery, 2 h cap.
# Priority 4 (normal): Task Scheduler's default 7 runs the job at below-normal CPU and low I/O priority,
# which starves it (minutes per git call) whenever anything else keeps the machine busy.
$settings = New-ScheduledTaskSettingsSet `
  -WakeToRun `
  -StartWhenAvailable `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
  -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 10) `
  -MultipleInstances IgnoreNew `
  -Priority 4

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
  -Description "CAOS_SEISMIC: daily fetch + infer + publish to $target from the dedicated job checkout (forecasts, never predictions). Region '$Region'. Runs scripts\job.ps1 -Job daily at $Time local." | Out-Null

Write-Step "Registered '$TaskName' - daily at $Time local, running $jobScript ($LogonType, $RunLevel)."
Write-Info  "Inspect:  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Info  "Run now:  Start-ScheduledTask -TaskName '$TaskName'"
Write-Info  "Logs:     $(Join-Path $repo 'logs')"
Write-Info  "Remove:   .\scripts\schedule-daily.ps1 -Remove"

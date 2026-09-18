# _common.ps1 - shared helpers for the CAOS_SEISMIC PowerShell scripts.
# Dot-sourced by setup.ps1, fetch.ps1, build-features.ps1, train.ps1, infer.ps1, daily.ps1, outlook.ps1,
# dev.ps1, check.ps1, job.ps1, setup-job-checkout.ps1, schedule-daily.ps1, schedule-outlook.ps1.
# Public-safe: no secrets, no machine-specific paths (everything is resolved relative to the repo root).

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Repo root = parent of the scripts/ directory that contains this file.
$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$script:VenvDir = Join-Path $RepoRoot '.venv'

function Get-RepoRoot { return $script:RepoRoot }

# The dedicated job checkout (scripts\setup-job-checkout.ps1) carries this marker file; the CLI checks the
# same name (cli.py JOB_MARKER) before `job-sync` or a publish. Keep the two in sync.
$script:JobMarker = '.caos-seismic-job'
function Test-JobCheckout { return (Test-Path (Join-Path $script:RepoRoot $script:JobMarker)) }

function Write-Step([string]$Message) {
  Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Info([string]$Message) {
  Write-Host "    $Message" -ForegroundColor DarkGray
}

function Write-Warn2([string]$Message) {
  Write-Host "WARN: $Message" -ForegroundColor Yellow
}

function Write-Err2([string]$Message) {
  Write-Host "ERROR: $Message" -ForegroundColor Red
}

# Resolve the .venv Python interpreter (Windows layout). Throws if the venv is absent.
function Get-VenvPython {
  $py = Join-Path $script:VenvDir 'Scripts\python.exe'
  if (-not (Test-Path $py)) {
    throw "virtualenv not found at '$($script:VenvDir)'. Run  scripts\setup.ps1  first."
  }
  return $py
}

# The interpreter that RUNS the pipeline (Invoke-Caos): the environment named by CAOS_SEISMIC_VENV when it
# is set, else this checkout's own .venv. The dedicated job checkout sets it (scripts\job.ps1 -VenvPath) to
# reuse an existing environment instead of installing a second multi-GB CUDA stack. setup.ps1 and dev.ps1
# keep using Get-VenvPython, so they never install into, or re-point, a shared environment.
function Get-CaosPython {
  if ($env:CAOS_SEISMIC_VENV) {
    $py = Join-Path $env:CAOS_SEISMIC_VENV 'Scripts\python.exe'
    if (-not (Test-Path $py)) {
      throw "CAOS_SEISMIC_VENV='$($env:CAOS_SEISMIC_VENV)' has no Scripts\python.exe."
    }
    return $py
  }
  return (Get-VenvPython)
}

# PYTHONPATH that makes the interpreter import THIS checkout's code: its src/ first. A shared environment's
# editable install points at whichever checkout ran `pip install -e .`; without this a job checkout would
# silently run the developer checkout's code, and with it that checkout's REPO_ROOT (its results/, data/).
function Get-CaosPythonPath {
  $src = Join-Path $script:RepoRoot 'src'
  if ($env:PYTHONPATH) { return "$src;$($env:PYTHONPATH)" }
  return $src
}

# Pick an interpreter to BOOTSTRAP the venv: prefer Python 3.12 via the launcher, else python/python3.
function Get-BootstrapPython {
  # Try the Windows launcher pinned to 3.12.
  $launcher = Get-Command 'py' -ErrorAction SilentlyContinue
  if ($launcher) {
    try {
      & py -3.12 --version *> $null
      if ($LASTEXITCODE -eq 0) { return @('py', '-3.12') }
    } catch { }
  }
  foreach ($name in @('python', 'python3')) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { return @($cmd.Source) }
  }
  throw "no Python interpreter found on PATH (need Python 3.12). Install it from https://www.python.org/downloads/"
}

# Invoke the package console entry point inside the venv:  caos-seismic <args...>
# We call the module form so it works even if the console-script shim is not on PATH. The code that runs
# is always this checkout's (Get-CaosPythonPath).
function Invoke-Caos {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
  $py = Get-CaosPython
  $previous = $env:PYTHONPATH
  $env:PYTHONPATH = Get-CaosPythonPath
  Push-Location $script:RepoRoot
  try {
    & $py '-m' 'caos_seismic.cli' @Args
    if ($LASTEXITCODE -ne 0) { throw "caos-seismic $($Args -join ' ')  exited with code $LASTEXITCODE." }
  } finally {
    Pop-Location
    $env:PYTHONPATH = $previous
  }
}

# Unattended variant for the scheduled jobs: append the command's stdout AND stderr to $LogFile and return
# its exit code (no throw). The output goes through cmd.exe redirection on purpose: Windows PowerShell 5.1
# turns every stderr line of a redirected native command into an error record, which under
# ErrorActionPreference=Stop would abort the job on the first log line Python writes to stderr.
function Invoke-CaosToLog {
  param(
    [Parameter(Mandatory = $true)][string]$LogFile,
    [Parameter(Mandatory = $true)][string[]]$CaosArgs
  )
  $py = Get-CaosPython
  $quoted = ($CaosArgs | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join ' '
  $cmdLine = "`"$py`" -m caos_seismic.cli $quoted >> `"$LogFile`" 2>&1"
  $comspec = if ($env:ComSpec) { $env:ComSpec } else { Join-Path $env:SystemRoot 'System32\cmd.exe' }

  $previous = @{ PYTHONPATH = $env:PYTHONPATH; PYTHONUTF8 = $env:PYTHONUTF8; PYTHONUNBUFFERED = $env:PYTHONUNBUFFERED }
  $env:PYTHONPATH = Get-CaosPythonPath
  $env:PYTHONUTF8 = '1'          # UTF-8 log bytes whatever the console code page
  $env:PYTHONUNBUFFERED = '1'    # keep stdout and stderr lines in order in the log
  try {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $comspec
    $psi.Arguments = "/d /s /c `"$cmdLine`""
    $psi.WorkingDirectory = $script:RepoRoot
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.WaitForExit()
    return $proc.ExitCode
  } finally {
    # Restore the caller's environment ($null removes a variable that was not set before).
    foreach ($name in $previous.Keys) { [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process') }
  }
}

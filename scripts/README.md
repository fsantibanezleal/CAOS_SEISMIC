<!-- markdownlint-disable MD013 -->
# `scripts/`: operator entry points

Thin, parallel **PowerShell (`*.ps1`)** and **bash (`*.sh`)** wrappers around the `caos-seismic` CLI
(`src/caos_seismic/cli.py`). Each subcommand exists in **both** flavours with identical behaviour, so the
same workflow runs on Windows (the local GPU workstation) and on a Linux VPS. Every script sources a
shared helper (`_common.ps1` / `_common.sh`) that resolves the repo root, locates the `.venv`
interpreter, and invokes the package as a module (`python -m caos_seismic.cli …`).

> **Forecasts, never predictions.** Everything here produces *bounded, calibrated, CSEP-scored
> probabilities*, never alarms, countdowns, or a "safe" state. See the repo `README.md`.

All paths are resolved **relative to the repo root**; there are **no machine-specific absolute paths and
no secrets** in any script (public-safe).

## The subcommands (1:1 across `.ps1` / `.sh`)

| Script | What it does | Underlying CLI |
|---|---|---|
| `setup` | Create the `.venv` (Python 3.12 if available), `pip install -r requirements.txt`, install the package editable, smoke-test. |, (env bootstrap) |
| `fetch` | Pull the recent + historical catalog (USGS ComCat spine + regional/anchor sources) and write a provenance manifest. | `caos-seismic fetch --region <id>` |
| `build-features` | `M_c` + b-value, Mw homogenization, **dual-catalog** declustering, feature extraction. | `caos-seismic build-features --region <id>` |
| `train` | Fit the stationary smoothed-seismicity null + space–time **ETAS** (+ Reasenberg–Jones fallback); reject fits that violate the stability gates. | `caos-seismic train --region <id>` |
| `infer` | Run the **forecast clock** for an issue date → one compact artifact under `results/`. | `caos-seismic infer --region <id> [--issue YYYY-MM-DD]` |
| `backanalysis` | Pseudo-prospective **CSEP** back-analysis over a date range (the clock advances day by day). | `caos-seismic backanalysis --region <id> --start … --end …` |
| `daily` | **The daily job:** fetch, infer (today + missed days), **scoped publish** (commit + push). Scheduled through `job` in the job checkout; run it directly for a dry run (`-NoPublish` / `--no-publish`). | `caos-seismic daily --region <id>` |
| `outlook` (`.ps1`) | **The weekly job:** fit the 30-day geodetic background, validate, scoped publish. Scheduled through `job -Job outlook`. | `caos-seismic outlook --region <id>` |
| `job` | **The scheduled entry point**, only in the dedicated job checkout: lock, `job-sync`, then `daily` or `outlook`, with a log per run under `logs/`. | `caos-seismic job-sync` + `daily` / `outlook` |
| `setup-job-checkout` | Create the dedicated job checkout (a detached worktree at `origin/main` with the `.caos-seismic-job` marker) and copy the gitignored data stores into it. | (git worktree only) |
| `dev` | Serve the **static** web app locally for preview (Vite HMR, or a dependency-free static server). **No processing backend.** |, (static server) |
| `check` | Environment + repo + config **sanity checks** (no network, no science deps). Exits non-zero on hard failure. | `caos-seismic check --region <id>` |

### Examples

```powershell
# Windows (PowerShell)
.\scripts\setup.ps1                 # create .venv + install
.\scripts\check.ps1                 # sanity check (no network)
.\scripts\fetch.ps1 -Region chile
.\scripts\build-features.ps1
.\scripts\train.ps1
.\scripts\infer.ps1 -Issue 2026-06-16
.\scripts\dev.ps1                   # preview the static SPA at http://127.0.0.1:5173
.\scripts\daily.ps1 -NoPublish      # full pipeline, local dry run (no commit/push)
```

```bash
# Linux / macOS / Git Bash
scripts/setup.sh
scripts/check.sh
scripts/fetch.sh --region chile
scripts/build-features.sh
scripts/train.sh
scripts/infer.sh --issue 2026-06-16
scripts/dev.sh                      # preview the static SPA at http://127.0.0.1:5173
scripts/daily.sh --no-publish       # full pipeline, local dry run
```

> First use on Linux/macOS: `chmod +x scripts/*.sh` (Git Bash on Windows runs them without the bit).

## `dev`: static preview, never a backend

The web app (`app/`) is a **pure static viewer**: at runtime it only reads the precomputed daily forecast
artifact (`app/public/data/` in preview; the committed `results/` JSON in production). It computes
nothing. `dev` therefore just serves files:

- with **npm/Vite** (HMR) if `node`/`npm` is available: `dev` runs `npm run dev`;
- otherwise it falls back to a **dependency-free** `python -m http.server` over `app/dist/` (or
  `app/public/`), so a preview is possible with only the Python `.venv`.

Flags: `-Build`/`--build` (build the SPA first, then serve `app/dist/`), `-Static`/`--static` (force the
plain static server), `-Port`/`--port` (default `5173`). It binds to `127.0.0.1` only.

## `daily`: the production job (scoped, git-as-data)

`daily` is the once-per-day production job (scheduled 03:00 local through `job`, see below):

1. **`fetch`** once: the freshest catalog covers every issue date in the batch.
2. **`infer`** for **today plus any missed prior days** (catch-up, bounded to the last 7 days so a
   long-dormant laptop does not try to backfill months of non-honest forecasts). A day already present
   under `results/` is skipped.
3. **Scoped publish**: stage **only** the `configs/publish.yaml` `git.add_allowlist` paths
   (`results/`, `manifests/`), **abort** if anything outside the allowlist is staged, commit with the
   configured `commit_message_prefix`, and fast-forward push that commit to `main`. This happens only in
   the dedicated job checkout (see "The job checkout" below); in a developer checkout the CLI warns and
   uses a deprecated legacy path that also commits on the checked-out branch.

### Scoped-publish discipline (hard rules)

This machine also holds `data/`, `models/`, `.venv/`, and `.env`. The publish step therefore:

- **NEVER** runs `git add -A` / `git add .`. It stages **only** the explicit allowlist entries.
- **Resets the index first** so a pre-existing staged change cannot ride along.
- **Aborts** (and resets the index) if any staged path is outside the allowlist: nothing is committed.
- Reads the allowlist, commit prefix, remote, and branch from `configs/publish.yaml` (`git.*`).

The push credential is a **least-privilege deploy key / fine-grained PAT scoped to THIS repo**, kept in
the host's git credential store, **never** committed and **never** in these scripts. If the push fails,
the commit stays in the job checkout, the run exits non-zero, and the next run pushes it.

Dry run (no commit/push): `daily.ps1 -NoPublish` / `daily.sh --no-publish`. Skip catch-up:
`-NoCatchUp` / `--no-catch-up`.

## The job checkout (where the scheduled jobs run)

The scheduled jobs never run in a developer checkout. They run in a **dedicated job checkout**: a git
worktree of this repository with a detached HEAD at `origin/main`, outside the developer checkout,
carrying the marker file `.caos-seismic-job`. `job` first runs `caos-seismic job-sync` (fast-forward to
`main`, pushing any data commit an earlier run could not push), then the job itself, so `main` has one
writer and the developer checkout's branch never matters. The scripts always run the code of the
checkout they live in (its `src/` goes first on `PYTHONPATH`), and `-VenvPath` / `--venv` lets the job
checkout reuse an existing environment. See `docs/deploy.md` §4 for the rationale and operations.

```powershell
# In the developer checkout (Windows):
.\scripts\setup-job-checkout.ps1                   # creates <parent>\<repo>.job + copies the data stores
.\scripts\setup-job-checkout.ps1 -RefreshData      # re-copy the data stores after rebuilding one here

# In the job checkout:
.\scripts\job.ps1 -Job daily -NoPublish -VenvPath <env>   # dry run: compute only, no git write
.\scripts\job.ps1 -SyncOnly -VenvPath <env>               # job-sync only
.\scripts\job.ps1 -Job daily -VenvPath <env>              # what the daily task runs
.\scripts\job.ps1 -Job outlook -VenvPath <env>            # what the weekly task runs
```

Each run writes `logs/job-<job>-<UTC stamp>.log` in the job checkout and holds `logs/job.lock`, so the
daily and weekly jobs never overlap. A step whose interpreter dies abnormally (not a Python-level
failure, which exits 1 or 2) is retried once, and the log records the retry.

## Scheduling the jobs

The job is the same on both platforms; only the scheduler differs. Both schedulers run `job` from the
job checkout.

### Windows: Task Scheduler (`schedule-daily.ps1`, `schedule-outlook.ps1`)

Run them **from the job checkout** (they refuse to run anywhere else); `setup-job-checkout.ps1` prints
the exact commands. `schedule-daily.ps1` registers a task that runs `job.ps1 -Job daily` **daily at the
local time from `configs/publish.yaml`** (`schedule.time_local`, default **03:00**, kept local across
daylight saving); `schedule-outlook.ps1` registers `job.ps1 -Job outlook` weekly (default Sunday 04:00).
Both **run whether the user is logged on or not**, **wake the computer to run**, start on next wake if a
fire was missed, run on battery, never start a second instance, and run at priority 4 (normal; the
Task Scheduler default of 7 starves the job's CPU and I/O on a busy machine).

```powershell
# from an ELEVATED PowerShell, in the job checkout:
.\scripts\schedule-daily.ps1   -VenvPath <env>   # register (idempotent; reads the time from publish.yaml)
.\scripts\schedule-outlook.ps1 -VenvPath <env>
.\scripts\schedule-daily.ps1 -Time 03:30 -VenvPath <env>   # override the time
.\scripts\schedule-daily.ps1 -Remove                       # unregister

Get-ScheduledTask -TaskName 'CAOS_SEISMIC daily forecast' | Get-ScheduledTaskInfo   # inspect
Start-ScheduledTask  -TaskName 'CAOS_SEISMIC daily forecast'                         # run now
```

The tasks use an **S4U** principal (run whether logged on or not, no stored password, no interactive
session) at the highest run level, so registration requires an elevated shell.
`-LogonType Interactive -RunLevel Limited` registers a run-only-when-logged-on task without elevation,
which is useful to test the task end to end.

### Linux VPS: systemd timer (portable fallback)

`caos-seismic-daily.service` (oneshot, runs `job.sh --job daily` in the job checkout) +
`caos-seismic-daily.timer` (`OnCalendar=*-*-* 03:00:00`, `Persistent=true` for missed-run catch-up).

```bash
# 1) clone the repo (e.g. /opt/caos-seismic), run scripts/setup.sh as the run user
# 2) as the run user, create the job checkout (/opt/caos-seismic.job):
/opt/caos-seismic/scripts/setup-job-checkout.sh
# 3) set the host timezone so 03:00 is local:
sudo timedatectl set-timezone America/Santiago
# 4) edit WorkingDirectory, --venv, User/Group and ReadWritePaths in caos-seismic-daily.service, then:
sudo cp scripts/caos-seismic-daily.service scripts/caos-seismic-daily.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now caos-seismic-daily.timer
systemctl list-timers caos-seismic-daily.timer        # verify next run
sudo systemctl start caos-seismic-daily.service        # test now
journalctl -u caos-seismic-daily -e                    # logs (also under logs/ in the job checkout)
```

`OnCalendar` uses the host's local time zone; set it (step 3) so **03:00** matches
`schedule.time_local`. `Persistent=true` runs the job on next boot if the host was off at 03:00; the
catch-up backfill in `caos-seismic daily` then fills any missed issue dates.

## Conventions

- **Parallel surfaces:** every subcommand is identical across `.ps1` and `.sh`. If you add a flag to one,
  add it to the other.
- **No science in the scripts.** They only wrap the CLI; all the science lives in `src/caos_seismic/`.
  Heavy deps (obspy / pycsep / geopandas / pygtide) are imported lazily by the stages that need them, so
  `setup` + `check` work with the core deps alone.
- **Public-safe:** no secrets, no machine-specific absolute paths, no reference to any private vault.

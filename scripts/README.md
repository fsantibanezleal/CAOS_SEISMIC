<!-- markdownlint-disable MD013 -->
# `scripts/` — operator entry points

Thin, parallel **PowerShell (`*.ps1`)** and **bash (`*.sh`)** wrappers around the `caos-seismic` CLI
(`src/caos_seismic/cli.py`). Each subcommand exists in **both** flavours with identical behaviour, so the
same workflow runs on Windows (the local GPU workstation) and on a Linux VPS. Every script sources a
shared helper (`_common.ps1` / `_common.sh`) that resolves the repo root, locates the `.venv`
interpreter, and invokes the package as a module (`python -m caos_seismic.cli …`).

> **Forecasts, never predictions.** Everything here produces *bounded, calibrated, CSEP-scored
> probabilities* — never alarms, countdowns, or a "safe" state. See the repo `README.md`.

All paths are resolved **relative to the repo root**; there are **no machine-specific absolute paths and
no secrets** in any script (public-safe).

## The subcommands (1:1 across `.ps1` / `.sh`)

| Script | What it does | Underlying CLI |
|---|---|---|
| `setup` | Create the `.venv` (Python 3.12 if available), `pip install -r requirements.txt`, install the package editable, smoke-test. | — (env bootstrap) |
| `fetch` | Pull the recent + historical catalog (USGS ComCat spine + regional/anchor sources) and write a provenance manifest. | `caos-seismic fetch --region <id>` |
| `build-features` | `M_c` + b-value, Mw homogenization, **dual-catalog** declustering, feature extraction. | `caos-seismic build-features --region <id>` |
| `train` | Fit the stationary smoothed-seismicity null + space–time **ETAS** (+ Reasenberg–Jones fallback); reject fits that violate the stability gates. | `caos-seismic train --region <id>` |
| `infer` | Run the **forecast clock** for an issue date → one compact artifact under `results/`. | `caos-seismic infer --region <id> [--issue YYYY-MM-DD]` |
| `backanalysis` | Pseudo-prospective **CSEP** back-analysis over a date range (the clock advances day by day). | `caos-seismic backanalysis --region <id> --start … --end …` |
| `daily` | **Production job:** fetch → infer (today + missed days) → **scoped publish** (commit + push). | (orchestrates `fetch` + `infer` + git) |
| `dev` | Serve the **static** web app locally for preview (Vite HMR, or a dependency-free static server). **No processing backend.** | — (static server) |
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

## `dev` — static preview, never a backend

The web app (`app/`) is a **pure static viewer**: at runtime it only reads the precomputed daily forecast
artifact (`app/public/data/` in preview; the committed `results/` JSON in production). It computes
nothing. `dev` therefore just serves files:

- with **npm/Vite** (HMR) if `node`/`npm` is available — `dev` runs `npm run dev`;
- otherwise it falls back to a **dependency-free** `python -m http.server` over `app/dist/` (or
  `app/public/`), so a preview is possible with only the Python `.venv`.

Flags: `-Build`/`--build` (build the SPA first, then serve `app/dist/`), `-Static`/`--static` (force the
plain static server), `-Port`/`--port` (default `5173`). It binds to `127.0.0.1` only.

## `daily` — the production job (scoped, git-as-data)

`daily` is the once-per-day production job (scheduled ~03:00 local, see below):

1. **`fetch`** once — the freshest catalog covers every issue date in the batch.
2. **`infer`** for **today plus any missed prior days** (catch-up, bounded to the last 7 days so a
   long-dormant laptop does not try to backfill months of non-honest forecasts). A day already present
   under `results/` is skipped.
3. **Scoped publish** — stage **only** the `configs/publish.yaml` `git.add_allowlist` paths
   (`results/`, `manifests/`), **abort** if anything outside the allowlist is staged, commit with the
   configured `commit_message_prefix`, and `git push`.

### Scoped-publish discipline (hard rules)

This machine also holds `data/`, `models/`, `.venv/`, and `.env`. The publish step therefore:

- **NEVER** runs `git add -A` / `git add .`. It stages **only** the explicit allowlist entries.
- **Resets the index first** so a pre-existing staged change cannot ride along.
- **Aborts** (and resets the index) if any staged path is outside the allowlist — nothing is committed.
- Reads the allowlist, commit prefix, remote, and branch from `configs/publish.yaml` (`git.*`).

The push credential is a **least-privilege deploy key / fine-grained PAT scoped to THIS repo**, kept in
the host's git credential store — **never** committed and **never** in these scripts. If the push fails
(no credential / no remote configured), the commit is preserved locally and the script exits non-zero
with an actionable message.

Dry run (no commit/push): `daily.ps1 -NoPublish` / `daily.sh --no-publish`. Skip catch-up:
`-NoCatchUp` / `--no-catch-up`.

## Scheduling the daily job

The job is the same on both platforms; only the scheduler differs.

### Windows — Task Scheduler (`schedule-daily.ps1`)

Registers a task that runs `daily.ps1` **daily at the local time from `configs/publish.yaml`**
(`schedule.time_local`, default **03:00**), configured to **run whether the user is logged on or not**,
**wake the computer to run**, start on next wake if a fire was missed, and run on battery.

```powershell
# from an ELEVATED PowerShell:
.\scripts\schedule-daily.ps1                 # register (idempotent; reads the time from publish.yaml)
.\scripts\schedule-daily.ps1 -Time 03:30     # override the time
.\scripts\schedule-daily.ps1 -Remove         # unregister

Get-ScheduledTask -TaskName 'CAOS_SEISMIC daily forecast' | Get-ScheduledTaskInfo   # inspect
Start-ScheduledTask  -TaskName 'CAOS_SEISMIC daily forecast'                         # run now
```

The task uses an **S4U** principal (run whether logged on or not, no stored password, no interactive
session) at the highest run level. Registration requires admin.

### Linux VPS — systemd timer (portable fallback)

`caos-seismic-daily.service` (oneshot, runs `daily.sh`) + `caos-seismic-daily.timer`
(`OnCalendar=*-*-* 03:00:00`, `Persistent=true` for missed-run catch-up).

```bash
# 1) clone the repo (e.g. /opt/caos-seismic), run scripts/setup.sh as the run user
# 2) set the host timezone so 03:00 is local:
sudo timedatectl set-timezone America/Santiago
# 3) edit WorkingDirectory + User/Group in caos-seismic-daily.service, then install both units:
sudo cp scripts/caos-seismic-daily.service scripts/caos-seismic-daily.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now caos-seismic-daily.timer
systemctl list-timers caos-seismic-daily.timer        # verify next run
sudo systemctl start caos-seismic-daily.service        # test now
journalctl -u caos-seismic-daily -e                    # logs
```

`OnCalendar` uses the host's local time zone; set it (step 2) so **03:00** matches
`schedule.time_local`. `Persistent=true` runs the job on next boot if the host was off at 03:00; the
catch-up backfill in `daily.sh` then fills any missed issue dates.

## Conventions

- **Parallel surfaces:** every subcommand is identical across `.ps1` and `.sh`. If you add a flag to one,
  add it to the other.
- **No science in the scripts.** They only wrap the CLI; all the science lives in `src/caos_seismic/`.
  Heavy deps (obspy / pycsep / geopandas / pygtide) are imported lazily by the stages that need them, so
  `setup` + `check` work with the core deps alone.
- **Public-safe:** no secrets, no machine-specific absolute paths, no reference to any private vault.

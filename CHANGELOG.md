# Changelog

All notable changes to this product. Format: `X.XX.XXX` (display, see the workspace `versioning.md`); stays `0.x` while pre-1.0. Tag every release.

## [0.02.002] · 2026-09-26

### Changed
- No em-dash in the product's content (ADR-0067): the site's strings and pages, the Python package,
  the docs, the experiment log the app serves, the configs and the scripts. The forecast records
  (`manifests/`, `results/`, `data/` and the app's generated data fallback) are results, not content,
  and keep the text their producers wrote; the content guard skips them and says so.
- A CI workflow of cheap guards on the trunk branches (content standard, CI budget, the package
  compiles); the Pages deploy stays its own workflow on `main`.
- `app/package.json` carries the semver form of `VERSION` (it had stayed at `0.1.0`).

### Housekeeping
- `develop` and `main` held the same tree with divergent histories: the daily forecast data commits
  had landed on both as different commits. Reconciled by merging `main` into `develop` and promoting
  it; the job publishes to `main` alone, and `develop` must merge `main` before a release.
- 12 merged task branches deleted.

## [0.02.001] · 2026-09-18

### Fixed
- `scripts/job.{ps1,sh}` retry a step once when its interpreter dies abnormally. A Python-level failure (exit 1 or 2) is still final. On 2026-09-18 the workstation's Microsoft Store Python 3.12 died repeatedly in `ntdll.dll` with `0xC000070A`, and one such death stopped a job midway. With the job checkout's publish, a retry either recomputes and commits or pushes the commit the dead run already made.

## [0.02.000] · 2026-09-18

### Changed
- The scheduled jobs publish from a **dedicated job checkout** (a git worktree with a detached HEAD at `origin/main`, marked by `.caos-seismic-job`) instead of the developer checkout. `main` becomes the single branch that receives data, with one commit per run, fast-forward pushed. Until now each run committed twice: on the branch checked out in the developer checkout, and as a separate `commit-tree` copy on `main`. That left `develop` 24 commits ahead of and 83 behind `main` (#49).
- `scripts/daily.sh` is a thin wrapper over `caos-seismic daily` (parity with `daily.ps1`). The CLI is the single implementation of the job, including the publish, on both platforms.
- `schedule-daily.ps1` and `schedule-outlook.ps1` register the tasks against `scripts\job.ps1` from the job checkout (they refuse to run elsewhere), keep the fire time local across daylight saving, and accept `-VenvPath`, `-LogonType` and `-RunLevel`.
- The in-app Architecture modal and the lanes diagram describe the new publish path.

### Added
- `caos-seismic job-sync`: before every job, fast-forward the job checkout to `main`, push any data commit an earlier run could not push, drop the uncommitted leftovers of an interrupted run, and refuse local code changes or non-data commits.
- `scripts/job.{ps1,sh}` (the scheduled entry point: lock, `job-sync`, `daily` or `outlook`, a log per run under `logs/`) and `scripts/setup-job-checkout.{ps1,sh}` (create the job checkout and copy the gitignored data stores).
- The scripts run the code of the checkout they live in (its `src/` first on `PYTHONPATH`) and accept `CAOS_SEISMIC_VENV` for the interpreter; `setup` never uses it.
- `tests/test_job_publish.py`: the publish path against real local git repositories.
- `.gitattributes`: shell scripts keep LF endings in Windows checkouts.

### Fixed
- A day whose push fails is no longer lost on `main`. The commit stays in the job checkout and the next run pushes it. Before, the next day's overlay carried only paths that changed against the developer branch, so the 2026-09-08 forecast reached `develop` but never `main`; `main`'s index listed it while the file answered 404.
- The publish no longer pushes the developer branch to `main` when the fetch fails.

### Deprecated
- Publishing from a developer checkout (the legacy path: a local commit plus a `commit-tree` copy on `main`). It still works, with a warning, until the scheduled tasks are re-registered against the job checkout; it will then be removed.

## [0.01.000] · 2026-07-03

### Added
- Adopt the `X.XX.XXX` versioning scheme: a `VERSION` file as the single source of truth, this `CHANGELOG`, and the first git tag. Baseline documenting the current shipped state; later changes are versioned by nature (major/minor/patch).

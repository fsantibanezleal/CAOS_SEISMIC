#!/usr/bin/env bash
# job.sh - the SCHEDULED entry point on Linux (systemd timer); parity with scripts/job.ps1.
#
# It runs ONLY in the dedicated job checkout: a git worktree with a DETACHED HEAD at the publish branch
# (main), outside the developer checkout, carrying the .caos-seismic-job marker (scripts/setup-job-checkout.sh
# creates it; docs/deploy.md section 4 explains it). One run takes a lock, then runs two processes:
#   1. `caos-seismic job-sync`: fast-forward to origin/main, after first pushing any data commit an earlier
#      run could not push (its own process, so step 2 imports one consistent version of the code);
#   2. `caos-seismic daily` (or `outlook`): compute and publish; the data commit lands on the detached HEAD
#      and that same commit is fast-forward pushed to main.
#
#   scripts/job.sh --job daily                           # the daily job
#   scripts/job.sh --job outlook                         # the weekly 30-day outlook
#   scripts/job.sh --job daily --venv <dir>              # run with an existing environment (CAOS_SEISMIC_VENV)
#   scripts/job.sh --job daily --no-publish              # compute only: no job-sync, no commit, no push
#   scripts/job.sh --sync-only                           # job-sync only
#   scripts/job.sh --job daily --publish-branch <name>   # end-to-end test against a scratch branch
#
# Every run appends to logs/job-<job>-<UTC stamp>.log (gitignored; the newest --keep-logs files are kept)
# and echoes to stdout (journald). flock on logs/job.lock keeps the daily and weekly jobs from overlapping.
# Public-safe: no secrets, no machine-specific paths.

set -euo pipefail
# shellcheck source=_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

JOB="daily"
REGION="global"
NO_PUBLISH=0
NO_CATCH_UP=0
SYNC_ONLY=0
LOCK_WAIT_MIN=90
KEEP_LOGS=120
while [ $# -gt 0 ]; do
  case "$1" in
    --job)               JOB="$2"; shift 2 ;;
    --region|-r)         REGION="$2"; shift 2 ;;
    --venv)              export CAOS_SEISMIC_VENV="$2"; shift 2 ;;
    --publish-branch)    export CAOS_SEISMIC_PUBLISH_BRANCH="$2"; shift 2 ;;
    --no-publish)        NO_PUBLISH=1; shift ;;
    --no-catch-up)       NO_CATCH_UP=1; shift ;;
    --sync-only)         SYNC_ONLY=1; shift ;;
    --lock-wait-minutes) LOCK_WAIT_MIN="$2"; shift 2 ;;
    --keep-logs)         KEEP_LOGS="$2"; shift 2 ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac
done
case "${JOB}" in
  daily|outlook) ;;
  *) err "--job must be daily or outlook (got '${JOB}')."; exit 2 ;;
esac
if [ "${SYNC_ONLY}" -eq 1 ] && [ "${NO_PUBLISH}" -eq 1 ]; then
  err "--sync-only and --no-publish exclude each other."
  exit 2
fi

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
label="${JOB}"
if [ "${SYNC_ONLY}" -eq 1 ]; then label="sync"; fi
LOG="${LOG_DIR}/job-${label}-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG}") 2>&1

log()  { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
fail() { log "ERROR: $*"; exit 1; }

log "job=${label} region=${REGION} checkout=${REPO_ROOT} user=$(id -un 2>/dev/null || printf '%s' "${USER:-?}")"
if ! is_job_checkout; then
  fail "not the dedicated job checkout (no '${JOB_MARKER}' in ${REPO_ROOT}). Create one with scripts/setup-job-checkout.sh (docs/deploy.md section 4); never schedule a developer checkout."
fi
log "python=$(caos_python)"
if [ -n "${CAOS_SEISMIC_PUBLISH_BRANCH:-}" ]; then
  log "publish branch override: ${CAOS_SEISMIC_PUBLISH_BRANCH} (test run)"
fi

# One job at a time in this checkout (released automatically when the process exits).
exec 9>"${LOG_DIR}/job.lock"
if command -v flock >/dev/null 2>&1; then
  flock -w $(( LOCK_WAIT_MIN * 60 )) 9 \
    || fail "another job still holds ${LOG_DIR}/job.lock after ${LOCK_WAIT_MIN} min; giving up (the next run catches up)."
else
  log "WARN: flock is not available here; running without the job lock."
fi

log "lock acquired; HEAD $(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
if [ "${NO_PUBLISH}" -eq 0 ]; then
  log "caos-seismic job-sync"
  invoke_caos job-sync || fail "job-sync exited with code $?"
fi
if [ "${SYNC_ONLY}" -eq 0 ]; then
  args=("${JOB}" --region "${REGION}")
  if [ "${NO_PUBLISH}" -eq 1 ]; then args+=(--no-publish); fi
  if [ "${NO_CATCH_UP}" -eq 1 ] && [ "${JOB}" = "daily" ]; then args+=(--no-catch-up); fi
  log "caos-seismic ${args[*]}"
  invoke_caos "${args[@]}" || fail "caos-seismic ${args[*]} exited with code $?"
fi
log "done; HEAD $(git -C "${REPO_ROOT}" rev-parse --short HEAD)"

# Keep the newest KEEP_LOGS run logs.
ls -1t "${LOG_DIR}"/job-*.log 2>/dev/null | tail -n +$(( KEEP_LOGS + 1 )) | while IFS= read -r old; do
  rm -f -- "${old}"
done

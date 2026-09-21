#!/usr/bin/env bash
# setup-job-checkout.sh - create the dedicated JOB checkout that the scheduled job runs from; parity with
# scripts/setup-job-checkout.ps1.
#
# Run it from the developer checkout. The job checkout is a git worktree of this repository with a
# DETACHED HEAD at origin/<publish branch>, outside the developer checkout and used by nothing but
# scripts/job.sh (docs/deploy.md section 4). This script adds the worktree (default: a sibling folder named
# <repo>.job), writes the .caos-seismic-job marker the CLI requires before it syncs or publishes, and
# copies every file git ignores under data/ and results/ (the catalogs, enricher caches, weights and
# checkpoints the pipeline reads). From then on the job checkout owns its copy.
#
#   scripts/setup-job-checkout.sh                        # creates <parent>/<repo>.job
#   scripts/setup-job-checkout.sh --path <dir>           # explicit location
#   scripts/setup-job-checkout.sh --refresh-data         # re-copy the data stores into an existing one
#
# Public-safe: no secrets; every path is derived from this checkout or passed in.

set -euo pipefail
# shellcheck source=_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DEST=""
REMOTE="origin"
BRANCH="main"
REFRESH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --path)         DEST="$2"; shift 2 ;;
    --remote)       REMOTE="$2"; shift 2 ;;
    --branch)       BRANCH="$2"; shift 2 ;;
    --refresh-data) REFRESH=1; shift ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac
done

if is_job_checkout; then
  err "run setup-job-checkout.sh from the developer checkout, not from a job checkout (${REPO_ROOT})."
  exit 1
fi
if [ -z "${DEST}" ]; then
  DEST="$(dirname "${REPO_ROOT}")/$(basename "${REPO_ROOT}").job"
fi

if [ -e "${DEST}" ]; then
  if [ ! -f "${DEST}/${JOB_MARKER}" ]; then
    err "'${DEST}' exists and is not a job checkout (no ${JOB_MARKER}); pass another --path."
    exit 1
  fi
  if [ "${REFRESH}" -eq 0 ]; then
    info "job checkout already present at ${DEST} (use --refresh-data to re-copy the data stores)."
  fi
else
  if [ "${REFRESH}" -eq 1 ]; then
    err "no job checkout at '${DEST}' to refresh."
    exit 1
  fi
  step "Fetching ${REMOTE}/${BRANCH}"
  git -C "${REPO_ROOT}" fetch "${REMOTE}" "${BRANCH}"
  base="$(git -C "${REPO_ROOT}" rev-parse FETCH_HEAD)"
  step "Adding the job worktree at ${DEST} (detached at ${REMOTE}/${BRANCH}, ${base:0:9})"
  git -C "${REPO_ROOT}" worktree add --detach "${DEST}" "${base}"
  printf '%s\n' "Dedicated CAOS_SEISMIC job checkout: the scheduled jobs run here (scripts/job.sh). Do not develop or switch branches here; see docs/deploy.md section 4." \
    > "${DEST}/${JOB_MARKER}"
  REFRESH=1   # a new job checkout always gets the data stores
fi

if [ "${REFRESH}" -eq 1 ]; then
  step "Copying the gitignored data stores from ${REPO_ROOT}"
  n=0
  while IFS= read -r rel; do
    [ -n "${rel}" ] || continue
    mkdir -p "$(dirname "${DEST}/${rel}")"
    cp -f -- "${REPO_ROOT}/${rel}" "${DEST}/${rel}"
    info "${rel}"
    n=$(( n + 1 ))
  done < <(git -C "${REPO_ROOT}" ls-files --others --ignored --exclude-standard -- data results)
  info "${n} file(s) copied."
fi

step "Job checkout ready: ${DEST}"
info "Dry run (no git write): ${DEST}/scripts/job.sh --job daily --no-publish"
info "Then point the systemd unit at it (WorkingDirectory=${DEST}; see scripts/caos-seismic-daily.service)."

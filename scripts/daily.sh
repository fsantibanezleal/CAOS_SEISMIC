#!/usr/bin/env bash
# daily.sh - the daily job: fetch -> infer -> SCOPED publish, via the self-sufficient CLI; parity with
# scripts/daily.ps1.
#
# A thin wrapper over `caos-seismic daily`, which performs fetch + infer (today + any MISSED prior days,
# bounded to a week) AND the scoped git publish entirely in Python (cli.py `_publish_scoped`): one
# implementation of the job on every platform. The scheduled job does not call this script; it runs
# scripts/job.sh from the dedicated job checkout, which syncs that checkout first and is the only place
# that publishes (docs/deploy.md section 4). Use this script for a local dry run.
#
#   scripts/daily.sh --no-publish     # fetch + infer only (local dry run, no commit/push)
#   scripts/daily.sh --no-catch-up    # only today's issue date (skip missed-day backfill)
#   scripts/daily.sh                  # full job, including the publish (meant for the job checkout)
#
# Scoped-publish discipline lives in the CLI: NEVER `git add -A`/`.`, abort on any out-of-allowlist staged
# path. Public-safe: no secrets, no machine-specific paths.

set -euo pipefail
# shellcheck source=_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

REGION="global"
NO_PUBLISH=0
CATCH_UP=1
while [ $# -gt 0 ]; do
  case "$1" in
    --region|-r)   REGION="$2"; shift 2 ;;
    --no-publish)  NO_PUBLISH=1; shift ;;
    --no-catch-up) CATCH_UP=0; shift ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac
done

args=(daily --region "${REGION}")
if [ "${NO_PUBLISH}" -eq 1 ]; then args+=(--no-publish); fi
if [ "${CATCH_UP}" -eq 0 ]; then args+=(--no-catch-up); fi

step "daily  (caos-seismic ${args[*]})"
invoke_caos "${args[@]}"
step "daily  done."

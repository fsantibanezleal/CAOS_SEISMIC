#!/usr/bin/env bash
# check.sh — environment + repo + config sanity checks (no network, no science deps required).
#
#   scripts/check.sh --region chile
#
# Thin wrapper over:  caos-seismic check --region <id>
# Exits non-zero if any hard check fails (used as a pre-flight in CI / before the daily job).

set -euo pipefail
# shellcheck source=_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

REGION="chile"
while [ $# -gt 0 ]; do
  case "$1" in
    --region|-r) REGION="$2"; shift 2 ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac
done

step "caos-seismic check --region ${REGION}"
invoke_caos check --region "${REGION}"

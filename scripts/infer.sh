#!/usr/bin/env bash
# infer.sh — run the daily forecast clock for an issue date -> compact artifact under results/.
#
#   scripts/infer.sh                                  # issue date = today (UTC)
#   scripts/infer.sh --region chile --issue 2026-06-16
#
# Thin wrapper over:  caos-seismic infer --region <id> [--issue YYYY-MM-DD]

set -euo pipefail
# shellcheck source=_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

REGION="global"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --region|-r) REGION="$2"; shift 2 ;;
    --issue)     ARGS+=(--issue "$2"); shift 2 ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac
done

step "caos-seismic infer --region ${REGION} ${ARGS[*]:-}"
invoke_caos infer --region "${REGION}" "${ARGS[@]}"

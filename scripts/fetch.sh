#!/usr/bin/env bash
# fetch.sh — pull the recent + historical catalog (ComCat spine + regional/anchor sources).
#
#   scripts/fetch.sh                              # default region (chile), configured fetch window
#   scripts/fetch.sh --region chile --days 30     # only the last 30 days
#   scripts/fetch.sh --focus north                # optional sub-region focus
#
# Thin wrapper over:  caos-seismic fetch --region <id> [--days N] [--focus key]

set -euo pipefail
# shellcheck source=_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

REGION="chile"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --region|-r) REGION="$2"; shift 2 ;;
    --days)      ARGS+=(--days "$2"); shift 2 ;;
    --focus)     ARGS+=(--focus "$2"); shift 2 ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac
done

step "caos-seismic fetch --region ${REGION} ${ARGS[*]:-}"
invoke_caos fetch --region "${REGION}" "${ARGS[@]}"

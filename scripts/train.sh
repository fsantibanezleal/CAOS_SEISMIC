#!/usr/bin/env bash
# train.sh — fit the stationary smoothed-seismicity null + space-time ETAS (+ R-J fallback).
#
#   scripts/train.sh --region chile
#
# Thin wrapper over:  caos-seismic train --region <id>

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

step "caos-seismic train --region ${REGION}"
invoke_caos train --region "${REGION}"

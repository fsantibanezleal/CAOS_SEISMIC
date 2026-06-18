#!/usr/bin/env bash
# build-features.sh — Mc + b-value, Mw homogenization, dual-catalog declustering, feature extraction.
#
#   scripts/build-features.sh --region chile
#
# Thin wrapper over:  caos-seismic build-features --region <id>

set -euo pipefail
# shellcheck source=_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

REGION="global"
while [ $# -gt 0 ]; do
  case "$1" in
    --region|-r) REGION="$2"; shift 2 ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac
done

step "caos-seismic build-features --region ${REGION}"
invoke_caos build-features --region "${REGION}"

#!/usr/bin/env bash
# backanalysis.sh — pseudo-prospective CSEP back-analysis over a date range.
#
#   scripts/backanalysis.sh --region chile --start 2024-01-01 --end 2024-12-31
#
# The forecast clock advances day by day; each forecast sees only the catalog slice (-inf, t) and is
# scored against the catalog as it was at issue time.
#
# Thin wrapper over:  caos-seismic backanalysis --region <id> --start YYYY-MM-DD --end YYYY-MM-DD

set -euo pipefail
# shellcheck source=_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

REGION="chile"
START=""
END=""
while [ $# -gt 0 ]; do
  case "$1" in
    --region|-r) REGION="$2"; shift 2 ;;
    --start)     START="$2"; shift 2 ;;
    --end)       END="$2"; shift 2 ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac
done

if [ -z "${START}" ] || [ -z "${END}" ]; then
  err "both --start and --end (YYYY-MM-DD) are required."
  exit 2
fi

step "caos-seismic backanalysis --region ${REGION} --start ${START} --end ${END}"
invoke_caos backanalysis --region "${REGION}" --start "${START}" --end "${END}"

#!/usr/bin/env bash
# _common.sh — shared helpers for the CAOS_SEISMIC bash scripts (Git Bash / Linux VPS).
# Sourced by setup.sh, fetch.sh, build-features.sh, train.sh, infer.sh, daily.sh, dev.sh, check.sh.
# Public-safe: no secrets, no machine-specific paths (everything is resolved relative to the repo root).

set -euo pipefail

# Repo root = parent of the scripts/ directory that contains this file.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

# Colour helpers (no-op if not a TTY).
if [ -t 1 ]; then
  _C_STEP=$'\033[36m'; _C_DIM=$'\033[90m'; _C_WARN=$'\033[33m'; _C_ERR=$'\033[31m'; _C_OFF=$'\033[0m'
else
  _C_STEP=''; _C_DIM=''; _C_WARN=''; _C_ERR=''; _C_OFF=''
fi

step() { printf '%s==> %s%s\n' "$_C_STEP" "$*" "$_C_OFF"; }
info() { printf '%s    %s%s\n' "$_C_DIM" "$*" "$_C_OFF"; }
warn() { printf '%sWARN: %s%s\n' "$_C_WARN" "$*" "$_C_OFF" >&2; }
err()  { printf '%sERROR: %s%s\n' "$_C_ERR" "$*" "$_C_OFF" >&2; }

# Resolve the .venv Python interpreter (POSIX layout, with a Windows/Git-Bash fallback). Exits if absent.
venv_python() {
  if [ -x "${VENV_DIR}/bin/python" ]; then
    printf '%s\n' "${VENV_DIR}/bin/python"
  elif [ -x "${VENV_DIR}/Scripts/python.exe" ]; then
    # Git Bash on Windows: a venv created by python.exe uses Scripts/.
    printf '%s\n' "${VENV_DIR}/Scripts/python.exe"
  else
    err "virtualenv not found at '${VENV_DIR}'. Run  scripts/setup.sh  first."
    exit 1
  fi
}

# Pick an interpreter to BOOTSTRAP the venv: prefer python3.12, then python3, then python.
bootstrap_python() {
  for name in python3.12 python3 python; do
    if command -v "$name" >/dev/null 2>&1; then
      printf '%s\n' "$name"
      return 0
    fi
  done
  err "no Python interpreter found on PATH (need Python 3.12)."
  exit 1
}

# Invoke the package console entry point inside the venv:  caos-seismic <args...>
# Module form, so it works even if the console-script shim is not on PATH.
invoke_caos() {
  local py
  py="$(venv_python)"
  ( cd "${REPO_ROOT}" && "${py}" -m caos_seismic.cli "$@" )
}

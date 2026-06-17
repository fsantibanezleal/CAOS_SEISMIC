#!/usr/bin/env bash
# setup.sh — create the project virtualenv and install the Python dependencies.
#
#   scripts/setup.sh           # create .venv (python3.12 if available) + pip install -r requirements.txt
#   scripts/setup.sh --force   # recreate the .venv from scratch
#
# Public-safe: resolves everything relative to the repo root; no secrets, no machine-specific paths.

set -euo pipefail
# shellcheck source=_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force|-f) FORCE=1 ;;
    *) err "unknown argument: $arg"; exit 2 ;;
  esac
done

if [ "$FORCE" -eq 1 ] && [ -d "${VENV_DIR}" ]; then
  step "Removing existing virtualenv (--force) at ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

if [ ! -d "${VENV_DIR}" ]; then
  boot="$(bootstrap_python)"
  step "Creating virtualenv with: ${boot} -m venv .venv"
  "${boot}" -m venv "${VENV_DIR}"
else
  info "virtualenv already present at ${VENV_DIR} (use --force to recreate)."
fi

PY="$(venv_python)"

step "Upgrading pip / setuptools / wheel"
"${PY}" -m pip install --upgrade pip setuptools wheel

step "Installing requirements from requirements.txt"
"${PY}" -m pip install -r "${REPO_ROOT}/requirements.txt"

step "Installing the caos-seismic package (editable)"
"${PY}" -m pip install -e "${REPO_ROOT}"

step "Smoke test: caos-seismic version"
"${PY}" -m caos_seismic.cli version

step "Setup complete."
info "Next:  scripts/check.sh   then   scripts/fetch.sh"

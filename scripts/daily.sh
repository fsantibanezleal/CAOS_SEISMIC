#!/usr/bin/env bash
# daily.sh — the PRODUCTION daily job: fetch -> infer -> SCOPED publish (commit + push).
#
# This is what the systemd timer (caos-seismic-daily.timer) fires once per day (~03:00 local,
# configs/publish.yaml `schedule.time_local`), and the portable VPS fallback for the Windows Task
# Scheduler task (scripts/schedule-daily.ps1 -> scripts/daily.ps1).
#
# Flow:
#   1. caos-seismic fetch  --region <id>          (refresh the catalog)
#   2. caos-seismic infer  --region <id>          (today + any MISSED prior days — catch-up)
#   3. SCOPED publish: stage ONLY the configs/publish.yaml `git.add_allowlist` paths (results/,
#      manifests/), ABORT if anything outside the allowlist is staged, commit with the configured
#      prefix, push.
#
# Scoped-publish discipline (hard rules — the host also has data/, models/, .venv/, .env):
#   * NEVER `git add -A` / `git add .` — only the explicit allowlist paths.
#   * ABORT the commit if any staged path falls outside the allowlist.
#
#   scripts/daily.sh                  # full job (fetch + infer + publish), region chile
#   scripts/daily.sh --no-publish     # fetch + infer only (local dry run, no commit/push)
#   scripts/daily.sh --no-catch-up    # only today's issue date (skip missed-day backfill)
#
# Public-safe: no secrets, no machine-specific paths. The push credential lives only in the host's git
# credential store (a least-privilege deploy key / fine-grained PAT scoped to THIS repo) — never here.

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

PY="$(venv_python)"

# ── Read the publish config (allowlist, commit prefix, remote, branch) as TSV lines ─────────────────
read_publish_config() {
  ( cd "${REPO_ROOT}" && "${PY}" - <<'PYEOF'
import sys
sys.path.insert(0, "src")
from caos_seismic.config import load
git = (load("publish").get("git", {}) or {})
allow = [str(p) for p in git.get("add_allowlist", [])]
print("PREFIX\t" + str(git.get("commit_message_prefix", "data: daily forecast")))
print("REMOTE\t" + str(git.get("remote", "origin")))
print("BRANCH\t" + str(git.get("publish_branch", "main")))
for a in allow:
    print("ALLOW\t" + a)
PYEOF
  )
}

PREFIX="data: daily forecast"
REMOTE="origin"
BRANCH="main"
ALLOWLIST=()
while IFS=$'\t' read -r key val; do
  case "$key" in
    PREFIX) PREFIX="$val" ;;
    REMOTE) REMOTE="$val" ;;
    BRANCH) BRANCH="$val" ;;
    ALLOW)  ALLOWLIST+=("$val") ;;
  esac
done < <(read_publish_config)

# ── Determine which issue dates to run (today + missed prior days, bounded to a week) ───────────────
today="$(date -u +%F)"
ISSUE_DATES=()
have_artifact() {
  local d="$1"
  ls "${REPO_ROOT}/results/forecast-${REGION}-${d}.json"* >/dev/null 2>&1
}
if [ "${CATCH_UP}" -eq 1 ]; then
  for back in $(seq 7 -1 0); do
    # Portable date arithmetic: GNU date and BSD/macOS date differ; try GNU first, then BSD.
    if d="$(date -u -d "${today} -${back} day" +%F 2>/dev/null)"; then :;
    elif d="$(date -u -v-"${back}"d -j -f %F "${today}" +%F 2>/dev/null)"; then :;
    else d="${today}"; fi
    have_artifact "${d}" || ISSUE_DATES+=("${d}")
  done
fi
# Ensure today is present.
present=0
for d in "${ISSUE_DATES[@]:-}"; do [ "${d}" = "${today}" ] && present=1; done
[ "${present}" -eq 1 ] || ISSUE_DATES+=("${today}")
# De-dup + sort.
mapfile -t ISSUE_DATES < <(printf '%s\n' "${ISSUE_DATES[@]}" | sort -u)

step "daily · region=${REGION} · issue_dates=${ISSUE_DATES[*]}"

# 1) Fetch once — the freshest catalog covers every issue date in this batch.
step "fetch"
invoke_caos fetch --region "${REGION}"

# 2) Infer for each issue date (oldest first, so results/index.json ends current).
for d in "${ISSUE_DATES[@]}"; do
  step "infer · issue=${d}"
  invoke_caos infer --region "${REGION}" --issue "${d}"
done

if [ "${NO_PUBLISH}" -eq 1 ]; then
  step "daily · --no-publish set; produced ${#ISSUE_DATES[@]} artifact(s), not committing."
  exit 0
fi

# ── 3) Scoped publish: stage ONLY the allowlist, abort on any out-of-allowlist staged path ──────────
step "publish (scoped)"
if [ "${#ALLOWLIST[@]}" -eq 0 ]; then
  err "publish.yaml git.add_allowlist is empty; refusing to publish."
  exit 1
fi
for entry in "${ALLOWLIST[@]}"; do
  case "$(printf '%s' "${entry}" | tr -d '[:space:]')" in
    .|-A|--all|\*) err "publish.yaml allowlist has a non-scoped entry '${entry}'; refusing to publish."; exit 1 ;;
  esac
done

cd "${REPO_ROOT}"
# Reset the index so a pre-existing staged change cannot ride along.
git reset -q
for entry in "${ALLOWLIST[@]}"; do
  git add -- "${entry}"
done

mapfile -t STAGED < <(git diff --cached --name-only | sed '/^[[:space:]]*$/d')
if [ "${#STAGED[@]}" -eq 0 ]; then
  info "nothing to commit (no new artifacts)."
  exit 0
fi

# Guard: every staged path MUST be under an allowlist entry.
offenders=()
for s in "${STAGED[@]}"; do
  ok=0
  for entry in "${ALLOWLIST[@]}"; do
    p="${entry%/}"
    if [ "${s}" = "${p}" ] || [ "${s#${p}/}" != "${s}" ]; then ok=1; break; fi
  done
  [ "${ok}" -eq 1 ] || offenders+=("${s}")
done
if [ "${#offenders[@]}" -gt 0 ]; then
  git reset -q
  err "scoped-publish guard tripped: out-of-allowlist paths were staged (${offenders[*]}). Index reset; nothing committed."
  exit 1
fi

suffix="${REGION} ${today}"
n="${#ISSUE_DATES[@]}"
if [ "${n}" -gt 1 ]; then suffix="${suffix} (+$((n - 1)) catch-up)"; fi
message="${PREFIX}: ${suffix}"
git commit -q -m "${message}"
step "committed: ${message}"

if git push "${REMOTE}" "HEAD:${BRANCH}"; then
  step "pushed to ${REMOTE} ${BRANCH}."
else
  warn "commit made locally but push failed (remote '${REMOTE}', branch '${BRANCH}'). Configure the deploy credential and retry; the commit is preserved."
  exit 1
fi
step "daily · done."

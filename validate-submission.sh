#!/usr/bin/env bash
#
# validate-submission.sh — Email Triage OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - openenv-core: pip install openenv-core
#   - curl (usually pre-installed)
#
# Usage:
#   ./validate-submission.sh <ping_url> [repo_dir]
#
# Examples:
#   ./validate-submission.sh https://your-username-email-triage-env.hf.space
#   ./validate-submission.sh https://your-username-email-triage-env.hf.space .
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi
PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  Email Triage OpenEnv — Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

# ── Step 1: Ping HF Space ──────────────────────────────────────────────────
log "${BOLD}Step 1/5: Pinging HF Space${NC} ($PING_URL/health) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" \
  "$PING_URL/health" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and /health returns 200"
else
  fail "HF Space /health returned HTTP $HTTP_CODE (expected 200)"
  hint "Check your network and that the Space is running."
  hint "Try: curl -s $PING_URL/health"
  stop_at "Step 1"
fi

# ── Step 2: Ping /reset ────────────────────────────────────────────────────
log "${BOLD}Step 2/5: Testing /reset endpoint${NC} ..."

RESET_OUTPUT=$(portable_mktemp "validate-reset")
CLEANUP_FILES+=("$RESET_OUTPUT")
RESET_CODE=$(curl -s -o "$RESET_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{"task_name":"easy"}' \
  "$PING_URL/reset" --max-time 30 2>/dev/null || printf "000")

if [ "$RESET_CODE" = "200" ]; then
  pass "/reset returns 200 with task_name=easy"
else
  fail "/reset returned HTTP $RESET_CODE (expected 200)"
  hint "Make sure /reset accepts POST {\"task_name\": \"easy\"}"
  stop_at "Step 2"
fi

# ── Step 3: Docker build ───────────────────────────────────────────────────
log "${BOLD}Step 3/5: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 3"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 3"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 3"
fi

# ── Step 4: openenv validate ───────────────────────────────────────────────
log "${BOLD}Step 4/5: Running openenv validate${NC} ..."

OPENENV_CMD="openenv"
if [ -f "$REPO_DIR/.venv/Scripts/openenv.exe" ]; then
  OPENENV_CMD="$REPO_DIR/.venv/Scripts/openenv.exe"
elif [ -f "$REPO_DIR/.venv/bin/openenv" ]; then
  OPENENV_CMD="$REPO_DIR/.venv/bin/openenv"
fi

if ! command -v "$OPENENV_CMD" &>/dev/null && [ ! -x "$OPENENV_CMD" ]; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 4"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && "$OPENENV_CMD" validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 4"
fi

# ── Step 5: Check inference.py exists ─────────────────────────────────────
log "${BOLD}Step 5/5: Checking inference.py${NC} ..."

if [ -f "$REPO_DIR/inference.py" ]; then
  pass "inference.py found in repo root"
else
  fail "inference.py not found in repo root"
  hint "The inference script must be named 'inference.py' and placed in the project root."
  stop_at "Step 5"
fi

# ── Done ───────────────────────────────────────────────────────────────────
printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 5/5 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
#!/usr/bin/env bash
# =============================================================================
# OpenEnv Hackathon Pre-Submission Validator
# Recreated from the hackathon-provided script.
# Checks: 1) Required files exist, 2) Docker builds, 3) openenv validate passes
# =============================================================================

set -euo pipefail

NC='\033[0m'
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'

DOCKER_BUILD_TIMEOUT=300
REPO_DIR="${1:-.}"

pass() { printf "  ${GREEN}✓ %s${NC}\n" "$1"; }
fail() { printf "  ${RED}✗ %s${NC}\n" "$1"; }
hint() { printf "    ${YELLOW}→ %s${NC}\n" "$1"; }
log()  { printf "%b\n" "$1"; }

stop_at() {
  printf "\n${RED}${BOLD}  Pre-validation failed at %s${NC}\n\n" "$1"
  exit 1
}

run_with_timeout() {
  local timeout=$1; shift
  if command -v gtimeout &>/dev/null; then
    gtimeout "$timeout" "$@"
  elif command -v timeout &>/dev/null; then
    timeout "$timeout" "$@"
  else
    "$@"
  fi
}

printf "\n${BOLD}OpenEnv Pre-Submission Validator${NC}\n"
printf "Checking: %s\n\n" "$(cd "$REPO_DIR" && pwd)"

# ── Step 1: Check required files ──────────────────────────────────────────
log "${BOLD}Step 1/3: Checking required files${NC} ..."

if [ -f "$REPO_DIR/inference.py" ]; then
  pass "inference.py found"
else
  fail "inference.py not found in repo root"
  stop_at "Step 1"
fi

if [ -f "$REPO_DIR/openenv.yaml" ]; then
  pass "openenv.yaml found"
else
  fail "openenv.yaml not found"
  stop_at "Step 1"
fi

if [ -f "$REPO_DIR/pyproject.toml" ]; then
  pass "pyproject.toml found"
else
  fail "pyproject.toml not found"
  stop_at "Step 1"
fi

# ── Step 2: Docker build ──────────────────────────────────────────────────
log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

# ── Step 3: openenv validate ──────────────────────────────────────────────
log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0

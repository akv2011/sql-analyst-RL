#!/usr/bin/env bash
# =============================================================================
# OpenEnv Round 1 Pre-Submission Validator
# Mirrors the judging flow more closely:
# 1) required files and config
# 2) local API reset/health contract
# 3) task + grader integrity
# 4) openenv validate
# 5) docker build
# 6) baseline inference (when credentials are available)
# 7) optional Hugging Face Space ping (when SPACE_URL is set)
# =============================================================================

set -euo pipefail

NC='\033[0m'
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'

REPO_DIR="${1:-.}"
DOCKER_BUILD_TIMEOUT="${DOCKER_BUILD_TIMEOUT:-300}"
BASELINE_TIMEOUT="${BASELINE_TIMEOUT:-1200}"
INFERENCE_TASK_IDS="${INFERENCE_TASK_IDS:-1,2,3}"
PARTIAL_VALIDATION=0

pass() { printf "  ${GREEN}✓ %s${NC}\n" "$1"; }
warn() { printf "  ${YELLOW}! %s${NC}\n" "$1"; }
fail() { printf "  ${RED}✗ %s${NC}\n" "$1"; }
hint() { printf "    ${YELLOW}→ %s${NC}\n" "$1"; }
log()  { printf "%b\n" "$1"; }

stop_at() {
  printf "\n${RED}${BOLD}  Pre-validation failed at %s${NC}\n\n" "$1"
  exit 1
}

run_with_timeout() {
  local timeout=$1
  shift
  if command -v gtimeout &>/dev/null; then
    gtimeout "$timeout" "$@"
  elif command -v timeout &>/dev/null; then
    timeout "$timeout" "$@"
  else
    "$@"
  fi
}

check_file() {
  local path=$1
  local label=$2
  if [ -f "$path" ]; then
    pass "$label found"
  else
    fail "$label not found at $path"
    return 1
  fi
}

printf "\n${BOLD}OpenEnv Round 1 Pre-Submission Validator${NC}\n"
printf "Checking: %s\n\n" "$(cd "$REPO_DIR" && pwd)"

# ── Step 1: Required files ────────────────────────────────────────────────
log "${BOLD}Step 1/7: Checking required files${NC} ..."

check_file "$REPO_DIR/inference.py" "inference.py" || stop_at "Step 1"
check_file "$REPO_DIR/openenv.yaml" "openenv.yaml" || stop_at "Step 1"
check_file "$REPO_DIR/pyproject.toml" "pyproject.toml" || stop_at "Step 1"
check_file "$REPO_DIR/README.md" "README.md" || stop_at "Step 1"

if [ -f "$REPO_DIR/Dockerfile" ]; then
  pass "Dockerfile found in repo root"
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  pass "Dockerfile found in server/"
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/"
  stop_at "Step 1"
fi

if [ -f "$REPO_DIR/.env.example" ]; then
  pass ".env.example found"
else
  warn ".env.example missing (recommended for submission handoff)"
fi

# ── Step 2: Local API contract ────────────────────────────────────────────
log "${BOLD}Step 2/7: Checking local API reset/health contract${NC} ..."

API_CHECK_OUTPUT=$(
  cd "$REPO_DIR" && python - <<'PY'
from pathlib import Path
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo))

from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

health = client.get("/health")
assert health.status_code == 200, f"/health returned {health.status_code}: {health.text}"

metadata = client.get("/metadata")
assert metadata.status_code == 200, f"/metadata returned {metadata.status_code}: {metadata.text}"
meta = metadata.json()
assert meta.get("name"), "metadata.name missing"
assert meta.get("version"), "metadata.version missing"

reset = client.post("/reset", json={})
assert reset.status_code == 200, f"/reset returned {reset.status_code}: {reset.text}"
payload = reset.json()
obs = payload.get("observation") or {}
assert obs.get("task_id") == 1, f"unexpected task_id: {obs.get('task_id')}"
assert payload.get("done") is False, "reset returned done=True"
assert payload.get("reward") == 0.0, f"reset returned non-zero reward: {payload.get('reward')}"
assert obs.get("schema_info"), "schema_info missing on reset"
assert obs.get("task_description"), "task_description missing on reset"

print(
    f"health={health.status_code} metadata={metadata.status_code} "
    f"reset={reset.status_code} task_id={obs.get('task_id')}"
)
PY
) || {
  fail "Local API contract check failed"
  hint "Make sure server.app:create_app exposes /health, /metadata, and /reset correctly."
  stop_at "Step 2"
}

pass "Local API contract looks healthy"
log "  $API_CHECK_OUTPUT"

# ── Step 3: Task + grader integrity ───────────────────────────────────────
log "${BOLD}Step 3/7: Checking tasks and graders${NC} ..."

TASK_CHECK_OUTPUT=$(
  cd "$REPO_DIR" && python - <<'PY'
from pathlib import Path
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo))

from server.database import TASK_DESCRIPTIONS, create_database
from server.graders import grade
from server.sql_analyst_env_environment import SqlAnalystEnvironment

task_ids = sorted(TASK_DESCRIPTIONS)
assert len(task_ids) >= 3, f"expected >=3 tasks, found {len(task_ids)}"

_, ground_truth = create_database()
for task_id in task_ids:
    gt_key = f"task{task_id}"
    assert gt_key in ground_truth, f"missing ground truth for {gt_key}"

env = SqlAnalystEnvironment()
for task_id in task_ids:
    obs = env.reset(task_id=task_id)
    assert obs.task_id == task_id, f"reset returned wrong task_id for task {task_id}"
    assert obs.task_description, f"task description missing for task {task_id}"

probe_answer = (
    "Validation probe answer with 42, 12.5%, electronics, marketing, "
    "and recommendation 1. recommendation 2. recommendation 3."
)
for task_id in task_ids:
    first_score, feedback = grade(task_id, probe_answer, ground_truth)
    second_score, _ = grade(task_id, probe_answer, ground_truth)
    empty_score, _ = grade(task_id, "", ground_truth)
    assert 0.0 <= first_score <= 1.0, f"task {task_id} score out of range: {first_score}"
    assert 0.0 <= empty_score <= 1.0, f"task {task_id} empty score out of range: {empty_score}"
    assert first_score == second_score, f"task {task_id} grader is non-deterministic"
    assert isinstance(feedback, str), f"task {task_id} feedback is not text"

env.close()
print(f"tasks={task_ids} ground_truth={sorted(ground_truth.keys())}")
PY
) || {
  fail "Task/grader integrity check failed"
  hint "Ensure every task has a resettable prompt, ground truth, and deterministic grader in [0,1]."
  stop_at "Step 3"
}

pass "Task and grader checks passed"
log "  $TASK_CHECK_OUTPUT"

# ── Step 4: openenv validate ──────────────────────────────────────────────
log "${BOLD}Step 4/7: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it with: pip install openenv-core"
  stop_at "Step 4"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 4"
fi

# ── Step 5: Docker build ──────────────────────────────────────────────────
log "${BOLD}Step 5/7: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker Desktop or Docker Engine."
  stop_at "Step 5"
fi

if ! docker info &>/dev/null; then
  fail "Docker daemon is not reachable"
  hint "Start Docker Desktop and rerun the validator."
  stop_at "Step 5"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 5"
fi

# ── Step 6: Baseline inference ────────────────────────────────────────────
log "${BOLD}Step 6/7: Running baseline inference (if credentials exist)${NC} ..."

if [ -n "${HF_TOKEN:-}" ] || [ -n "${OPENAI_API_KEY:-}" ]; then
  BASELINE_OK=false
  BASELINE_OUTPUT=$(
    cd "$REPO_DIR" && \
    run_with_timeout "$BASELINE_TIMEOUT" \
      env \
        API_BASE_URL="${API_BASE_URL:-https://api.openai.com/v1}" \
        MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}" \
        HF_TOKEN="${HF_TOKEN:-}" \
        OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
        python inference.py --tasks "$INFERENCE_TASK_IDS" --summary-path /tmp/sql_analyst_baseline.json 2>&1
  ) && BASELINE_OK=true

  if [ "$BASELINE_OK" = true ]; then
    pass "Baseline inference completed"
    printf "%s\n" "$BASELINE_OUTPUT" | tail -20
  else
    fail "Baseline inference failed or timed out (timeout=${BASELINE_TIMEOUT}s)"
    printf "%s\n" "$BASELINE_OUTPUT" | tail -40
    stop_at "Step 6"
  fi
else
  PARTIAL_VALIDATION=1
  warn "Skipping live baseline run because HF_TOKEN / OPENAI_API_KEY is not set"
  hint "Set API_BASE_URL, MODEL_NAME, and HF_TOKEN to exercise the judge-facing inference path."
fi

# ── Step 7: Optional HF Space ping ────────────────────────────────────────
log "${BOLD}Step 7/7: Pinging Hugging Face Space (optional)${NC} ..."

if [ -n "${SPACE_URL:-}" ]; then
  SPACE_RESET_URL="${SPACE_URL%/}/reset"
  HTTP_CODE=$(curl -sS -o /tmp/openenv_space_reset.json -w "%{http_code}" \
    -X POST "$SPACE_RESET_URL" \
    -H "Content-Type: application/json" \
    -d '{}' || true)

  if [ "$HTTP_CODE" = "200" ]; then
    pass "Space /reset responded with HTTP 200"
  else
    fail "Space /reset returned HTTP $HTTP_CODE"
    hint "Check that SPACE_URL points at the live Space root and the app is awake."
    stop_at "Step 7"
  fi
else
  PARTIAL_VALIDATION=1
  warn "Skipping live Space ping because SPACE_URL is not set"
  hint "Set SPACE_URL to your deployed HF Space root to validate the hosted /reset endpoint."
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
if [ "$PARTIAL_VALIDATION" -eq 0 ]; then
  printf "${GREEN}${BOLD}  All 7/7 checks passed.${NC}\n"
  printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
else
  printf "${YELLOW}${BOLD}  Core local checks passed.${NC}\n"
  printf "${YELLOW}${BOLD}  Live baseline and/or Space checks were skipped.${NC}\n"
fi
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0

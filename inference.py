"""
Baseline inference — runs an LLM agent against all 3 tasks.

The agent reads the schema, writes SQL queries to explore the data,
and submits its analysis. We use the OpenAI client so it works with
any compatible API.

Setup:
    conda activate meta
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o"
    export OPENAI_API_KEY="sk-..."
    python inference.py
"""

import os
import re
import json
import time
import sys

from openai import OpenAI

# ── Environment setup (direct import, no HTTP needed for baseline) ────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.sql_analyst_env_environment import SqlAnalystEnvironment
from models import SqlAnalystAction

# ── LLM client setup ─────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("HF_TOKEN", ""))

if not API_KEY:
    print("ERROR: Set OPENAI_API_KEY or HF_TOKEN environment variable.")
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

MAX_STEPS_PER_TASK = 15
FORCE_SUBMIT_STEP = 12


def extract_sql(text: str) -> str | None:
    """Extract SQL from ```sql ... ``` blocks."""
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: look for SELECT statements
    match = re.search(r"(SELECT\s+.+?;)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_submit(text: str) -> str | None:
    """Extract answer after SUBMIT: marker."""
    match = re.search(r"SUBMIT:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def truncate_result(result_json: str, max_rows: int = 50) -> str:
    """Truncate query results to avoid blowing up context."""
    try:
        data = json.loads(result_json)
        if isinstance(data, list) and len(data) > max_rows:
            data = data[:max_rows]
            return json.dumps(data, indent=2) + f"\n... (truncated to {max_rows} rows)"
        return result_json
    except (json.JSONDecodeError, TypeError):
        return result_json[:3000] if result_json else ""


def run_task(env: SqlAnalystEnvironment, task_id: int) -> float:
    """Run one task and return the final reward."""
    print(f"\n{'='*60}")
    print(f"TASK {task_id}")
    print(f"{'='*60}")

    obs = env.reset(task_id=task_id)
    schema = obs.schema_info
    task_desc = obs.task_description

    system_prompt = (
        "You are a data analyst. You have access to an e-commerce SQLite database.\n"
        "Your job is to answer the given task by writing SQL queries and analyzing results.\n\n"
        "INSTRUCTIONS:\n"
        "- Write SQL queries inside ```sql ... ``` blocks to explore the database.\n"
        "- Only SELECT queries are allowed.\n"
        "- When you have enough data to answer, write SUBMIT: followed by your complete answer.\n"
        "- Be precise with numbers. Include units ($ for money, % for percentages).\n"
        "- For complex tasks, explore data systematically before answering.\n\n"
        f"DATABASE SCHEMA:\n{schema}\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_desc},
    ]

    for step in range(1, MAX_STEPS_PER_TASK + 1):
        print(f"\n  Step {step}/{MAX_STEPS_PER_TASK}...")

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
            )
            llm_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  LLM Error: {e}")
            messages.append({"role": "assistant", "content": f"Error calling LLM: {e}"})
            continue

        messages.append({"role": "assistant", "content": llm_text})

        # Check for SUBMIT
        answer = extract_submit(llm_text)
        if answer:
            print(f"  Submitting answer ({len(answer)} chars)...")
            obs = env.step(SqlAnalystAction(action_type="submit_answer", content=answer))
            print(f"  Reward: {obs.reward}")
            print(f"  Feedback: {obs.message}")
            return obs.reward

        # Check for SQL
        sql = extract_sql(llm_text)
        if sql:
            print(f"  Executing SQL: {sql[:80]}...")
            obs = env.step(SqlAnalystAction(action_type="execute_sql", content=sql))

            if obs.error_message:
                result_msg = f"SQL Error: {obs.error_message}"
            else:
                truncated = truncate_result(obs.query_result or "[]")
                result_msg = f"Query returned {obs.row_count} rows:\n{truncated}"

            messages.append({"role": "user", "content": result_msg})
        else:
            # No SQL and no SUBMIT — ask LLM to continue
            if step >= FORCE_SUBMIT_STEP:
                # Force submit the last response as answer
                print(f"  Force-submitting at step {step}...")
                obs = env.step(SqlAnalystAction(action_type="submit_answer", content=llm_text))
                print(f"  Reward: {obs.reward}")
                print(f"  Feedback: {obs.message}")
                return obs.reward

            messages.append({
                "role": "user",
                "content": "Please write a SQL query (in ```sql ... ``` block) or submit your final answer with SUBMIT: prefix.",
            })

    # If we reach here, force submit
    print("  Max steps reached. Force-submitting last response...")
    obs = env.step(SqlAnalystAction(action_type="submit_answer", content=llm_text))
    print(f"  Reward: {obs.reward}")
    return obs.reward


def main():
    """Run all 3 tasks and report scores."""
    start_time = time.time()

    env = SqlAnalystEnvironment()
    scores = {}

    for task_id in [1, 2, 3]:
        reward = run_task(env, task_id)
        scores[f"task_{task_id}"] = reward

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("FINAL SCORES")
    print(f"{'='*60}")
    for task, score in scores.items():
        print(f"  {task}: {score:.4f}")
    total = sum(scores.values())
    avg = total / len(scores)
    print(f"  Total:  {total:.4f}")
    print(f"  Average: {avg:.4f}")
    print(f"  Time:   {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

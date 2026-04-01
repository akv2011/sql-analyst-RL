"""
Baseline inference — runs an LLM agent against all 5 tasks.

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
MIN_QUERIES_BEFORE_SUBMIT = 2
FORCE_SUBMIT_STEP = 12


def extract_sql(text: str) -> str | None:
    """Extract SQL from ```sql ... ``` blocks or raw SELECT statements."""
    # First try fenced code blocks
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Also try generic code blocks that contain SELECT
    match = re.search(r"```\s*(SELECT\s+.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: bare SELECT statement ending with semicolon
    match = re.search(r"(SELECT\s+.+?;)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_final_answer(text: str) -> str | None:
    """Extract the final answer from the model's response.

    Looks for FINAL ANSWER: or SUBMIT: markers.
    """
    for marker in [r"FINAL\s*ANSWER\s*:", r"SUBMIT\s*:"]:
        match = re.search(marker + r"\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def truncate_result(result_json: str, max_rows: int = 50) -> str:
    """Truncate query results to keep context manageable."""
    try:
        data = json.loads(result_json)
        if isinstance(data, list) and len(data) > max_rows:
            data = data[:max_rows]
            return json.dumps(data, indent=2) + f"\n... (truncated to {max_rows} rows)"
        return result_json
    except (json.JSONDecodeError, TypeError):
        return result_json[:3000] if result_json else ""


def build_system_prompt(schema: str) -> str:
    """Build a clear system prompt that forces the model to query first."""
    return (
        "You are an expert SQL data analyst. You have access to an e-commerce SQLite database.\n\n"
        "YOUR WORKFLOW (you MUST follow this order):\n"
        "1. FIRST, write SQL queries to explore the data. Write each query in a ```sql ... ``` block.\n"
        "2. Analyze the results from your queries.\n"
        "3. Run MORE queries if you need additional data.\n"
        "4. ONLY AFTER you have gathered enough data, write your final answer.\n\n"
        "RULES:\n"
        "- You MUST run at least 2 SQL queries before answering.\n"
        "- Write exactly ONE SQL query per response inside a ```sql\\n...\\n``` block.\n"
        "- Only SELECT queries are allowed (read-only database).\n"
        "- When ready to answer, start your response with FINAL ANSWER: followed by your complete analysis.\n"
        "- Be precise with numbers. Use $ for money and % for percentages.\n"
        "- Include the exact values from your query results in your answer.\n"
        "- For complex tasks, number your recommendations (1. 2. 3.) with at least one sentence each.\n\n"
        f"DATABASE SCHEMA:\n{schema}\n"
    )


def run_task(env: SqlAnalystEnvironment, task_id: int) -> float:
    """Run one task and return the final reward."""
    print(f"\n{'='*60}")
    print(f"TASK {task_id}")
    print(f"{'='*60}")

    obs = env.reset(task_id=task_id)
    schema = obs.schema_info
    task_desc = obs.task_description

    system_prompt = build_system_prompt(schema)
    queries_run = 0
    llm_text = ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"TASK:\n{task_desc}\n\nStart by writing a SQL query to explore the relevant data."},
    ]

    for step in range(1, MAX_STEPS_PER_TASK + 1):
        print(f"\n  Step {step}/{MAX_STEPS_PER_TASK} (queries so far: {queries_run})...")

        try:
            # Build API params — reasoning models (o1/o3/gpt-5.x) don't support temperature
            api_params = {
                "model": MODEL_NAME,
                "messages": messages,
                "max_completion_tokens": 2048,
            }
            model_lower = MODEL_NAME.lower()
            is_reasoning = any(x in model_lower for x in ["o1", "o3", "gpt-5"])
            if not is_reasoning:
                api_params["temperature"] = 0.1

            response = client.chat.completions.create(**api_params)
            llm_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  LLM Error: {e}")
            messages.append({"role": "assistant", "content": f"Error: {e}"})
            continue

        messages.append({"role": "assistant", "content": llm_text})

        # Always check for SQL first — prioritize querying
        sql = extract_sql(llm_text)
        if sql:
            print(f"  Executing SQL: {sql[:100]}...")
            obs = env.step(SqlAnalystAction(action_type="execute_sql", content=sql))
            queries_run += 1

            if obs.error_message:
                result_msg = f"SQL Error: {obs.error_message}\nPlease fix the query and try again."
            else:
                truncated = truncate_result(obs.query_result or "[]")
                result_msg = (
                    f"Query returned {obs.row_count} rows (reward: {obs.reward:+.3f}):\n{truncated}\n\n"
                    f"You've run {queries_run} queries so far. "
                )
                if queries_run < MIN_QUERIES_BEFORE_SUBMIT:
                    result_msg += "Run more queries to gather the data you need."
                else:
                    result_msg += "You can run more queries or write FINAL ANSWER: followed by your complete analysis."

            messages.append({"role": "user", "content": result_msg})
            continue

        # Check for final answer — only accept if enough queries were run
        answer = extract_final_answer(llm_text)
        if answer and queries_run >= MIN_QUERIES_BEFORE_SUBMIT:
            print(f"  Submitting answer ({len(answer)} chars) after {queries_run} queries...")
            obs = env.step(SqlAnalystAction(action_type="submit_answer", content=answer))
            print(f"  Reward: {obs.reward}")
            print(f"  Feedback: {obs.message}")
            return obs.reward
        elif answer and queries_run < MIN_QUERIES_BEFORE_SUBMIT:
            # Model tried to answer too early — redirect to querying
            print(f"  Model tried to answer after only {queries_run} queries — redirecting...")
            messages.append({
                "role": "user",
                "content": (
                    f"You've only run {queries_run} SQL queries. You need to run at least "
                    f"{MIN_QUERIES_BEFORE_SUBMIT} queries to gather actual data before answering. "
                    "Please write a SQL query to explore the database."
                ),
            })
            continue

        # No SQL and no answer — nudge the model
        if step >= FORCE_SUBMIT_STEP:
            print(f"  Force-submitting at step {step}...")
            obs = env.step(SqlAnalystAction(action_type="submit_answer", content=llm_text))
            print(f"  Reward: {obs.reward}")
            print(f"  Feedback: {obs.message}")
            return obs.reward

        messages.append({
            "role": "user",
            "content": "Please write a SQL query inside a ```sql ... ``` block to explore the data.",
        })

    # Max steps reached
    print("  Max steps reached. Force-submitting last response...")
    obs = env.step(SqlAnalystAction(action_type="submit_answer", content=llm_text))
    print(f"  Reward: {obs.reward}")
    return obs.reward


def main():
    """Run all 5 tasks and report scores."""
    start_time = time.time()

    env = SqlAnalystEnvironment()
    scores = {}

    for task_id in [1, 2, 3, 4, 5]:
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

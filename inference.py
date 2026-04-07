"""
Baseline inference — runs an LLM agent against all tasks.

Outputs the required [START]/[STEP]/[END] structured blocks that the
Phase 2 evaluator parses from stdout.

Setup:
    conda activate meta
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o"
    export HF_TOKEN="your-key"
    python inference.py
"""

import os
import re
import json
import time
import sys
import traceback

from openai import OpenAI

# ── Environment setup ─────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.sql_analyst_env_environment import SqlAnalystEnvironment
from models import SqlAnalystAction

# ── Configuration ─────────────────────────────────────────────────────────

MAX_STEPS_PER_TASK = 15
MIN_QUERIES_BEFORE_SUBMIT = 2
FORCE_SUBMIT_STEP = 12

TASK_NAMES = {1: "easy", 2: "medium", 3: "hard"}


def log(msg: str):
    """Print to stdout with flush so the evaluator sees output immediately."""
    print(msg, flush=True)


def get_config():
    """Read API config from environment variables."""
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o")
    api_key = (
        os.environ.get("HF_TOKEN", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    return api_base_url, model_name, api_key


def extract_sql(text: str) -> str | None:
    """Extract SQL from ```sql ... ``` blocks or raw SELECT statements."""
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(SELECT\s+.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"(SELECT\s+.+?;)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_final_answer(text: str) -> str | None:
    """Extract answer after FINAL ANSWER: or SUBMIT: markers."""
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
    """Build a system prompt that forces the model to query first."""
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


def call_llm(client: OpenAI, model_name: str, messages: list) -> str:
    """Call the LLM with model-appropriate parameters."""
    api_params = {
        "model": model_name,
        "messages": messages,
        "max_completion_tokens": 2048,
    }
    model_lower = model_name.lower()
    is_reasoning = any(x in model_lower for x in ["o1", "o3", "gpt-5"])
    if not is_reasoning:
        api_params["temperature"] = 0.1

    response = client.chat.completions.create(**api_params)
    return response.choices[0].message.content or ""


def run_task(env: SqlAnalystEnvironment, client: OpenAI, model_name: str, task_id: int) -> float:
    """Run one task. Emits [START], [STEP], and [END] blocks for the evaluator."""
    task_name = TASK_NAMES.get(task_id, f"task{task_id}")

    # ── [START] block ─────────────────────────────────────────────────
    log(f"[START] task={task_name}")

    obs = env.reset(task_id=task_id)
    schema = obs.schema_info
    task_desc = obs.task_description

    system_prompt = build_system_prompt(schema)
    queries_run = 0
    llm_text = ""
    step_num = 0
    last_reward = 0.0

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"TASK:\n{task_desc}\n\nStart by writing a SQL query to explore the relevant data."},
    ]

    for step in range(1, MAX_STEPS_PER_TASK + 1):
        step_num = step

        try:
            llm_text = call_llm(client, model_name, messages)
        except Exception as e:
            log(f"[STEP] step={step} reward=0.0 action=llm_error info=\"{e}\"")
            messages.append({"role": "assistant", "content": f"Error: {e}"})
            continue

        messages.append({"role": "assistant", "content": llm_text})

        # Check for SQL first
        sql = extract_sql(llm_text)
        if sql:
            obs = env.step(SqlAnalystAction(action_type="execute_sql", content=sql))
            queries_run += 1
            last_reward = obs.reward or 0.0

            # ── [STEP] block for SQL execution ────────────────────────
            log(f"[STEP] step={step} reward={last_reward} action=execute_sql")

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

        # Check for final answer
        answer = extract_final_answer(llm_text)
        if answer and queries_run >= MIN_QUERIES_BEFORE_SUBMIT:
            obs = env.step(SqlAnalystAction(action_type="submit_answer", content=answer))
            last_reward = obs.reward or 0.0
            log(f"[STEP] step={step} reward={last_reward} action=submit_answer")
            log(f"[END] task={task_name} score={last_reward} steps={step}")
            return last_reward
        elif answer and queries_run < MIN_QUERIES_BEFORE_SUBMIT:
            log(f"[STEP] step={step} reward=0.0 action=redirect_to_query")
            messages.append({
                "role": "user",
                "content": (
                    f"You've only run {queries_run} SQL queries. You need to run at least "
                    f"{MIN_QUERIES_BEFORE_SUBMIT} queries to gather actual data before answering. "
                    "Please write a SQL query to explore the database."
                ),
            })
            continue

        # No SQL and no answer — nudge or force-submit
        if step >= FORCE_SUBMIT_STEP:
            obs = env.step(SqlAnalystAction(action_type="submit_answer", content=llm_text))
            last_reward = obs.reward or 0.0
            log(f"[STEP] step={step} reward={last_reward} action=force_submit")
            log(f"[END] task={task_name} score={last_reward} steps={step}")
            return last_reward

        log(f"[STEP] step={step} reward=0.0 action=nudge")
        messages.append({
            "role": "user",
            "content": "Please write a SQL query inside a ```sql ... ``` block to explore the data.",
        })

    # Max steps reached
    obs = env.step(SqlAnalystAction(action_type="submit_answer", content=llm_text))
    last_reward = obs.reward or 0.0
    log(f"[STEP] step={step_num} reward={last_reward} action=max_steps_submit")
    log(f"[END] task={task_name} score={last_reward} steps={step_num}")
    return last_reward


def main():
    """Run all tasks and report scores."""
    api_base_url, model_name, api_key = get_config()

    if not api_key:
        log("ERROR: Set HF_TOKEN or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    log(f"API_BASE_URL: {api_base_url}")
    log(f"MODEL_NAME:   {model_name}")

    client = OpenAI(base_url=api_base_url, api_key=api_key)
    env = SqlAnalystEnvironment()
    start_time = time.time()

    task_ids = [1, 2, 3]
    scores = {}

    for task_id in task_ids:
        try:
            reward = run_task(env, client, model_name, task_id)
        except Exception as e:
            task_name = TASK_NAMES.get(task_id, f"task{task_id}")
            log(f"[STEP] step=0 reward=0.0 action=error info=\"{e}\"")
            log(f"[END] task={task_name} score=0.0 steps=0")
            traceback.print_exc()
            reward = 0.0
        scores[f"task_{task_id}"] = reward

    elapsed = time.time() - start_time
    avg = sum(scores.values()) / len(scores)

    log(f"\nFINAL SCORES:")
    for task, score in scores.items():
        log(f"  {task}: {score:.4f}")
    log(f"  Average: {avg:.4f}")
    log(f"  Time:   {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)

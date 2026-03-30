
# SQL Analyst Environment

An OpenEnv environment where an AI agent interacts with a simulated e-commerce SQLite database to answer business intelligence questions of increasing difficulty.

## Motivation

Data analysis with SQL is one of the most common real-world tasks performed by knowledge workers. This environment provides a controlled, reproducible setting to train and evaluate AI agents on:
- Writing correct SQL queries against relational databases
- Multi-step data exploration and reasoning
- Synthesizing quantitative findings into actionable business insights

## Environment Setup

```bash
conda activate meta
pip install -e .
# or
pip install openenv-core[core]>=0.2.2 openai>=1.0.0
```

## Running the Server

```bash
conda activate meta
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Or with Docker:
```bash
docker build -t sql-analyst-env .
docker run -p 8000:8000 sql-analyst-env
```

## Action Space

| Action Type | Description |
|------------|-------------|
| `execute_sql` | Run a SELECT query against the database. Returns JSON results (max 100 rows). |
| `submit_answer` | Submit final analysis. Triggers grading and ends the episode. |

```python
SqlAnalystAction(action_type="execute_sql", content="SELECT COUNT(*) FROM orders")
SqlAnalystAction(action_type="submit_answer", content="The total revenue was $266,532.48...")
```

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_description` | str | Current task prompt |
| `schema_info` | str | Database schema (on reset) |
| `query_result` | str\|None | JSON query results |
| `error_message` | str\|None | SQL error if query failed |
| `row_count` | int\|None | Rows returned |
| `columns` | list\|None | Column names |
| `task_id` | int | Current task (1, 2, or 3) |
| `step_number` | int | Current step |
| `max_steps` | int | Max steps (20) |
| `reward` | float | Reward (0.0-1.0, set on submit) |
| `done` | bool | Episode ended |
| `message` | str | Feedback from environment |

## Database Schema

7-table e-commerce database with ~25K rows (deterministic, seed=42):

- **customers** (500 rows): customer_id, name, email, signup_date, country, segment
- **products** (100 rows): product_id, name, category, subcategory, unit_price, cost_price
- **orders** (5000 rows): order_id, customer_id, order_date, status, total_amount, discount_amount, channel
- **order_items** (15000 rows): item_id, order_id, product_id, quantity, unit_price, subtotal
- **returns** (~750 rows): return_id, order_id, product_id, return_date, reason
- **marketing_campaigns** (20 rows): campaign_id, name, channel, start_date, end_date, budget, target_segment
- **campaign_attributions** (~3000 rows): attribution_id, campaign_id, order_id, attribution_type

## Tasks

### Task 1 — Easy: Basic Data Retrieval
Answer two questions requiring single-table SELECT queries:
1. Total revenue for completed orders in December 2024
2. Top 5 customers by completed order count

**Expected difficulty**: Straightforward for any SQL-capable agent.

### Task 2 — Medium: Multi-Table Analysis
Answer two questions requiring JOINs, GROUP BY, and subqueries:
1. Product category with the highest return rate
2. Customers active in Q1 but churned in Q3 — count and average spend

**Expected difficulty**: Requires multi-table reasoning and careful aggregation.

### Task 3 — Hard: Business Intelligence
Comprehensive analysis of Q3 2024 revenue decline:
1. Quantify the Q2-to-Q3 decline (absolute and percentage)
2. Identify top 2 root causes with data evidence
3. Provide 3 data-backed recommendations

**Expected difficulty**: Requires multi-step exploration, pattern recognition, and synthesis.

## Reward Function

```
final_reward = raw_score × efficiency_multiplier × step_decay
```

- **Raw score** (0.0–1.0): Partial credit from deterministic grader
- **Efficiency multiplier**: `max(0.8, 1.0 - 0.02 × max(0, queries - 5))` — penalizes excessive queries
- **Step decay**: `max(0.85, 1.0 - 0.01 × step_count)` — rewards faster solutions

## Running the Baseline Inference

```bash
conda activate meta
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export OPENAI_API_KEY="your-key-here"
python inference.py
```

### Baseline Scores (GPT-4o)

| Task | Score |
|------|-------|
| Task 1 (Easy) | ~0.80 |
| Task 2 (Medium) | ~0.55 |
| Task 3 (Hard) | ~0.45 |
| **Average** | **~0.60** |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (e.g., `https://api.openai.com/v1`) |
| `MODEL_NAME` | Model identifier (e.g., `gpt-4o`) |
| `OPENAI_API_KEY` | API key for LLM calls |
| `HF_TOKEN` | HuggingFace token (for deployment) |

## HuggingFace Space

Deployed at: `https://huggingface.co/spaces/akv2011/sql-analyst-env`

## License

BSD-style license. See LICENSE file.

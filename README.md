---
title: SQL Analyst Environment
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
tags:
  - openenv
---

# SQL Analyst Environment

An OpenEnv environment where an AI agent interacts with a simulated e-commerce SQLite database to answer business intelligence questions of increasing difficulty across 5 tasks.

## Motivation

Data analysis with SQL is one of the most common real-world tasks performed by knowledge workers. This environment provides a controlled, reproducible setting to train and evaluate AI agents on:
- Writing correct SQL queries against relational databases
- Multi-step data exploration and reasoning
- Synthesizing quantitative findings into actionable business insights
- Auditing data quality and building executive summaries

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

| Action Type | Description | Step Cost |
|------------|-------------|-----------|
| `execute_sql` | Run a SELECT query against the database. Returns JSON + ASCII table (max 100 rows). | Yes |
| `submit_answer` | Submit final analysis. Triggers grading and ends the episode. | Yes |
| `request_schema` | View the full database schema again. | Free |
| `request_hint` | Get a progressive hint for the current task (small reward penalty). | Free |
| `explain_sql` | Run EXPLAIN QUERY PLAN on a query to see execution strategy. | Free |

```python
SqlAnalystAction(action_type="execute_sql", content="SELECT COUNT(*) FROM orders")
SqlAnalystAction(action_type="submit_answer", content="The total revenue was $288,461.05...")
SqlAnalystAction(action_type="request_schema", content="")
SqlAnalystAction(action_type="request_hint", content="")
SqlAnalystAction(action_type="explain_sql", content="SELECT * FROM orders JOIN customers ON ...")
```

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_description` | str | Current task prompt |
| `schema_info` | str | Database schema (on reset and request_schema) |
| `query_result` | str\|None | JSON query results |
| `error_message` | str\|None | SQL error if query failed |
| `row_count` | int\|None | Rows returned |
| `columns` | list\|None | Column names |
| `task_id` | int | Current task (1-5) |
| `step_number` | int | Current step |
| `max_steps` | int | Max steps (20) |
| `reward` | float | Reward for this action |
| `done` | bool | Episode ended |
| `message` | str | Feedback with ASCII-formatted tables |
| `reward_breakdown` | dict\|None | Decomposed reward components (on submit and SQL) |
| `query_history` | list\|None | Last 5 queries executed this episode |

## Database Schema

7-table e-commerce database with ~25K rows (deterministic, seed=42):

- **customers** (500 rows): customer_id, name, email, signup_date, country, segment
- **products** (100 rows): product_id, name, category, subcategory, unit_price, cost_price
- **orders** (~5000 rows): order_id, customer_id, order_date, status, total_amount, discount_amount, channel
- **order_items** (~15000 rows): item_id, order_id, product_id, quantity, unit_price, subtotal
- **returns** (~750 rows): return_id, order_id, product_id, return_date, reason
- **marketing_campaigns** (20 rows): campaign_id, name, channel, start_date, end_date, budget, target_segment
- **campaign_attributions** (~3000 rows): attribution_id, campaign_id, order_id, attribution_type

### Planted Patterns

The data contains deliberate patterns for agents to discover:
- **Q3 revenue dip** driven by reduced order volume
- **Electronics return spike** in Q3 with elevated "defective" reason
- **Marketing campaign gap** with zero campaigns starting in Q3
- **Data quality discrepancies** in ~3% of orders (total vs item sum mismatch)
- **Negative margin products** in 3 categories (unit_price < cost_price)

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

### Task 4 — Medium: Data Quality Audit
Audit the database for data integrity issues:
1. Find orders where recorded total diverges from item subtotals by >1%
2. Identify product categories with negative-margin products

**Expected difficulty**: Requires careful comparison of related tables and anomaly detection.

### Task 5 — Hard: Executive Dashboard
Build a comprehensive 2024 business summary:
1. Monthly revenue trend with best/worst months
2. Customer cohort retention analysis (signup period → Q4 activity)
3. Channel performance ranking by revenue-per-order and return rate

**Expected difficulty**: Requires multi-faceted analysis, cohort logic, and cross-cutting metrics.

## Reward Function

```
total_reward = (raw_score x efficiency x step_decay) + exploration_bonus
```

- **Raw score** (0.0-1.0): Partial credit from deterministic grader
- **Efficiency multiplier**: `max(0.65, 1.0 - 0.03 x max(0, queries - 5))` — penalizes excessive queries
- **Step decay**: Two-tier — gentle for first 10 steps, steeper after
- **Exploration bonus**: Accumulated intermediate rewards (capped at 0.15) for discovering relevant tables and running analytical queries

### Intermediate Rewards
- +0.02 per newly discovered relevant table (via FROM/JOIN parsing)
- +0.03 bonus for covering all relevant tables
- +0.01 for non-empty result sets
- +0.01 for analytical queries (GROUP BY, COUNT, etc.)
- -0.02 for SQL syntax errors
- -0.05 for destructive SQL attempts (DROP, DELETE, etc.)

### Reward Breakdown
Every observation includes a `reward_breakdown` dict showing exactly how the reward was computed (raw score, efficiency, step decay, exploration bonus, and grader feedback).

## Running the Baseline Inference

```bash
conda activate meta
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export OPENAI_API_KEY="your-key-here"
python inference.py
```

### Baseline Scores

| Model | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Average |
|-------|--------|--------|--------|--------|--------|---------|
| GPT-5.4 | 0.96 | 0.96 | 0.88 | 0.96 | 0.94 | **0.94** |
| GPT-5.3 | 0.97 | 0.96 | 0.86 | 0.92 | 0.78 | **0.90** |
| GPT-4o-mini | 0.58 | 0.43 | 0.57 | 0.45 | 0.35 | **0.48** |

*Scores are reproducible. The script auto-detects reasoning models and adjusts API parameters accordingly.*

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (e.g., `https://api.openai.com/v1`) |
| `MODEL_NAME` | Model identifier (e.g., `gpt-4o`) |
| `OPENAI_API_KEY` | API key for LLM calls |
| `HF_TOKEN` | HuggingFace token (for deployment) |

See `.env.example` for a template.

## HuggingFace Space

Deployed at: `https://huggingface.co/spaces/akv2011/sql-analyst-env`

## License

BSD 3-Clause license. See LICENSE file.

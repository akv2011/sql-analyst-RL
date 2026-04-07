"""Tests for the core environment — reset, step, state, reward flow."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.sql_analyst_env_environment import SqlAnalystEnvironment
from models import SqlAnalystAction


def make_env():
    return SqlAnalystEnvironment()


# ── Reset tests ───────────────────────────────────────────────────────────

def test_reset_returns_observation():
    env = make_env()
    obs = env.reset(task_id=1)
    assert obs.done is False
    assert obs.reward == 0.0
    assert len(obs.task_description) > 50
    assert len(obs.schema_info) > 500
    assert obs.task_id == 1
    assert obs.step_number == 0


def test_reset_all_five_tasks():
    env = make_env()
    for tid in [1, 2, 3, 4, 5]:
        obs = env.reset(task_id=tid)
        assert obs.task_id == tid
        assert f"TASK {tid}" in obs.task_description


def test_reset_invalid_task_defaults_to_1():
    env = make_env()
    obs = env.reset(task_id=99)
    assert obs.task_id == 1


def test_reset_clears_state():
    """After stepping, reset should give a clean slate."""
    env = make_env()
    env.reset(task_id=1)
    env.step(SqlAnalystAction(action_type="execute_sql", content="SELECT 1"))
    obs = env.reset(task_id=2)
    assert obs.step_number == 0
    assert obs.task_id == 2
    assert obs.done is False


# ── SQL execution tests ──────────────────────────────────────────────────

def test_valid_sql_returns_results():
    env = make_env()
    env.reset(task_id=1)
    obs = env.step(SqlAnalystAction(
        action_type="execute_sql",
        content="SELECT COUNT(*) as cnt FROM orders"
    ))
    assert obs.done is False
    assert obs.error_message is None
    assert obs.query_result is not None
    assert obs.row_count == 1
    assert "cnt" in obs.columns


def test_invalid_sql_returns_error():
    env = make_env()
    env.reset(task_id=1)
    obs = env.step(SqlAnalystAction(
        action_type="execute_sql",
        content="SELECT * FROM nonexistent_table"
    ))
    assert obs.error_message is not None
    assert obs.done is False
    assert obs.reward < 0  # Negative reward for error


def test_destructive_sql_blocked():
    env = make_env()
    env.reset(task_id=1)
    for sql in ["DROP TABLE orders", "DELETE FROM customers", "UPDATE orders SET status='x'",
                 "INSERT INTO orders VALUES (1,1,1,1,1,1,1)", "ALTER TABLE orders ADD col"]:
        obs = env.step(SqlAnalystAction(action_type="execute_sql", content=sql))
        assert obs.error_message is not None
        assert "SELECT" in obs.error_message or "read-only" in obs.error_message.lower() or "Only SELECT" in obs.error_message
        assert obs.reward < 0  # Penalty


def test_empty_sql():
    env = make_env()
    env.reset(task_id=1)
    obs = env.step(SqlAnalystAction(action_type="execute_sql", content=""))
    assert obs.error_message is not None


def test_sql_results_capped_at_100():
    """Large queries should return at most 100 rows."""
    env = make_env()
    env.reset(task_id=1)
    obs = env.step(SqlAnalystAction(
        action_type="execute_sql",
        content="SELECT * FROM orders"
    ))
    assert obs.row_count <= 100


# ── Intermediate rewards ─────────────────────────────────────────────────

def test_relevant_table_query_gives_positive_reward():
    """Querying a task-relevant table should give positive intermediate reward."""
    env = make_env()
    env.reset(task_id=1)  # Relevant tables: orders, customers
    obs = env.step(SqlAnalystAction(
        action_type="execute_sql",
        content="SELECT COUNT(*) FROM orders"
    ))
    assert obs.reward > 0, "Querying a relevant table should give positive reward"


def test_analytical_query_bonus():
    """Queries with GROUP BY, COUNT etc. should get a bonus."""
    env = make_env()
    env.reset(task_id=1)
    obs = env.step(SqlAnalystAction(
        action_type="execute_sql",
        content="SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY COUNT(*) DESC"
    ))
    assert obs.reward > 0.02  # Basic + analytical bonus


# ── Submit tests ──────────────────────────────────────────────────────────

def test_submit_ends_episode():
    env = make_env()
    env.reset(task_id=1)
    obs = env.step(SqlAnalystAction(
        action_type="submit_answer",
        content="The revenue was $266,532.48"
    ))
    assert obs.done is True
    assert obs.reward >= 0


def test_submit_empty_answer():
    env = make_env()
    env.reset(task_id=1)
    obs = env.step(SqlAnalystAction(action_type="submit_answer", content=""))
    assert obs.done is True
    assert obs.reward == 0.0


def test_submit_after_done_returns_done():
    """Stepping after episode ends should still return done."""
    env = make_env()
    env.reset(task_id=1)
    env.step(SqlAnalystAction(action_type="submit_answer", content="answer"))
    obs = env.step(SqlAnalystAction(action_type="execute_sql", content="SELECT 1"))
    assert obs.done is True


def test_perfect_task1_scores_high():
    """Submitting the exact correct answer should score close to 1.0."""
    from server.database import create_database
    _, gt = create_database()
    t1 = gt["task1"]
    rev = t1["dec_revenue"]
    cust_str = ", ".join(f"{n} ({c})" for n, c in t1["top_customers"])

    env = make_env()
    env.reset(task_id=1)
    # Run one query first to simulate real usage
    env.step(SqlAnalystAction(action_type="execute_sql", content="SELECT 1"))
    obs = env.step(SqlAnalystAction(
        action_type="submit_answer",
        content=f"1. Total revenue for completed orders in December 2024: ${rev:.2f}\n2. Top 5 customers: {cust_str}."
    ))
    assert obs.reward >= 0.9, f"Perfect answer scored {obs.reward}"


# ── Max steps test ────────────────────────────────────────────────────────

def test_max_steps_terminates_episode():
    env = make_env()
    env.reset(task_id=1)
    for i in range(25):
        obs = env.step(SqlAnalystAction(action_type="execute_sql", content="SELECT 1"))
        if obs.done:
            break
    assert obs.done is True


# ── State test ────────────────────────────────────────────────────────────

def test_state_updates():
    env = make_env()
    env.reset(task_id=2)
    assert env.state.step_count == 0
    assert env.state.episode_id is not None
    env.step(SqlAnalystAction(action_type="execute_sql", content="SELECT 1"))
    assert env.state.step_count == 1


# ── Invalid action type ──────────────────────────────────────────────────

def test_unknown_action_type():
    env = make_env()
    env.reset(task_id=1)
    obs = env.step(SqlAnalystAction(action_type="fly_to_moon", content="wheee"))
    assert obs.error_message is not None
    assert obs.done is False


# ── Reward shaping tests ─────────────────────────────────────────────────

def test_efficiency_penalty():
    """Many queries should reduce the final reward vs few queries."""
    from server.database import create_database
    _, gt = create_database()
    t1 = gt["task1"]
    rev = t1["dec_revenue"]
    cust_str = ", ".join(f"{n} ({c})" for n, c in t1["top_customers"])
    answer = f"Revenue: ${rev:.2f}. Top 5: {cust_str}."

    env1 = make_env()
    env1.reset(task_id=1)
    # Few queries
    env1.step(SqlAnalystAction(action_type="execute_sql", content="SELECT 1"))
    obs1 = env1.step(SqlAnalystAction(action_type="submit_answer", content=answer))

    env2 = make_env()
    env2.reset(task_id=1)
    # Many queries
    for _ in range(10):
        env2.step(SqlAnalystAction(action_type="execute_sql", content="SELECT 1"))
    obs2 = env2.step(SqlAnalystAction(action_type="submit_answer", content=answer))

    assert obs1.reward > obs2.reward, (
        f"Fewer queries ({obs1.reward}) should score higher than many ({obs2.reward})"
    )


# ── Task 4 & 5 tests ────────────────────────────────────────────────

def test_task4_reset_and_submit():
    """Task 4 should be resettable and gradable."""
    env = make_env()
    obs = env.reset(task_id=4)
    assert obs.task_id == 4
    assert "Data Quality" in obs.task_description
    obs = env.step(SqlAnalystAction(action_type="submit_answer", content="Found 157 discrepancies."))
    assert obs.done is True
    assert 0.0 <= obs.reward <= 1.0


def test_task5_reset_and_submit():
    """Task 5 should be resettable and gradable."""
    env = make_env()
    obs = env.reset(task_id=5)
    assert obs.task_id == 5
    assert "Dashboard" in obs.task_description or "Executive" in obs.task_description
    obs = env.step(SqlAnalystAction(action_type="submit_answer", content="Monthly revenue analysis..."))
    assert obs.done is True
    assert 0.0 <= obs.reward <= 1.0


# ── New action type tests ───────────────────────────────────────────

def test_request_schema_action():
    """Schema requests should return schema without costing a step."""
    env = make_env()
    env.reset(task_id=1)
    env.step(SqlAnalystAction(action_type="execute_sql", content="SELECT 1"))
    step_before = env.state.step_count
    obs = env.step(SqlAnalystAction(action_type="request_schema", content=""))
    assert len(obs.schema_info) > 500
    assert obs.done is False
    assert env.state.step_count == step_before


def test_request_hint():
    """Hints should return with increasing penalty."""
    env = make_env()
    env.reset(task_id=1)
    obs1 = env.step(SqlAnalystAction(action_type="request_hint", content=""))
    assert "Hint:" in obs1.message
    assert obs1.reward < 0  # Penalty
    obs2 = env.step(SqlAnalystAction(action_type="request_hint", content=""))
    assert "Hint:" in obs2.message
    assert obs2.reward < obs1.reward  # Increasing penalty
    # Third hint should say no more
    obs3 = env.step(SqlAnalystAction(action_type="request_hint", content=""))
    assert "No more hints" in obs3.message


def test_explain_sql():
    """EXPLAIN should return query plan without step cost."""
    env = make_env()
    env.reset(task_id=1)
    step_before = env.state.step_count
    obs = env.step(SqlAnalystAction(
        action_type="explain_sql",
        content="SELECT * FROM orders JOIN customers ON orders.customer_id = customers.customer_id"
    ))
    assert obs.query_result is not None
    assert "SCAN" in obs.query_result or "SEARCH" in obs.query_result
    assert env.state.step_count == step_before


# ── Reward and observation enhancement tests ────────────────────────

def test_reward_not_overwritten_on_submit():
    """Intermediate rewards should be incorporated into final reward."""
    env = make_env()
    env.reset(task_id=1)
    env.step(SqlAnalystAction(action_type="execute_sql", content="SELECT * FROM orders LIMIT 5"))
    env.step(SqlAnalystAction(action_type="execute_sql", content="SELECT * FROM customers LIMIT 5"))
    intermediate = env._total_reward
    assert intermediate > 0

    from server.database import create_database
    _, gt = create_database()
    t1 = gt["task1"]
    rev = t1["dec_revenue"]
    cust_str = ", ".join(f"{n} ({c})" for n, c in t1["top_customers"])
    obs = env.step(SqlAnalystAction(
        action_type="submit_answer",
        content=f"Revenue: ${rev:.2f}. Top 5: {cust_str}."
    ))
    assert obs.reward > 0.8


def test_table_detection_ignores_comments():
    """Table names in SQL comments should not count as discovered."""
    env = make_env()
    env.reset(task_id=1)
    env.step(SqlAnalystAction(
        action_type="execute_sql",
        content="-- SELECT * FROM orders\nSELECT 1"
    ))
    assert "orders" not in env._tables_queried


def test_reward_breakdown_on_submit():
    """Submit observations should include reward breakdown."""
    env = make_env()
    env.reset(task_id=1)
    obs = env.step(SqlAnalystAction(action_type="submit_answer", content="Revenue: $100"))
    assert obs.reward_breakdown is not None
    assert "raw_score" in obs.reward_breakdown
    assert "efficiency_multiplier" in obs.reward_breakdown


def test_query_history_tracked():
    """Query history should be tracked and returned in observations."""
    env = make_env()
    env.reset(task_id=1)
    obs = env.step(SqlAnalystAction(action_type="execute_sql", content="SELECT COUNT(*) FROM orders"))
    assert obs.query_history is not None
    assert len(obs.query_history) == 1
    assert obs.query_history[0]["row_count"] == 1


def test_get_metadata():
    """get_metadata should return valid EnvironmentMetadata."""
    env = make_env()
    meta = env.get_metadata()
    assert meta.name == "SQL Analyst Environment"
    assert meta.version == "0.2.0"
    assert "SQL" in meta.description


def test_close():
    """close() should clean up without errors."""
    env = make_env()
    env.reset(task_id=1)
    env.close()
    assert env._db_conn is None


if __name__ == "__main__":
    passed = failed = errored = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"  PASS: {name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL: {name} — {e}")
                failed += 1
            except Exception as e:
                print(f"  ERROR: {name} — {e}")
                errored += 1
    print(f"\n  Results: {passed} passed, {failed} failed, {errored} errors")

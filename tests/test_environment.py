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


def test_reset_all_three_tasks():
    env = make_env()
    for tid in [1, 2, 3]:
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
    env = make_env()
    env.reset(task_id=1)
    # Run one query first to simulate real usage
    env.step(SqlAnalystAction(action_type="execute_sql", content="SELECT 1"))
    obs = env.step(SqlAnalystAction(
        action_type="submit_answer",
        content=(
            "1. Total revenue for completed orders in December 2024: $266,532.48\n"
            "2. Top 5 customers: Alexander White (14), Edward White (14), "
            "Frank Wright (14), Andrew Roberts (13), Donna Torres (13)."
        )
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
    env1 = make_env()
    env1.reset(task_id=1)
    # Few queries
    env1.step(SqlAnalystAction(action_type="execute_sql", content="SELECT 1"))
    obs1 = env1.step(SqlAnalystAction(
        action_type="submit_answer",
        content="Revenue: $266,532.48. Top 5: Alexander White (14), Edward White (14), Frank Wright (14), Andrew Roberts (13), Donna Torres (13)."
    ))

    env2 = make_env()
    env2.reset(task_id=1)
    # Many queries
    for _ in range(10):
        env2.step(SqlAnalystAction(action_type="execute_sql", content="SELECT 1"))
    obs2 = env2.step(SqlAnalystAction(
        action_type="submit_answer",
        content="Revenue: $266,532.48. Top 5: Alexander White (14), Edward White (14), Frank Wright (14), Andrew Roberts (13), Donna Torres (13)."
    ))

    assert obs1.reward > obs2.reward, (
        f"Fewer queries ({obs1.reward}) should score higher than many ({obs2.reward})"
    )


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

"""Tests for database generation and ground truth computation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.database import create_database, get_schema_info, TASK_DESCRIPTIONS


def test_database_creates_all_tables():
    """All 7 tables should exist with expected row counts."""
    conn, _ = create_database()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    assert set(tables) == {
        "customers", "products", "orders", "order_items",
        "returns", "marketing_campaigns", "campaign_attributions",
    }
    conn.close()


def test_row_counts_reasonable():
    """Tables should have reasonable row counts."""
    conn, _ = create_database()
    cur = conn.cursor()
    expected_mins = {
        "customers": 450, "products": 90, "orders": 4000,
        "order_items": 8000, "returns": 500,
        "marketing_campaigns": 15, "campaign_attributions": 1000,
    }
    for table, min_rows in expected_mins.items():
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        assert count >= min_rows, f"{table} has only {count} rows, expected >= {min_rows}"
    conn.close()


def test_deterministic_generation():
    """Two calls should produce identical data (same seed)."""
    _, gt1 = create_database()
    _, gt2 = create_database()
    assert gt1 == gt2, "Database generation is not deterministic"


def test_ground_truth_task1():
    """Task 1 ground truth should contain expected keys and valid values."""
    _, gt = create_database()
    t1 = gt["task1"]
    assert "dec_revenue" in t1
    assert isinstance(t1["dec_revenue"], (int, float))
    assert t1["dec_revenue"] > 0
    assert "top_customers" in t1
    assert len(t1["top_customers"]) == 5
    for name, count in t1["top_customers"]:
        assert isinstance(name, str)
        assert isinstance(count, int)
        assert count > 0


def test_ground_truth_task2():
    """Task 2 ground truth should have valid return rate and churned customer data."""
    _, gt = create_database()
    t2 = gt["task2"]
    assert t2["highest_return_category"] in ["electronics", "clothing", "home", "sports", "books"]
    assert 0 < t2["highest_return_rate"] < 100
    assert t2["churned_count"] > 0
    assert t2["churned_avg_spend"] > 0


def test_ground_truth_task3():
    """Task 3 ground truth: Q3 revenue should be LESS than Q2 (planted dip)."""
    _, gt = create_database()
    t3 = gt["task3"]
    assert t3["q3_revenue"] < t3["q2_revenue"], "Q3 should have lower revenue than Q2"
    assert t3["revenue_decline_pct"] > 10, "Revenue decline should be significant"
    assert t3["electronics_return_q3"] > t3["electronics_return_other"], (
        "Electronics returns should spike in Q3"
    )
    # No Q3 campaigns
    assert "Q3" not in t3["campaigns_by_quarter"], "There should be no Q3 campaigns"


def test_no_q3_marketing_campaigns():
    """The planted marketing gap: zero campaigns should start in Q3."""
    conn, _ = create_database()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM marketing_campaigns
        WHERE start_date >= '2024-07-01' AND start_date < '2024-10-01'
    """)
    assert cur.fetchone()[0] == 0, "There should be no campaigns starting in Q3"
    conn.close()


def test_schema_info():
    """Schema info should contain all table names and be non-empty."""
    conn, _ = create_database()
    schema = get_schema_info(conn)
    assert len(schema) > 1000
    for table in ["customers", "orders", "products", "returns", "marketing_campaigns"]:
        assert table in schema
    conn.close()


def test_task_descriptions_exist():
    """All 3 task descriptions should exist and be substantial."""
    for tid in [1, 2, 3]:
        assert tid in TASK_DESCRIPTIONS
        assert len(TASK_DESCRIPTIONS[tid]) > 100


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"  PASS: {name}")
            except AssertionError as e:
                print(f"  FAIL: {name} — {e}")
            except Exception as e:
                print(f"  ERROR: {name} — {e}")

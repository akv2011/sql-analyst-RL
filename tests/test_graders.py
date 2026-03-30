"""Tests for the grading logic — ensures scores are in [0, 1] and partial credit works."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.database import create_database
from server.graders import grade, extract_numbers, numeric_closeness


# ── Helper function tests ─────────────────────────────────────────────────

def test_extract_numbers():
    """Should extract numbers from various formats."""
    assert 1234.0 in extract_numbers("revenue was $1,234")
    assert 42.0 in extract_numbers("there are 42 rows")
    assert 99.5 in extract_numbers("rate is 99.5%")
    assert len(extract_numbers("no numbers here")) == 0


def test_numeric_closeness_exact():
    assert numeric_closeness(100.0, 100.0, 1.0) == 1.0


def test_numeric_closeness_within_tolerance():
    assert numeric_closeness(101.0, 100.0, 2.0) == 1.0  # 1% off, 2% tolerance


def test_numeric_closeness_far():
    assert numeric_closeness(200.0, 100.0, 1.0) == 0.0


# ── Task 1 grading ────────────────────────────────────────────────────────

def test_task1_perfect_answer():
    """A perfect answer should score close to 1.0."""
    _, gt = create_database()
    t1 = gt["task1"]
    rev = t1["dec_revenue"]
    customers = t1["top_customers"]
    cust_str = ", ".join(f"{n} ({c})" for n, c in customers)
    answer = f"Revenue: ${rev:.2f}. Top 5: {cust_str}."
    score, _ = grade(1, answer, gt)
    assert score >= 0.9, f"Perfect answer scored only {score}"


def test_task1_empty_answer():
    _, gt = create_database()
    score, _ = grade(1, "", gt)
    assert score == 0.0


def test_task1_partial_revenue_only():
    """Getting only revenue right should give ~0.5."""
    _, gt = create_database()
    rev = gt["task1"]["dec_revenue"]
    score, _ = grade(1, f"Total revenue was ${rev:.2f}.", gt)
    assert 0.4 <= score <= 0.6, f"Revenue-only answer scored {score}"


def test_task1_wrong_answer():
    _, gt = create_database()
    score, _ = grade(1, "The revenue was $1.00 and the top customer is Nobody.", gt)
    assert score < 0.15


# ── Task 2 grading ────────────────────────────────────────────────────────

def test_task2_perfect_answer():
    _, gt = create_database()
    t2 = gt["task2"]
    answer = (
        f"The {t2['highest_return_category']} category has the highest return rate "
        f"at {t2['highest_return_rate']}%. "
        f"There are {t2['churned_count']} customers who were active in Q1 but "
        f"not in Q3, with average spend of ${t2['churned_avg_spend']:.2f}."
    )
    score, _ = grade(2, answer, gt)
    assert score >= 0.9, f"Perfect answer scored only {score}"


def test_task2_wrong_category():
    """Wrong category should still give partial credit for other parts."""
    _, gt = create_database()
    t2 = gt["task2"]
    answer = (
        f"The electronics category has the highest return rate. "
        f"There are {t2['churned_count']} churned customers with avg spend ${t2['churned_avg_spend']:.2f}."
    )
    score, _ = grade(2, answer, gt)
    # Should get some credit for churned customer data even if category is wrong
    assert 0.3 <= score <= 0.7, f"Partially correct scored {score}"


# ── Task 3 grading ────────────────────────────────────────────────────────

def test_task3_perfect_answer():
    _, gt = create_database()
    t3 = gt["task3"]
    answer = f"""
    Revenue Decline Analysis:
    Q2 2024 revenue: ${t3['q2_revenue']:.2f}, Q3 2024 revenue: ${t3['q3_revenue']:.2f}.
    This represents a decline of ${t3['revenue_decline_abs']:.2f} ({t3['revenue_decline_pct']:.1f}%).

    Root Cause 1: Electronics Return Spike
    Electronics products had a return rate of {t3['electronics_return_q3']}% in Q3
    vs {t3['electronics_return_other']}% in other quarters, primarily due to defective items.

    Root Cause 2: Marketing Campaign Gap
    There were zero marketing campaigns in Q3, compared to 5 in Q1 and Q2 each.
    The absence of marketing reduced customer acquisition and engagement.

    Recommendations:
    1. Implement quality control for electronics suppliers to reduce the defective rate from 10% to 6%.
    2. Schedule at least 3 marketing campaigns per quarter with minimum $30,000 budget.
    3. Create a re-engagement program for the 36 churned customers with $8,158 average lifetime value.
    """
    score, _ = grade(3, answer, gt)
    assert score >= 0.85, f"Comprehensive answer scored only {score}"


def test_task3_partial_only_revenue():
    """Mentioning decline without root causes should give partial credit."""
    _, gt = create_database()
    t3 = gt["task3"]
    answer = f"Revenue declined from ${t3['q2_revenue']:.2f} to ${t3['q3_revenue']:.2f}, a {t3['revenue_decline_pct']:.1f}% drop."
    score, _ = grade(3, answer, gt)
    assert 0.1 <= score <= 0.4, f"Revenue-only scored {score}"


def test_task3_empty():
    _, gt = create_database()
    score, _ = grade(3, "", gt)
    assert score == 0.0


# ── Score range invariants ────────────────────────────────────────────────

def test_all_scores_in_range():
    """All grader outputs must be in [0.0, 1.0]."""
    _, gt = create_database()
    test_answers = [
        "", "random gibberish xyz", "42", "$1,000,000",
        "The answer is electronics at 50%",
        "Revenue was $100,000. Decline was 30%.",
    ]
    for task_id in [1, 2, 3]:
        for answer in test_answers:
            score, feedback = grade(task_id, answer, gt)
            assert 0.0 <= score <= 1.0, f"Task {task_id}, answer '{answer[:30]}': score {score} out of range"
            assert isinstance(feedback, str)


def test_graders_are_deterministic():
    """Same input should always produce same output."""
    _, gt = create_database()
    answer = "Revenue: $266,532.48. Top customer is Alexander White with 14 orders."
    scores = [grade(1, answer, gt)[0] for _ in range(5)]
    assert len(set(scores)) == 1, f"Grading is non-deterministic: {scores}"


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

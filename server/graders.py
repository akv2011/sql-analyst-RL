"""
Grading logic for the five tasks.

Each grader takes the agent's free-text answer and scores it 0.0 to 1.0.
We give partial credit — if you got the right category but wrong percentage,
you still get points. Grading is deterministic: we compare against
precomputed ground truth using numeric tolerance and keyword matching.
"""

import re
from typing import Dict, Any, List, Tuple


def extract_numbers(text: str) -> List[float]:
    """Extract all numeric values from text, handling commas and dollar signs."""
    # Match numbers like 1234, 1,234, 1234.56, $1,234.56, 0.5, etc.
    # First alt requires commas (1,234), second handles plain numbers (1234.56)
    matches = re.findall(
        r'(?<![.\w])\$?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|0?\.\d+)',
        text,
    )
    results = []
    for m in matches:
        clean = m.replace(",", "")
        try:
            results.append(float(clean))
        except ValueError:
            continue
    return results


def numeric_closeness(actual: float, expected: float, tolerance_pct: float) -> float:
    """Score 0.0-1.0 based on how close actual is to expected."""
    if expected == 0:
        return 1.0 if actual == 0 else 0.0
    pct_diff = abs(actual - expected) / abs(expected) * 100
    if pct_diff <= tolerance_pct:
        return 1.0
    elif pct_diff <= tolerance_pct * 2:
        return 0.7
    elif pct_diff <= tolerance_pct * 5:
        return 0.4
    elif pct_diff <= tolerance_pct * 10:
        return 0.2
    return 0.0


def fuzzy_contains(text: str, keyword: str) -> bool:
    """Check if text contains keyword (case-insensitive)."""
    return keyword.lower() in text.lower()


def find_closest_number(numbers: List[float], target: float, tolerance_pct: float = 50) -> float:
    """Find the number in the list closest to the target, within tolerance."""
    if not numbers:
        return 0.0
    best_score = 0.0
    for n in numbers:
        score = numeric_closeness(n, target, tolerance_pct)
        best_score = max(best_score, score)
    return best_score


def grade_task1(answer: str, ground_truth: Dict[str, Any]) -> Tuple[float, str]:
    """Grade Task 1: Basic Data Retrieval.

    Part A (0.5): December 2024 completed order revenue
    Part B (0.5): Top 5 customers by completed order count
    """
    score = 0.0
    feedback_parts = []
    text = answer.lower()
    numbers = extract_numbers(answer)

    # Part A: December revenue
    expected_rev = ground_truth["dec_revenue"]
    rev_score = find_closest_number(numbers, expected_rev, tolerance_pct=1)

    if rev_score >= 1.0:
        score += 0.5
        feedback_parts.append(f"Part A: Correct revenue ${expected_rev:.2f} (+0.50)")
    elif rev_score >= 0.7:
        score += 0.4
        feedback_parts.append(f"Part A: Close to correct revenue (+0.40)")
    elif rev_score >= 0.4:
        score += 0.2
        feedback_parts.append(f"Part A: Roughly correct revenue (+0.20)")
    elif any(n > 10000 for n in numbers):
        score += 0.1
        feedback_parts.append("Part A: Found a revenue figure but not accurate (+0.10)")
    else:
        feedback_parts.append("Part A: Revenue not found or incorrect (+0.00)")

    # Part B: Top 5 customers
    expected_customers = ground_truth["top_customers"]
    expected_names = [c[0] for c in expected_customers]
    expected_counts = [c[1] for c in expected_customers]

    names_found = 0
    for name in expected_names:
        # Check for full name or last name match
        if fuzzy_contains(text, name.lower()):
            names_found += 1
        else:
            parts = name.split()
            if len(parts) == 2 and (fuzzy_contains(text, parts[0]) and fuzzy_contains(text, parts[1])):
                names_found += 1

    # Check if counts are mentioned
    counts_found = 0
    for count in expected_counts:
        if find_closest_number(numbers, float(count), tolerance_pct=5) >= 0.7:
            counts_found += 1

    if names_found >= 5 and counts_found >= 3:
        score += 0.5
        feedback_parts.append("Part B: All 5 customers identified correctly (+0.50)")
    elif names_found >= 3:
        partial = 0.15 + (names_found / 5) * 0.25
        score += round(partial, 2)
        feedback_parts.append(f"Part B: {names_found}/5 customers found (+{partial:.2f})")
    elif names_found >= 1:
        score += 0.1
        feedback_parts.append(f"Part B: {names_found}/5 customers found (+0.10)")
    else:
        feedback_parts.append("Part B: No customers correctly identified (+0.00)")

    return round(min(score, 1.0), 4), " | ".join(feedback_parts)


def grade_task2(answer: str, ground_truth: Dict[str, Any]) -> Tuple[float, str]:
    """Grade Task 2: Multi-Table Analysis.

    Part A (0.5): Highest return rate category + rate
    Part B (0.5): Churned customer count + average spend
    """
    score = 0.0
    feedback_parts = []
    text = answer.lower()
    numbers = extract_numbers(answer)

    # Part A: Return category
    expected_cat = ground_truth["highest_return_category"]
    expected_rate = ground_truth["highest_return_rate"]

    if fuzzy_contains(text, expected_cat):
        score += 0.3
        feedback_parts.append(f"Part A: Correct category '{expected_cat}' (+0.30)")

        rate_score = find_closest_number(numbers, expected_rate, tolerance_pct=5)
        if rate_score >= 0.7:
            score += 0.2
            feedback_parts.append(f"Part A: Return rate accurate (+0.20)")
        elif rate_score >= 0.4:
            score += 0.1
            feedback_parts.append(f"Part A: Return rate roughly correct (+0.10)")
        else:
            feedback_parts.append("Part A: Return rate inaccurate (+0.00)")
    else:
        # Check if they got a category at all
        for cat in ["electronics", "clothing", "home", "sports", "books"]:
            if fuzzy_contains(text, cat):
                score += 0.05
                feedback_parts.append(f"Part A: Identified a category but not the correct one (+0.05)")
                break
        else:
            feedback_parts.append("Part A: Category not identified (+0.00)")

    # Part B: Churned customers
    expected_count = ground_truth["churned_count"]
    expected_spend = ground_truth["churned_avg_spend"]

    count_score = find_closest_number(numbers, float(expected_count), tolerance_pct=5)
    if count_score >= 0.7:
        score += 0.25
        feedback_parts.append(f"Part B: Customer count correct ({expected_count}) (+0.25)")
    elif count_score >= 0.4:
        score += 0.15
        feedback_parts.append(f"Part B: Customer count close (+0.15)")
    else:
        feedback_parts.append(f"Part B: Customer count incorrect (+0.00)")

    spend_score = find_closest_number(numbers, expected_spend, tolerance_pct=5)
    if spend_score >= 0.7:
        score += 0.25
        feedback_parts.append(f"Part B: Average spend accurate (+0.25)")
    elif spend_score >= 0.4:
        score += 0.15
        feedback_parts.append(f"Part B: Average spend roughly correct (+0.15)")
    else:
        feedback_parts.append(f"Part B: Average spend incorrect (+0.00)")

    return round(min(score, 1.0), 4), " | ".join(feedback_parts)


def grade_task3(answer: str, ground_truth: Dict[str, Any]) -> Tuple[float, str]:
    """Grade Task 3: Business Intelligence Analysis.

    Revenue quantification (0.2)
    Root cause 1 - electronics returns (0.25)
    Root cause 2 - marketing gap (0.25)
    Recommendations (0.3)
    """
    score = 0.0
    feedback_parts = []
    text = answer.lower()
    numbers = extract_numbers(answer)

    # ── Revenue quantification (0.2) ──────────────────────────────────

    q2_rev = ground_truth["q2_revenue"]
    q3_rev = ground_truth["q3_revenue"]
    decline_abs = ground_truth["revenue_decline_abs"]
    decline_pct = ground_truth["revenue_decline_pct"]

    rev_subscore = 0.0

    # Check for Q2 and Q3 revenue figures
    q2_match = find_closest_number(numbers, q2_rev, tolerance_pct=2)
    q3_match = find_closest_number(numbers, q3_rev, tolerance_pct=2)
    decline_match = find_closest_number(numbers, decline_abs, tolerance_pct=5)
    pct_match = find_closest_number(numbers, decline_pct, tolerance_pct=5)

    if q2_match >= 0.7 and q3_match >= 0.7:
        rev_subscore += 0.10
    elif q2_match >= 0.4 or q3_match >= 0.4:
        rev_subscore += 0.05

    if decline_match >= 0.7 or pct_match >= 0.7:
        rev_subscore += 0.10
    elif decline_match >= 0.4 or pct_match >= 0.4:
        rev_subscore += 0.05

    # At minimum, detect that there IS a decline mentioned
    if rev_subscore == 0 and (fuzzy_contains(text, "decline") or fuzzy_contains(text, "decrease") or fuzzy_contains(text, "drop")):
        rev_subscore = 0.03

    score += rev_subscore
    feedback_parts.append(f"Revenue quantification: +{rev_subscore:.2f}/0.20")

    # ── Root cause 1: Electronics returns spike (0.25) ────────────────

    rc1_subscore = 0.0
    has_electronics = fuzzy_contains(text, "electronics") or fuzzy_contains(text, "electronic")
    has_returns = fuzzy_contains(text, "return") or fuzzy_contains(text, "refund")
    has_defective = fuzzy_contains(text, "defective") or fuzzy_contains(text, "defect") or fuzzy_contains(text, "quality")

    if has_electronics and (has_returns or has_defective):
        rc1_subscore += 0.15
        # Check for supporting numbers
        elec_q3_rate = ground_truth["electronics_return_q3"]
        elec_other_rate = ground_truth["electronics_return_other"]
        if find_closest_number(numbers, elec_q3_rate, tolerance_pct=15) >= 0.4:
            rc1_subscore += 0.10
        elif any(n > 0 for n in numbers):
            rc1_subscore += 0.05
    elif has_electronics or (has_returns and has_defective):
        rc1_subscore += 0.05

    score += rc1_subscore
    feedback_parts.append(f"Root cause 1 (electronics returns): +{rc1_subscore:.2f}/0.25")

    # ── Root cause 2: Marketing campaign gap (0.25) ───────────────────

    rc2_subscore = 0.0
    has_marketing = fuzzy_contains(text, "marketing") or fuzzy_contains(text, "campaign")
    has_gap = (
        fuzzy_contains(text, "gap") or fuzzy_contains(text, "no campaign")
        or fuzzy_contains(text, "zero campaign") or fuzzy_contains(text, "absence")
        or fuzzy_contains(text, "lack") or fuzzy_contains(text, "missing")
        or fuzzy_contains(text, "no active") or fuzzy_contains(text, "0 campaign")
        or fuzzy_contains(text, "none") or fuzzy_contains(text, "stopped")
        or fuzzy_contains(text, "paused") or fuzzy_contains(text, "reduced")
        or fuzzy_contains(text, "fewer") or fuzzy_contains(text, "declined")
    )

    if has_marketing and has_gap:
        rc2_subscore += 0.15
        # Check for supporting data about Q3 having 0 campaigns
        campaigns_q3 = ground_truth["campaigns_by_quarter"].get("Q3", {}).get("count", 0)
        has_q3_zero = (
            fuzzy_contains(text, "q3")
            and (fuzzy_contains(text, "0 campaign") or fuzzy_contains(text, "zero campaign")
                 or fuzzy_contains(text, "no campaign") or fuzzy_contains(text, "none"))
        )
        if campaigns_q3 == 0 and has_q3_zero:
            rc2_subscore += 0.10
        elif campaigns_q3 == 0 and (fuzzy_contains(text, "q3") or fuzzy_contains(text, "third quarter")):
            rc2_subscore += 0.05
    elif has_marketing:
        rc2_subscore += 0.05

    score += rc2_subscore
    feedback_parts.append(f"Root cause 2 (marketing gap): +{rc2_subscore:.2f}/0.25")

    # ── Recommendations (0.3) ─────────────────────────────────────────

    rec_subscore = 0.0

    # Look for recommendation markers
    rec_patterns = [
        r"recommend", r"\d\.\s", r"suggestion", r"propose",
        r"should\s", r"could\s", r"action\s*item",
        r"going forward", r"to prevent", r"to address",
    ]

    rec_indicators = sum(1 for p in rec_patterns if re.search(p, text))

    # Check for numbered recommendations (require substantive content, min 40 chars)
    numbered = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s*.{40,}', answer, re.MULTILINE)

    # Check if recommendations reference data
    has_data_refs = len([n for n in numbers if n > 0]) >= 3

    if len(numbered) >= 3 and has_data_refs and rec_indicators >= 2:
        rec_subscore = 0.30
    elif len(numbered) >= 3 or (rec_indicators >= 3 and has_data_refs):
        rec_subscore = 0.25
    elif len(numbered) >= 2 or rec_indicators >= 2:
        rec_subscore = 0.20
    elif len(numbered) >= 1 or rec_indicators >= 1:
        rec_subscore = 0.10
    elif len(answer) > 200:
        rec_subscore = 0.05

    score += rec_subscore
    feedback_parts.append(f"Recommendations: +{rec_subscore:.2f}/0.30")

    return round(min(score, 1.0), 4), " | ".join(feedback_parts)


def grade_task4(answer: str, ground_truth: Dict[str, Any]) -> Tuple[float, str]:
    """Grade Task 4: Data Quality Audit.

    Part A (0.5): Discrepancy count + average discrepancy percentage
    Part B (0.5): Negative margin categories
    """
    score = 0.0
    feedback_parts = []
    text = answer.lower()
    numbers = extract_numbers(answer)

    # Part A: Discrepancy count and average percentage
    expected_count = ground_truth["discrepancy_count"]
    expected_pct = ground_truth["avg_discrepancy_pct"]

    count_score = find_closest_number(numbers, float(expected_count), tolerance_pct=10)
    if count_score >= 0.7:
        score += 0.25
        feedback_parts.append(f"Part A: Discrepancy count correct (~{expected_count}) (+0.25)")
    elif count_score >= 0.4:
        score += 0.15
        feedback_parts.append(f"Part A: Discrepancy count close (+0.15)")
    else:
        feedback_parts.append(f"Part A: Discrepancy count incorrect (+0.00)")

    pct_score = find_closest_number(numbers, expected_pct, tolerance_pct=15)
    if pct_score >= 0.7:
        score += 0.25
        feedback_parts.append(f"Part A: Avg discrepancy % accurate ({expected_pct}%) (+0.25)")
    elif pct_score >= 0.4:
        score += 0.15
        feedback_parts.append(f"Part A: Avg discrepancy % roughly correct (+0.15)")
    else:
        feedback_parts.append(f"Part A: Avg discrepancy % incorrect (+0.00)")

    # Part B: Negative margin categories
    expected_cats = ground_truth["negative_margin_categories"]
    expected_cat_count = ground_truth["negative_margin_count"]

    cats_found = sum(1 for cat in expected_cats if fuzzy_contains(text, cat))

    cat_count_score = find_closest_number(numbers, float(expected_cat_count), tolerance_pct=10)
    if cat_count_score >= 0.7:
        score += 0.15
        feedback_parts.append(f"Part B: Margin category count correct ({expected_cat_count}) (+0.15)")
    elif cat_count_score >= 0.4:
        score += 0.08
        feedback_parts.append(f"Part B: Margin category count close (+0.08)")
    else:
        feedback_parts.append(f"Part B: Margin category count incorrect (+0.00)")

    if cats_found >= len(expected_cats):
        score += 0.35
        feedback_parts.append(f"Part B: All {len(expected_cats)} categories identified (+0.35)")
    elif cats_found >= 1:
        partial = round(0.10 + (cats_found / len(expected_cats)) * 0.20, 2) if expected_cats else 0.10
        score += partial
        feedback_parts.append(f"Part B: {cats_found}/{len(expected_cats)} categories found (+{partial:.2f})")
    else:
        feedback_parts.append(f"Part B: No categories identified (+0.00)")

    return round(min(score, 1.0), 4), " | ".join(feedback_parts)


def grade_task5(answer: str, ground_truth: Dict[str, Any]) -> Tuple[float, str]:
    """Grade Task 5: Executive Dashboard Summary.

    Part A (0.35): Monthly revenue — best/worst month identification + figures
    Part B (0.35): Cohort retention rates
    Part C (0.30): Channel performance ranking
    """
    score = 0.0
    feedback_parts = []
    text = answer.lower()
    numbers = extract_numbers(answer)

    # ── Part A: Monthly revenue trend (0.35) ──────────────────────
    best = ground_truth["best_month"]
    worst = ground_truth["worst_month"]

    month_aliases = {
        "January": ["january", "jan"], "February": ["february", "feb"],
        "March": ["march", "mar"], "April": ["april", "apr"],
        "May": ["may"], "June": ["june", "jun"],
        "July": ["july", "jul"], "August": ["august", "aug"],
        "September": ["september", "sep", "sept"], "October": ["october", "oct"],
        "November": ["november", "nov"], "December": ["december", "dec"],
    }

    def month_mentioned(month_name):
        for alias in month_aliases.get(month_name, [month_name.lower()]):
            if alias in text:
                return True
        return False

    a_subscore = 0.0
    if month_mentioned(best["name"]):
        a_subscore += 0.10
        rev_match = find_closest_number(numbers, best["revenue"], tolerance_pct=2)
        if rev_match >= 0.7:
            a_subscore += 0.08
    if month_mentioned(worst["name"]):
        a_subscore += 0.10
        rev_match = find_closest_number(numbers, worst["revenue"], tolerance_pct=2)
        if rev_match >= 0.7:
            a_subscore += 0.07

    score += a_subscore
    feedback_parts.append(f"Part A (monthly revenue): +{a_subscore:.2f}/0.35")

    # ── Part B: Cohort retention (0.35) ───────────────────────────
    cohort_data = ground_truth["cohort_retention"]
    b_subscore = 0.0
    retention_matches = 0

    for cohort_name, info in cohort_data.items():
        rate = info["retention_pct"]
        if find_closest_number(numbers, rate, tolerance_pct=3) >= 0.7:
            retention_matches += 1

    if retention_matches >= 3:
        b_subscore = 0.35
    elif retention_matches >= 2:
        b_subscore = 0.25
    elif retention_matches >= 1:
        b_subscore = 0.15
    elif fuzzy_contains(text, "retention") or fuzzy_contains(text, "cohort"):
        b_subscore = 0.05

    score += b_subscore
    feedback_parts.append(f"Part B (cohort retention): +{b_subscore:.2f}/0.35 ({retention_matches} rates matched)")

    # ── Part C: Channel performance (0.30) ────────────────────────
    channel_data = ground_truth["channel_stats"]
    c_subscore = 0.0

    # Check if channels are mentioned
    channels_mentioned = sum(1 for ch in ["web", "mobile", "in-store"] if fuzzy_contains(text, ch))

    # Check channel ranking by avg revenue per order
    sorted_channels = sorted(channel_data.items(), key=lambda x: x[1]["avg_revenue_per_order"], reverse=True)
    top_channel = sorted_channels[0][0] if sorted_channels else ""

    if channels_mentioned >= 3:
        c_subscore += 0.10

    # Check for revenue per order figures
    rev_matches = 0
    for ch, info in channel_data.items():
        if find_closest_number(numbers, info["avg_revenue_per_order"], tolerance_pct=3) >= 0.7:
            rev_matches += 1
    if rev_matches >= 2:
        c_subscore += 0.10
    elif rev_matches >= 1:
        c_subscore += 0.05

    # Check for return rate figures
    rate_matches = 0
    for ch, info in channel_data.items():
        if find_closest_number(numbers, info["return_rate"], tolerance_pct=5) >= 0.7:
            rate_matches += 1
    if rate_matches >= 2:
        c_subscore += 0.10
    elif rate_matches >= 1:
        c_subscore += 0.05

    score += c_subscore
    feedback_parts.append(f"Part C (channel performance): +{c_subscore:.2f}/0.30")

    return round(min(score, 1.0), 4), " | ".join(feedback_parts)


def grade(task_id: int, answer: str, ground_truth: Dict[str, Any]) -> Tuple[float, str]:
    """Route to the appropriate task grader."""
    task_gt = ground_truth.get(f"task{task_id}", {})

    if task_id == 1:
        return grade_task1(answer, task_gt)
    elif task_id == 2:
        return grade_task2(answer, task_gt)
    elif task_id == 3:
        return grade_task3(answer, task_gt)
    elif task_id == 4:
        return grade_task4(answer, task_gt)
    elif task_id == 5:
        return grade_task5(answer, task_gt)
    else:
        return 0.0, f"Unknown task_id: {task_id}"

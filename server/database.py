"""
Builds the e-commerce SQLite database that powers the environment.

Everything is seeded (seed=42) so results are reproducible across runs.
The data has some deliberate patterns baked in — like a Q3 revenue dip
caused by a spike in electronics returns and a gap in marketing campaigns.
These give the hard task something real to discover.

Ground truth is computed by actually running SQL against the generated data,
not by hardcoding numbers.
"""

import sqlite3
import random
import json
from datetime import date, timedelta
from typing import Dict, Any, Tuple

# ── Constants ──────────────────────────────────────────────────────────────

SEED = 42

COUNTRIES = ["US", "US", "US", "UK", "UK", "DE", "FR", "IN", "JP", "BR", "AU"]
SEGMENTS = ["enterprise", "mid-market", "smb", "consumer", "consumer", "consumer"]

CATEGORIES = {
    "electronics": {
        "subcategories": ["phones", "laptops", "tablets", "headphones", "cameras"],
        "price_range": (49.99, 1299.99),
        "cost_ratio": (0.45, 0.65),
    },
    "clothing": {
        "subcategories": ["shirts", "pants", "jackets", "shoes", "accessories"],
        "price_range": (14.99, 199.99),
        "cost_ratio": (0.30, 0.50),
    },
    "home": {
        "subcategories": ["furniture", "kitchen", "decor", "bedding", "lighting"],
        "price_range": (19.99, 599.99),
        "cost_ratio": (0.40, 0.60),
    },
    "sports": {
        "subcategories": ["fitness", "outdoor", "team_sports", "water_sports", "yoga"],
        "price_range": (9.99, 349.99),
        "cost_ratio": (0.35, 0.55),
    },
    "books": {
        "subcategories": ["fiction", "non_fiction", "technical", "children", "academic"],
        "price_range": (7.99, 79.99),
        "cost_ratio": (0.25, 0.45),
    },
}

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Daniel",
    "Lisa", "Matthew", "Nancy", "Anthony", "Betty", "Mark", "Margaret",
    "Donald", "Sandra", "Steven", "Ashley", "Paul", "Dorothy", "Andrew",
    "Kimberly", "Joshua", "Emily", "Kenneth", "Donna", "Kevin", "Michelle",
    "Brian", "Carol", "George", "Amanda", "Timothy", "Melissa", "Ronald",
    "Deborah", "Edward", "Stephanie", "Jason", "Rebecca", "Jeffrey", "Sharon",
    "Ryan", "Laura", "Jacob", "Cynthia", "Gary", "Kathleen", "Nicholas",
    "Amy", "Eric", "Angela", "Jonathan", "Shirley", "Stephen", "Anna",
    "Larry", "Brenda", "Justin", "Pamela", "Scott", "Emma", "Brandon",
    "Nicole", "Benjamin", "Helen", "Samuel", "Samantha", "Raymond", "Katherine",
    "Gregory", "Christine", "Frank", "Debra", "Alexander", "Rachel", "Patrick",
    "Carolyn", "Jack", "Janet", "Dennis", "Catherine",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts",
]

CHANNELS = ["web", "web", "web", "mobile", "mobile", "in-store"]
ORDER_STATUSES = ["completed", "completed", "completed", "completed", "returned", "cancelled", "pending"]
RETURN_REASONS = ["defective", "wrong_item", "not_as_described", "changed_mind", "late_delivery"]

CAMPAIGN_NAMES = [
    "New Year Sale", "Valentine Special", "Spring Collection",
    "Easter Deals", "Summer Kickoff", "Back to School",
    "Fall Favorites", "Halloween Special", "Black Friday",
    "Cyber Monday", "Holiday Season", "Year End Clearance",
    "Flash Sale Q1", "Flash Sale Q2", "Loyalty Rewards Q1",
    "Loyalty Rewards Q2", "Social Media Push", "Email Blast Q1",
    "Influencer Collab", "Retargeting Campaign",
]

CAMPAIGN_CHANNELS = ["email", "social", "search", "display"]


def _random_date(start: date, end: date, rng: random.Random) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def create_database() -> Tuple[sqlite3.Connection, Dict[str, Any]]:
    """Create the e-commerce database and compute ground truth answers.

    Returns:
        (connection, ground_truth) where ground_truth contains precomputed
        answers for all 3 tasks.
    """
    rng = random.Random(SEED)
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    cur = conn.cursor()

    # ── Create tables ──────────────────────────────────────────────────

    cur.executescript("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            signup_date DATE NOT NULL,
            country TEXT NOT NULL,
            segment TEXT NOT NULL
        );

        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT NOT NULL,
            unit_price REAL NOT NULL,
            cost_price REAL NOT NULL,
            created_date DATE NOT NULL
        );

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_date DATE NOT NULL,
            status TEXT NOT NULL,
            total_amount REAL NOT NULL,
            discount_amount REAL NOT NULL,
            channel TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        CREATE TABLE order_items (
            item_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            subtotal REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        CREATE TABLE returns (
            return_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            return_date DATE NOT NULL,
            reason TEXT NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        CREATE TABLE marketing_campaigns (
            campaign_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            channel TEXT NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            budget REAL NOT NULL,
            target_segment TEXT NOT NULL
        );

        CREATE TABLE campaign_attributions (
            attribution_id INTEGER PRIMARY KEY,
            campaign_id INTEGER NOT NULL,
            order_id INTEGER NOT NULL,
            attribution_type TEXT NOT NULL,
            FOREIGN KEY (campaign_id) REFERENCES marketing_campaigns(campaign_id),
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );
    """)

    # ── Generate customers (500) ───────────────────────────────────────

    used_emails = set()
    customers = []
    for i in range(1, 501):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        email_base = f"{first.lower()}.{last.lower()}"
        email = f"{email_base}{i}@example.com"
        while email in used_emails:
            email = f"{email_base}{rng.randint(1000, 9999)}@example.com"
        used_emails.add(email)
        signup = _random_date(date(2023, 1, 1), date(2024, 6, 30), rng)
        country = rng.choice(COUNTRIES)
        segment = rng.choice(SEGMENTS)
        customers.append((i, name, email, signup.isoformat(), country, segment))

    cur.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)", customers
    )

    # ── Generate products (100) ────────────────────────────────────────

    products = []
    pid = 1
    for cat, info in CATEGORIES.items():
        for _ in range(20):
            subcat = rng.choice(info["subcategories"])
            price = round(rng.uniform(*info["price_range"]), 2)
            cost = round(price * rng.uniform(*info["cost_ratio"]), 2)
            name = f"{subcat.replace('_', ' ').title()} {rng.randint(100, 999)}"
            created = _random_date(date(2023, 6, 1), date(2024, 3, 31), rng)
            products.append((pid, name, cat, subcat, price, cost, created.isoformat()))
            pid += 1

    cur.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", products
    )

    # ── Generate orders (5000) with seasonal patterns ──────────────────
    # Q1: baseline, Q2: growth, Q3: deliberate DIP, Q4: holiday spike

    product_lookup = {p[0]: p for p in products}  # pid -> tuple
    electronics_pids = [p[0] for p in products if p[2] == "electronics"]

    quarter_weights = {
        1: 1.0,   # Q1 baseline
        2: 1.25,  # Q2 growth
        3: 0.75,  # Q3 dip (planted)
        4: 1.5,   # Q4 holiday spike
    }

    orders = []
    order_items_list = []
    oid = 1
    iid = 1

    for month in range(1, 13):
        quarter = (month - 1) // 3 + 1
        weight = quarter_weights[quarter]
        n_orders = int(420 * weight * rng.uniform(0.9, 1.1))

        for _ in range(n_orders):
            cid = rng.randint(1, 500)
            order_date = _random_date(
                date(2024, month, 1),
                date(2024, month, 28),
                rng,
            )
            channel = rng.choice(CHANNELS)

            # Determine number of items (1-5)
            n_items = rng.choices([1, 2, 3, 4, 5], weights=[35, 30, 20, 10, 5])[0]

            items = []
            total = 0.0
            for _ in range(n_items):
                prod = product_lookup[rng.randint(1, 100)]
                qty = rng.choices([1, 2, 3], weights=[60, 30, 10])[0]
                price = prod[4]
                subtotal = round(price * qty, 2)
                items.append((iid, oid, prod[0], qty, price, subtotal))
                total += subtotal
                iid += 1

            discount = round(total * rng.uniform(0, 0.15), 2)
            total_after = round(total - discount, 2)

            # Status: more returns in Q3 for electronics-heavy orders
            status = rng.choice(ORDER_STATUSES)
            if quarter == 3 and any(it[2] in electronics_pids for it in items):
                if rng.random() < 0.15:  # Extra 15% return rate for Q3 electronics
                    status = "returned"

            orders.append((oid, cid, order_date.isoformat(), status, total_after, discount, channel))
            order_items_list.extend(items)
            oid += 1

    cur.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)", orders
    )
    cur.executemany(
        "INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?)", order_items_list
    )

    # ── Generate returns ───────────────────────────────────────────────

    returns = []
    rid = 1
    for o in orders:
        if o[3] == "returned":
            order_date = date.fromisoformat(o[2])
            return_date = order_date + timedelta(days=rng.randint(1, 30))
            if return_date > date(2024, 12, 31):
                return_date = date(2024, 12, 31)

            # Pick a random item from this order
            order_items_for_order = [it for it in order_items_list if it[1] == o[0]]
            if order_items_for_order:
                item = rng.choice(order_items_for_order)
                prod_id = item[2]

                # Electronics in Q3 get "defective" reason more often
                order_month = order_date.month
                quarter = (order_month - 1) // 3 + 1
                if quarter == 3 and prod_id in electronics_pids:
                    reason = rng.choices(
                        RETURN_REASONS,
                        weights=[50, 10, 15, 15, 10],  # Heavy "defective"
                    )[0]
                else:
                    reason = rng.choice(RETURN_REASONS)

                returns.append((rid, o[0], prod_id, return_date.isoformat(), reason))
                rid += 1

    cur.executemany(
        "INSERT INTO returns VALUES (?, ?, ?, ?, ?)", returns
    )

    # ── Generate marketing campaigns ───────────────────────────────────
    # Deliberate GAP in Q3: campaigns cluster in Q1-Q2 and Q4

    campaigns = []
    campaign_periods = [
        # Q1 campaigns
        (date(2024, 1, 1), date(2024, 1, 31)),
        (date(2024, 2, 1), date(2024, 2, 28)),
        (date(2024, 3, 1), date(2024, 3, 31)),
        (date(2024, 1, 15), date(2024, 3, 15)),
        (date(2024, 2, 10), date(2024, 3, 31)),
        # Q2 campaigns
        (date(2024, 4, 1), date(2024, 4, 30)),
        (date(2024, 5, 1), date(2024, 6, 15)),
        (date(2024, 4, 15), date(2024, 6, 30)),
        (date(2024, 5, 15), date(2024, 6, 30)),
        (date(2024, 6, 1), date(2024, 6, 30)),
        # NO Q3 campaigns (planted gap)
        # Q4 campaigns
        (date(2024, 10, 1), date(2024, 10, 31)),
        (date(2024, 10, 15), date(2024, 11, 30)),
        (date(2024, 11, 1), date(2024, 11, 30)),
        (date(2024, 11, 15), date(2024, 12, 31)),
        (date(2024, 11, 25), date(2024, 12, 2)),
        (date(2024, 12, 1), date(2024, 12, 31)),
        (date(2024, 12, 15), date(2024, 12, 31)),
        (date(2024, 10, 1), date(2024, 12, 31)),
        (date(2024, 11, 1), date(2024, 12, 15)),
        (date(2024, 12, 10), date(2024, 12, 31)),
    ]

    for cid_idx in range(20):
        start, end = campaign_periods[cid_idx]
        budget = round(rng.uniform(5000, 50000), 2)
        segment = rng.choice(SEGMENTS)
        ch = rng.choice(CAMPAIGN_CHANNELS)
        campaigns.append((
            cid_idx + 1, CAMPAIGN_NAMES[cid_idx], ch,
            start.isoformat(), end.isoformat(), budget, segment,
        ))

    cur.executemany(
        "INSERT INTO marketing_campaigns VALUES (?, ?, ?, ?, ?, ?, ?)", campaigns
    )

    # ── Generate campaign attributions ─────────────────────────────────

    attributions = []
    aid = 1
    for camp in campaigns:
        camp_start = date.fromisoformat(camp[3])
        camp_end = date.fromisoformat(camp[4])
        # Attribute some orders that fall within campaign period
        matching_orders = [
            o for o in orders
            if camp_start <= date.fromisoformat(o[2]) <= camp_end
            and o[3] in ("completed", "returned")
        ]
        n_attr = min(len(matching_orders), rng.randint(50, 200))
        selected = rng.sample(matching_orders, min(n_attr, len(matching_orders)))
        for o in selected:
            attr_type = rng.choice(["first_touch", "last_touch"])
            attributions.append((aid, camp[0], o[0], attr_type))
            aid += 1

    cur.executemany(
        "INSERT INTO campaign_attributions VALUES (?, ?, ?, ?)", attributions
    )

    conn.commit()

    # ── Compute ground truth ───────────────────────────────────────────

    ground_truth = _compute_ground_truth(conn)

    return conn, ground_truth


def _compute_ground_truth(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Run reference SQL queries to compute ground truth answers."""
    cur = conn.cursor()

    # Task 1A: Total revenue for completed orders in December 2024
    cur.execute("""
        SELECT ROUND(SUM(total_amount), 2)
        FROM orders
        WHERE status = 'completed'
          AND order_date >= '2024-12-01'
          AND order_date <= '2024-12-31'
    """)
    dec_revenue = cur.fetchone()[0]

    # Task 1B: Top 5 customers by completed order count
    cur.execute("""
        SELECT c.name, COUNT(*) as order_count
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.status = 'completed'
        GROUP BY o.customer_id, c.name
        ORDER BY order_count DESC, c.name ASC
        LIMIT 5
    """)
    top_customers = [(row[0], row[1]) for row in cur.fetchall()]

    # Task 2A: Product category with highest return rate
    cur.execute("""
        SELECT
            p.category,
            ROUND(
                100.0 * COUNT(DISTINCT r.return_id) / COUNT(DISTINCT oi.item_id),
                2
            ) as return_rate
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN returns r ON r.order_id = oi.order_id AND r.product_id = oi.product_id
        GROUP BY p.category
        ORDER BY return_rate DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    highest_return_category = row[0]
    highest_return_rate = row[1]

    # Task 2B: Customers with 3+ orders in Q1 but 0 in Q3
    cur.execute("""
        SELECT customer_id
        FROM orders
        WHERE order_date >= '2024-01-01' AND order_date <= '2024-03-31'
        GROUP BY customer_id
        HAVING COUNT(*) >= 3
    """)
    q1_active = set(row[0] for row in cur.fetchall())

    cur.execute("""
        SELECT DISTINCT customer_id
        FROM orders
        WHERE order_date >= '2024-07-01' AND order_date <= '2024-09-30'
    """)
    q3_active = set(row[0] for row in cur.fetchall())

    churned_customers = q1_active - q3_active
    churned_count = len(churned_customers)

    if churned_customers:
        placeholders = ",".join("?" * len(churned_customers))
        cur.execute(f"""
            SELECT ROUND(AVG(total_spend), 2)
            FROM (
                SELECT customer_id, SUM(total_amount) as total_spend
                FROM orders
                WHERE customer_id IN ({placeholders})
                GROUP BY customer_id
            )
        """, list(churned_customers))
        churned_avg_spend = cur.fetchone()[0]
    else:
        churned_avg_spend = 0.0

    # Task 3: Q2 vs Q3 revenue
    cur.execute("""
        SELECT ROUND(SUM(total_amount), 2)
        FROM orders
        WHERE status = 'completed'
          AND order_date >= '2024-04-01' AND order_date <= '2024-06-30'
    """)
    q2_revenue = cur.fetchone()[0]

    cur.execute("""
        SELECT ROUND(SUM(total_amount), 2)
        FROM orders
        WHERE status = 'completed'
          AND order_date >= '2024-07-01' AND order_date <= '2024-09-30'
    """)
    q3_revenue = cur.fetchone()[0]

    revenue_decline_abs = round(q2_revenue - q3_revenue, 2)
    revenue_decline_pct = round(100.0 * revenue_decline_abs / q2_revenue, 2)

    # Electronics return rate in Q3 vs other quarters
    cur.execute("""
        SELECT
            CASE
                WHEN o.order_date >= '2024-07-01' AND o.order_date <= '2024-09-30' THEN 'Q3'
                ELSE 'Other'
            END as period,
            ROUND(100.0 * COUNT(DISTINCT r.return_id) / COUNT(DISTINCT oi.item_id), 2) as return_rate
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        JOIN orders o ON oi.order_id = o.order_id
        LEFT JOIN returns r ON r.order_id = oi.order_id AND r.product_id = oi.product_id
        WHERE p.category = 'electronics'
        GROUP BY period
    """)
    electronics_return_rates = {row[0]: row[1] for row in cur.fetchall()}

    # Campaign count by quarter
    cur.execute("""
        SELECT
            CASE
                WHEN start_date < '2024-04-01' THEN 'Q1'
                WHEN start_date < '2024-07-01' THEN 'Q2'
                WHEN start_date < '2024-10-01' THEN 'Q3'
                ELSE 'Q4'
            END as quarter,
            COUNT(*) as campaign_count,
            ROUND(SUM(budget), 2) as total_budget
        FROM marketing_campaigns
        GROUP BY quarter
        ORDER BY quarter
    """)
    campaigns_by_quarter = {row[0]: {"count": row[1], "budget": row[2]} for row in cur.fetchall()}

    return {
        "task1": {
            "dec_revenue": dec_revenue,
            "top_customers": top_customers,
        },
        "task2": {
            "highest_return_category": highest_return_category,
            "highest_return_rate": highest_return_rate,
            "churned_count": churned_count,
            "churned_avg_spend": churned_avg_spend,
        },
        "task3": {
            "q2_revenue": q2_revenue,
            "q3_revenue": q3_revenue,
            "revenue_decline_abs": revenue_decline_abs,
            "revenue_decline_pct": revenue_decline_pct,
            "electronics_return_q3": electronics_return_rates.get("Q3", 0),
            "electronics_return_other": electronics_return_rates.get("Other", 0),
            "campaigns_by_quarter": campaigns_by_quarter,
        },
    }


def get_schema_info(conn: sqlite3.Connection) -> str:
    """Generate a human-readable schema description."""
    cur = conn.cursor()

    lines = ["DATABASE SCHEMA", "=" * 60, ""]

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]

    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        lines.append(f"Table: {table} ({count} rows)")

        cur.execute(f"PRAGMA table_info({table})")
        for col in cur.fetchall():
            col_name, col_type, not_null, default, pk = col[1], col[2], col[3], col[4], col[5]
            extras = []
            if pk:
                extras.append("PRIMARY KEY")
            if not_null:
                extras.append("NOT NULL")
            extra_str = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"  - {col_name} {col_type}{extra_str}")

        lines.append("")

    # Add sample data
    lines.extend(["SAMPLE DATA (first 3 rows per table)", "=" * 60, ""])
    for table in tables:
        cur.execute(f"SELECT * FROM {table} LIMIT 3")
        rows = cur.fetchall()
        cur.execute(f"PRAGMA table_info({table})")
        col_names = [col[1] for col in cur.fetchall()]
        lines.append(f"{table}:")
        lines.append(f"  Columns: {', '.join(col_names)}")
        for row in rows:
            lines.append(f"  {row}")
        lines.append("")

    # Add relationship hints
    lines.extend([
        "RELATIONSHIPS",
        "=" * 60,
        "orders.customer_id -> customers.customer_id",
        "order_items.order_id -> orders.order_id",
        "order_items.product_id -> products.product_id",
        "returns.order_id -> orders.order_id",
        "returns.product_id -> products.product_id",
        "campaign_attributions.campaign_id -> marketing_campaigns.campaign_id",
        "campaign_attributions.order_id -> orders.order_id",
        "",
        "NOTES",
        "=" * 60,
        "- All dates are in YYYY-MM-DD format (year 2024)",
        "- order status values: completed, returned, cancelled, pending",
        "- return reasons: defective, wrong_item, not_as_described, changed_mind, late_delivery",
        "- channels: web, mobile, in-store",
        "- segments: enterprise, mid-market, smb, consumer",
        "- product categories: electronics, clothing, home, sports, books",
    ])

    return "\n".join(lines)


# Task descriptions
TASK_DESCRIPTIONS = {
    1: (
        "TASK 1 (Easy): Basic Data Retrieval\n"
        "Answer the following two questions about the e-commerce database:\n\n"
        "1. What was the total revenue (sum of total_amount) for COMPLETED orders "
        "in December 2024? Provide the exact dollar amount.\n\n"
        "2. List the top 5 customers by total number of COMPLETED orders. "
        "For each customer, provide their name and order count, ordered by "
        "count descending (break ties alphabetically by name).\n\n"
        "Use execute_sql actions to query the database, then submit your "
        "final answer with submit_answer when ready."
    ),
    2: (
        "TASK 2 (Medium): Multi-Table Analysis\n"
        "Answer the following two questions:\n\n"
        "1. Which product CATEGORY has the highest return rate? The return rate "
        "is calculated as: (number of distinct returns / number of distinct order items) * 100. "
        "Provide the category name and its return rate as a percentage.\n\n"
        "2. Identify customers who placed at least 3 orders in Q1 2024 "
        "(January-March) but ZERO orders in Q3 2024 (July-September). "
        "How many such customers are there? What is their average total spend "
        "(sum of total_amount across ALL their orders)?\n\n"
        "Use execute_sql actions to query the database, then submit your "
        "final answer with submit_answer when ready."
    ),
    3: (
        "TASK 3 (Hard): Business Intelligence Analysis\n"
        "The company experienced a revenue decline in Q3 2024 (Jul-Sep) compared "
        "to Q2 2024 (Apr-Jun). Perform a comprehensive analysis:\n\n"
        "1. QUANTIFY the Q2-to-Q3 revenue decline: provide both the absolute "
        "dollar amount and the percentage decline (for completed orders only).\n\n"
        "2. IDENTIFY the top 2 root causes of the decline. Use data evidence "
        "from the database to support each cause. Look at returns patterns, "
        "marketing campaigns, product categories, and customer behavior.\n\n"
        "3. Provide 3 specific, data-backed RECOMMENDATIONS to prevent this "
        "in the future. Each recommendation must reference specific numbers "
        "from your analysis.\n\n"
        "Support every claim with specific numbers from the database. "
        "Use execute_sql actions to query the database, then submit your "
        "comprehensive analysis with submit_answer when ready."
    ),
}


if __name__ == "__main__":
    conn, gt = create_database()
    print(json.dumps(gt, indent=2, default=str))
    print(f"\nSchema info length: {len(get_schema_info(conn))} chars")

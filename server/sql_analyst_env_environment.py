"""
The core environment — where the magic happens.

The agent gets a task, queries the database with SQL, and submits an analysis.
We grade the answer, factor in how efficiently the agent worked, and return
a reward. Five difficulty levels from easy to hard.
"""

import json
import re
import sqlite3
from uuid import uuid4
from typing import Optional, Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

try:
    from ..models import SqlAnalystAction, SqlAnalystObservation
    from .database import create_database, get_schema_info, TASK_DESCRIPTIONS
    from .graders import grade
except (ImportError, ModuleNotFoundError):
    from models import SqlAnalystAction, SqlAnalystObservation
    from server.database import create_database, get_schema_info, TASK_DESCRIPTIONS
    from server.graders import grade


class SqlAnalystEnvironment(Environment):
    """SQL Data Analysis environment for training and evaluating AI agents.

    The agent writes SQL queries against an e-commerce database and submits
    analyses to answer business intelligence questions.

    Five tasks with increasing difficulty:
    - Task 1 (Easy): Basic data retrieval with single-table queries
    - Task 2 (Medium): Multi-table analysis with JOINs and aggregations
    - Task 3 (Hard): Complex BI analysis requiring multi-step reasoning
    - Task 4 (Medium): Data quality audit — finding discrepancies and anomalies
    - Task 5 (Hard): Executive dashboard — multi-faceted business summary
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Tables that are relevant to each task — used for intermediate rewards
    TASK_RELEVANT_TABLES = {
        1: {"orders", "customers"},
        2: {"orders", "order_items", "products", "returns", "customers"},
        3: {"orders", "order_items", "products", "returns", "marketing_campaigns", "campaign_attributions", "customers"},
        4: {"orders", "order_items", "products", "customers"},
        5: {"orders", "order_items", "products", "returns", "customers"},
    }

    # Map string task IDs to integers so evaluators can use either format
    TASK_ID_MAP = {
        "easy": 1, "1": 1, 1: 1,
        "medium": 2, "2": 2, 2: 2,
        "hard": 3, "3": 3, 3: 3,
        "data_quality": 4, "4": 4, 4: 4,
        "dashboard": 5, "executive": 5, "5": 5, 5: 5,
    }

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = 1
        self._queries_executed = 0
        self._successful_queries = 0
        self._failed_queries = 0
        self._tables_queried: set = set()
        self._query_history: list = []
        self._hints_given: int = 0
        self._max_steps = 20
        self._done = False
        self._total_reward = 0.0
        # Init DB eagerly so stateless HTTP endpoints work (each /step creates a fresh instance)
        self._db_conn, self._ground_truth = create_database()
        self._schema_info = get_schema_info(self._db_conn)

    def _init_db(self):
        """Reinitialize the database for a fresh episode."""
        if self._db_conn:
            try:
                self._db_conn.close()
            except Exception:
                pass  # Ignore thread-safety errors on close
        self._db_conn, self._ground_truth = create_database()
        self._schema_info = get_schema_info(self._db_conn)

    def _resolve_task_id(self, raw_task_id) -> int:
        """Convert any task_id format to int. Accepts 1/2/3, 'easy'/'medium'/'hard', etc."""
        if isinstance(raw_task_id, str):
            raw_task_id = raw_task_id.strip().lower()
        return self.TASK_ID_MAP.get(raw_task_id, 1)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SqlAnalystObservation:
        """Reset the environment for a new episode.

        Keyword Args:
            task_id (int): Which task to run (1, 2, or 3). Default 1.
        """
        self._task_id = self._resolve_task_id(kwargs.get("task_id", 1))

        self._init_db()
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._queries_executed = 0
        self._successful_queries = 0
        self._failed_queries = 0
        self._tables_queried = set()
        self._query_history = []
        self._hints_given = 0
        self._done = False
        self._total_reward = 0.0

        task_desc = TASK_DESCRIPTIONS.get(self._task_id, TASK_DESCRIPTIONS[1])

        return SqlAnalystObservation(
            task_description=task_desc,
            schema_info=self._schema_info,
            task_id=self._task_id,
            step_number=0,
            max_steps=self._max_steps,
            done=False,
            reward=0.0,
            message="Environment reset. Database ready. Use execute_sql to query, submit_answer to submit.",
        )

    # Actions that don't cost a step (informational/educational)
    FREE_ACTIONS = {"request_schema", "request_hint", "explain_sql"}

    def step(
        self,
        action: SqlAnalystAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SqlAnalystObservation:
        """Execute an action (SQL query or answer submission)."""
        if self._done:
            return SqlAnalystObservation(
                task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                schema_info=self._schema_info,
                task_id=self._task_id,
                step_number=self._state.step_count,
                max_steps=self._max_steps,
                done=True,
                reward=0.0,
                message="Episode already finished. Call reset() to start a new episode.",
            )

        action_type = action.action_type.strip().lower()
        content = action.content.strip()

        # Free actions don't increment step count
        if action_type not in self.FREE_ACTIONS:
            self._state.step_count += 1

            # Check max steps
            if self._state.step_count >= self._max_steps:
                self._done = True
                return SqlAnalystObservation(
                    task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                    schema_info=self._schema_info,
                    task_id=self._task_id,
                    step_number=self._state.step_count,
                    max_steps=self._max_steps,
                    done=True,
                    reward=0.0,
                    message=f"Maximum steps ({self._max_steps}) reached. Episode ended with no submission.",
                )

        if action_type == "execute_sql":
            return self._handle_sql(content)
        elif action_type == "submit_answer":
            return self._handle_submit(content)
        elif action_type == "request_schema":
            return self._handle_schema_request()
        elif action_type == "request_hint":
            return self._handle_hint_request()
        elif action_type == "explain_sql":
            return self._handle_explain(content)
        else:
            return SqlAnalystObservation(
                task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                schema_info=self._schema_info,
                task_id=self._task_id,
                step_number=self._state.step_count,
                max_steps=self._max_steps,
                done=False,
                reward=0.0,
                error_message=(
                    f"Unknown action_type '{action_type}'. Use 'execute_sql', 'submit_answer', "
                    f"'request_schema', 'request_hint', or 'explain_sql'."
                ),
                message="Invalid action type.",
            )

    @staticmethod
    def _format_ascii_table(columns: list, rows: list, max_rows: int = 20) -> str:
        """Format query results as an ASCII table for readability."""
        if not columns or not rows:
            return "(empty result set)"
        display_rows = rows[:max_rows]
        col_widths = []
        for i, c in enumerate(columns):
            max_data = max((len(str(r[i])) for r in display_rows), default=0)
            col_widths.append(min(max(len(str(c)), max_data), 30))
        header = " | ".join(str(c).ljust(w)[:w] for c, w in zip(columns, col_widths))
        separator = "-+-".join("-" * w for w in col_widths)
        data_lines = [
            " | ".join(str(v).ljust(w)[:w] for v, w in zip(row, col_widths))
            for row in display_rows
        ]
        table = f"{header}\n{separator}\n" + "\n".join(data_lines)
        if len(rows) > max_rows:
            table += f"\n... ({len(rows) - max_rows} more rows)"
        return table

    @staticmethod
    def _format_ascii_bar(columns: list, rows: list) -> str:
        """If result looks like a grouped aggregation, render ASCII bars."""
        if len(columns) != 2 or len(rows) < 2 or len(rows) > 15:
            return ""
        try:
            labels = [str(r[0]) for r in rows]
            values = [float(r[1]) for r in rows]
        except (ValueError, TypeError):
            return ""
        max_val = max(values) if values else 1
        max_label = max(len(l) for l in labels)
        bar_width = 30
        lines = [f"\n{'':>{max_label}}  {columns[1]}"]
        for label, val in zip(labels, values):
            bar_len = int(bar_width * val / max_val) if max_val else 0
            lines.append(f"{label:>{max_label}}  {'=' * bar_len} {val:,.2f}")
        return "\n".join(lines)

    def _handle_sql(self, sql: str) -> SqlAnalystObservation:
        """Execute a SQL query and return results."""
        self._queries_executed += 1

        if not sql:
            return SqlAnalystObservation(
                task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                schema_info=self._schema_info,
                task_id=self._task_id,
                step_number=self._state.step_count,
                max_steps=self._max_steps,
                done=False,
                reward=0.0,
                error_message="Empty SQL query.",
                message="Please provide a valid SQL query.",
            )

        # Block dangerous operations — negative reward for trying
        sql_upper = sql.upper().strip()
        if any(kw in sql_upper for kw in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]):
            self._failed_queries += 1
            reward = -0.05  # Penalize destructive attempts
            self._total_reward += reward
            return SqlAnalystObservation(
                task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                schema_info=self._schema_info,
                task_id=self._task_id,
                step_number=self._state.step_count,
                max_steps=self._max_steps,
                done=False,
                reward=reward,
                error_message="Only SELECT queries are allowed. Data modification is not permitted.",
                message="Read-only environment. Use SELECT queries only.",
            )

        try:
            cur = self._db_conn.cursor()
            cur.execute(sql)
            rows = cur.fetchmany(100)
            columns = [desc[0] for desc in cur.description] if cur.description else []

            result_data = [dict(zip(columns, row)) for row in rows]
            result_json = json.dumps(result_data, indent=2, default=str)

            total_rows = len(rows)
            if total_rows == 100:
                extra = cur.fetchone()
                if extra:
                    total_rows = "100+ (truncated)"

            self._successful_queries += 1

            # Intermediate reward: reward exploring relevant tables
            reward = self._compute_query_reward(sql, len(rows))
            self._total_reward += reward

            # Track query history
            self._query_history.append({
                "step": self._state.step_count,
                "sql": sql[:200],
                "row_count": len(rows),
                "reward": reward,
            })

            # Build rich message with ASCII formatting
            ascii_table = self._format_ascii_table(columns, rows)
            message = f"Query executed successfully. {total_rows} rows returned. (reward: {reward:+.3f})\n\n{ascii_table}"
            if "GROUP BY" in sql_upper:
                bar = self._format_ascii_bar(columns, rows[:15])
                if bar:
                    message += "\n" + bar

            # Reward breakdown for this query
            reward_breakdown = {
                "query_reward": reward,
                "tables_discovered": list(self._extract_tables_from_sql(sql)),
                "cumulative_reward": round(self._total_reward, 4),
            }

            return SqlAnalystObservation(
                task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                schema_info="",
                query_result=result_json,
                columns=columns,
                row_count=len(rows),
                task_id=self._task_id,
                step_number=self._state.step_count,
                max_steps=self._max_steps,
                done=False,
                reward=reward,
                message=message,
                reward_breakdown=reward_breakdown,
                query_history=self._query_history[-5:],
            )
        except Exception as e:
            self._failed_queries += 1
            reward = -0.02  # Small penalty for syntax errors
            self._total_reward += reward
            return SqlAnalystObservation(
                task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                schema_info="",
                task_id=self._task_id,
                step_number=self._state.step_count,
                max_steps=self._max_steps,
                done=False,
                reward=reward,
                error_message=str(e),
                message=f"SQL query failed. (reward: {reward:+.3f})",
            )

    @staticmethod
    def _extract_tables_from_sql(sql: str) -> set:
        """Extract table names from FROM/JOIN clauses, ignoring comments."""
        # Strip single-line comments
        sql_clean = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        # Strip block comments
        sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)

        known_tables = {"customers", "products", "orders", "order_items",
                        "returns", "marketing_campaigns", "campaign_attributions"}

        # Match table names after FROM or JOIN keywords
        pattern = r'(?:FROM|JOIN)\s+(\w+)'
        found = set(m.lower() for m in re.findall(pattern, sql_clean, re.IGNORECASE))
        return found & known_tables

    def _compute_query_reward(self, sql: str, row_count: int) -> float:
        """Compute intermediate reward for a successful SQL query.

        Rewards the agent for:
        - Querying tables relevant to the current task (via FROM/JOIN parsing)
        - Getting non-empty results (suggests a productive query)
        - Using aggregations (GROUP BY, COUNT, SUM, AVG — signs of analysis)
        """
        sql_upper = sql.upper()
        reward = 0.0

        # Reward for querying relevant tables (only first time each table is hit)
        relevant = self.TASK_RELEVANT_TABLES.get(self._task_id, set())
        tables_in_query = self._extract_tables_from_sql(sql)
        newly_discovered = set()
        for table in tables_in_query:
            if table not in self._tables_queried:
                self._tables_queried.add(table)
                if table in relevant:
                    newly_discovered.add(table)

        if newly_discovered:
            # Reward proportional to how many relevant tables are now covered
            coverage = len(self._tables_queried & relevant) / len(relevant)
            reward += 0.02 * len(newly_discovered)
            if coverage >= 1.0:
                reward += 0.03  # Bonus for covering all relevant tables

        # Small reward for non-empty results
        if row_count > 0:
            reward += 0.01

        # Reward for analytical queries (aggregations suggest deeper analysis)
        analytical_keywords = ["GROUP BY", "COUNT(", "SUM(", "AVG(", "HAVING", "ORDER BY"]
        if any(kw in sql_upper for kw in analytical_keywords):
            reward += 0.01

        return round(min(reward, 0.1), 4)  # Cap per-query reward at 0.1

    def _handle_submit(self, answer: str) -> SqlAnalystObservation:
        """Grade a submitted answer and end the episode."""
        self._done = True

        if not answer.strip():
            return SqlAnalystObservation(
                task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                schema_info="",
                task_id=self._task_id,
                step_number=self._state.step_count,
                max_steps=self._max_steps,
                done=True,
                reward=0.0,
                message="Empty answer submitted. Score: 0.0",
            )

        # Get raw score from grader
        raw_score, feedback = grade(self._task_id, answer, self._ground_truth)

        # Apply efficiency multiplier: penalize using more than 5 queries
        efficiency = max(0.65, 1.0 - 0.03 * max(0, self._queries_executed - 5))

        # Apply step decay: two-tier — gentle for first 10 steps, steeper after
        if self._state.step_count <= 10:
            step_decay = max(0.85, 1.0 - 0.015 * self._state.step_count)
        else:
            step_decay = max(0.70, 0.85 - 0.03 * (self._state.step_count - 10))

        # Incorporate intermediate exploration rewards (capped at 0.15 bonus)
        exploration_bonus = min(self._total_reward, 0.15)
        final_reward = round(raw_score * efficiency * step_decay, 4)
        self._total_reward = final_reward + exploration_bonus

        reward_breakdown = {
            "raw_score": raw_score,
            "efficiency_multiplier": round(efficiency, 4),
            "step_decay": round(step_decay, 4),
            "exploration_bonus": round(exploration_bonus, 4),
            "graded_reward": final_reward,
            "total_reward": round(self._total_reward, 4),
            "grader_feedback": feedback,
        }

        return SqlAnalystObservation(
            task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
            schema_info="",
            task_id=self._task_id,
            step_number=self._state.step_count,
            max_steps=self._max_steps,
            done=True,
            reward=final_reward,
            message=(
                f"Answer submitted. Raw score: {raw_score:.4f}, "
                f"Efficiency: {efficiency:.2f} ({self._queries_executed} queries), "
                f"Step decay: {step_decay:.2f} ({self._state.step_count} steps), "
                f"Exploration bonus: {exploration_bonus:.4f}, "
                f"Final reward: {final_reward:.4f}\n"
                f"Feedback: {feedback}"
            ),
            reward_breakdown=reward_breakdown,
            query_history=self._query_history[-5:],
        )

    # ── Free action handlers ─────────────────────────────────────────

    def _handle_schema_request(self) -> SqlAnalystObservation:
        """Return the database schema without costing a step."""
        return SqlAnalystObservation(
            task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
            schema_info=self._schema_info,
            task_id=self._task_id,
            step_number=self._state.step_count,
            max_steps=self._max_steps,
            done=False,
            reward=0.0,
            message="Schema information provided. No step cost for schema requests.",
        )

    TASK_HINTS = {
        1: [
            "Hint: Filter orders by status='completed' and order_date for December 2024.",
            "Hint: Use GROUP BY customer_id with COUNT(*) and JOIN customers for names.",
        ],
        2: [
            "Hint: Join order_items with returns (on order_id AND product_id) to compute return rates per category.",
            "Hint: Compare Q1 (Jan-Mar) customers with Q3 (Jul-Sep) using subqueries to find churn.",
        ],
        3: [
            "Hint: Compare SUM(total_amount) between Q2 and Q3 for completed orders.",
            "Hint: Look at returns table joined with products — check category-level patterns in Q3.",
            "Hint: Check marketing_campaigns for Q3 coverage gaps — are there any campaigns starting in Jul-Sep?",
        ],
        4: [
            "Hint: Compare orders.total_amount + orders.discount_amount with SUM(order_items.subtotal) per order.",
            "Hint: Check products table for items where unit_price < cost_price.",
        ],
        5: [
            "Hint: Use strftime('%m', order_date) or substr(order_date, 6, 2) for monthly grouping.",
            "Hint: For cohort analysis, determine each customer's signup quarter, then check if they ordered in Q4.",
            "Hint: Group by channel and compute AVG(total_amount) and return rate separately.",
        ],
    }

    def _handle_hint_request(self) -> SqlAnalystObservation:
        """Return the next progressive hint with a small reward penalty."""
        hints = self.TASK_HINTS.get(self._task_id, [])
        if self._hints_given < len(hints):
            hint = hints[self._hints_given]
            self._hints_given += 1
            penalty = round(-0.02 * self._hints_given, 4)
            self._total_reward += penalty
            return SqlAnalystObservation(
                task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                schema_info="",
                task_id=self._task_id,
                step_number=self._state.step_count,
                max_steps=self._max_steps,
                done=False,
                reward=penalty,
                message=f"{hint} (hint penalty: {penalty:+.3f})",
            )
        return SqlAnalystObservation(
            task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
            schema_info="",
            task_id=self._task_id,
            step_number=self._state.step_count,
            max_steps=self._max_steps,
            done=False,
            reward=0.0,
            message="No more hints available for this task.",
        )

    def _handle_explain(self, sql: str) -> SqlAnalystObservation:
        """Run EXPLAIN QUERY PLAN on a SQL query to show execution strategy."""
        if not sql:
            return SqlAnalystObservation(
                task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                schema_info="",
                task_id=self._task_id,
                step_number=self._state.step_count,
                max_steps=self._max_steps,
                done=False,
                reward=0.0,
                error_message="Empty SQL query for EXPLAIN.",
                message="Please provide a SQL query to explain.",
            )
        try:
            cur = self._db_conn.cursor()
            cur.execute(f"EXPLAIN QUERY PLAN {sql}")
            plan_rows = cur.fetchall()
            plan_text = "\n".join(
                f"  {'  ' * row[1]}{row[3]}" for row in plan_rows
            )
            return SqlAnalystObservation(
                task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                schema_info="",
                task_id=self._task_id,
                step_number=self._state.step_count,
                max_steps=self._max_steps,
                done=False,
                reward=0.0,
                query_result=plan_text,
                message=f"Query execution plan (no step cost):\n{plan_text}",
            )
        except Exception as e:
            return SqlAnalystObservation(
                task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                schema_info="",
                task_id=self._task_id,
                step_number=self._state.step_count,
                max_steps=self._max_steps,
                done=False,
                reward=0.0,
                error_message=str(e),
                message=f"EXPLAIN failed: {e}",
            )

    # ── Framework interface ────────────────────────────────────────

    @property
    def state(self) -> State:
        """Get current environment state."""
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            task_id=self._task_id,
            queries_executed=self._queries_executed,
            total_reward=self._total_reward,
            done=self._done,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        """Return environment metadata for the /metadata endpoint."""
        readme_content = None
        try:
            import os
            readme_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md"
            )
            with open(readme_path, "r") as f:
                readme_content = f.read()
        except Exception:
            pass
        return EnvironmentMetadata(
            name="SQL Analyst Environment",
            description=(
                "An RL environment where AI agents write SQL queries against an "
                "e-commerce database to answer business intelligence questions of "
                "increasing difficulty (5 tasks: easy to hard)."
            ),
            version="0.2.0",
            author="Hari Harasudhan",
            documentation_url="https://huggingface.co/spaces/akv2011/sql-analyst-env",
            readme_content=readme_content,
        )

    def close(self) -> None:
        """Clean up the database connection."""
        if self._db_conn:
            try:
                self._db_conn.close()
            except Exception:
                pass
            self._db_conn = None

"""
The core environment — where the magic happens.

The agent gets a task, queries the database with SQL, and submits an analysis.
We grade the answer, factor in how efficiently the agent worked, and return
a reward. Three difficulty levels: easy, medium, and hard.
"""

import json
import sqlite3
from uuid import uuid4
from typing import Optional, Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

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

    Three tasks with increasing difficulty:
    - Task 1 (Easy): Basic data retrieval with single-table queries
    - Task 2 (Medium): Multi-table analysis with JOINs and aggregations
    - Task 3 (Hard): Complex BI analysis requiring multi-step reasoning
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Tables that are relevant to each task — used for intermediate rewards
    TASK_RELEVANT_TABLES = {
        1: {"orders", "customers"},
        2: {"orders", "order_items", "products", "returns", "customers"},
        3: {"orders", "order_items", "products", "returns", "marketing_campaigns", "campaign_attributions", "customers"},
    }

    # Map string task IDs to integers so evaluators can use either format
    TASK_ID_MAP = {
        "easy": 1, "1": 1, 1: 1,
        "medium": 2, "2": 2, 2: 2,
        "hard": 3, "3": 3, 3: 3,
    }

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = 1
        self._queries_executed = 0
        self._successful_queries = 0
        self._failed_queries = 0
        self._tables_queried: set = set()
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

        action_type = action.action_type.strip().lower()
        content = action.content.strip()

        if action_type == "execute_sql":
            return self._handle_sql(content)
        elif action_type == "submit_answer":
            return self._handle_submit(content)
        else:
            return SqlAnalystObservation(
                task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                schema_info=self._schema_info,
                task_id=self._task_id,
                step_number=self._state.step_count,
                max_steps=self._max_steps,
                done=False,
                reward=0.0,
                error_message=f"Unknown action_type '{action_type}'. Use 'execute_sql' or 'submit_answer'.",
                message="Invalid action type.",
            )

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
                message=f"Query executed successfully. {total_rows} rows returned. (reward: {reward:+.3f})",
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

    def _compute_query_reward(self, sql: str, row_count: int) -> float:
        """Compute intermediate reward for a successful SQL query.

        Rewards the agent for:
        - Querying tables relevant to the current task
        - Getting non-empty results (suggests a productive query)
        - Using aggregations (GROUP BY, COUNT, SUM, AVG — signs of analysis)
        """
        sql_upper = sql.upper()
        reward = 0.0

        # Reward for querying relevant tables (only first time each table is hit)
        relevant = self.TASK_RELEVANT_TABLES.get(self._task_id, set())
        all_tables = ["customers", "products", "orders", "order_items",
                       "returns", "marketing_campaigns", "campaign_attributions"]
        newly_discovered = set()
        for table in all_tables:
            if table.upper() in sql_upper and table not in self._tables_queried:
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
        efficiency = max(0.8, 1.0 - 0.02 * max(0, self._queries_executed - 5))

        # Apply step decay: reward faster solutions
        step_decay = max(0.85, 1.0 - 0.01 * self._state.step_count)

        final_reward = round(raw_score * efficiency * step_decay, 4)
        self._total_reward = final_reward

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
                f"Final reward: {final_reward:.4f}\n"
                f"Feedback: {feedback}"
            ),
        )

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

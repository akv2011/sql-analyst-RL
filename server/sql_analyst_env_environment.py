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

    def __init__(self):
        super().__init__()
        self._db_conn: Optional[sqlite3.Connection] = None
        self._ground_truth = {}
        self._schema_info = ""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = 1
        self._queries_executed = 0
        self._max_steps = 20
        self._done = False
        self._total_reward = 0.0

    def _init_db(self):
        """Initialize (or reinitialize) the database."""
        if self._db_conn:
            self._db_conn.close()
        self._db_conn, self._ground_truth = create_database()
        self._schema_info = get_schema_info(self._db_conn)

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
        self._task_id = kwargs.get("task_id", 1)
        if self._task_id not in (1, 2, 3):
            self._task_id = 1

        self._init_db()
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._queries_executed = 0
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

        # Block dangerous operations
        sql_upper = sql.upper().strip()
        if any(kw in sql_upper for kw in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]):
            return SqlAnalystObservation(
                task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                schema_info=self._schema_info,
                task_id=self._task_id,
                step_number=self._state.step_count,
                max_steps=self._max_steps,
                done=False,
                reward=0.0,
                error_message="Only SELECT queries are allowed. Data modification is not permitted.",
                message="Read-only environment. Use SELECT queries only.",
            )

        try:
            cur = self._db_conn.cursor()
            cur.execute(sql)
            rows = cur.fetchmany(100)  # Limit to 100 rows
            columns = [desc[0] for desc in cur.description] if cur.description else []

            result_data = [dict(zip(columns, row)) for row in rows]
            result_json = json.dumps(result_data, indent=2, default=str)

            total_rows = len(rows)
            if total_rows == 100:
                # Check if there are more
                extra = cur.fetchone()
                if extra:
                    total_rows = "100+ (truncated)"

            return SqlAnalystObservation(
                task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
                schema_info="",  # Don't repeat schema on every step
                query_result=result_json,
                columns=columns,
                row_count=len(rows),
                task_id=self._task_id,
                step_number=self._state.step_count,
                max_steps=self._max_steps,
                done=False,
                reward=0.0,
                message=f"Query executed successfully. {total_rows} rows returned.",
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
                message="SQL query failed. Check your syntax and table/column names.",
            )

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

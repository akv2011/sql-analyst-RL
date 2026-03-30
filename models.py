"""
Models for the SQL Analyst Environment.

These define what the agent sends (actions) and what it gets back (observations)
when interacting with the e-commerce database.
"""

from typing import Optional, List, Dict, Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SqlAnalystAction(Action):
    """Action the agent can take in the SQL Analyst environment."""

    action_type: str = Field(
        ...,
        description=(
            "Type of action: 'execute_sql' to run a query, 'submit_answer' to submit final answer, "
            "'request_schema' to view the database schema (free, no step cost), "
            "'request_hint' for a progressive hint (small reward penalty), "
            "or 'explain_sql' to see the query execution plan (free, no step cost)"
        ),
    )
    content: str = Field(
        ...,
        description="SQL query string (for execute_sql/explain_sql) or final answer text (for submit_answer)",
    )


class SqlAnalystObservation(Observation):
    """Observation returned after each action in the SQL Analyst environment."""

    task_description: str = Field(default="", description="Current task prompt")
    schema_info: str = Field(default="", description="Database schema description")
    query_result: Optional[str] = Field(
        default=None, description="JSON-serialized query results (if SQL was executed)"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if query failed"
    )
    row_count: Optional[int] = Field(
        default=None, description="Number of rows returned by query"
    )
    columns: Optional[List[str]] = Field(
        default=None, description="Column names from query result"
    )
    task_id: int = Field(default=1, description="Current task ID (1-5)")
    step_number: int = Field(default=0, description="Current step in the episode")
    max_steps: int = Field(default=20, description="Maximum steps allowed")
    message: str = Field(default="", description="Feedback message from environment")
    reward_breakdown: Optional[Dict[str, Any]] = Field(
        default=None, description="Detailed breakdown of reward components"
    )
    query_history: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Summary of queries executed this episode"
    )

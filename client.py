"""Client for connecting to a running SQL Analyst Environment server."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SqlAnalystAction, SqlAnalystObservation


class SqlAnalystEnv(
    EnvClient[SqlAnalystAction, SqlAnalystObservation, State]
):
    """Client for the SQL Analyst Environment.

    Example:
        >>> with SqlAnalystEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task_id=1)
        ...     print(result.observation.task_description)
        ...
        ...     result = client.step(SqlAnalystAction(
        ...         action_type="execute_sql",
        ...         content="SELECT COUNT(*) FROM orders"
        ...     ))
        ...     print(result.observation.query_result)
    """

    def _step_payload(self, action: SqlAnalystAction) -> Dict:
        return {
            "action_type": action.action_type,
            "content": action.content,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SqlAnalystObservation]:
        obs_data = payload.get("observation", {})
        observation = SqlAnalystObservation(
            task_description=obs_data.get("task_description", ""),
            schema_info=obs_data.get("schema_info", ""),
            query_result=obs_data.get("query_result"),
            error_message=obs_data.get("error_message"),
            row_count=obs_data.get("row_count"),
            columns=obs_data.get("columns"),
            task_id=obs_data.get("task_id", 1),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 20),
            message=obs_data.get("message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

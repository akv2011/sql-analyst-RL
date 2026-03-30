"""
Server entry point — wires up the environment with FastAPI.

This gives us /reset, /step, /state, /health, and WebSocket endpoints
out of the box via OpenEnv's create_app helper.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import SqlAnalystAction, SqlAnalystObservation
    from .sql_analyst_env_environment import SqlAnalystEnvironment
except (ImportError, ModuleNotFoundError):
    from models import SqlAnalystAction, SqlAnalystObservation
    from server.sql_analyst_env_environment import SqlAnalystEnvironment


app = create_app(
    SqlAnalystEnvironment,
    SqlAnalystAction,
    SqlAnalystObservation,
    env_name="sql_analyst_env",
    max_concurrent_envs=10,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

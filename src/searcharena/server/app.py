"""FastAPI application factory for SearchArena."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from openenv.core.env_server import HTTPEnvServer

from ..env import SearchEnvironment
from ..models import SearchAction, SearchObservation
from .environment import EnvironmentManager, create_environment


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Attach the environment manager for the app lifetime."""
    if not hasattr(app.state, "environment_manager"):
        factory = getattr(app.state, "environment_factory", create_environment)
        app.state.environment_manager = EnvironmentManager(factory=factory)
    yield
    app.state.environment_manager.close_all()


def create_app(
    factory: Callable[[], SearchEnvironment] | None = None,
) -> FastAPI:
    """Build the FastAPI app with OpenEnv-compatible transport routes."""
    app = FastAPI(
        title="SearchArena",
        description="A stateful search environment for RL training",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.environment_factory = factory or create_environment
    app.state.environment_manager = EnvironmentManager(factory=app.state.environment_factory)

    server = HTTPEnvServer(
        env=lambda: app.state.environment_manager.create_environment(),
        action_cls=SearchAction,
        observation_cls=SearchObservation,
    )
    server.register_routes(app)
    app.state.openenv_server = server

    @app.get("/")
    async def root() -> dict[str, Any]:
        """Welcome metadata for the service."""
        return {
            "name": "SearchArena",
            "description": "A stateful search environment for training retrieval agents.",
            "endpoints": {
                "GET /": "Service info",
                "GET /health": "Health check",
                "GET /schema": "Action, observation, and state schemas",
                "GET /state": "Current environment state",
                "GET /metadata": "Environment metadata",
                "GET /tasks": "Available tasks",
                "POST /reset": "Reset one episode",
                "POST /step": "Execute an action",
                "WS /ws": "Persistent OpenEnv session",
                "WS /mcp": "Persistent MCP session",
            },
        }

    @app.get("/tasks")
    async def list_tasks(request: Request) -> dict[str, Any]:
        """List the bundled tasks available from the current factory."""
        manager: EnvironmentManager = request.app.state.environment_manager
        return {"tasks": manager.list_task_summaries()}

    return app


app = create_app()


def main() -> None:
    """Run the server directly."""
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run("searcharena.server.app:app", host=args.host, port=args.port)

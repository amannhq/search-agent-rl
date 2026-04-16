"""FastAPI application factory for SearchArena."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .environment import EnvironmentManager
from .routers.episodes import router as episodes_router
from .routers.system import router as system_router
from .routers.tasks import router as tasks_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Attach the environment manager for the app lifetime."""
    app.state.environment_manager = EnvironmentManager()
    yield
    app.state.environment_manager.close_all()


def create_app() -> FastAPI:
    """Build the FastAPI app."""
    app = FastAPI(
        title="SearchArena",
        description="A stateful search environment for RL training",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.include_router(system_router)
    app.include_router(tasks_router)
    app.include_router(episodes_router)
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


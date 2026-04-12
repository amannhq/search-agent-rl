"""Server entrypoint for the search environment.

This provides a stateful HTTP server that maintains environment state
between requests. For production use with multiple concurrent users,
consider using WebSocket connections or separate environment instances per session.
"""

import argparse
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from searcharena import (
    SearchAction,
    SearchEnvConfig,
    SearchEnvironment,
    SearchObservation,
    create_sample_corpus,
    create_sample_tasks,
)


# Request/Response models
class ResetRequest(BaseModel):
    seed: int | None = None
    task_id: str | None = None


class ResetResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool


class StepRequest(BaseModel):
    action: SearchAction


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool


class HealthResponse(BaseModel):
    status: str


# Global environment instance
_env: SearchEnvironment | None = None


def create_environment() -> SearchEnvironment:
    """Create one environment instance."""
    config = SearchEnvConfig()
    return SearchEnvironment(
        config=config,
        corpus=create_sample_corpus(config),
        tasks=create_sample_tasks(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize environment on startup."""
    global _env
    _env = create_environment()
    yield
    if _env:
        _env.close()


app = FastAPI(
    title="Search RL Environment",
    description="A stateful search environment for RL training",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/")
async def root() -> dict[str, Any]:
    """Welcome page with API info."""
    return {
        "name": "Search RL Environment",
        "description": "A stateful search environment for RL training",
        "endpoints": {
            "GET /": "This welcome page",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /health": "Health check",
            "GET /tasks": "List available tasks",
            "GET /state": "Get current environment state",
            "GET /schema": "Get action/observation schemas",
            "POST /reset": "Reset environment and get first observation",
            "POST /step": "Execute an action (search/read/prune/answer)",
        },
        "usage": {
            "1_reset": "POST /reset to get a task",
            "2_search": "POST /step with action_type='search' to find documents",
            "3_read": "POST /step with action_type='read' to read chunks",
            "4_answer": "POST /step with action_type='answer' to submit answer",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@app.get("/schema")
async def schema() -> dict[str, Any]:
    """Get action and observation schemas."""
    return {
        "action": SearchAction.model_json_schema(),
        "observation": SearchObservation.model_json_schema(),
    }


@app.get("/state")
async def state() -> dict[str, Any]:
    """Get current environment state."""
    if _env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    return _env.state.model_dump()


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest | None = None) -> ResetResponse:
    """Reset the environment and get the first observation."""
    global _env
    if _env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    task_id = request.task_id if request else None
    obs = _env.reset(task_id=task_id)

    return ResetResponse(
        observation=obs.model_dump(),
        reward=0.0,
        done=False,
    )


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest) -> StepResponse:
    """Execute an action and get the resulting observation."""
    global _env
    if _env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    obs = _env.step(request.action)

    return StepResponse(
        observation=obs.model_dump(),
        reward=obs.reward if obs.reward is not None else 0.0,
        done=obs.done,
    )


@app.get("/metadata")
async def metadata() -> dict[str, Any]:
    """Get environment metadata."""
    if _env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    return _env.get_metadata().model_dump()


@app.get("/tasks")
async def list_tasks() -> dict[str, Any]:
    """List available tasks."""
    if _env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "level": t.level,
                "domain": t.domain,
                "question": t.question,
            }
            for t in _env.tasks
        ]
    }


def main() -> None:
    """Run the server directly."""
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

"""Task and state routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from ...env import SearchEnvironment
from ..dependencies import get_search_environment

router = APIRouter(tags=["tasks"])


@router.get("/state")
async def state(
    env: Annotated[SearchEnvironment, Depends(get_search_environment)],
) -> dict[str, Any]:
    """Return current OpenEnv state."""
    return env.state.model_dump()


@router.get("/metadata")
async def metadata(
    env: Annotated[SearchEnvironment, Depends(get_search_environment)],
) -> dict[str, Any]:
    """Return environment metadata."""
    return env.get_metadata().model_dump()


@router.get("/tasks")
async def list_tasks(
    env: Annotated[SearchEnvironment, Depends(get_search_environment)],
) -> dict[str, Any]:
    """List tasks available for the current session."""
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "level": task.level,
                "domain": task.domain,
                "question": task.question,
            }
            for task in env.tasks
        ]
    }

"""Episode lifecycle routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ...env import SearchEnvironment
from ..dependencies import get_search_environment
from ..environment import ResetRequest, ResetResponse, StepRequest, StepResponse

router = APIRouter(tags=["episodes"])


@router.post("/reset", response_model=ResetResponse)
async def reset(
    env: Annotated[SearchEnvironment, Depends(get_search_environment)],
    request: ResetRequest | None = None,
) -> ResetResponse:
    """Reset the current session."""
    task = None
    if request and request.task_id:
        task = next((item for item in env.tasks if item.task_id == request.task_id), None)
        if task is None:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found: {request.task_id}",
            )

    observation = env.reset(seed=request.seed if request else None, task=task)
    reward = float(observation.reward if observation.reward is not None else 0.0)
    return ResetResponse(
        observation=observation.model_dump(),
        reward=reward,
        done=observation.done,
    )


@router.post("/step", response_model=StepResponse)
async def step(
    request: StepRequest,
    env: Annotated[SearchEnvironment, Depends(get_search_environment)],
) -> StepResponse:
    """Execute one action for the current session."""
    observation = env.step(request.action)
    reward = float(observation.reward if observation.reward is not None else 0.0)
    return StepResponse(
        observation=observation.model_dump(),
        reward=reward,
        done=observation.done,
    )

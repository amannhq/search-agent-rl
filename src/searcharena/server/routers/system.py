"""System-level server routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ...models import SearchAction, SearchObservation
from ..environment import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/")
async def root() -> dict[str, Any]:
    """Welcome metadata for the service."""
    return {
        "name": "SearchArena",
        "description": "A stateful search environment for training retrieval agents.",
        "endpoints": {
            "GET /": "Service info",
            "GET /health": "Health check",
            "GET /schema": "Action and observation schemas",
            "GET /state": "Current session state",
            "GET /metadata": "Environment metadata",
            "GET /tasks": "Available tasks",
            "POST /reset": "Reset one session",
            "POST /step": "Execute an action",
        },
    }


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return health status."""
    return HealthResponse(status="healthy")


@router.get("/schema")
async def schema() -> dict[str, Any]:
    """Return action and observation JSON schemas."""
    return {
        "action": SearchAction.model_json_schema(),
        "observation": SearchObservation.model_json_schema(),
    }

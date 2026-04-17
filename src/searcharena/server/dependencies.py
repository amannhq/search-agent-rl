"""FastAPI dependencies for SearchArena."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from .environment import EnvironmentManager


def get_environment_manager(request: Request) -> EnvironmentManager:
    """Return the app-scoped environment manager."""
    return request.app.state.environment_manager


def get_session_id(
    x_session_id: Annotated[str | None, Header(alias="X-Session-Id")] = None,
) -> str:
    """Resolve the current client session id."""
    return x_session_id or "default"


def get_search_environment(
    session_id: Annotated[str, Depends(get_session_id)],
    manager: Annotated[EnvironmentManager, Depends(get_environment_manager)],
):
    """Resolve the request-scoped environment instance."""
    return manager.get(session_id)

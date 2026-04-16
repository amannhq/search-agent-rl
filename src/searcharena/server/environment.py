"""Server-side environment factory and shared response models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from ..env import SearchEnvironment
from ..models import SearchAction, SearchEnvConfig
from ..tasks import create_sample_corpus, create_sample_tasks


class ResetRequest(BaseModel):
    """Reset payload."""

    seed: int | None = None
    task_id: str | None = None


class ResetResponse(BaseModel):
    """Reset response payload."""

    observation: dict[str, Any]
    reward: float
    done: bool


class StepRequest(BaseModel):
    """Step payload."""

    action: SearchAction


class StepResponse(BaseModel):
    """Step response payload."""

    observation: dict[str, Any]
    reward: float
    done: bool


class HealthResponse(BaseModel):
    """Health response payload."""

    status: str


def create_environment() -> SearchEnvironment:
    """Create a fresh environment instance."""
    config = SearchEnvConfig()
    return SearchEnvironment(
        config=config,
        corpus=create_sample_corpus(config),
        tasks=create_sample_tasks(),
    )


class EnvironmentManager:
    """App-level environment manager with per-session isolation."""

    def __init__(self, factory: Callable[[], SearchEnvironment] = create_environment):
        self.factory = factory
        self._environments: dict[str, SearchEnvironment] = {}

    def get(self, session_id: str = "default") -> SearchEnvironment:
        """Fetch or create an environment for one session."""
        env = self._environments.get(session_id)
        if env is None:
            env = self.factory()
            self._environments[session_id] = env
        return env

    def list_task_summaries(self, session_id: str = "default") -> list[dict[str, Any]]:
        """Return task metadata for one session."""
        env = self.get(session_id)
        return [
            {
                "task_id": task.task_id,
                "level": task.level,
                "domain": task.domain,
                "question": task.question,
            }
            for task in env.tasks
        ]

    def close_all(self) -> None:
        """Close every managed environment."""
        for env in self._environments.values():
            env.close()
        self._environments.clear()

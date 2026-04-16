"""
Server environment wrapper.

This is a thin wrapper - all logic lives in searcharena.engine.
Follows OpsArena pattern: server/ only contains the OpenEnv interface.
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.interfaces import Environment, EnvironmentMetadata
from openenv.core.env_server.types import State

from searcharena.engine import (
    SearchEnvironment as _SearchEnvironment,
    create_sample_corpus,
    create_sample_tasks,
)
from searcharena.models import SearchAction, SearchEnvConfig, SearchObservation


class SearchEnvironment(Environment[SearchAction, SearchObservation, State]):
    """
    OpenEnv-compatible wrapper for SearchArena environment.

    This thin wrapper delegates all logic to searcharena.engine.SearchEnvironment.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(
        self,
        config: SearchEnvConfig | None = None,
        corpus: Any | None = None,
        tasks: list | None = None,
    ):
        super().__init__()
        self._env = _SearchEnvironment(config=config, corpus=corpus, tasks=tasks)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> SearchObservation:
        return self._env.reset(seed=seed, episode_id=episode_id, **kwargs)

    def step(
        self,
        action: SearchAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> SearchObservation:
        return self._env.step(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._env.state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="SearchArena",
            description="Multi-hop document retrieval environment for training search agents.",
            version="0.1.0",
        )


__all__ = [
    "SearchEnvironment",
    "create_sample_corpus",
    "create_sample_tasks",
]

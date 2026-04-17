"""Tests for the FastAPI server wrapper behavior."""

from __future__ import annotations

import asyncio
import importlib
from typing import Any, Protocol, cast

from searcharena import SearchEnvironment


class ServerAppModule(Protocol):
    _env: SearchEnvironment | None

    async def reset(self, request: Any | None = None) -> Any: ...


def test_reset_response_preserves_terminal_observation(
    empty_corpus,
    monkeypatch,
) -> None:
    """The reset endpoint should mirror a terminal observation from the environment."""
    server_app = cast(ServerAppModule, importlib.import_module("server.app"))
    env = SearchEnvironment(corpus=empty_corpus, tasks=[])
    monkeypatch.setattr(server_app, "_env", env)

    response = asyncio.run(server_app.reset())

    assert response.done is True
    assert response.reward == 0.001
    assert response.observation["done"] is True
    assert response.observation["reward"] == 0.001
    assert "No tasks available" in response.observation["question"]

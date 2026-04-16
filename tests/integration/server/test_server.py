"""Tests for the FastAPI server behavior."""

from __future__ import annotations

from fastapi.testclient import TestClient

from searcharena import SearchEnvironment
from searcharena.server import EnvironmentManager, create_app


def test_reset_response_preserves_terminal_observation(
    empty_corpus,
) -> None:
    """The reset endpoint should mirror a terminal observation from the environment."""
    app = create_app()

    with TestClient(app) as client:
        app.state.environment_manager = EnvironmentManager(
            lambda: SearchEnvironment(corpus=empty_corpus, tasks=[])
        )
        response = client.post("/reset")

    assert response.status_code == 200
    body = response.json()
    assert body["done"] is True
    assert body["reward"] == 0.001
    assert body["observation"]["done"] is True
    assert body["observation"]["reward"] == 0.001
    assert body["observation"]["terminated"] is True
    assert body["observation"]["termination_reason"] == "no_tasks"
    assert "No tasks available" in body["observation"]["question"]

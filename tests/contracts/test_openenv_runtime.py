"""Runtime contract checks for the packaged FastAPI app."""

from __future__ import annotations

from fastapi.testclient import TestClient

from searcharena.server import create_app


def test_app_health_and_schema() -> None:
    """The packaged app should expose basic runtime endpoints."""
    with TestClient(create_app()) as client:
        health = client.get("/health")
        schema = client.get("/schema")

    assert health.status_code == 200
    assert health.json() == {"status": "healthy"}
    assert schema.status_code == 200
    assert "action" in schema.json()
    assert "observation" in schema.json()

from __future__ import annotations

from apps.api.main import app
from fastapi.testclient import TestClient


def test_health_returns_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["service"]


def test_ready_returns_ready_after_startup() -> None:
    # Using `with` triggers the FastAPI lifespan (startup ingests the
    # knowledge base), matching how the app actually runs in production —
    # a bare TestClient() without `with` never runs startup and would
    # report "degraded", which is correct behavior, not a bug.
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["checks"]["knowledge_base"] == "ok"


def test_root() -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["name"] == "VoiceOps AI"

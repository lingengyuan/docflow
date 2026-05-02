from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app as api_app


def _patch_health_checks(
    monkeypatch,
    sqlite_status="ok",
    qdrant_status="ok",
    ollama_status="ok",
    models_status="ok",
):
    monkeypatch.setattr(
        api_app,
        "_check_sqlite",
        lambda cfg: {"status": sqlite_status, "quick_check": "ok"},
    )
    monkeypatch.setattr(
        api_app,
        "_check_qdrant",
        lambda cfg: {"status": qdrant_status, "collection": "docflow", "points_count": 1},
    )
    monkeypatch.setattr(
        api_app,
        "_check_ollama",
        lambda cfg: {
            "status": ollama_status,
            "models": {
                "ocr": {"model": "glm-ocr", "available": ollama_status == "ok"},
                "contextual_prefix": {"model": "qwen2.5:7b", "available": ollama_status == "ok"},
            },
            "missing_models": [] if ollama_status == "ok" else ["glm-ocr"],
        },
    )
    monkeypatch.setattr(
        api_app,
        "_check_models",
        lambda cfg: {
            "status": models_status,
            "local_cache": {},
            "missing_local_cache": [] if models_status == "ok" else ["mlx-community/Qwen3-4B-4bit"],
        },
    )


def test_health_returns_ok_when_all_checks_pass(monkeypatch):
    _patch_health_checks(monkeypatch)
    client = TestClient(api_app.app)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["sqlite"]["status"] == "ok"
    assert body["checks"]["qdrant"]["status"] == "ok"
    assert body["capabilities"]["query"] is True
    assert body["capabilities"]["contextual_prefix"] is False
    assert body["capabilities"]["contextual_prefix_enabled"] is False


def test_health_is_unavailable_when_critical_dependency_fails(monkeypatch):
    _patch_health_checks(monkeypatch, qdrant_status="unavailable")
    client = TestClient(api_app.app)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["capabilities"]["query"] is False
    assert body["capabilities"]["ingest"] is False


def test_health_is_degraded_when_optional_dependency_fails(monkeypatch):
    _patch_health_checks(monkeypatch, ollama_status="unavailable")
    client = TestClient(api_app.app)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["capabilities"]["query"] is True
    assert body["capabilities"]["ocr"] is False


def test_health_catches_check_exceptions(monkeypatch):
    _patch_health_checks(monkeypatch)

    def broken_sqlite(cfg):
        raise RuntimeError("database locked")

    monkeypatch.setattr(api_app, "_check_sqlite", broken_sqlite)
    client = TestClient(api_app.app)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["checks"]["sqlite"]["status"] == "unavailable"
    assert "database locked" in body["checks"]["sqlite"]["error"]

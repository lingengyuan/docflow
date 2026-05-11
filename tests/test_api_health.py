from __future__ import annotations

import httpx
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
            "local_cache": {
                "embedding": {
                    "model": "Qwen/Qwen3-Embedding-0.6B",
                    "cached": models_status == "ok",
                },
                "reranker": {"model": "Qwen/Qwen3-Reranker-0.6B", "cached": models_status == "ok"},
                "llm": {"model": "mlx-community/Qwen3-4B-4bit", "cached": models_status == "ok"},
                "llm_enhanced": {
                    "model": "mlx-community/Qwen3-8B-4bit",
                    "cached": models_status == "ok",
                },
                "vlm": {
                    "model": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
                    "cached": models_status == "ok",
                },
            },
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
    assert body["groups"]["core"]["label"] == "核心功能"
    assert [item["key"] for item in body["groups"]["core"]["items"]] == [
        "query",
        "ingest",
        "sqlite",
        "qdrant",
    ]
    assert body["groups"]["optional"]["label"] == "可选能力"
    assert body["groups"]["runtime"]["label"] == "模型运行时"
    assert [item["key"] for item in body["groups"]["runtime"]["items"]] == [
        "embedding",
        "reranker",
        "llm",
        "llm_enhanced",
        "ocr_runtime",
        "vlm",
    ]
    assert body["actions"][0]["command"] == "python main.py check --json"


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
    ocr_item = next(item for item in body["groups"]["optional"]["items"] if item["key"] == "ocr")
    assert ocr_item["status"] == "optional_unavailable"
    assert "只影响扫描 PDF" in ocr_item["detail"]
    assert any("ollama pull glm-ocr" in action for action in ocr_item["actions"])
    assert any(action["command"] == "ollama pull glm-ocr" for action in body["actions"])


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
    assert any(action["command"] == "python main.py doctor --strict" for action in body["actions"])


def test_health_runtime_group_reports_missing_model_cache(monkeypatch):
    _patch_health_checks(monkeypatch, models_status="degraded")
    client = TestClient(api_app.app)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    runtime = {item["key"]: item for item in body["groups"]["runtime"]["items"]}
    assert runtime["llm"]["status"] == "degraded"
    assert "本地缓存缺失" in runtime["llm"]["detail"]
    assert any("mlx-community/Qwen3-4B-4bit" in action["label"] for action in body["actions"])


def test_ollama_check_reports_guidance_when_service_is_closed(monkeypatch):
    def broken_get(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", broken_get)

    result = api_app._check_ollama(
        {
            "ollama": {"base_url": "http://localhost:11434", "ocr_model": "glm-ocr"},
            "ingest": {"contextual_prefix_mode": "metadata"},
            "llm": {"backend": "mlx"},
        }
    )

    assert result["status"] == "degraded"
    assert result["models"]["ocr"] == {"model": "glm-ocr", "available": False}
    assert result["missing_models"] == ["glm-ocr"]
    assert "打开 Ollama" in result["actions"][0]


def test_runtime_sqlite_health_skips_deep_quick_check(monkeypatch):
    statements = []

    class FakeCursor:
        def __init__(self, rows=None):
            self.rows = rows or []

        def fetchone(self):
            return self.rows[0] if self.rows else {"value": 1}

        def fetchall(self):
            return self.rows

    class FakeConn:
        def execute(self, sql, params=()):
            statements.append(sql)
            if "sqlite_master" in sql:
                return FakeCursor(
                    [
                        {"name": "chunks_fts"},
                        {"name": "chunks_fts_trigram"},
                        {"name": "history_fts"},
                    ]
                )
            return FakeCursor()

        def commit(self):
            pass

        def rollback(self):
            pass

    class FakeStore:
        def _conn(self):
            class ConnContext:
                def __enter__(self):
                    return FakeConn()

                def __exit__(self, exc_type, exc, tb):
                    return False

            return ConnContext()

    monkeypatch.setattr(api_app, "store", FakeStore())

    result = api_app._check_sqlite({"paths": {"db_path": "unused.db"}})

    assert result["status"] == "ok"
    assert result["mode"] == "runtime"
    assert result["missing_fts_tables"] == []
    assert result["quick_check"] == "skipped during app runtime"
    assert not any("PRAGMA quick_check" in statement for statement in statements)

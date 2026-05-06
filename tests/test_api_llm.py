from __future__ import annotations

import time

from fastapi.testclient import TestClient

from src.api import app as api_app
from src.api.model_tasks import ModelTaskController


class FakeGenerator:
    backend = "mlx"

    def __init__(self, current="mlx-community/Qwen3-4B-4bit", delay=0.0):
        self.mlx_model_name = current
        self.delay = delay

    @property
    def current_model(self):
        return self.mlx_model_name


class FakeQueryEngine:
    def __init__(self, generator):
        self.generator = generator


def test_llm_endpoint_reports_model_availability(monkeypatch):
    generator = FakeGenerator()
    monkeypatch.setattr(api_app, "query_engine", FakeQueryEngine(generator))
    monkeypatch.setattr(
        api_app,
        "llm_options",
        ["mlx-community/Qwen3-4B-4bit", "mlx-community/Qwen3-8B-4bit"],
    )
    monkeypatch.setattr(api_app, "_is_hf_model_cached", lambda model: model.endswith("4B-4bit"))
    client = TestClient(api_app.app)

    response = client.get("/api/llm")

    assert response.status_code == 200
    body = response.json()
    assert body["current"] == "mlx-community/Qwen3-4B-4bit"
    assert body["models"][0]["current"] is True
    assert body["models"][0]["available"] is True
    assert body["models"][0]["detail"] == "本地缓存可用。"
    assert body["models"][1]["available"] is False
    assert body["models"][1]["actions"] == ["联网后准备模型缓存：mlx-community/Qwen3-8B-4bit"]


def test_llm_switch_unknown_model_returns_400(monkeypatch):
    monkeypatch.setattr(api_app, "query_engine", FakeQueryEngine(FakeGenerator()))
    monkeypatch.setattr(api_app, "llm_options", ["mlx-community/Qwen3-4B-4bit"])
    client = TestClient(api_app.app)

    response = client.post("/api/llm", json={"model": "unknown-model"})

    assert response.status_code == 400


def test_llm_switch_current_model_is_idempotent(monkeypatch):
    generator = FakeGenerator()
    monkeypatch.setattr(api_app, "query_engine", FakeQueryEngine(generator))
    monkeypatch.setattr(api_app, "llm_options", ["mlx-community/Qwen3-4B-4bit"])
    monkeypatch.setattr(
        api_app,
        "_load_mlx_model_candidate",
        lambda model: (_ for _ in ()).throw(AssertionError("current model should not load")),
    )
    client = TestClient(api_app.app)

    response = client.post("/api/llm", json={"model": "mlx-community/Qwen3-4B-4bit"})

    assert response.status_code == 200
    assert response.json()["unchanged"] is True


def test_llm_switch_success_applies_loaded_candidate(monkeypatch):
    generator = FakeGenerator()
    monkeypatch.setattr(api_app, "query_engine", FakeQueryEngine(generator))
    monkeypatch.setattr(
        api_app,
        "llm_options",
        ["mlx-community/Qwen3-4B-4bit", "mlx-community/Qwen3-8B-4bit"],
    )
    monkeypatch.setattr(api_app, "_is_hf_model_cached", lambda model: True)
    monkeypatch.setattr(api_app, "_load_mlx_model_candidate", lambda model: ("model-object", "tokenizer"))
    client = TestClient(api_app.app)

    response = client.post("/api/llm", json={"model": "mlx-community/Qwen3-8B-4bit"})

    assert response.status_code == 200
    assert generator.current_model == "mlx-community/Qwen3-8B-4bit"
    assert generator._mlx_model == "model-object"
    assert generator._mlx_tokenizer == "tokenizer"


def test_llm_switch_timeout_preserves_current_model(monkeypatch):
    controller = ModelTaskController(thread_name_prefix="test-api-model-task")
    generator = FakeGenerator(delay=0.2)
    monkeypatch.setattr(api_app, "model_tasks", controller)
    monkeypatch.setattr(api_app, "MODEL_TASK_TIMEOUT_S", 0.02)
    monkeypatch.setattr(api_app, "query_engine", FakeQueryEngine(generator))
    monkeypatch.setattr(
        api_app,
        "llm_options",
        ["mlx-community/Qwen3-4B-4bit", "mlx-community/Qwen3-8B-4bit"],
    )
    monkeypatch.setattr(api_app, "_is_hf_model_cached", lambda model: True)
    monkeypatch.setattr(
        api_app,
        "_load_mlx_model_candidate",
        lambda model: (time.sleep(0.2), ("late-model", "late-tokenizer"))[1],
    )
    client = TestClient(api_app.app)

    try:
        response = client.post("/api/llm", json={"model": "mlx-community/Qwen3-8B-4bit"})

        assert response.status_code == 504
        assert "模型任务超时" in response.json()["detail"]
        assert generator.current_model == "mlx-community/Qwen3-4B-4bit"
        assert api_app.llm_switch_state["state"] == "error"
    finally:
        controller.shutdown()

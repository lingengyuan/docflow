from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from src.api import app as api_app


def test_storage_usage_endpoint_reports_real_local_usage(monkeypatch, tmp_path):
    source = tmp_path / "source.md"
    source.write_text("hello", encoding="utf-8")
    model_dir = tmp_path / "model-cache"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"model")
    db_path = tmp_path / "docflow.db"
    db_path.write_bytes(b"database")
    counter_path = tmp_path / "qdrant_id_counter.txt"
    counter_path.write_text("7", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {"db_path": str(db_path), "id_counter": str(counter_path)},
                "embedding": {},
                "reranker": {},
                "llm": {},
                "ollama": {},
                "vlm": {},
                "ingest": {},
            }
        ),
        encoding="utf-8",
    )
    usage = namedtuple("usage", "total used free")

    class FakeStore:
        def list_files(self):
            return [
                {"file_path": str(source), "collection": "Inbox"},
                {"file_path": str(tmp_path / "missing.md"), "collection": "Research"},
            ]

    monkeypatch.setattr(api_app, "CONFIG_PATH", config_path)
    monkeypatch.setattr(api_app, "store", FakeStore())
    monkeypatch.setattr(
        api_app.shutil, "disk_usage", lambda path: usage(total=1000, used=400, free=600)
    )
    monkeypatch.setattr(api_app, "_configured_model_cache_paths", lambda cfg: [model_dir])
    client = TestClient(api_app.app)

    response = client.get("/api/storage/usage")

    assert response.status_code == 200
    body = response.json()
    categories = {item["id"]: item for item in body["categories"]}
    assert body["disk"] == {
        "path": str(Path.home()),
        "total_bytes": 1000,
        "used_bytes": 400,
        "free_bytes": 600,
        "used_percent": 40.0,
    }
    assert categories["library"]["bytes"] == 5
    assert categories["models"]["bytes"] == 5
    assert categories["app_data"]["bytes"] == 9
    assert categories["other"]["bytes"] == 381
    assert body["library"] == {
        "file_count": 2,
        "existing_file_count": 1,
        "missing_file_count": 1,
        "collection_count": 2,
    }

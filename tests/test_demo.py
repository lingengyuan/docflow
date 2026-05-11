from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from src.api import app as api_app
from src.maintenance.demo import create_demo_files


def _write_demo_config(tmp_path: Path) -> Path:
    config = {
        "paths": {
            "watch_dirs": [{"path": "watch", "recursive": True, "extensions": [".md", ".py"]}],
            "db_path": "data/docflow.db",
            "id_counter": "data/qdrant_id_counter.txt",
        },
        "qdrant": {"host": "localhost", "port": 6333, "collection": "docflow"},
        "embedding": {"model": "demo", "backend": "torch", "batch_size": 1},
        "chunking": {"chunk_size": 512, "chunk_overlap": 51},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_create_demo_files_writes_small_first_run_library(tmp_path):
    config_path = _write_demo_config(tmp_path)

    result = create_demo_files(config_path)

    demo_dir = tmp_path / "watch" / "DocFlow Demo"
    assert result["status"] == "created"
    assert result["file_count"] == 4
    assert demo_dir.exists()
    assert (
        (demo_dir / "docflow-overview.md")
        .read_text(encoding="utf-8")
        .startswith("# DocFlow Demo Overview")
    )

    second = create_demo_files(config_path)
    assert {item["status"] for item in second["files"]} == {"unchanged"}


def test_demo_endpoint_creates_files_and_queues_them(monkeypatch, tmp_path):
    config_path = _write_demo_config(tmp_path)
    queued = []

    class FakeQueue:
        def submit(self, path: Path):
            queued.append(path)
            return {"status": "queued", "file": path.name}

    monkeypatch.setattr(api_app, "CONFIG_PATH", config_path)
    monkeypatch.setattr(api_app, "ingest_queue", FakeQueue())
    client = TestClient(api_app.app)

    response = client.post("/api/demo")

    assert response.status_code == 200
    body = response.json()
    assert body["queued"] == 4
    assert len(queued) == 4
    assert all(path.exists() for path in queued)

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app as api_app


def test_files_api_passes_library_filters(monkeypatch):
    calls = {}

    class FakeStore:
        def list_files(self, **kwargs):
            calls["filters"] = kwargs
            return [{"id": 1, "file_name": "note.md"}]

    monkeypatch.setattr(api_app, "store", FakeStore())
    client = TestClient(api_app.app)

    response = client.get("/api/files?status=done&collection=Research&tag=paper&favorite=true")

    assert response.status_code == 200
    assert calls["filters"] == {
        "status": "done",
        "collection": "Research",
        "tag": "paper",
        "favorite": True,
    }


def test_batch_file_metadata_endpoint(monkeypatch):
    class FakeStore:
        def update_files_metadata(self, file_ids, collection=None, user_tags=None):
            return [
                {
                    "id": file_ids[0],
                    "collection": collection,
                    "user_tags": user_tags,
                }
            ]

    monkeypatch.setattr(api_app, "store", FakeStore())
    client = TestClient(api_app.app)

    response = client.post(
        "/api/files/batch/metadata",
        json={"file_ids": [7], "collection": "Research", "user_tags": ["paper"]},
    )

    assert response.status_code == 200
    assert response.json()["files"][0] == {
        "id": 7,
        "collection": "Research",
        "user_tags": ["paper"],
    }


def test_batch_rebuild_queues_existing_files(monkeypatch, tmp_path):
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")
    queued = {}

    class FakeStore:
        def get_file_by_id(self, file_id):
            if file_id == 1:
                return {"id": 1, "file_path": str(source), "file_name": source.name}
            return None

    class FakeQueue:
        def submit_many(self, paths: list[Path]):
            queued["paths"] = paths
            return {"queued": len(paths)}

    monkeypatch.setattr(api_app, "store", FakeStore())
    monkeypatch.setattr(api_app, "ingest_queue", FakeQueue())
    client = TestClient(api_app.app)

    response = client.post("/api/files/batch/rebuild", json={"file_ids": [1, 999]})

    assert response.status_code == 200
    assert response.json()["queued"] == 1
    assert response.json()["missing_ids"] == [999]
    assert queued["paths"] == [source]

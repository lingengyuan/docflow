from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app as api_app
from src.ingest.store import DocStore
from src.query.generator import Answer


class ScopeQueryEngine:
    def __init__(self):
        self.calls = []

    def query(
        self,
        question,
        file_filter=None,
        retrieval_mode="hybrid",
        conversation_context=None,
        retrieval_query=None,
    ):
        self.calls.append(
            {
                "question": question,
                "file_filter": file_filter,
                "retrieval_mode": retrieval_mode,
                "retrieval_query": retrieval_query,
            }
        )
        return Answer(text="scoped answer", citations=[])


def _add_done_file(store: DocStore, path: Path, collection: str = "Inbox") -> int:
    path.write_text("body", encoding="utf-8")
    file_id = store.upsert_file(
        path,
        path.name,
        DocStore.compute_hash(path),
        status="done",
        total_pages=1,
        mtime_ns=path.stat().st_mtime_ns,
    )
    store.update_file_metadata(file_id, collection=collection)
    return file_id


def test_query_collection_scope_resolves_to_indexed_file_names(monkeypatch, tmp_path):
    active_store = DocStore(tmp_path / "docflow.db")
    _add_done_file(active_store, tmp_path / "research.md", collection="Research")
    _add_done_file(active_store, tmp_path / "inbox.md", collection="Inbox")
    fake_engine = ScopeQueryEngine()
    monkeypatch.setattr(api_app, "store", active_store)
    monkeypatch.setattr(api_app, "query_engine", fake_engine)
    client = TestClient(api_app.app)

    response = client.post(
        "/api/query",
        json={"question": "只问研究集合", "scope_mode": "collection", "collection": "Research"},
    )

    assert response.status_code == 200
    assert fake_engine.calls[-1]["file_filter"] == ["research.md"]
    assert fake_engine.calls[-1]["retrieval_mode"] == "hybrid"
    assert response.json()["scope"]["mode"] == "collection"


def test_query_file_scope_resolves_file_id(monkeypatch, tmp_path):
    active_store = DocStore(tmp_path / "docflow.db")
    file_id = _add_done_file(active_store, tmp_path / "single.md")
    fake_engine = ScopeQueryEngine()
    monkeypatch.setattr(api_app, "store", active_store)
    monkeypatch.setattr(api_app, "query_engine", fake_engine)
    client = TestClient(api_app.app)

    response = client.post(
        "/api/query",
        json={"question": "只问这个文件", "scope_mode": "file", "file_id": file_id},
    )

    assert response.status_code == 200
    assert fake_engine.calls[-1]["file_filter"] == ["single.md"]
    assert response.json()["scope"]["file_id"] == file_id


def test_query_full_text_scope_forces_full_text_mode(monkeypatch, tmp_path):
    active_store = DocStore(tmp_path / "docflow.db")
    fake_engine = ScopeQueryEngine()
    monkeypatch.setattr(api_app, "store", active_store)
    monkeypatch.setattr(api_app, "query_engine", fake_engine)
    client = TestClient(api_app.app)

    response = client.post(
        "/api/query",
        json={"question": "精确关键词", "scope_mode": "full_text"},
    )

    assert response.status_code == 200
    assert fake_engine.calls[-1]["file_filter"] is None
    assert fake_engine.calls[-1]["retrieval_mode"] == "full_text"
    assert response.json()["scope"]["retrieval_mode"] == "full_text"


def test_query_collection_scope_reports_empty_collection(monkeypatch, tmp_path):
    active_store = DocStore(tmp_path / "docflow.db")
    monkeypatch.setattr(api_app, "store", active_store)
    monkeypatch.setattr(api_app, "query_engine", ScopeQueryEngine())
    client = TestClient(api_app.app)

    response = client.post(
        "/api/query",
        json={"question": "没有文件", "scope_mode": "collection", "collection": "Missing"},
    )

    assert response.status_code == 404
    assert "No indexed files found" in response.json()["detail"]

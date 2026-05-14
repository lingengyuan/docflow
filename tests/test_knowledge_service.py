from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app as api_app
from src.api.services.knowledge_service import KnowledgeService
from src.domain_types import FileStatus
from src.ingest.store import DocStore


def _add_file(store: DocStore, root: Path, name: str, text: str, collection: str = "Inbox") -> int:
    path = root / name
    path.write_text(text, encoding="utf-8")
    file_id = store.upsert_file(
        path,
        name,
        DocStore.compute_hash(path),
        status=FileStatus.DONE,
        total_pages=1,
        mtime_ns=path.stat().st_mtime_ns,
    )
    store.update_file_metadata(file_id, collection=collection, user_tags=["phase57"])
    store.add_chunks(
        file_id,
        [
            {
                "qdrant_id": file_id,
                "chunk_type": "text",
                "page_num": 1,
                "section": "Overview",
                "char_count": len(text),
                "raw_text": text,
                "embedding_text": text,
                "tokenized_text": text.lower(),
            }
        ],
    )
    return file_id


def test_knowledge_service_derives_topics_similar_documents_and_cards(tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    try:
        _add_file(
            store,
            tmp_path,
            "privacy-roadmap.md",
            "privacy local offline model cache source control",
        )
        _add_file(
            store,
            tmp_path,
            "privacy-check.md",
            "privacy local offline checks source preview",
        )
        history_id = store.add_history("Is privacy local?", "Yes", citations_json="[]")
        store.set_answer_feedback(history_id, "useful")

        overview = KnowledgeService().overview(store)

        assert overview["topics"]
        assert overview["similar_documents"]
        assert overview["knowledge_cards"]
        assert overview["knowledge_cards"][0]["source_file"]["file_name"]
        assert overview["knowledge_graph"]["nodes"]
        assert overview["knowledge_graph"]["edges"]
        assert overview["feedback"]["useful"] == 1
    finally:
        store.close()


def test_knowledge_overview_api_uses_current_store(monkeypatch, tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    try:
        file_id = _add_file(
            store,
            tmp_path,
            "knowledge-card.md",
            "knowledge cards connect answers notes and source files",
            collection="Knowledge Outputs",
        )
        monkeypatch.setattr(api_app, "store", store)

        client = TestClient(api_app.app)
        response = client.get(f"/api/knowledge/overview?file_id={file_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["stats"]["files"] == 1
        assert body["knowledge_cards"][0]["source_file"]["id"] == file_id
        assert body["backlinks"] == []
        assert body["knowledge_graph"]["stats"]["nodes"] >= 1
    finally:
        store.close()

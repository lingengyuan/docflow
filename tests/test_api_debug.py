from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app as api_app


class FakeStore:
    def get_file_by_id(self, file_id):
        if file_id != 1:
            return None
        return {
            "id": 1,
            "file_name": "README.md",
            "file_path": "/tmp/README.md",
            "status": "done",
        }

    def list_file_chunks(self, file_id):
        return [
            {
                "id": 10,
                "file_id": file_id,
                "qdrant_id": 100,
                "chunk_type": "text",
                "page_num": 1,
                "section": "Intro",
                "char_count": 42,
                "created_at": "2026-05-02",
            }
        ]


class FakeRetriever:
    def fetch_chunks_by_ids(self, qdrant_ids, max_text_chars=500):
        return {
            100: {
                "text_preview": "DocFlow local private document QA",
                "text_length": 34,
            }
        }

    def debug_retrieve(
        self,
        question,
        file_filter=None,
        prefer_tables=False,
        include_rerank=True,
        max_text_chars=300,
    ):
        return {
            "query": question,
            "file_filter": file_filter or [],
            "prefer_tables": prefer_tables,
            "stages": {
                "vector": [],
                "fts": [],
                "fused": [],
                "deduped": [],
                "reranked": [],
            },
            "timings": {"total_ms": 1.0},
            "include_rerank": include_rerank,
            "max_text_chars": max_text_chars,
        }


class FakeQueryEngine:
    retriever = FakeRetriever()

    @staticmethod
    def _is_table_query(question):
        return "表格" in question


def test_file_chunks_endpoint_returns_payload_previews(monkeypatch):
    monkeypatch.setattr(api_app, "store", FakeStore())
    monkeypatch.setattr(api_app, "query_engine", FakeQueryEngine())
    client = TestClient(api_app.app)

    response = client.get("/api/file/1/chunks")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["chunks"][0]["qdrant_id"] == 100
    assert body["chunks"][0]["text_preview"].startswith("DocFlow")


def test_file_chunks_endpoint_404s_missing_file(monkeypatch):
    monkeypatch.setattr(api_app, "store", FakeStore())
    monkeypatch.setattr(api_app, "query_engine", FakeQueryEngine())
    client = TestClient(api_app.app)

    response = client.get("/api/file/999/chunks")

    assert response.status_code == 404


def test_debug_retrieve_endpoint_uses_retriever(monkeypatch):
    monkeypatch.setattr(api_app, "query_engine", FakeQueryEngine())
    client = TestClient(api_app.app)

    response = client.post(
        "/api/debug/retrieve",
        json={"question": "表格数据", "include_rerank": False, "max_text_chars": 123},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "表格数据"
    assert body["prefer_tables"] is True
    assert body["include_rerank"] is False
    assert body["max_text_chars"] == 123

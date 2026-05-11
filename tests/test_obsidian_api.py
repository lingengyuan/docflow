from fastapi.testclient import TestClient

from src.api import app as api_app


class FakeObsidianRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(
        self,
        query,
        file_filter=None,
        retrieval_mode="hybrid",
        prefer_tables=False,
        cancel_event=None,
        related_k=0,
    ):
        self.calls.append(
            {
                "query": query,
                "retrieval_mode": retrieval_mode,
                "related_k": related_k,
            }
        )
        return [
            {
                "text": "current note should be excluded",
                "file_name": "Current.md",
                "file_path": "notes/Current.md",
                "page_num": 1,
                "rrf_score": 0.9,
            },
            {
                "text": "related note alpha",
                "file_name": "Alpha.md",
                "file_path": "notes/Alpha.md",
                "page_num": 2,
                "section": "Alpha",
                "rrf_score": 0.7,
            },
            {
                "text": "related note beta",
                "file_name": "Beta.md",
                "file_path": "notes/Beta.md",
                "page_num": 3,
                "section": "Beta",
                "rrf_score": 0.6,
            },
        ]


class FakeObsidianQueryEngine:
    def __init__(self):
        self.retriever = FakeObsidianRetriever()


def test_obsidian_related_notes_endpoint_excludes_current_note(monkeypatch):
    engine = FakeObsidianQueryEngine()
    monkeypatch.setattr(api_app, "query_engine", engine)
    client = TestClient(api_app.app)

    response = client.post(
        "/api/obsidian/related",
        json={
            "note_title": "Current",
            "note_path": "notes/Current.md",
            "note_content": "Current note body about local knowledge.",
            "selection": "local knowledge",
            "retrieval_mode": "full_text",
            "limit": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [item["file_name"] for item in body["related_notes"]] == ["Alpha.md", "Beta.md"]
    assert engine.retriever.calls[0]["retrieval_mode"] == "full_text"
    assert engine.retriever.calls[0]["related_k"] == 2


def test_obsidian_related_notes_requires_query_text(monkeypatch):
    monkeypatch.setattr(api_app, "query_engine", FakeObsidianQueryEngine())
    client = TestClient(api_app.app)

    response = client.post("/api/obsidian/related", json={})

    assert response.status_code == 400

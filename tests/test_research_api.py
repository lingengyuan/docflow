from fastapi.testclient import TestClient

from src.api import app as api_app
from src.ingest.store import DocStore
from src.query.generator import Answer, Citation


class FakeResearchQueryEngine:
    def __init__(self):
        self.calls = []

    def deep_research(
        self,
        question,
        file_filter=None,
        retrieval_mode="hybrid",
        max_steps=3,
        conversation_context=None,
    ):
        self.calls.append(
            {
                "question": question,
                "file_filter": file_filter,
                "retrieval_mode": retrieval_mode,
                "max_steps": max_steps,
                "conversation_context": conversation_context,
            }
        )
        return Answer(
            text="research answer",
            citations=[
                Citation(
                    file_name="Research.md",
                    file_path="/tmp/Research.md",
                    page_num=1,
                    snippet="research",
                    score=0.9,
                )
            ],
            related_notes=[{"file_name": "Related.md"}],
            research_steps=[
                {"step": 1, "query": question, "result_count": 2, "new_results": 2, "top_files": ["Research.md"]}
            ],
        )


def test_research_endpoint_returns_steps_and_saves_history(monkeypatch, tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    engine = FakeResearchQueryEngine()
    monkeypatch.setattr(api_app, "store", store)
    monkeypatch.setattr(api_app, "query_engine", engine)
    client = TestClient(api_app.app)

    response = client.post(
        "/api/research",
        json={"question": "compare project notes", "scope_mode": "all", "max_steps": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "research answer"
    assert body["research_steps"][0]["new_results"] == 2
    assert body["related_notes"][0]["file_name"] == "Related.md"
    assert engine.calls[0]["max_steps"] == 2
    assert store.list_history()[0]["question"] == "compare project notes"

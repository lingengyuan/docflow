from fastapi.testclient import TestClient

from src.api import app as api_app
from src.ingest.store import DocStore
from src.query.generator import Answer, Citation


class FakeQueryEngine:
    def __init__(self):
        self.calls = []

    def query(
        self,
        question,
        file_filter=None,
        conversation_context=None,
        retrieval_query=None,
    ):
        self.calls.append(
            {
                "question": question,
                "file_filter": file_filter,
                "conversation_context": conversation_context,
                "retrieval_query": retrieval_query,
            }
        )
        return Answer(
            text=f"answer for {question}",
            citations=[
                Citation(
                    file_name="README.md",
                    file_path="/tmp/README.md",
                    page_num=1,
                    snippet="DocFlow",
                    score=0.9,
                )
            ],
        )

    def query_stream(
        self,
        question,
        file_filter=None,
        cancel_event=None,
        conversation_context=None,
        retrieval_query=None,
    ):
        self.calls.append(
            {
                "question": question,
                "file_filter": file_filter,
                "conversation_context": conversation_context,
                "retrieval_query": retrieval_query,
            }
        )
        chunks = [
            {
                "text": "DocFlow",
                "file_name": "README.md",
                "file_path": "/tmp/README.md",
                "page_num": 1,
                "rerank_score": 0.9,
            }
        ]
        return chunks, iter(["stream answer"])


def test_query_creates_conversation_and_rewrites_followup(monkeypatch, tmp_path):
    active_store = DocStore(tmp_path / "docflow.db")
    fake_engine = FakeQueryEngine()
    monkeypatch.setattr(api_app, "store", active_store)
    monkeypatch.setattr(api_app, "query_engine", fake_engine)
    client = TestClient(api_app.app)

    first = client.post("/api/query", json={"question": "总结 DocFlow"})

    assert first.status_code == 200
    conversation_id = first.json()["conversation_id"]
    assert conversation_id is not None

    second = client.post(
        "/api/query",
        json={"question": "展开第二点", "conversation_id": conversation_id},
    )

    assert second.status_code == 200
    second_call = fake_engine.calls[-1]
    assert second_call["retrieval_query"] == "总结 DocFlow\n展开第二点"
    assert [item["role"] for item in second_call["conversation_context"]] == ["user", "assistant"]

    messages = client.get(f"/api/conversations/{conversation_id}/messages").json()
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[1]["citations"][0]["file_name"] == "README.md"

    conversations = client.get("/api/conversations").json()
    assert conversations[0]["id"] == conversation_id
    assert conversations[0]["message_count"] == 4


def test_conversation_create_list_and_delete(monkeypatch, tmp_path):
    active_store = DocStore(tmp_path / "docflow.db")
    monkeypatch.setattr(api_app, "store", active_store)
    client = TestClient(api_app.app)

    created = client.post("/api/conversations", json={"title": "手动会话"})

    assert created.status_code == 200
    conversation_id = created.json()["id"]
    assert client.get("/api/conversations").json()[0]["title"] == "手动会话"

    deleted = client.delete(f"/api/conversations/{conversation_id}")

    assert deleted.status_code == 200
    assert client.get(f"/api/conversations/{conversation_id}/messages").status_code == 404


def test_stream_query_emits_conversation_and_saves_messages(monkeypatch, tmp_path):
    active_store = DocStore(tmp_path / "docflow.db")
    fake_engine = FakeQueryEngine()
    monkeypatch.setattr(api_app, "store", active_store)
    monkeypatch.setattr(api_app, "query_engine", fake_engine)
    client = TestClient(api_app.app)

    with client.stream("POST", "/api/query/stream", json={"question": "流式问题"}) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "event: conversation" in body
    assert "event: token" in body
    conversation_id = active_store.list_conversations()[0]["id"]
    messages = active_store.list_messages(conversation_id)
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "stream answer"

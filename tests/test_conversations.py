import threading
import time

from fastapi.testclient import TestClient

from src.api import app as api_app
from src.api.model_tasks import ModelTaskController
from src.ingest.store import DocStore
from src.query.generator import Answer, Citation


class FakeQueryEngine:
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
                "conversation_context": conversation_context,
                "retrieval_query": retrieval_query,
            }
        )
        answer = Answer(
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
        answer.related_notes = [
            {
                "file_name": "NOTES.md",
                "file_path": "/tmp/NOTES.md",
                "page_num": 1,
                "section": "Related",
                "snippet": "Related note",
                "score": 0.5,
            }
        ]
        return answer

    def query_stream(
        self,
        question,
        file_filter=None,
        retrieval_mode="hybrid",
        cancel_event=None,
        conversation_context=None,
        retrieval_query=None,
        include_related=False,
    ):
        self.calls.append(
            {
                "question": question,
                "file_filter": file_filter,
                "retrieval_mode": retrieval_mode,
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
        if include_related:
            return chunks, iter(["stream answer"]), [{"file_name": "related-stream.md"}]
        return chunks, iter(["stream answer"])


class TimeoutThenFastQueryEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._calls = 0

    def query(
        self,
        question,
        file_filter=None,
        retrieval_mode="hybrid",
        conversation_context=None,
        retrieval_query=None,
    ):
        with self._lock:
            self._calls += 1
            call_number = self._calls
        if call_number == 1:
            time.sleep(0.2)
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


class SlowStreamQueryEngine(FakeQueryEngine):
    def query_stream(
        self,
        question,
        file_filter=None,
        retrieval_mode="hybrid",
        cancel_event=None,
        conversation_context=None,
        retrieval_query=None,
        include_related=False,
    ):
        time.sleep(0.2)
        return super().query_stream(
            question,
            file_filter=file_filter,
            retrieval_mode=retrieval_mode,
            cancel_event=cancel_event,
            conversation_context=conversation_context,
            retrieval_query=retrieval_query,
            include_related=include_related,
        )


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
    assert first.json()["related_notes"][0]["file_name"] == "NOTES.md"

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


def test_independent_english_question_is_not_rewritten_as_followup(monkeypatch, tmp_path):
    active_store = DocStore(tmp_path / "docflow.db")
    fake_engine = FakeQueryEngine()
    monkeypatch.setattr(api_app, "store", active_store)
    monkeypatch.setattr(api_app, "query_engine", fake_engine)
    client = TestClient(api_app.app)

    first = client.post("/api/query", json={"question": "summarize docflow"})
    conversation_id = first.json()["conversation_id"]

    second = client.post(
        "/api/query",
        json={"question": "what is bitcoin", "conversation_id": conversation_id},
    )

    assert second.status_code == 200
    assert fake_engine.calls[-1]["retrieval_query"] == "what is bitcoin"


def test_english_followup_uses_whole_word_marker(monkeypatch, tmp_path):
    active_store = DocStore(tmp_path / "docflow.db")
    fake_engine = FakeQueryEngine()
    monkeypatch.setattr(api_app, "store", active_store)
    monkeypatch.setattr(api_app, "query_engine", fake_engine)
    client = TestClient(api_app.app)

    first = client.post("/api/query", json={"question": "summarize docflow"})
    conversation_id = first.json()["conversation_id"]

    second = client.post(
        "/api/query",
        json={"question": "what about it", "conversation_id": conversation_id},
    )

    assert second.status_code == 200
    assert fake_engine.calls[-1]["retrieval_query"] == "summarize docflow\nwhat about it"


def test_query_timeout_recovers_without_restart(monkeypatch):
    controller = ModelTaskController(thread_name_prefix="test-api-model-task")
    monkeypatch.setattr(api_app, "model_tasks", controller)
    monkeypatch.setattr(api_app, "MODEL_TASK_TIMEOUT_S", 0.02)
    monkeypatch.setattr(api_app, "store", None)
    monkeypatch.setattr(api_app, "query_engine", TimeoutThenFastQueryEngine())
    client = TestClient(api_app.app)

    try:
        first = client.post("/api/query", json={"question": "慢问题"})
        second = client.post("/api/query", json={"question": "快问题"})

        assert first.status_code == 504
        assert "模型任务超时" in first.json()["detail"]
        assert second.status_code == 200
        assert second.json()["answer"] == "answer for 快问题"
    finally:
        controller.shutdown()


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
    assert "event: related_notes" in body
    assert "related-stream.md" in body
    assert "event: token" in body
    conversation_id = active_store.list_conversations()[0]["id"]
    messages = active_store.list_messages(conversation_id)
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "stream answer"


def test_stream_query_timeout_reports_error_and_skips_assistant_message(monkeypatch, tmp_path):
    controller = ModelTaskController(thread_name_prefix="test-api-model-task")
    active_store = DocStore(tmp_path / "docflow.db")
    monkeypatch.setattr(api_app, "model_tasks", controller)
    monkeypatch.setattr(api_app, "STREAM_FIRST_CONTENT_TIMEOUT_S", 0.02)
    monkeypatch.setattr(api_app, "STREAM_QUEUE_POLL_S", 0.01)
    monkeypatch.setattr(api_app, "store", active_store)
    monkeypatch.setattr(api_app, "query_engine", SlowStreamQueryEngine())
    client = TestClient(api_app.app)

    try:
        with client.stream("POST", "/api/query/stream", json={"question": "慢流式问题"}) as response:
            body = response.read().decode("utf-8")

        assert response.status_code == 200
        assert "event: error" in body
        assert "模型任务超时" in body
        conversation_id = active_store.list_conversations()[0]["id"]
        messages = active_store.list_messages(conversation_id)
        assert [message["role"] for message in messages] == ["user"]
    finally:
        controller.shutdown()

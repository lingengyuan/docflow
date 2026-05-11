from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app as api_app
from src.domain_types import FileStatus
from src.ingest.store import DocStore
from src.ingest.watcher import WatchDir
from src.query.generator import Answer, Citation


class SynchronousIngestQueue:
    def __init__(self, store: DocStore):
        self.store = store
        self.next_qdrant_id = 1000
        self.submitted: list[Path] = []

    def submit(self, path: Path):
        path = Path(path)
        self.submitted.append(path)
        record = self.store.get_file_by_path(path)
        if record is None:
            raise AssertionError(f"Missing file record for {path}")
        text = path.read_text(encoding="utf-8")
        self.next_qdrant_id += 1
        self.store.add_chunks(
            record["id"],
            [
                {
                    "qdrant_id": self.next_qdrant_id,
                    "chunk_type": "text",
                    "page_num": 1,
                    "section": "",
                    "char_count": len(text),
                    "raw_text": text,
                    "embedding_text": text,
                    "tokenized_text": text.lower(),
                }
            ],
        )
        self.store.set_chunk_count(path, 1)
        self.store.set_status(path, FileStatus.DONE)
        return {"status": "done", "file": path.name}


class StoreBackedQueryEngine:
    def __init__(self, store: DocStore):
        self.store = store

    def query(
        self,
        question: str,
        file_filter=None,
        retrieval_mode: str = "hybrid",
        conversation_context=None,
        retrieval_query: str | None = None,
    ) -> Answer:
        files = self.store.list_files(status=FileStatus.DONE)
        if file_filter:
            files = [record for record in files if record["file_name"] in file_filter]
        for record in files:
            for chunk in self.store.list_file_chunks(record["id"]):
                text = chunk["raw_text"]
                if "Apollo budget" not in text:
                    continue
                return Answer(
                    text="Apollo budget is approved for Phase36.",
                    citations=[
                        Citation(
                            file_name=record["file_name"],
                            file_path=record["file_path"],
                            page_num=chunk["page_num"],
                            snippet=text[:200],
                            score=0.99,
                        )
                    ],
                )
        return Answer(text="在现有文档中未找到相关信息。", citations=[])


def test_create_note_ingest_query_citation_and_save_answer_flow(monkeypatch, tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    queue = SynchronousIngestQueue(store)

    monkeypatch.setattr(api_app, "store", store)
    monkeypatch.setattr(api_app, "ingest_queue", queue)
    monkeypatch.setattr(api_app, "query_engine", StoreBackedQueryEngine(store))
    monkeypatch.setattr(api_app, "watch_dirs", [WatchDir(path=tmp_path)])
    client = TestClient(api_app.app)

    note_response = client.post(
        "/api/notes",
        json={
            "title": "Apollo Plan",
            "content": "Apollo budget is approved for the Phase36 end-to-end test.",
            "collection": "Plans",
            "user_tags": ["phase36"],
        },
    )

    assert note_response.status_code == 200
    note_body = note_response.json()
    note_path = Path(note_body["path"])
    note_record = store.get_file_by_id(note_body["file"]["id"])
    assert note_path.exists()
    assert note_record["status"] == "done"
    assert store.get_file_qdrant_ids(note_record["id"]) == [1001]

    query_response = client.post(
        "/api/query",
        json={"question": "What is approved?", "scope_mode": "all"},
    )

    assert query_response.status_code == 200
    query_body = query_response.json()
    assert "Apollo budget is approved" in query_body["answer"]
    assert query_body["citations"][0]["file_name"] == note_path.name
    assert store.list_history()[0]["question"] == "What is approved?"

    save_response = client.post(
        "/api/notes/from-answer",
        json={
            "title": "Saved Apollo Answer",
            "question": "What is approved?",
            "answer": query_body["answer"],
            "citations": query_body["citations"],
            "collection": "Saved Answers",
            "user_tags": ["answer", "phase36"],
        },
    )

    assert save_response.status_code == 200
    save_body = save_response.json()
    saved_path = Path(save_body["path"])
    saved_record = store.get_file_by_id(save_body["file"]["id"])
    assert saved_path.exists()
    assert "Apollo budget is approved" in saved_path.read_text(encoding="utf-8")
    assert saved_record["status"] == "done"
    assert len(queue.submitted) == 2

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app as api_app
from src.ingest.imports import (
    MarkdownImport,
    build_answer_note_markdown,
    build_knowledge_output_markdown,
    build_quick_note_markdown,
    html_to_markdown,
    safe_filename,
)
from src.ingest.store import DocStore
from src.ingest.watcher import WatchDir


def test_html_to_markdown_extracts_title_and_readable_text():
    title, markdown = html_to_markdown(
        """
        <html><head><title>Example Article</title><style>.x{}</style></head>
        <body><h1>Main</h1><p>Hello <a href="https://example.com/a">link</a>.</p>
        <script>ignore()</script></body></html>
        """
    )

    assert title == "Example Article"
    assert "## Main" in markdown
    assert "[link](https://example.com/a)" in markdown
    assert "ignore()" not in markdown


def test_note_markdown_builders_validate_content():
    note = build_quick_note_markdown("Quick Idea", "Remember this", tags=["idea"])
    answer = build_answer_note_markdown(
        "Saved Answer",
        "The answer",
        question="What is it?",
        citations=[{"file_name": "README.md", "page_num": 1}],
    )

    assert 'title: "Quick Idea"' in note.markdown
    assert "# Quick Idea" in note.markdown
    assert "## Question" in answer.markdown
    assert "README.md p.1" in answer.markdown
    assert safe_filename("A/B: C") == "a-b-c"


def test_knowledge_output_markdown_builder_adds_type_sources_and_tags():
    output = build_knowledge_output_markdown(
        "Sprint Brief",
        "project_brief",
        "## 当前状态\n\n已完成第一版。",
        source_files=["README.md", "docs/phase16-handoff.md"],
        tags=["phase17"],
    )

    assert 'title: "Sprint Brief"' in output.markdown
    assert 'output_type: "project_brief"' in output.markdown
    assert "knowledge-output" in output.markdown
    assert "phase17" in output.markdown
    assert "docs/phase16-handoff.md" in output.markdown
    assert "## 内容" in output.markdown


def test_create_note_endpoint_writes_markdown_and_queues(monkeypatch, tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    queued = []

    class FakeQueue:
        def submit(self, path: Path):
            queued.append(path)
            return {"status": "queued", "file": path.name}

    monkeypatch.setattr(api_app, "store", store)
    monkeypatch.setattr(api_app, "ingest_queue", FakeQueue())
    monkeypatch.setattr(api_app, "watch_dirs", [WatchDir(path=tmp_path)])
    client = TestClient(api_app.app)

    response = client.post(
        "/api/notes",
        json={
            "title": "Phase 13 Note",
            "content": "Local note body",
            "collection": "Notes",
            "user_tags": ["phase13"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    path = Path(body["path"])
    assert path.exists()
    assert path.read_text(encoding="utf-8").count("Local note body") == 1
    assert queued == [path]
    record = store.get_file_by_id(body["file"]["id"])
    assert record["collection"] == "Notes"
    assert record["user_tags"] == ["phase13"]


def test_import_url_endpoint_uses_fetcher_and_queues(monkeypatch, tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    queued = []

    class FakeQueue:
        def submit(self, path: Path):
            queued.append(path)
            return {"status": "queued", "file": path.name}

    monkeypatch.setattr(api_app, "store", store)
    monkeypatch.setattr(api_app, "ingest_queue", FakeQueue())
    monkeypatch.setattr(api_app, "watch_dirs", [WatchDir(path=tmp_path)])
    monkeypatch.setattr(
        api_app,
        "fetch_webpage_markdown",
        lambda url, title=None: MarkdownImport(title or "Fetched Page", "# Fetched Page\n\nBody\n"),
    )
    client = TestClient(api_app.app)

    response = client.post(
        "/api/import/url",
        json={
            "url": "https://example.com/article",
            "title": "Imported Page",
            "collection": "Web Imports",
            "user_tags": ["web"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert Path(body["path"]).exists()
    assert queued == [Path(body["path"])]
    assert body["file"]["collection"] == "Web Imports"


def test_save_answer_endpoint_writes_note(monkeypatch, tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    source_path = tmp_path / "source.md"
    source_path.write_text("Source fact", encoding="utf-8")
    source_id = store.upsert_file(
        source_path,
        source_path.name,
        DocStore.compute_hash(source_path),
        status="done",
        mtime_ns=source_path.stat().st_mtime_ns,
    )

    class FakeQueue:
        def submit(self, path: Path):
            return {"status": "queued", "file": path.name}

    monkeypatch.setattr(api_app, "store", store)
    monkeypatch.setattr(api_app, "ingest_queue", FakeQueue())
    monkeypatch.setattr(api_app, "watch_dirs", [WatchDir(path=tmp_path)])
    client = TestClient(api_app.app)

    response = client.post(
        "/api/notes/from-answer",
        json={
            "title": "Answer Note",
            "question": "Question?",
            "answer": "Answer body",
            "citations": [{"file_name": source_path.name, "file_path": str(source_path)}],
            "collection": "Saved Answers",
            "user_tags": ["answer"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    path = Path(body["path"])
    assert "Answer body" in path.read_text(encoding="utf-8")
    assert body["source_links"] == [source_id]


def test_knowledge_output_endpoint_writes_generated_markdown(monkeypatch, tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    generated = {}

    class FakeQueue:
        def submit(self, path: Path):
            return {"status": "queued", "file": path.name}

    class FakeQueryEngine:
        def generate_knowledge_output(self, output_type, title, source_text):
            generated["output_type"] = output_type
            generated["title"] = title
            generated["source_text"] = source_text
            return "## 当前状态\n\n资料已经整理。"

    monkeypatch.setattr(api_app, "store", store)
    monkeypatch.setattr(api_app, "ingest_queue", FakeQueue())
    monkeypatch.setattr(api_app, "query_engine", FakeQueryEngine())
    monkeypatch.setattr(api_app, "watch_dirs", [WatchDir(path=tmp_path)])
    client = TestClient(api_app.app)

    response = client.post(
        "/api/knowledge-output",
        json={
            "output_type": "project_brief",
            "title": "Phase 17 Brief",
            "source_text": "Phase 17 adds reusable knowledge outputs.",
            "collection": "Knowledge Outputs",
            "user_tags": ["phase17"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    path = Path(body["path"])
    assert path.exists()
    saved = path.read_text(encoding="utf-8")
    assert "## 当前状态" in saved
    assert 'output_type: "project_brief"' in saved
    assert generated["output_type"] == "project_brief"
    assert "Phase 17 adds" in generated["source_text"]
    assert body["file"]["collection"] == "Knowledge Outputs"
    assert body["file"]["user_tags"] == ["knowledge-output", "project_brief", "phase17"]


def test_knowledge_output_endpoint_uses_selected_file_chunks(monkeypatch, tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    source_path = tmp_path / "source.md"
    source_path.write_text("# Source\n\nBody", encoding="utf-8")
    file_id = store.upsert_file(
        source_path,
        source_path.name,
        "hash-source",
        status="done",
        total_pages=1,
        mtime_ns=source_path.stat().st_mtime_ns,
    )
    store.add_chunks(
        file_id,
        [
            {
                "qdrant_id": 101,
                "chunk_type": "text",
                "page_num": 1,
                "section": "Intro",
                "char_count": 21,
            }
        ],
    )
    generated = {}

    class FakeQueue:
        def submit(self, path: Path):
            return {"status": "queued", "file": path.name}

    class FakeRetriever:
        def fetch_file_chunks(self, qdrant_ids, max_chunks=12):
            assert qdrant_ids == [101]
            return [
                {
                    "text": "Source text from selected file.",
                    "file_name": "source.md",
                    "page_num": 1,
                    "section": "Intro",
                }
            ]

    class FakeQueryEngine:
        retriever = FakeRetriever()

        def generate_knowledge_output(self, output_type, title, source_text):
            generated["source_text"] = source_text
            return "## 核心要点\n\n- 来自选中文件。"

    monkeypatch.setattr(api_app, "store", store)
    monkeypatch.setattr(api_app, "ingest_queue", FakeQueue())
    monkeypatch.setattr(api_app, "query_engine", FakeQueryEngine())
    monkeypatch.setattr(api_app, "watch_dirs", [WatchDir(path=tmp_path)])
    client = TestClient(api_app.app)

    response = client.post(
        "/api/knowledge-output",
        json={"output_type": "summary", "file_ids": [file_id]},
    )

    assert response.status_code == 200
    body = response.json()
    saved = Path(body["path"]).read_text(encoding="utf-8")
    assert "文件：source.md" in generated["source_text"]
    assert "Source text from selected file" in generated["source_text"]
    assert "source.md" in saved
    assert body["source_files"] == ["source.md"]


def test_knowledge_output_endpoint_requires_source(monkeypatch, tmp_path):
    store = DocStore(tmp_path / "docflow.db")

    class FakeQueue:
        def submit(self, path: Path):
            return {"status": "queued", "file": path.name}

    class FakeQueryEngine:
        def generate_knowledge_output(self, output_type, title, source_text):
            return "unused"

    monkeypatch.setattr(api_app, "store", store)
    monkeypatch.setattr(api_app, "ingest_queue", FakeQueue())
    monkeypatch.setattr(api_app, "query_engine", FakeQueryEngine())
    monkeypatch.setattr(api_app, "watch_dirs", [WatchDir(path=tmp_path)])
    client = TestClient(api_app.app)

    response = client.post(
        "/api/knowledge-output",
        json={"output_type": "summary", "source_text": ""},
    )

    assert response.status_code == 400

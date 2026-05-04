from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app as api_app
from src.ingest.imports import (
    MarkdownImport,
    build_answer_note_markdown,
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

    assert "title: \"Quick Idea\"" in note.markdown
    assert "# Quick Idea" in note.markdown
    assert "## Question" in answer.markdown
    assert "README.md p.1" in answer.markdown
    assert safe_filename("A/B: C") == "a-b-c"


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
            "collection": "Saved Answers",
            "user_tags": ["answer"],
        },
    )

    assert response.status_code == 200
    path = Path(response.json()["path"])
    assert "Answer body" in path.read_text(encoding="utf-8")

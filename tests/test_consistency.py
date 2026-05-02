from pathlib import Path

import yaml

from src.ingest.store import DocStore
from src.maintenance.consistency import (
    check_consistency,
    compare_index_state,
    rebuild_qdrant_only,
)


def _write_config(tmp_path: Path, db_path: Path) -> Path:
    config = {
        "ollama": {
            "base_url": "http://localhost:11434",
            "ocr_model": "glm-ocr",
            "llm_model": "qwen2.5:7b",
        },
        "embedding": {"model": "fake-model", "backend": "torch", "batch_size": 2},
        "ingest": {},
        "reranker": {},
        "qdrant": {"host": "localhost", "port": 6333, "collection": "docflow"},
        "chunking": {"chunk_size": 64, "chunk_overlap": 6},
        "paths": {
            "db_path": str(db_path),
            "id_counter": str(tmp_path / "qdrant_id_counter.txt"),
            "watch_dirs": [{"path": str(tmp_path), "recursive": False, "extensions": [".md"]}],
            "supported_extensions": [".md"],
        },
        "vlm": {"enabled": False},
        "llm": {"backend": "local"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_compare_index_state_reports_mismatches():
    report = compare_index_state(
        sqlite_chunk_ids={1, 2, 3},
        qdrant_point_ids={2, 3, 4},
        file_counts=[
            {
                "id": 1,
                "file_name": "a.md",
                "file_path": "/tmp/a.md",
                "chunk_count": 3,
                "actual_chunk_count": 2,
            }
        ],
        missing_source_files=[{"id": 2, "file_name": "missing.md", "file_path": "/tmp/missing.md"}],
    )

    assert report.status == "inconsistent"
    assert report.missing_qdrant_points == [1]
    assert report.orphan_qdrant_points == [4]
    assert report.file_chunk_mismatches[0]["file_name"] == "a.md"
    assert report.missing_source_files[0]["file_name"] == "missing.md"


def test_check_consistency_accepts_matching_sqlite_and_qdrant(tmp_path):
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")
    db_path = tmp_path / "docflow.db"
    store = DocStore(db_path)
    file_id = store.upsert_file(source, source.name, "hash", status="done")
    store.add_chunks(
        file_id,
        [
            {
                "qdrant_id": 10,
                "chunk_type": "text",
                "page_num": 1,
                "section": "",
                "char_count": 5,
                "raw_text": "hello",
            }
        ],
    )
    store.set_chunk_count(source, 1)
    config_path = _write_config(tmp_path, db_path)

    class Record:
        def __init__(self, qid):
            self.id = qid

    class FakeQdrant:
        def scroll(self, **kwargs):
            return [Record(10)], None

    report = check_consistency(config_path, store=store, qdrant_client=FakeQdrant())

    assert report.status == "ok"
    assert report.sqlite_chunks == 1
    assert report.qdrant_points == 1


def test_rebuild_qdrant_only_dry_run_reports_sqlite_chunks(tmp_path):
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")
    db_path = tmp_path / "docflow.db"
    store = DocStore(db_path)
    file_id = store.upsert_file(source, source.name, "hash", status="done")
    store.add_chunks(
        file_id,
        [
            {
                "qdrant_id": 20,
                "chunk_type": "text",
                "page_num": 1,
                "section": "",
                "char_count": 5,
                "raw_text": "hello",
                "embedding_text": "hello",
                "parent_text": "hello",
            }
        ],
    )
    config_path = _write_config(tmp_path, db_path)

    result = rebuild_qdrant_only(config_path, dry_run=True)

    assert result == {
        "status": "dry_run",
        "mode": "qdrant_only",
        "chunks": 1,
        "min_qdrant_id": 20,
        "max_qdrant_id": 20,
    }


def test_rebuild_qdrant_only_dry_run_handles_empty_index(tmp_path):
    db_path = tmp_path / "docflow.db"
    DocStore(db_path)
    config_path = _write_config(tmp_path, db_path)

    result = rebuild_qdrant_only(config_path, dry_run=True)

    assert result == {
        "status": "dry_run",
        "mode": "qdrant_only",
        "chunks": 0,
        "min_qdrant_id": None,
        "max_qdrant_id": None,
    }

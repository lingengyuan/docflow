from pathlib import Path

import yaml

import src.maintenance.consistency as consistency
from src.ingest.store import DocStore
from src.maintenance.consistency import (
    check_consistency,
    compare_index_state,
    inspect_id_counter,
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


def test_compare_index_state_reports_duplicate_qdrant_ids():
    report = compare_index_state(
        sqlite_chunk_ids={10},
        qdrant_point_ids={10},
        file_counts=[],
        missing_source_files=[],
        duplicate_qdrant_ids=[
            {
                "qdrant_id": 10,
                "count": 2,
                "chunk_ids": [1, 2],
                "files": [
                    {"file_id": 1, "file_name": "a.md", "file_path": "/tmp/a.md"},
                    {"file_id": 2, "file_name": "b.md", "file_path": "/tmp/b.md"},
                ],
            }
        ],
        sqlite_chunk_count=2,
    )

    assert report.status == "inconsistent"
    assert report.sqlite_chunks == 2
    assert report.qdrant_points == 1
    assert report.duplicate_qdrant_ids[0]["qdrant_id"] == 10


def test_inspect_id_counter_reports_stale_counter(tmp_path):
    counter_path = tmp_path / "qdrant_id_counter.txt"
    counter_path.write_text("10", encoding="utf-8")

    result = inspect_id_counter(counter_path, [{"qdrant_id": 12}])

    assert result["status"] == "stale"
    assert result["value"] == 10
    assert result["expected_min"] == 13


def test_inspect_id_counter_uses_qdrant_point_floor(tmp_path):
    counter_path = tmp_path / "qdrant_id_counter.txt"
    counter_path.write_text("10", encoding="utf-8")

    result = inspect_id_counter(counter_path, [{"qdrant_id": 4}], qdrant_point_ids={18})

    assert result["status"] == "stale"
    assert result["expected_min"] == 19
    assert result["qdrant_max"] == 18


def test_repair_index_ids_dry_run_reports_counter_and_duplicate_actions(monkeypatch, tmp_path):
    source_a = tmp_path / "a.md"
    source_b = tmp_path / "b.md"
    source_a.write_text("a", encoding="utf-8")
    source_b.write_text("b", encoding="utf-8")
    db_path = tmp_path / "docflow.db"
    store = DocStore(db_path)
    file_a = store.upsert_file(source_a, source_a.name, "hash-a", status="done")
    file_b = store.upsert_file(source_b, source_b.name, "hash-b", status="done")
    for file_id, text in [(file_a, "a"), (file_b, "b")]:
        store.add_chunks(
            file_id,
            [
                {
                    "qdrant_id": 10,
                    "chunk_type": "text",
                    "page_num": 1,
                    "section": "",
                    "char_count": 1,
                    "raw_text": text,
                }
            ],
        )
        store.set_chunk_count(source_a if file_id == file_a else source_b, 1)
    config_path = _write_config(tmp_path, db_path)
    (tmp_path / "qdrant_id_counter.txt").write_text("5", encoding="utf-8")
    monkeypatch.setattr(consistency, "QdrantClient", lambda **_kwargs: object())
    monkeypatch.setattr(consistency, "_scroll_qdrant_ids", lambda *_args, **_kwargs: {10, 15})

    result = consistency.repair_index_ids(config_path, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["min_next_id"] == 16
    assert len(result["duplicate_qdrant_ids"]) == 1
    assert len(result["affected_files"]) == 2
    assert any("advance id counter" in action for action in result["actions"])
    assert any("reingest 2 files" in action for action in result["actions"])


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
    (tmp_path / "qdrant_id_counter.txt").write_text("11", encoding="utf-8")

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

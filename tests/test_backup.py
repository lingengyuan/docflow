import json
import os
import tarfile
from pathlib import Path

import yaml

from src.ingest.store import DocStore
from src.maintenance.backup import (
    apply_retention,
    create_backup,
    export_chunks_jsonl,
    restore_plan,
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


def _store_with_chunk(tmp_path: Path) -> tuple[Path, DocStore]:
    source = tmp_path / "note.md"
    source.write_text("hello backup", encoding="utf-8")
    db_path = tmp_path / "docflow.db"
    store = DocStore(db_path)
    file_id = store.upsert_file(source, source.name, "hash", status="done")
    store.add_chunks(
        file_id,
        [
            {
                "qdrant_id": 7,
                "chunk_type": "text",
                "page_num": 1,
                "section": "Intro",
                "char_count": 12,
                "raw_text": "hello backup",
                "embedding_text": "hello backup",
                "parent_text": "hello backup",
            }
        ],
    )
    store.set_chunk_count(source, 1)
    return _write_config(tmp_path, db_path), store


def test_export_chunks_jsonl_writes_rows(tmp_path):
    config_path, store = _store_with_chunk(tmp_path)
    output_path = tmp_path / "chunks.jsonl"

    result = export_chunks_jsonl(config_path, output_path=output_path, store=store)

    assert result["status"] == "done"
    assert result["chunks"] == 1
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["qdrant_id"] == 7
    assert rows[0]["raw_text"] == "hello backup"
    assert rows[0]["file_name"] == "note.md"


def test_create_backup_archive_contains_expected_files(tmp_path):
    config_path, _store = _store_with_chunk(tmp_path)

    result = create_backup(config_path, output_dir=tmp_path / "backups", keep=5)

    assert result["status"] == "done"
    assert result["files"] == 1
    assert result["chunks"] == 1
    with tarfile.open(result["path"], "r:gz") as tar:
        names = sorted(member.name for member in tar.getmembers())
        assert names == ["chunks.jsonl", "config.yaml", "docflow.db", "manifest.json"]
        manifest = json.loads(tar.extractfile("manifest.json").read().decode("utf-8"))
        chunk_rows = tar.extractfile("chunks.jsonl").read().decode("utf-8").splitlines()

    assert manifest["file_count"] == 1
    assert manifest["chunk_count"] == 1
    assert json.loads(chunk_rows[0])["raw_text"] == "hello backup"


def test_restore_plan_reads_archive_without_writing(tmp_path):
    config_path, _store = _store_with_chunk(tmp_path)
    backup = create_backup(config_path, output_dir=tmp_path / "backups", keep=5)

    result = restore_plan(backup["path"])

    assert result["status"] == "ok"
    assert result["missing"] == []
    assert "config.yaml" in result["members"]
    assert any("rebuild --qdrant-only" in step for step in result["steps"])


def test_create_backup_dry_run_does_not_write_archive(tmp_path):
    config_path, _store = _store_with_chunk(tmp_path)

    result = create_backup(config_path, output_dir=tmp_path / "backups", keep=3, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["chunks"] == 1
    assert not Path(result["path"]).exists()


def test_retention_keeps_newest_archives(tmp_path):
    output_dir = tmp_path / "backups"
    output_dir.mkdir()
    paths = [
        output_dir / "docflow-backup-20260101-000000-000000.tar.gz",
        output_dir / "docflow-backup-20260102-000000-000000.tar.gz",
        output_dir / "docflow-backup-20260103-000000-000000.tar.gz",
    ]
    for index, path in enumerate(paths):
        path.write_text("backup", encoding="utf-8")
        os.utime(path, (index + 1, index + 1))

    deleted = apply_retention(output_dir, keep=2)

    assert deleted == [str(paths[0])]
    assert not paths[0].exists()
    assert paths[1].exists()
    assert paths[2].exists()

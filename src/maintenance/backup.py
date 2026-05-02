"""Backup, chunk export, and restore planning helpers."""

from __future__ import annotations

import json
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.ingest.store import DocStore

BACKUP_PREFIX = "docflow-backup-"
BACKUP_SUFFIX = ".tar.gz"


def export_chunks_jsonl(
    config_path: str | Path = "config.yaml",
    output_path: str | Path | None = None,
    store: DocStore | None = None,
) -> dict[str, Any]:
    """Export the SQLite chunk index as JSONL."""
    cfg = _load_config(config_path)
    active_store = store or DocStore(Path(cfg["paths"]["db_path"]).expanduser())
    rows = active_store.list_chunk_index()
    target = Path(output_path) if output_path else Path("backups") / f"chunks-{_timestamp()}.jsonl"
    _write_chunks_jsonl(rows, target)
    return {
        "status": "done",
        "path": str(target),
        "chunks": len(rows),
    }


def create_backup(
    config_path: str | Path = "config.yaml",
    output_dir: str | Path = "backups",
    keep: int = 5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create a tar.gz archive with config, SQLite snapshot, manifest, and chunks JSONL."""
    cfg = _load_config(config_path)
    config_file = Path(config_path).expanduser()
    db_path = Path(cfg["paths"]["db_path"]).expanduser()
    out_dir = Path(output_dir).expanduser()
    archive_path = out_dir / f"{BACKUP_PREFIX}{_timestamp()}{BACKUP_SUFFIX}"

    store = DocStore(db_path)
    files = store.list_files()
    chunks = store.list_chunk_index()
    manifest = _manifest(config_file, db_path, files, chunks)
    includes = ["manifest.json", "config.yaml", "docflow.db", "chunks.jsonl"]

    if dry_run:
        return {
            "status": "dry_run",
            "path": str(archive_path),
            "would_include": includes,
            "files": len(files),
            "chunks": len(chunks),
            "keep": keep,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="docflow-backup-") as tmp:
        tmp_dir = Path(tmp)
        manifest_path = tmp_dir / "manifest.json"
        config_copy = tmp_dir / "config.yaml"
        db_copy = tmp_dir / "docflow.db"
        chunks_path = tmp_dir / "chunks.jsonl"

        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        config_copy.write_text(config_file.read_text(encoding="utf-8"), encoding="utf-8")
        _sqlite_backup(db_path, db_copy)
        _write_chunks_jsonl(chunks, chunks_path)

        with tarfile.open(archive_path, "w:gz") as tar:
            for item in (manifest_path, config_copy, db_copy, chunks_path):
                tar.add(item, arcname=item.name)

    deleted = apply_retention(out_dir, keep=keep)
    return {
        "status": "done",
        "path": str(archive_path),
        "files": len(files),
        "chunks": len(chunks),
        "deleted": deleted,
    }


def restore_plan(archive_path: str | Path) -> dict[str, Any]:
    """Read a backup archive and return a non-destructive restore checklist."""
    archive = Path(archive_path).expanduser()
    if not archive.exists():
        return {
            "status": "missing",
            "path": str(archive),
            "error": "archive not found",
        }

    with tarfile.open(archive, "r:gz") as tar:
        members = sorted(member.name for member in tar.getmembers())
        manifest = {}
        if "manifest.json" in members:
            extracted = tar.extractfile("manifest.json")
            if extracted is not None:
                manifest = json.loads(extracted.read().decode("utf-8"))

    required = {"config.yaml", "docflow.db", "chunks.jsonl"}
    missing = sorted(required - set(members))
    return {
        "status": "ok" if not missing else "incomplete",
        "path": str(archive),
        "members": members,
        "manifest": manifest,
        "missing": missing,
        "steps": [
            "Stop DocFlow before replacing local files.",
            "Extract this archive into a temporary folder.",
            "Back up the current config.yaml and SQLite database before copying restored files in.",
            "Copy config.yaml and docflow.db from the archive to the intended project paths.",
            "Run `.venv/bin/python main.py rebuild --qdrant-only` to restore Qdrant from SQLite chunks.",
            "Run `.venv/bin/python main.py check` to confirm SQLite and Qdrant match.",
        ],
    }


def apply_retention(output_dir: str | Path, keep: int = 5) -> list[str]:
    """Delete older backup archives, keeping the newest N archives in output_dir."""
    keep_count = max(1, int(keep))
    out_dir = Path(output_dir).expanduser()
    if not out_dir.exists():
        return []

    archives = sorted(
        out_dir.glob(f"{BACKUP_PREFIX}*{BACKUP_SUFFIX}"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    deleted: list[str] = []
    for archive in archives[keep_count:]:
        archive.unlink()
        deleted.append(str(archive))
    return deleted


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with open(Path(config_path).expanduser(), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def _manifest(
    config_path: Path,
    db_path: Path,
    files: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "db_path": str(db_path),
        "file_count": len(files),
        "chunk_count": len(chunks),
        "includes": ["config.yaml", "docflow.db", "chunks.jsonl"],
        "restore_hint": "Restore config.yaml and docflow.db, then run rebuild --qdrant-only and check.",
    }


def _sqlite_backup(source: Path, target: Path) -> None:
    source.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
        src.backup(dst)


def _write_chunks_jsonl(rows: list[dict[str, Any]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

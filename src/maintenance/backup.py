"""Backup, chunk export, and restore planning helpers."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.ingest.store import DocStore
from src.maintenance.consistency import find_duplicate_qdrant_ids, inspect_id_counter

BACKUP_PREFIX = "docflow-backup-"
BACKUP_SUFFIX = ".tar.gz"
DEFAULT_RESTORE_DRILL_DIR = Path("/tmp/docflow-phase22-restore-drill")
RESTORE_DRILL_MARKER = ".docflow-restore-drill"


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
            "Run `docflow admin rebuild --qdrant-only` "
            "to restore Qdrant from SQLite chunks.",
            "Run `docflow admin check` to confirm SQLite and Qdrant match.",
        ],
    }


def restore_drill(
    config_path: str | Path = "config.yaml",
    output_dir: str | Path = DEFAULT_RESTORE_DRILL_DIR,
    keep: int = 2,
) -> dict[str, Any]:
    """Run a non-destructive backup restore rehearsal in a disposable folder."""
    drill_dir = Path(output_dir).expanduser()
    backup_dir = drill_dir / "backups"
    extract_dir = drill_dir / "extracted"
    checks: list[dict[str, Any]] = []

    _reset_restore_drill_dir(drill_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    backup = create_backup(config_path, output_dir=backup_dir, keep=keep, dry_run=False)
    archive = Path(backup["path"])
    members = _safe_extract_tar(archive, extract_dir)
    member_set = set(members)
    required = {"manifest.json", "config.yaml", "docflow.db", "chunks.jsonl"}

    manifest_path = extract_dir / "manifest.json"
    db_path = extract_dir / "docflow.db"
    chunks_path = extract_dir / "chunks.jsonl"
    manifest = _read_manifest(manifest_path)
    restored_store = DocStore(db_path)
    restored_files = restored_store.list_files()
    restored_chunks = restored_store.list_chunk_index()
    duplicate_qdrant_ids = find_duplicate_qdrant_ids(restored_chunks)
    cfg = _load_config(config_path)
    id_counter = inspect_id_counter(
        Path(cfg["paths"].get("id_counter", "qdrant_id_counter.txt")).expanduser(),
        restored_chunks,
    )
    chunk_jsonl_count = _count_jsonl_rows(chunks_path)
    file_chunk_mismatches = [
        item
        for item in restored_store.list_file_chunk_counts()
        if item.get("chunk_count", 0) != item.get("actual_chunk_count", 0)
    ]
    missing_source_files = [
        {
            "id": item["id"],
            "file_name": item["file_name"],
            "file_path": item["file_path"],
        }
        for item in restored_store.list_file_chunk_counts()
        if not Path(item["file_path"]).exists()
    ]
    quick_check = _sqlite_quick_check(db_path)
    plan = restore_plan(archive)

    _record_check(
        checks,
        "archive_created",
        backup.get("status") == "done" and archive.exists(),
        {"archive": str(archive)},
    )
    _record_check(
        checks,
        "required_members",
        required.issubset(member_set),
        {"missing": sorted(required - member_set), "members": members},
    )
    _record_check(
        checks,
        "sqlite_quick_check",
        quick_check == "ok",
        {"quick_check": quick_check},
    )
    _record_check(
        checks,
        "manifest_file_count",
        manifest.get("file_count") == len(restored_files),
        {"manifest": manifest.get("file_count"), "restored": len(restored_files)},
    )
    _record_check(
        checks,
        "manifest_chunk_count",
        manifest.get("chunk_count") == len(restored_chunks),
        {"manifest": manifest.get("chunk_count"), "restored": len(restored_chunks)},
    )
    _record_check(
        checks,
        "chunks_jsonl_count",
        chunk_jsonl_count == len(restored_chunks),
        {"chunks_jsonl": chunk_jsonl_count, "sqlite_chunks": len(restored_chunks)},
    )
    _record_check(
        checks,
        "qdrant_id_uniqueness",
        not duplicate_qdrant_ids,
        {"duplicates": duplicate_qdrant_ids},
    )
    _record_check(
        checks,
        "id_counter_safe",
        id_counter.get("status") == "ok",
        id_counter,
    )
    _record_check(
        checks,
        "file_chunk_counts",
        not file_chunk_mismatches,
        {"mismatches": file_chunk_mismatches},
    )
    _record_check(
        checks,
        "source_paths_exist",
        not missing_source_files,
        {"checked": len(restored_files), "missing": missing_source_files},
    )
    _record_check(
        checks,
        "restore_plan",
        plan.get("status") == "ok",
        {"status": plan.get("status"), "missing": plan.get("missing", [])},
    )

    failed = sum(1 for check in checks if not check["passed"])
    return {
        "schema": "docflow.restore_drill.v1",
        "status": "ok" if failed == 0 else "failed",
        "output_dir": str(drill_dir),
        "archive": str(archive),
        "manifest": {
            "file_count": manifest.get("file_count", 0),
            "chunk_count": manifest.get("chunk_count", 0),
        },
        "restored": {
            "files": len(restored_files),
            "chunks": len(restored_chunks),
            "chunks_jsonl": chunk_jsonl_count,
            "quick_check": quick_check,
        },
        "source_paths": {
            "checked": len(restored_files),
            "missing": missing_source_files,
        },
        "id_counter": id_counter,
        "duplicate_qdrant_ids": duplicate_qdrant_ids,
        "file_chunk_mismatches": file_chunk_mismatches,
        "restore_plan_status": plan.get("status"),
        "passed": len(checks) - failed,
        "failed": failed,
        "checks": checks,
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
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")


def _manifest(
    config_path: Path,
    db_path: Path,
    files: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "db_path": str(db_path),
        "file_count": len(files),
        "chunk_count": len(chunks),
        "includes": ["config.yaml", "docflow.db", "chunks.jsonl"],
        "restore_hint": (
            "Restore config.yaml and docflow.db, then run rebuild --qdrant-only and check."
        ),
    }


def _sqlite_backup(source: Path, target: Path) -> None:
    source.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(source)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def _write_chunks_jsonl(rows: list[dict[str, Any]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _safe_extract_tar(archive: Path, target_dir: Path) -> list[str]:
    """Extract a backup archive while rejecting path traversal entries."""
    root = target_dir.resolve()
    names: list[str] = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_target = (root / member.name).resolve()
            if member_target != root and root not in member_target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.isdir():
                member_target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                member_target.parent.mkdir(parents=True, exist_ok=True)
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ValueError(f"archive member could not be read: {member.name}")
                with member_target.open("wb") as f:
                    shutil.copyfileobj(extracted, f)
            else:
                raise ValueError(f"unsupported archive member: {member.name}")
            names.append(member.name)
    return sorted(names)


def _reset_restore_drill_dir(drill_dir: Path) -> None:
    target = drill_dir.resolve()
    protected = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if target in protected:
        raise ValueError(f"refusing to clear protected restore drill directory: {target}")

    marker = target / RESTORE_DRILL_MARKER
    default_target = DEFAULT_RESTORE_DRILL_DIR.resolve()
    if target.exists():
        if not marker.exists() and target != default_target:
            raise ValueError(f"refusing to clear unmarked restore drill directory: {target}")
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)
    marker.write_text("docflow restore drill workspace\n", encoding="utf-8")


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _sqlite_quick_check(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("PRAGMA quick_check").fetchone()
    finally:
        conn.close()
    return str(row[0]) if row else "missing result"


def _record_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    details: dict[str, Any],
) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "details": details,
        }
    )

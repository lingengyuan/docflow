"""Consistency checks and rebuild helpers for SQLite and Qdrant indexes."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.embedding_backend import embedding_backend_config_from_dict
from src.ingest.chunker import Chunk
from src.ingest.pipeline import IngestPipeline
from src.ingest.store import DocStore

COLLECTION_NAME = "docflow"


@dataclass
class ConsistencyReport:
    status: str
    sqlite_chunks: int
    qdrant_points: int
    id_counter: dict
    missing_qdrant_points: list[int]
    orphan_qdrant_points: list[int]
    duplicate_qdrant_ids: list[dict]
    file_chunk_mismatches: list[dict]
    missing_source_files: list[dict]

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        return asdict(self)


def collect_watch_files(config_path: str | Path = "config.yaml") -> list[Path]:
    from src.api.app import _parse_watch_dirs
    from src.ingest.watcher import _is_excluded

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pipeline = IngestPipeline.from_config(config_path)
    paths: list[Path] = []
    for wd in _parse_watch_dirs(cfg):
        exts = wd.extensions if wd.extensions else pipeline.registry.supported_extensions
        for ext in exts:
            pattern = f"**/*{ext}" if wd.recursive else f"*{ext}"
            paths.extend(path for path in wd.path.glob(pattern) if not _is_excluded(path))
    return sorted(set(paths), key=lambda path: str(path))


def compare_index_state(
    sqlite_chunk_ids: set[int],
    qdrant_point_ids: set[int],
    file_counts: list[dict],
    missing_source_files: list[dict],
    duplicate_qdrant_ids: list[dict] | None = None,
    sqlite_chunk_count: int | None = None,
    id_counter: dict | None = None,
) -> ConsistencyReport:
    duplicates = duplicate_qdrant_ids or []
    counter = id_counter or {}
    missing_qdrant_points = sorted(sqlite_chunk_ids - qdrant_point_ids)
    orphan_qdrant_points = sorted(qdrant_point_ids - sqlite_chunk_ids)
    file_chunk_mismatches = [
        item
        for item in file_counts
        if item.get("chunk_count", 0) != item.get("actual_chunk_count", 0)
    ]
    status = "ok"
    if (
        missing_qdrant_points
        or orphan_qdrant_points
        or duplicates
        or counter.get("status") not in {None, "ok"}
        or file_chunk_mismatches
        or missing_source_files
    ):
        status = "inconsistent"
    return ConsistencyReport(
        status=status,
        sqlite_chunks=sqlite_chunk_count if sqlite_chunk_count is not None else len(sqlite_chunk_ids),
        qdrant_points=len(qdrant_point_ids),
        id_counter=counter,
        missing_qdrant_points=missing_qdrant_points,
        orphan_qdrant_points=orphan_qdrant_points,
        duplicate_qdrant_ids=duplicates,
        file_chunk_mismatches=file_chunk_mismatches,
        missing_source_files=missing_source_files,
    )


def check_consistency(
    config_path: str | Path = "config.yaml",
    store: DocStore | None = None,
    qdrant_client: QdrantClient | None = None,
) -> ConsistencyReport:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    active_store = store or DocStore(Path(cfg["paths"]["db_path"]).expanduser())
    qdrant = qdrant_client or QdrantClient(
        host=cfg["qdrant"]["host"],
        port=cfg["qdrant"]["port"],
        timeout=5,
    )
    collection = cfg.get("qdrant", {}).get("collection", COLLECTION_NAME)

    chunk_rows = active_store.list_chunk_index()
    sqlite_ids = {int(row["qdrant_id"]) for row in chunk_rows}
    qdrant_ids = _scroll_qdrant_ids(qdrant, collection)
    file_counts = active_store.list_file_chunk_counts()
    duplicate_qdrant_ids = find_duplicate_qdrant_ids(chunk_rows)
    id_counter = inspect_id_counter(
        Path(cfg["paths"].get("id_counter", "qdrant_id_counter.txt")).expanduser(),
        chunk_rows,
    )
    missing_source_files = [
        {
            "id": item["id"],
            "file_name": item["file_name"],
            "file_path": item["file_path"],
        }
        for item in file_counts
        if not Path(item["file_path"]).exists()
    ]
    return compare_index_state(
        sqlite_ids,
        qdrant_ids,
        file_counts,
        missing_source_files,
        duplicate_qdrant_ids=duplicate_qdrant_ids,
        sqlite_chunk_count=len(chunk_rows),
        id_counter=id_counter,
    )


def inspect_id_counter(counter_path: Path, chunk_rows: list[dict]) -> dict:
    """Check whether the next Qdrant point ID counter is ahead of SQLite IDs."""
    max_qdrant_id = max((int(row["qdrant_id"]) for row in chunk_rows), default=-1)
    expected_min = max_qdrant_id + 1
    result: dict[str, int | str | None] = {
        "path": str(counter_path),
        "value": None,
        "expected_min": expected_min,
        "status": "ok",
    }
    if not counter_path.exists():
        result["status"] = "missing" if expected_min > 0 else "ok"
        return result
    try:
        value = int(counter_path.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        result["status"] = "invalid"
        return result
    result["value"] = value
    if value < expected_min:
        result["status"] = "stale"
    return result


def find_duplicate_qdrant_ids(chunk_rows: list[dict]) -> list[dict]:
    """Return SQLite chunk rows that reuse the same Qdrant point ID."""
    counts = Counter(int(row["qdrant_id"]) for row in chunk_rows)
    duplicated_ids = {qid for qid, count in counts.items() if count > 1}
    if not duplicated_ids:
        return []

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in chunk_rows:
        qdrant_id = int(row["qdrant_id"])
        if qdrant_id in duplicated_ids:
            grouped[qdrant_id].append(row)

    result: list[dict] = []
    for qdrant_id in sorted(grouped):
        rows = grouped[qdrant_id]
        result.append(
            {
                "qdrant_id": qdrant_id,
                "count": len(rows),
                "chunk_ids": [int(row["id"]) for row in rows],
                "files": [
                    {
                        "file_id": int(row["file_id"]),
                        "file_name": row["file_name"],
                        "file_path": row["file_path"],
                    }
                    for row in rows
                ],
            }
        )
    return result


def rebuild_index(config_path: str | Path = "config.yaml", dry_run: bool = False) -> dict:
    files = collect_watch_files(config_path)
    if dry_run:
        return {
            "status": "dry_run",
            "mode": "full",
            "files": len(files),
            "paths": [str(path) for path in files],
        }

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    store = DocStore(Path(cfg["paths"]["db_path"]).expanduser())
    pipeline = IngestPipeline.from_config(config_path, store=store)

    store.clear_index()
    _recreate_qdrant_collection(pipeline.embedder._qdrant, cfg)
    pipeline.embedder._reset_id_counter()

    results = [pipeline.ingest(path) for path in files]
    done = sum(1 for item in results if item.get("status") == "done")
    return {
        "status": "done",
        "mode": "full",
        "files": len(files),
        "done": done,
        "results": results,
    }


def rebuild_qdrant_only(config_path: str | Path = "config.yaml", dry_run: bool = False) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    store = DocStore(Path(cfg["paths"]["db_path"]).expanduser())
    rows = store.list_chunk_index()
    if dry_run:
        return {
            "status": "dry_run",
            "mode": "qdrant_only",
            "chunks": len(rows),
            "min_qdrant_id": min((row["qdrant_id"] for row in rows), default=None),
            "max_qdrant_id": max((row["qdrant_id"] for row in rows), default=None),
        }

    pipeline = IngestPipeline.from_config(config_path, store=store)
    embedder = pipeline.embedder
    _recreate_qdrant_collection(embedder._qdrant, cfg)
    if not rows:
        embedder._reset_id_counter()
        return {
            "status": "done",
            "mode": "qdrant_only",
            "chunks": 0,
            "next_qdrant_id": 0,
        }

    chunks = [_chunk_from_row(row) for row in rows]
    texts = [chunk.embedding_text or chunk.raw_text or chunk.text for chunk in chunks]
    vectors = embedder.encode_texts(texts)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("encoded vectors must be 2D")
    embedder._vector_dim = vectors.shape[1]
    embedder._ensure_collection(embedder._vector_dim)

    points = [
        PointStruct(
            id=int(row["qdrant_id"]),
            vector=vectors[i].tolist(),
            payload=_chunk_payload(chunks[i]),
        )
        for i, row in enumerate(rows)
    ]
    for i in range(0, len(points), 100):
        embedder._qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i:i + 100])

    next_id = max((int(row["qdrant_id"]) for row in rows), default=-1) + 1
    embedder._qdrant_next_id = next_id
    embedder._id_counter_path.write_text(str(next_id))
    return {
        "status": "done",
        "mode": "qdrant_only",
        "chunks": len(rows),
        "next_qdrant_id": next_id,
    }


def _scroll_qdrant_ids(qdrant: QdrantClient, collection: str) -> set[int]:
    ids: set[int] = set()
    offset = None
    while True:
        records, offset = qdrant.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(int(record.id) for record in records)
        if offset is None:
            break
    return ids


def _recreate_qdrant_collection(qdrant: QdrantClient, cfg: dict) -> None:
    collection = cfg.get("qdrant", {}).get("collection", COLLECTION_NAME)
    if qdrant.collection_exists(collection):
        qdrant.delete_collection(collection)


def _chunk_from_row(row: dict) -> Chunk:
    chunk = Chunk(
        text=row.get("raw_text", ""),
        chunk_type=row.get("chunk_type", "text"),
        file_name=row.get("file_name", ""),
        file_path=row.get("file_path", ""),
        page_num=row.get("page_num", 0),
        section=row.get("section", ""),
        raw_text=row.get("raw_text", ""),
        embedding_text=row.get("embedding_text") or row.get("raw_text", ""),
        parent_id=row.get("parent_id", 0),
        parent_text=row.get("parent_text", ""),
        contextual_prefix=row.get("contextual_prefix", ""),
    )
    chunk.char_count = row.get("char_count", len(chunk.raw_text))
    return chunk


def _chunk_payload(chunk: Chunk) -> dict:
    return {
        "file_name": chunk.file_name,
        "file_path": chunk.file_path,
        "page_num": chunk.page_num,
        "section": chunk.section,
        "chunk_type": chunk.chunk_type,
        "text": chunk.raw_text,
        "raw_text": chunk.raw_text,
        "embedding_text": chunk.embedding_text,
        "child_text": chunk.raw_text,
        "parent_id": chunk.parent_id,
        "parent_text": chunk.parent_text,
        "contextual_prefix": chunk.contextual_prefix,
        "char_count": chunk.char_count,
    }


def print_report(report: ConsistencyReport, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return

    print(f"status: {report.status}")
    print(f"sqlite_chunks: {report.sqlite_chunks}")
    print(f"qdrant_points: {report.qdrant_points}")
    counter = report.id_counter or {}
    print(f"id_counter: {counter.get('status', 'unknown')}")
    print(f"missing_qdrant_points: {len(report.missing_qdrant_points)}")
    print(f"orphan_qdrant_points: {len(report.orphan_qdrant_points)}")
    print(f"duplicate_qdrant_ids: {len(report.duplicate_qdrant_ids)}")
    print(f"file_chunk_mismatches: {len(report.file_chunk_mismatches)}")
    print(f"missing_source_files: {len(report.missing_source_files)}")

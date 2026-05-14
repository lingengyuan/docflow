"""Shared ingest pipeline types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from src.ingest.chunker import Chunk
from src.ingest.pdf_analyzer import ParsedDocument

ProgressCallback = Callable[[dict], None]


@dataclass
class IngestMetrics:
    parse_s: float = 0.0
    chunk_s: float = 0.0
    embed_s: float = 0.0
    qdrant_s: float = 0.0
    sqlite_s: float = 0.0
    total_s: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    chunk_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreparedIngestFile:
    path: Path
    file_id: int
    file_hash: str
    mtime_ns: int
    doc: ParsedDocument
    tags_json: str
    chunks: list[Chunk]
    is_cjk: bool
    old_qdrant_ids: list[int]
    metrics: IngestMetrics

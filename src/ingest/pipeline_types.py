"""Shared ingest pipeline types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

from src.ingest.chunker import Chunk
from src.ingest.pdf_analyzer import ParsedDocument


class ProgressPayload(TypedDict, total=False):
    stage: str
    current_file: str | None
    current_path: str | None
    processed_chunks: int
    total_chunks: int
    batch_files: list[str]
    batch_size: int
    cache_hits: int
    cache_misses: int
    adaptive_batch_size: int | None
    updated_at: float
    pause_reason: str
    paused_since: float | None
    encoded_texts: int
    total_texts: int


ProgressCallback = Callable[[ProgressPayload], None]
ProgressFn = ProgressCallback | None


class IngestEmbedder(Protocol):
    @property
    def embedding_cache_key(self) -> str: ...

    def encode_texts(self, texts: list[str], progress_callback: ProgressFn = None) -> Any: ...

    def upsert_embeddings(
        self, chunks: list[Chunk], dense_vecs: Any, min_next_id: int | None = None
    ) -> list[int]: ...

    def delete_file_vectors(self, qdrant_ids: list[int]) -> None: ...

    def close(self) -> None: ...


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

    def to_dict(self) -> dict[str, float | int]:
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

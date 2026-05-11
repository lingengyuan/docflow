"""Shared typed records for DocFlow module boundaries."""

from __future__ import annotations

from enum import StrEnum
from typing import NotRequired, TypedDict


class FileStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class FileRecord(TypedDict):
    id: int
    file_path: str
    file_name: str
    file_hash: str
    status: str
    total_pages: int
    is_scanned: bool
    chunk_count: int
    error_msg: str
    tags: list[str]
    collection: str
    user_tags: list[str]
    mtime_ns: int
    created_at: str
    updated_at: str
    favorited: bool


class ChunkRecord(TypedDict):
    qdrant_id: int
    chunk_type: str
    page_num: int
    section: str
    char_count: int
    parent_id: NotRequired[int]
    raw_text: NotRequired[str]
    embedding_text: NotRequired[str]
    parent_text: NotRequired[str]
    contextual_prefix: NotRequired[str]
    tokenized_text: NotRequired[str]


class RetrievalResult(TypedDict, total=False):
    qdrant_id: int
    score: float
    vec_score: float
    fts_score: float
    rrf_score: float
    rerank_score: float
    rerank_fallback: bool
    text: str
    raw_text: str
    embedding_text: str
    file_name: str
    file_path: str
    page_num: int
    section: str
    chunk_type: str
    char_count: int
    parent_id: int
    parent_text: str
    contextual_prefix: str
    degradations: list[dict]


class HealthAction(TypedDict):
    label: str
    detail: str
    command: str
    kind: str

"""Typed ingestion and queue settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IngestSettings:
    parse_workers: int = 2
    microbatch_max_files: int = 8
    microbatch_max_chunks: int = 128
    microbatch_linger_ms: int = 75
    pause_check_interval_ms: int = 500
    embedding_cache: bool = True
    parent_context_chars: int = 2048
    contextual_prefix: bool = False
    contextual_prefix_mode: str = "metadata"
    contextual_prefix_model: str = ""
    adaptive_batch_char_budget: int | None = None
    adaptive_batch_max: int | None = None

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any], default_model: str) -> IngestSettings:
        ingest_cfg = cfg.get("ingest", {})
        return cls(
            parse_workers=int(ingest_cfg.get("parse_workers", 2)),
            microbatch_max_files=int(ingest_cfg.get("microbatch_max_files", 8)),
            microbatch_max_chunks=int(ingest_cfg.get("microbatch_max_chunks", 128)),
            microbatch_linger_ms=int(ingest_cfg.get("microbatch_linger_ms", 75)),
            pause_check_interval_ms=int(ingest_cfg.get("pause_check_interval_ms", 500)),
            embedding_cache=bool(ingest_cfg.get("embedding_cache", True)),
            parent_context_chars=int(ingest_cfg.get("parent_context_chars", 2048)),
            contextual_prefix=bool(ingest_cfg.get("contextual_prefix", False)),
            contextual_prefix_mode=str(ingest_cfg.get("contextual_prefix_mode", "metadata")),
            contextual_prefix_model=str(
                ingest_cfg.get("contextual_prefix_model", default_model)
            ),
            adaptive_batch_char_budget=_optional_int(ingest_cfg.get("adaptive_batch_char_budget")),
            adaptive_batch_max=_optional_int(ingest_cfg.get("adaptive_batch_max")),
        )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)

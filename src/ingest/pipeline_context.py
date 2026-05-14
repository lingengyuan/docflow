"""Chunk context helpers for ingest pipelines."""

from __future__ import annotations

import logging
from pathlib import Path

from src import net
from src.ingest.chunker import Chunk

logger = logging.getLogger(__name__)


def is_cjk_dominant(text: str, threshold: float = 0.2) -> bool:
    """CJK 字符占比超过阈值则视为中文主导。"""
    if not text:
        return False
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")
    return cjk_count / len(text) > threshold


def fts_tokenize(text: str, is_cjk: bool | None = None) -> str:
    """
    分词 -> 空格分隔字符串，用于 FTS5 精确匹配索引。
    中文主导：jieba 分词；英文主导：直接小写（利用 FTS5 unicode61 英文处理）。
    传入 is_cjk 可跳过逐 chunk 的语言检测。
    """
    if is_cjk is None:
        is_cjk = is_cjk_dominant(text)
    if is_cjk:
        import jieba

        return " ".join(t for t in jieba.cut(text.lower()) if t.strip())
    return text.lower()


def prepare_chunk_contexts(pipeline, chunks: list[Chunk]) -> None:
    assign_parent_contexts(pipeline, chunks)
    if pipeline.contextual_prefix_enabled:
        apply_contextual_prefixes(pipeline, chunks)


def assign_parent_contexts(pipeline, chunks: list[Chunk]) -> None:
    parent_id = 0
    current: list[Chunk] = []
    current_key: tuple[str, int, str] | None = None
    current_chars = 0

    def flush_current():
        nonlocal parent_id, current, current_key, current_chars
        if not current:
            return
        parent_id += 1
        parent_text = "\n\n".join(chunk.raw_text or chunk.text for chunk in current)
        for chunk in current:
            chunk.parent_id = parent_id
            chunk.parent_text = parent_text
        current = []
        current_key = None
        current_chars = 0

    for chunk in chunks:
        raw_text = chunk.raw_text or chunk.text
        key = (chunk.file_path, chunk.page_num, chunk.section)
        next_chars = current_chars + len(raw_text) + 2
        if current and (key != current_key or next_chars > pipeline.parent_context_chars):
            flush_current()

        current.append(chunk)
        current_key = key
        current_chars += len(raw_text) + 2

    flush_current()


def apply_contextual_prefixes(pipeline, chunks: list[Chunk]) -> None:
    for chunk in chunks:
        if not should_prefix_chunk(chunk):
            chunk.embedding_text = chunk.raw_text or chunk.text
            continue

        prefix = build_contextual_prefix(pipeline, chunk)
        chunk.contextual_prefix = prefix
        raw_text = chunk.raw_text or chunk.text
        chunk.embedding_text = f"{prefix}\n\n{raw_text}" if prefix else raw_text


def should_prefix_chunk(chunk: Chunk) -> bool:
    suffix = Path(chunk.file_path).suffix.lower()
    return suffix in {".md", ".markdown"} or chunk.chunk_type in {"table", "table_summary"}


def build_contextual_prefix(pipeline, chunk: Chunk) -> str:
    if pipeline.contextual_prefix_mode == "ollama" and pipeline.contextual_prefix_model:
        prefix = build_ollama_contextual_prefix(pipeline, chunk)
        if prefix:
            return prefix

    parts = [f"File: {chunk.file_name}"]
    if chunk.section:
        parts.append(f"Section: {chunk.section}")
    if chunk.chunk_type in {"table", "table_summary"}:
        parts.append("Content type: table")
    return " | ".join(parts)


def build_ollama_contextual_prefix(pipeline, chunk: Chunk) -> str:
    prompt = (
        "Write one short retrieval context prefix for this document chunk. "
        "Use facts only from the metadata and text. Do not add new facts.\n\n"
        f"File: {chunk.file_name}\n"
        f"Section: {chunk.section or '(none)'}\n"
        f"Type: {chunk.chunk_type}\n"
        f"Text:\n{(chunk.raw_text or chunk.text)[:1200]}\n\n"
        "Prefix:"
    )
    try:
        response = net.post(
            f"{pipeline.ollama_base_url}/api/generate",
            json={
                "model": pipeline.contextual_prefix_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 80},
            },
            timeout=net.Timeout(20.0, connect=5.0),
        )
        response.raise_for_status()
        prefix = response.json().get("response", "").strip()
        return " ".join(prefix.split())[:300]
    except (net.HTTPError, ValueError) as exc:
        logger.warning("[ingest] contextual prefix generation failed: %s", exc, exc_info=True)
        return ""

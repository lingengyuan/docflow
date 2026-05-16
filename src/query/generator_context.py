"""Prompt context formatting helpers."""

from __future__ import annotations


def build_context(chunks: list[dict]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, 1):
        section = f" > {chunk['section']}" if chunk.get("section") else ""
        qdrant_id = chunk.get("qdrant_id")
        chunk_id = chunk.get("chunk_id") or (f"q:{qdrant_id}" if qdrant_id is not None else "")
        cite_hint = f"引用格式: [[cite:{chunk_id}]]\n" if chunk_id else ""
        parts.append(
            f"[片段{index}] 来源: {chunk['file_name']}, 第{chunk['page_num']}页{section}\n"
            f"chunk_id: {chunk_id or '无'}\n"
            f"{cite_hint}{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def build_user_message(
    query: str,
    context: str,
    conversation_context: list[dict] | None = None,
) -> str:
    conversation = build_conversation_context(conversation_context or [])
    if conversation:
        return f"最近对话：\n{conversation}\n\n当前问题：{query}\n\n文档片段：\n{context}"
    return f"问题：{query}\n\n文档片段：\n{context}"


def build_conversation_context(messages: list[dict]) -> str:
    parts = []
    labels = {"user": "用户", "assistant": "DocFlow"}
    for message in messages:
        role = labels.get(message.get("role", ""), message.get("role", ""))
        content = " ".join(str(message.get("content", "")).split())
        if not role or not content:
            continue
        parts.append(f"{role}：{content}")
    return "\n".join(parts)

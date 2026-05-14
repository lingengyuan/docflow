"""Answer quality states surfaced to API and UI."""

from __future__ import annotations

from typing import Any


def grounded_quality() -> dict[str, Any]:
    return {
        "status": "grounded",
        "severity": "ok",
        "answer_mode": "generated",
        "label": "已基于本地资料回答",
        "reason": "回答由检索到的本地资料支持，请按引用核对关键事实。",
        "degradations": [],
    }


def insufficient_evidence_quality() -> dict[str, Any]:
    return {
        "status": "insufficient_evidence",
        "severity": "warning",
        "answer_mode": "no_answer",
        "label": "资料不足，未生成完整回答",
        "reason": "当前范围内没有找到足够可靠的片段，请扩大范围、换个问法，或等待资料完成入库。",
        "degradations": [],
    }


def local_model_unavailable_quality(exc: Exception | None = None) -> dict[str, Any]:
    quality: dict[str, Any] = {
        "status": "local_model_unavailable",
        "severity": "warning",
        "answer_mode": "snippet_fallback",
        "label": "本地回答模型暂不可用",
        "reason": "已找到相关资料，但这次只能显示引用片段；稍后可重试完整回答。",
        "degradations": [],
    }
    if exc is not None:
        quality["error_type"] = exc.__class__.__name__
    return quality


def retrieval_quality_from_chunks(chunks: list[dict]) -> dict[str, Any]:
    degradations = _collect_degradations(chunks)
    stages = {str(item.get("stage") or "") for item in degradations}
    if "vector" in stages:
        return {
            "status": "vector_store_unavailable",
            "severity": "warning",
            "answer_mode": "generated",
            "label": "向量检索暂不可用",
            "reason": "已改用关键词检索生成回答，相关度可能下降，请重点核对引用片段。",
            "degradations": degradations,
        }
    if "reranker" in stages:
        return {
            "status": "degraded_retrieval",
            "severity": "warning",
            "answer_mode": "generated",
            "label": "排序能力暂时降级",
            "reason": "已使用基础检索结果生成回答，来源顺序可能不如平时稳定。",
            "degradations": degradations,
        }
    return grounded_quality()


def _collect_degradations(chunks: list[dict]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    degradations: list[dict[str, Any]] = []
    for chunk in chunks:
        for item in chunk.get("degradations") or []:
            stage = str(item.get("stage") or "")
            error_type = str(item.get("error_type") or "")
            key = (stage, error_type)
            if key in seen:
                continue
            seen.add(key)
            degradations.append(
                {
                    "stage": stage,
                    "status": str(item.get("status") or "degraded"),
                    "error_type": error_type,
                }
            )
    return degradations

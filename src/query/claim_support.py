"""Deterministic sentence-level citation coverage checks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

SOURCE_RE = re.compile(r"\[来源:\s*([^,\]，]+)(?:[,，]\s*第?(\d+)页)?\]")
UNVERIFIED_SOURCE = "[未验证来源]"

NO_ANSWER_MARKERS = (
    "在现有文档中未找到相关信息",
    "资料不足",
    "没有找到足够",
    "无法获取文档内容",
)


def audit_answer_claim_support(answer_text: str, citations: Sequence[Any]) -> dict[str, Any]:
    claims = split_answer_claims(answer_text)
    verified_pairs, verified_files = verified_source_refs(citations)
    details: list[dict[str, str]] = []
    supported = 0
    unsupported: list[str] = []
    unverified: list[str] = []

    for claim in claims:
        if UNVERIFIED_SOURCE in claim:
            status = "unverified"
            unverified.append(claim)
        elif has_verified_source_marker(claim, verified_pairs, verified_files):
            status = "supported"
            supported += 1
        else:
            status = "unsupported"
            unsupported.append(claim)
        details.append({"text": claim, "status": status})

    total = len(claims)
    if total == 0:
        level = "none"
    elif unsupported or unverified:
        level = "partial" if supported else "unsupported"
    else:
        level = "supported"

    return {
        "level": level,
        "total_claims": total,
        "supported_claims": supported,
        "unsupported_claims": len(unsupported),
        "unverified_claims": len(unverified),
        "coverage": round(supported / total, 4) if total else 1.0,
        "unsupported_examples": unsupported[:3],
        "unverified_examples": unverified[:3],
        "claims": details[:20],
    }


def split_answer_claims(answer_text: str) -> list[str]:
    text = str(answer_text or "").strip()
    if not text or any(marker in text for marker in NO_ANSWER_MARKERS):
        return []

    claims: list[str] = []
    for line in re.split(r"\n+", text):
        for piece in re.split(r"(?<=[。！？；!?;])", line):
            claim = normalize_claim(piece)
            if claim:
                claims.append(claim)
    return claims


def normalize_claim(text: str) -> str:
    claim = re.sub(r"^\s*(?:[-*+•]|\d+[.)、]|[一二三四五六七八九十]+[、.])\s*", "", text)
    claim = " ".join(claim.strip().split())
    if not claim:
        return ""
    without_sources = SOURCE_RE.sub("", claim).replace(UNVERIFIED_SOURCE, "")
    meaningful_chars = re.findall(r"[\w\u4e00-\u9fff]", without_sources)
    if len(meaningful_chars) < 4:
        return ""
    return claim


def verified_source_refs(citations: Sequence[Any]) -> tuple[set[tuple[str, str]], set[str]]:
    pairs: set[tuple[str, str]] = set()
    files: set[str] = set()
    for citation in citations:
        file_name = str(_citation_value(citation, "file_name") or "").strip()
        if not file_name:
            continue
        page_num = _citation_value(citation, "page_num")
        page_text = str(page_num or "").strip()
        files.add(file_name)
        if page_text:
            pairs.add((file_name, page_text))
    return pairs, files


def has_verified_source_marker(
    claim: str,
    verified_pairs: set[tuple[str, str]],
    verified_files: set[str],
) -> bool:
    for match in SOURCE_RE.finditer(claim):
        file_name = match.group(1).strip()
        page_num = (match.group(2) or "").strip()
        if page_num and (file_name, page_num) in verified_pairs:
            return True
        if not page_num and file_name in verified_files:
            return True
    return False


def _citation_value(citation: Any, field: str) -> Any:
    if isinstance(citation, dict):
        return citation.get(field)
    return getattr(citation, field, None)

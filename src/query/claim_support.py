"""Deterministic sentence-level source support checks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

SOURCE_RE = re.compile(r"\[来源:\s*([^,\]，]+)(?:[,，]\s*第?(\d+)页)?\]")
UNVERIFIED_SOURCE = "[未验证来源]"
MIN_SOURCE_OVERLAP = 0.25
MIN_SHARED_TERMS = 2

STOP_TERMS = {
    "about",
    "answer",
    "claim",
    "conclusion",
    "fact",
    "from",
    "source",
    "that",
    "the",
    "this",
    "with",
    "引用",
    "来源",
    "结论",
    "资料",
    "说明",
}

NO_ANSWER_MARKERS = (
    "在现有文档中未找到相关信息",
    "资料不足",
    "没有找到足够",
    "无法获取文档内容",
)


def audit_answer_claim_support(answer_text: str, citations: Sequence[Any]) -> dict[str, Any]:
    claims = split_answer_claims(answer_text)
    citation_refs = verified_source_refs(citations)
    details: list[dict[str, Any]] = []
    supported = 0
    unsupported: list[str] = []
    unverified: list[str] = []
    weak_support: list[str] = []

    for claim in claims:
        source_score: dict[str, Any] | None = None
        if UNVERIFIED_SOURCE in claim:
            status = "unverified"
            unverified.append(claim)
        else:
            claim_refs = verified_citations_for_claim(claim, citation_refs)
            if not claim_refs:
                status = "unsupported"
                unsupported.append(claim)
            else:
                source_score = source_support_score(claim, claim_refs)
                if source_score["supported"]:
                    status = "supported"
                    supported += 1
                else:
                    status = "weak_source"
                    weak_support.append(claim)
        details.append({"text": claim, "status": status})
        if source_score is not None:
            details[-1]["source_score"] = source_score

    total = len(claims)
    if total == 0:
        level = "none"
    elif unsupported or unverified or weak_support:
        level = "partial" if supported else "unsupported"
    else:
        level = "supported"

    return {
        "level": level,
        "total_claims": total,
        "supported_claims": supported,
        "unsupported_claims": len(unsupported),
        "unverified_claims": len(unverified),
        "weak_source_claims": len(weak_support),
        "coverage": round(supported / total, 4) if total else 1.0,
        "unsupported_examples": unsupported[:3],
        "unverified_examples": unverified[:3],
        "weak_source_examples": weak_support[:3],
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


def verified_source_refs(citations: Sequence[Any]) -> dict[tuple[str, str], list[Any]]:
    refs: dict[tuple[str, str], list[Any]] = {}
    for citation in citations:
        file_name = str(_citation_value(citation, "file_name") or "").strip()
        if not file_name:
            continue
        page_num = _citation_value(citation, "page_num")
        page_text = str(page_num or "").strip()
        refs.setdefault((file_name, ""), []).append(citation)
        if page_text:
            refs.setdefault((file_name, page_text), []).append(citation)
    return refs


def has_verified_source_marker(
    claim: str,
    verified_pairs: set[tuple[str, str]] | dict[tuple[str, str], list[Any]],
    verified_files: set[str] | None = None,
) -> bool:
    if isinstance(verified_pairs, dict):
        return bool(verified_citations_for_claim(claim, verified_pairs))
    verified_files = verified_files or set()
    for match in SOURCE_RE.finditer(claim):
        file_name = match.group(1).strip()
        page_num = (match.group(2) or "").strip()
        if page_num and (file_name, page_num) in verified_pairs:
            return True
        if not page_num and file_name in verified_files:
            return True
    return False


def verified_citations_for_claim(
    claim: str,
    citation_refs: dict[tuple[str, str], list[Any]],
) -> list[Any]:
    matched: list[Any] = []
    seen: set[int] = set()
    for match in SOURCE_RE.finditer(claim):
        file_name = match.group(1).strip()
        page_num = (match.group(2) or "").strip()
        keys = [(file_name, page_num)] if page_num else [(file_name, "")]
        for key in keys:
            for citation in citation_refs.get(key, []):
                ident = id(citation)
                if ident in seen:
                    continue
                seen.add(ident)
                matched.append(citation)
    return matched


def source_support_score(claim: str, citations: Sequence[Any]) -> dict[str, Any]:
    claim_terms = meaningful_terms(SOURCE_RE.sub("", claim).replace(UNVERIFIED_SOURCE, ""))
    source_text = " ".join(
        str(_citation_value(citation, "snippet") or "") for citation in citations
    )
    source_terms = meaningful_terms(source_text)
    if not claim_terms:
        return {"supported": True, "score": 1.0, "shared_terms": []}
    if not source_terms:
        return {"supported": False, "score": 0.0, "shared_terms": []}
    shared = sorted(claim_terms & source_terms)
    denominator = max(1, min(len(claim_terms), 12))
    score = len(shared) / denominator
    return {
        "supported": score >= MIN_SOURCE_OVERLAP or len(shared) >= MIN_SHARED_TERMS,
        "score": round(score, 4),
        "shared_terms": shared[:8],
    }


def meaningful_terms(text: str) -> set[str]:
    normalized = str(text or "").lower()
    ascii_terms = {
        term
        for term in re.findall(r"[a-z][a-z0-9'\-]{2,}", normalized)
        if term not in STOP_TERMS and not term.isdigit()
    }
    chinese_terms = {
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
        if re.match(r"[\u4e00-\u9fff]{2}", normalized[index : index + 2])
        and normalized[index : index + 2] not in STOP_TERMS
    }
    return ascii_terms | chinese_terms


def _citation_value(citation: Any, field: str) -> Any:
    if isinstance(citation, dict):
        return citation.get(field)
    return getattr(citation, field, None)

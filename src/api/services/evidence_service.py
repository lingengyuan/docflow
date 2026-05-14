"""Evidence labels and answer-trust summaries."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

STRONG_SCORE = 0.75
MEDIUM_SCORE = 0.45
STALE_DAYS = 365

CONFLICT_GROUPS = {
    "status": (
        {
            "approved",
            "accept",
            "accepted",
            "yes",
            "true",
            "enabled",
            "批准",
            "通过",
            "开启",
            "支持",
        },
        {
            "rejected",
            "deny",
            "denied",
            "no",
            "false",
            "disabled",
            "拒绝",
            "失败",
            "关闭",
            "不支持",
        },
        "不同来源对状态给出了相反描述。",
    ),
    "direction": (
        {"increase", "increased", "higher", "up", "提升", "增加", "上涨", "变高"},
        {"decrease", "decreased", "lower", "down", "下降", "减少", "下跌", "变低"},
        "不同来源对变化方向给出了相反描述。",
    ),
    "freshness": (
        {"current", "latest", "new", "最新", "当前", "新版"},
        {"old", "deprecated", "obsolete", "过期", "废弃", "旧版"},
        "不同来源对资料时效给出了相反描述。",
    ),
}


class EvidenceService:
    def enrich_citations(self, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._enrich_citation(citation) for citation in citations]

    def summarize(self, citations: list[dict[str, Any]]) -> dict[str, Any]:
        if not citations:
            return {
                "level": "none",
                "label": "资料不足",
                "summary": "这次回答没有可核查来源。",
                "conflicts": [],
                "recommendations": ["补充资料后再问，或缩小到具体文件范围。"],
            }
        conflicts = self.detect_conflicts(citations)
        weak_count = sum(1 for item in citations if item.get("evidence_level") == "weak")
        strong_count = sum(1 for item in citations if item.get("evidence_level") == "strong")
        stale_count = sum(1 for item in citations if item.get("freshness") == "stale")
        if conflicts:
            level = "conflict"
            label = "存在冲突"
            summary = "不同来源之间有明显不一致，需要先打开来源核对。"
        elif strong_count:
            level = "strong"
            label = "来源较强"
            summary = "主要结论有高相关来源支撑。"
        elif weak_count == len(citations):
            level = "weak"
            label = "来源较弱"
            summary = "当前来源相关度偏低，回答需要谨慎使用。"
        else:
            level = "medium"
            label = "需要核对"
            summary = "回答有来源支撑，但建议打开片段确认。"
        recommendations = []
        if stale_count:
            recommendations.append("优先核对较旧来源是否仍然有效。")
        if weak_count:
            recommendations.append("可以换更具体的问题，或限定到可信文件。")
        if conflicts:
            recommendations.append("先比较冲突来源，再保存为笔记。")
        return {
            "level": level,
            "label": label,
            "summary": summary,
            "conflicts": conflicts,
            "recommendations": recommendations[:3],
        }

    def detect_conflicts(self, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        conflicts = []
        for group, (positive, negative, message) in CONFLICT_GROUPS.items():
            positives = [
                item
                for item in citations
                if self._contains_any(item.get("snippet", ""), positive)
            ]
            negatives = [
                item
                for item in citations
                if self._contains_any(item.get("snippet", ""), negative)
            ]
            if not positives or not negatives:
                continue
            if {item.get("file_name") for item in positives} == {
                item.get("file_name") for item in negatives
            }:
                continue
            conflicts.append(
                {
                    "type": group,
                    "message": message,
                    "files": sorted(
                        {
                            str(item.get("file_name") or "未知来源")
                            for item in positives + negatives
                        }
                    )[:4],
                }
            )
        return conflicts[:3]

    def _enrich_citation(self, citation: dict[str, Any]) -> dict[str, Any]:
        score = float(citation.get("score") or 0)
        freshness = self._freshness(citation.get("file_path", ""))
        level = self._level(score, bool(citation.get("snippet")))
        enriched = dict(citation)
        enriched.update(
            {
                "verified": bool(citation.get("file_name") or citation.get("file_path")),
                "evidence_level": level,
                "evidence_label": self._label(level),
                "evidence_reason": self._reason(level, freshness),
                "freshness": freshness["state"],
                "source_age_days": freshness["days"],
            }
        )
        return enriched

    @staticmethod
    def _level(score: float, has_snippet: bool) -> str:
        if score >= STRONG_SCORE and has_snippet:
            return "strong"
        if score >= MEDIUM_SCORE:
            return "medium"
        return "weak"

    @staticmethod
    def _label(level: str) -> str:
        return {
            "strong": "强来源",
            "medium": "可核对",
            "weak": "弱来源",
        }.get(level, "可核对")

    @staticmethod
    def _reason(level: str, freshness: dict[str, Any]) -> str:
        if freshness["state"] == "stale":
            return "来源较旧，建议打开原文确认。"
        return {
            "strong": "片段相关度高，可定位到原文。",
            "medium": "片段相关，建议打开来源核对。",
            "weak": "相关度偏低，需要更多资料支撑。",
        }.get(level, "建议打开来源核对。")

    @staticmethod
    def _freshness(file_path: str) -> dict[str, Any]:
        path = Path(str(file_path or ""))
        if not file_path or not path.exists():
            return {"state": "unknown", "days": None}
        days = max(0, int((time.time() - path.stat().st_mtime) / 86400))
        return {"state": "stale" if days >= STALE_DAYS else "current", "days": days}

    @staticmethod
    def _contains_any(text: Any, terms: set[str]) -> bool:
        normalized = str(text or "").lower()
        return any(term in normalized for term in terms)

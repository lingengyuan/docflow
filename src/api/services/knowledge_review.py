"""Active review signals for the knowledge workspace."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any

from src.api.services.knowledge_depth import KnowledgeDepthService
from src.domain_types import FileStatus
from src.ingest.store import DocStore


class KnowledgeReviewService:
    def __init__(self) -> None:
        self._depth_service = KnowledgeDepthService()

    def review(self, store: DocStore, *, base: Any, limit: int = 6) -> dict[str, Any]:
        files = [dict(file) for file in store.list_files(status=FileStatus.DONE)]
        profiles = [base._build_file_profile(store, file) for file in files]
        history = store.list_history(limit=40)
        feedback = store.get_feedback_summary()
        topics = base._topics(
            [profile for profile in profiles if profile["terms"]],
            limit=limit,
        )
        citation_counts = self._citation_counts(history, files)
        review_queue = self._review_queue(
            store,
            profiles,
            citation_counts=citation_counts,
            limit=limit,
        )
        relationship_timeline = self._relationship_timeline(store, files, limit=limit)
        knowledge_depth = self._depth_service.summarize(
            store,
            profiles=profiles,
            history=history,
            citation_counts=citation_counts,
            base=base,
            limit=limit,
        )
        return {
            "signals": self._review_signals(files, history, feedback, store),
            "recent_activity": {
                "files": [base._file_summary(file) for file in files[:limit]],
                "questions": [
                    {
                        "id": int(item.get("id") or 0),
                        "question": item.get("question", ""),
                        "created_at": item.get("created_at", ""),
                    }
                    for item in history[:limit]
                ],
            },
            "topic_activity": self._topic_activity(topics, history, base=base, limit=limit),
            "review_queue": review_queue,
            "relationship_timeline": relationship_timeline,
            "knowledge_depth": knowledge_depth,
            "recommendations": self._recommendations(
                files,
                history,
                topics,
                feedback,
                review_queue,
                limit=limit,
            ),
        }

    def _review_queue(
        self,
        store: DocStore,
        profiles: list[dict[str, Any]],
        *,
        citation_counts: Counter[int],
        limit: int,
    ) -> list[dict[str, Any]]:
        items = []
        for profile in profiles:
            file = profile["file"]
            file_id = int(file["id"])
            backlinks = store.list_backlinks(file_id)
            outbound_links = store.list_outbound_links(file_id)
            chunk_count = len(profile.get("chunks") or [])
            citation_count = citation_counts[file_id]
            days_since_update = self._days_since(file.get("updated_at", ""))
            score = (
                len(backlinks) * 4
                + len(outbound_links) * 2
                + citation_count * 3
                + min(chunk_count, 5)
                + (2 if days_since_update >= 14 else 0)
                + (2 if file.get("favorited") else 0)
            )
            if not backlinks and not outbound_links:
                score += 2
            reasons = self._review_reasons(
                backlinks=len(backlinks),
                outbound_links=len(outbound_links),
                citation_count=citation_count,
                days_since_update=days_since_update,
                chunk_count=chunk_count,
            )
            items.append(
                {
                    "file": file,
                    "priority": min(100, score * 10),
                    "reason": "、".join(reasons),
                    "signals": {
                        "backlinks": len(backlinks),
                        "source_links": len(outbound_links),
                        "citations": citation_count,
                        "chunks": chunk_count,
                        "days_since_update": days_since_update,
                    },
                    "keywords": [word for word, _ in profile["terms"].most_common(4)],
                }
            )
        items.sort(key=lambda item: (item["priority"], item["file"]["updated_at"]), reverse=True)
        return items[:limit]

    def _relationship_timeline(
        self,
        store: DocStore,
        files: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        items = []
        for file in files:
            note_file = self._file_summary(file)
            for link in store.list_outbound_links(int(file["id"])):
                source = self._file_summary(link.get("file") or {})
                if not source["id"]:
                    continue
                items.append(
                    {
                        "type": str(link.get("relation") or "answer_note"),
                        "label": self._relationship_label(str(link.get("relation") or "")),
                        "created_at": link.get("created_at", ""),
                        "note": note_file,
                        "source": source,
                    }
                )
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return items[:limit]

    def _review_reasons(
        self,
        *,
        backlinks: int,
        outbound_links: int,
        citation_count: int,
        days_since_update: int,
        chunk_count: int,
    ) -> list[str]:
        reasons = []
        if citation_count:
            reasons.append("最近回答引用过")
        if backlinks:
            reasons.append("已有笔记回连")
        if outbound_links:
            reasons.append("连接了来源资料")
        if days_since_update >= 14:
            reasons.append("有一段时间未更新")
        if chunk_count >= 3:
            reasons.append("内容较多")
        if not reasons:
            reasons.append("还缺少笔记关联")
        return reasons[:3]

    def _review_signals(
        self,
        files: list[dict[str, Any]],
        history: list[dict[str, Any]],
        feedback: dict[str, Any],
        store: DocStore,
    ) -> dict[str, Any]:
        collections = Counter(str(file.get("collection") or "") for file in files)
        file_ids = [int(file["id"]) for file in files]
        backlink_count = sum(len(store.list_backlinks(file_id)) for file_id in file_ids)
        source_link_count = sum(len(store.list_outbound_links(file_id)) for file_id in file_ids)
        return {
            "files": len(files),
            "questions": len(history),
            "feedback": feedback,
            "saved_answers": collections["Saved Answers"],
            "knowledge_outputs": collections["Knowledge Outputs"],
            "notes": collections["Notes"],
            "source_links": source_link_count,
            "backlinks": backlink_count,
        }

    def _topic_activity(
        self,
        topics: list[dict[str, Any]],
        history: list[dict[str, Any]],
        *,
        base: Any,
        limit: int,
    ) -> list[dict[str, Any]]:
        question_terms: Counter[str] = Counter()
        for item in history:
            question_terms.update(base._tokens(str(item.get("question") or "")))
        activity = []
        for topic in topics[: limit * 2]:
            terms = {str(topic.get("id") or "")}
            terms.update(str(word) for word in topic.get("keywords") or [])
            question_count = sum(question_terms[term.lower()] for term in terms)
            activity.append(
                {
                    "title": topic.get("title", ""),
                    "file_count": topic.get("file_count", 0),
                    "keywords": topic.get("keywords", []),
                    "question_count": question_count,
                    "weight": int(topic.get("file_count", 0)) + question_count,
                }
            )
        activity.sort(key=lambda item: item["weight"], reverse=True)
        return activity[:limit]

    def _recommendations(
        self,
        files: list[dict[str, Any]],
        history: list[dict[str, Any]],
        topics: list[dict[str, Any]],
        feedback: dict[str, Any],
        review_queue: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        recommendations = []
        if review_queue:
            first = review_queue[0]
            recommendations.append(
                {
                    "type": "review",
                    "title": "回顾高关联资料",
                    "detail": first["file"]["file_name"],
                    "file_id": first["file"]["id"],
                }
            )
        if topics:
            recommendations.append(
                {
                    "type": "topic",
                    "title": "整理活跃主题",
                    "detail": str(topics[0].get("title") or ""),
                    "file_id": None,
                }
            )
        if history and not feedback.get("total"):
            recommendations.append(
                {
                    "type": "feedback",
                    "title": "标记回答是否有用",
                    "detail": "让后续回顾更贴近你的资料使用方式",
                    "file_id": None,
                }
            )
        if files and not any(item["signals"]["source_links"] for item in review_queue):
            recommendations.append(
                {
                    "type": "connect",
                    "title": "把笔记连接到来源",
                    "detail": "保存回答或知识产物后，来源关系会进入回顾列表",
                    "file_id": None,
                }
            )
        if not recommendations:
            recommendations.append(
                {
                    "type": "start",
                    "title": "先导入资料并提问",
                    "detail": "有资料和问题后，这里会生成主动回顾建议",
                    "file_id": None,
                }
            )
        return recommendations[:limit]

    @staticmethod
    def _relationship_label(relation: str) -> str:
        labels = {
            "answer_note": "保存回答引用了来源",
            "source_note": "笔记摘录自来源",
            "knowledge_output": "知识产物基于来源",
            "manual_relationship": "确认了相关资料",
        }
        return labels.get(relation or "answer_note", "连接了来源资料")

    @staticmethod
    def _file_summary(file: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": int(file.get("id") or 0),
            "file_name": file.get("file_name", ""),
            "collection": file.get("collection", ""),
            "updated_at": file.get("updated_at", ""),
        }

    def _citation_counts(
        self,
        history: list[dict[str, Any]],
        files: list[dict[str, Any]],
    ) -> Counter[int]:
        by_name = {str(file.get("file_name") or ""): int(file["id"]) for file in files}
        by_path = {str(file.get("file_path") or ""): int(file["id"]) for file in files}
        counts: Counter[int] = Counter()
        for item in history:
            citations = self._decode_citations(item.get("citations"))
            for citation in citations:
                file_id = by_path.get(str(citation.get("file_path") or ""))
                file_id = file_id or by_name.get(str(citation.get("file_name") or ""))
                if file_id:
                    counts[file_id] += 1
        return counts

    @staticmethod
    def _decode_citations(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, str) or not value.strip():
            return []
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return []
        if not isinstance(decoded, list):
            return []
        return [item for item in decoded if isinstance(item, dict)]

    @staticmethod
    def _days_since(value: str) -> int:
        text = str(value or "").replace("Z", "").strip()
        if not text:
            return 0
        for candidate in (text, text.replace(" ", "T")):
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                continue
            delta = datetime.now() - parsed.replace(tzinfo=None)
            return max(0, delta.days)
        return 0

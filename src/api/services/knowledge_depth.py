"""Data-backed knowledge-depth signals for the personal workspace."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from src.ingest.store import DocStore


class KnowledgeDepthService:
    def summarize(
        self,
        store: DocStore,
        *,
        profiles: list[dict[str, Any]],
        history: list[dict[str, Any]],
        citation_counts: Counter[int],
        base: Any,
        limit: int = 6,
    ) -> dict[str, Any]:
        concepts = self._concepts(profiles, history, base=base, limit=limit)
        source_trails = self._source_trails(store, profiles, history, limit=limit)
        coverage_gaps = self._coverage_gaps(
            store,
            profiles,
            citation_counts=citation_counts,
            limit=limit,
        )
        return {
            "concepts": concepts,
            "source_trails": source_trails,
            "coverage_gaps": coverage_gaps,
            "next_actions": self._next_actions(
                concepts,
                source_trails,
                coverage_gaps,
                limit=limit,
            ),
        }

    def _concepts(
        self,
        profiles: list[dict[str, Any]],
        history: list[dict[str, Any]],
        *,
        base: Any,
        limit: int,
    ) -> list[dict[str, Any]]:
        files_by_term: dict[str, list[dict[str, Any]]] = defaultdict(list)
        term_weights: Counter[str] = Counter()
        for profile in profiles:
            file = profile["file"]
            for term, count in profile["terms"].most_common(16):
                term_weights[term] += int(count)
                if file not in files_by_term[term]:
                    files_by_term[term].append(file)

        question_terms: Counter[str] = Counter()
        for item in history:
            question_terms.update(base._tokens(str(item.get("question") or "")))

        concepts: list[dict[str, Any]] = []
        for term, weight in term_weights.items():
            files = files_by_term[term]
            if not files:
                continue
            question_count = int(question_terms[term])
            score = weight + len(files) * 3 + question_count * 4
            concepts.append(
                {
                    "title": term,
                    "file_count": len(files),
                    "question_count": question_count,
                    "weight": score,
                    "files": files[:4],
                }
            )
        concepts.sort(
            key=lambda item: (
                int(item["question_count"]),
                int(item["file_count"]),
                int(item["weight"]),
            ),
            reverse=True,
        )
        return concepts[:limit]

    def _source_trails(
        self,
        store: DocStore,
        profiles: list[dict[str, Any]],
        history: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        files = [profile["file"] for profile in profiles]
        by_name = {str(file.get("file_name") or ""): file for file in files}
        trails: list[dict[str, Any]] = []
        for item in history:
            matched_files = []
            seen: set[int] = set()
            for citation in self._decode_citations(item.get("citations")):
                file = by_name.get(str(citation.get("file_name") or ""))
                if not file:
                    continue
                file_id = int(file.get("id") or 0)
                if not file_id or file_id in seen:
                    continue
                seen.add(file_id)
                matched_files.append(file)
            if not matched_files:
                continue
            history_id = int(item.get("id") or 0)
            trails.append(
                {
                    "history_id": history_id,
                    "question": item.get("question", ""),
                    "created_at": item.get("created_at", ""),
                    "files": matched_files[:4],
                    "citation_count": len(seen),
                    "feedback": store.get_answer_feedback(history_id) if history_id else None,
                }
            )
            if len(trails) >= limit:
                break
        return trails

    def _coverage_gaps(
        self,
        store: DocStore,
        profiles: list[dict[str, Any]],
        *,
        citation_counts: Counter[int],
        limit: int,
    ) -> list[dict[str, Any]]:
        gaps: list[dict[str, Any]] = []
        for profile in profiles:
            file = profile["file"]
            file_id = int(file.get("id") or 0)
            if not file_id:
                continue
            backlinks = len(store.list_backlinks(file_id))
            outbound_links = len(store.list_outbound_links(file_id))
            citations = int(citation_counts[file_id])
            days_since_update = self._days_since(str(file.get("updated_at") or ""))
            term_count = len(profile.get("term_set") or [])

            if citations and not backlinks:
                gaps.append(
                    self._gap(
                        "cited_without_note",
                        "把被引用资料沉淀成笔记",
                        "最近回答引用过，但还没有笔记回连。",
                        file,
                        90 + citations * 5,
                    )
                )
            elif term_count >= 5 and not backlinks and not outbound_links:
                gaps.append(
                    self._gap(
                        "unlinked_source",
                        "补一条资料关联",
                        "内容已经入库，但还没有和笔记或来源建立关系。",
                        file,
                        55 + min(term_count, 12),
                    )
                )
            elif days_since_update >= 30 and (backlinks or citations):
                gaps.append(
                    self._gap(
                        "stale_review",
                        "回看旧资料",
                        "这份资料已有使用痕迹，但有一段时间没有更新。",
                        file,
                        60 + min(days_since_update, 60),
                    )
                )
        gaps.sort(key=lambda item: int(item["priority"]), reverse=True)
        return gaps[:limit]

    def _next_actions(
        self,
        concepts: list[dict[str, Any]],
        source_trails: list[dict[str, Any]],
        coverage_gaps: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        if coverage_gaps:
            first = coverage_gaps[0]
            actions.append(
                {
                    "type": "coverage_gap",
                    "title": first["title"],
                    "detail": first["file"]["file_name"],
                    "file_id": first["file"]["id"],
                }
            )
        unscored = next((trail for trail in source_trails if not trail.get("feedback")), None)
        if unscored:
            actions.append(
                {
                    "type": "feedback",
                    "title": "标记最近回答是否有用",
                    "detail": str(unscored.get("question") or "")[:80],
                    "file_id": None,
                }
            )
        active = next((concept for concept in concepts if int(concept["question_count"]) > 0), None)
        if active:
            actions.append(
                {
                    "type": "concept",
                    "title": "围绕活跃概念整理卡片",
                    "detail": str(active.get("title") or ""),
                    "file_id": int((active.get("files") or [{}])[0].get("id") or 0) or None,
                }
            )
        if not actions:
            actions.append(
                {
                    "type": "start",
                    "title": "继续导入资料并提问",
                    "detail": "问题、来源和笔记越多，这里的建议越具体。",
                    "file_id": None,
                }
            )
        return actions[:limit]

    @staticmethod
    def _gap(
        gap_type: str,
        title: str,
        detail: str,
        file: dict[str, Any],
        priority: int,
    ) -> dict[str, Any]:
        return {
            "type": gap_type,
            "title": title,
            "detail": detail,
            "file": file,
            "priority": min(100, priority),
        }

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

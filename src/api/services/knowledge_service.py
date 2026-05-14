"""Knowledge-management summaries derived from indexed local content."""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from src.api.services.knowledge_review import KnowledgeReviewService
from src.domain_types import FileStatus
from src.ingest.store import DocStore

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{2,}|[\u4e00-\u9fff]{2,}")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "docflow",
    "local",
    "file",
    "资料",
    "文件",
    "本地",
    "内容",
    "可以",
    "一个",
    "这个",
    "inbox",
    "notes",
    "web",
    "imports",
    "saved",
    "answers",
    "knowledge",
    "outputs",
}


class KnowledgeService:
    def __init__(self) -> None:
        self._review_service = KnowledgeReviewService()

    def overview(
        self,
        store: DocStore,
        *,
        active_file_id: int | None = None,
        limit: int = 6,
    ) -> dict[str, Any]:
        files = store.list_files(status=FileStatus.DONE)
        profiles = [self._build_file_profile(store, file) for file in files]
        profiles = [profile for profile in profiles if profile["terms"]]
        backlinks = store.list_backlinks(active_file_id) if active_file_id else []
        outbound_links = store.list_outbound_links(active_file_id) if active_file_id else []
        topics = self._topics(profiles, limit=limit)
        similar_documents = self._similar_documents(
            profiles,
            active_file_id=active_file_id,
            limit=limit,
        )
        knowledge_cards = self._knowledge_cards(store, profiles, limit=limit)
        return {
            "topics": topics,
            "similar_documents": similar_documents,
            "knowledge_cards": knowledge_cards,
            "knowledge_graph": self._knowledge_graph(
                topics,
                similar_documents,
                knowledge_cards,
                backlinks,
                outbound_links,
                active_file_id=active_file_id,
            ),
            "feedback": store.get_feedback_summary(),
            "backlinks": backlinks,
            "outbound_links": outbound_links,
            "stats": {
                "files": len(files),
                "profiled_files": len(profiles),
                "backlinks": len(backlinks),
                "outbound_links": len(outbound_links),
            },
        }

    def review(self, store: DocStore, *, limit: int = 6) -> dict[str, Any]:
        return self._review_service.review(store, base=self, limit=limit)

    def _build_file_profile(self, store: DocStore, file: dict) -> dict[str, Any]:
        chunks = store.list_file_chunks(int(file["id"]))[:8]
        chunk_text = "\n".join(
            str(chunk.get("raw_text") or chunk.get("parent_text") or "")[:800]
            for chunk in chunks
        )
        tags = " ".join(str(tag) for tag in file.get("user_tags") or [])
        text = " ".join(
            [
                str(file.get("file_name") or ""),
                str(file.get("collection") or ""),
                tags,
                chunk_text,
            ]
        )
        terms = Counter(self._tokens(text))
        return {
            "file": self._file_summary(file),
            "terms": terms,
            "term_set": set(terms),
            "chunks": chunks,
        }

    def _topics(self, profiles: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
        corpus = Counter()
        for profile in profiles:
            corpus.update(profile["terms"])
        topics = []
        for term, _count in corpus.most_common(limit * 3):
            matches = [profile for profile in profiles if term in profile["term_set"]]
            if len(matches) < 1:
                continue
            related = Counter()
            for profile in matches:
                related.update(profile["terms"])
            keywords = [word for word, _ in related.most_common(5) if word != term]
            topics.append(
                {
                    "id": term,
                    "title": term,
                    "file_count": len(matches),
                    "keywords": keywords[:4],
                    "files": [profile["file"] for profile in matches[:4]],
                }
            )
            if len(topics) >= limit:
                break
        return topics

    def _similar_documents(
        self,
        profiles: list[dict[str, Any]],
        *,
        active_file_id: int | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        pairs = []
        for left, right in combinations(profiles, 2):
            left_terms = left["term_set"]
            right_terms = right["term_set"]
            shared = sorted(left_terms & right_terms)
            if not shared:
                continue
            union_size = max(1, len(left_terms | right_terms))
            score = len(shared) / union_size
            if active_file_id and active_file_id not in {
                left["file"]["id"],
                right["file"]["id"],
            }:
                continue
            if score < 0.08 and len(shared) < 2:
                continue
            pairs.append(
                {
                    "score": round(score, 3),
                    "shared_terms": shared[:5],
                    "files": [left["file"], right["file"]],
                }
            )
        pairs.sort(key=lambda item: item["score"], reverse=True)
        return pairs[:limit]

    def _knowledge_cards(
        self,
        store: DocStore,
        profiles: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        cards = []
        for profile in profiles:
            file = profile["file"]
            chunks = profile["chunks"] or store.list_file_chunks(int(file["id"]))[:1]
            if not chunks:
                continue
            chunk = chunks[0]
            text = self._compact_text(
                str(chunk.get("raw_text") or chunk.get("parent_text") or "")
            )
            if not text:
                continue
            title = str(chunk.get("section") or Path(file["file_name"]).stem or file["file_name"])
            terms = [word for word, _ in profile["terms"].most_common(3)]
            cards.append(
                {
                    "title": title[:60],
                    "summary": text[:180],
                    "source_file": file,
                    "page_num": chunk.get("page_num", 1),
                    "keywords": terms,
                }
            )
            if len(cards) >= limit:
                break
        return cards

    @staticmethod
    def _tokens(text: str) -> list[str]:
        tokens = []
        for raw in TOKEN_RE.findall(text):
            token = raw.lower()
            if token in STOPWORDS or len(token) < 2:
                continue
            tokens.append(token)
        return tokens

    @staticmethod
    def _compact_text(text: str) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def _file_summary(file: dict) -> dict[str, Any]:
        return {
            "id": int(file["id"]),
            "file_name": file.get("file_name", ""),
            "collection": file.get("collection", ""),
            "updated_at": file.get("updated_at", ""),
        }

    def _knowledge_graph(
        self,
        topics: list[dict[str, Any]],
        similar_documents: list[dict[str, Any]],
        knowledge_cards: list[dict[str, Any]],
        backlinks: list[dict[str, Any]],
        outbound_links: list[dict[str, Any]],
        *,
        active_file_id: int | None,
    ) -> dict[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []

        def add_node(node_id: str, node_type: str, label: str, **meta: Any) -> None:
            nodes.setdefault(
                node_id,
                {
                    "id": node_id,
                    "type": node_type,
                    "label": label,
                    **meta,
                },
            )

        def add_file_node(file: dict[str, Any]) -> str:
            file_id = int(file.get("id") or 0)
            node_id = f"file:{file_id}"
            add_node(
                node_id,
                "file",
                str(file.get("file_name") or "资料"),
                file_id=file_id,
                collection=file.get("collection", ""),
            )
            return node_id

        for topic in topics[:6]:
            topic_id = f"topic:{topic['id']}"
            add_node(topic_id, "topic", str(topic["title"]), keywords=topic.get("keywords", []))
            for file in (topic.get("files") or [])[:4]:
                file_id = add_file_node(file)
                edges.append({"source": topic_id, "target": file_id, "type": "topic_file"})

        for item in similar_documents[:6]:
            files = item.get("files") or []
            if len(files) < 2:
                continue
            left = add_file_node(files[0])
            right = add_file_node(files[1])
            edges.append(
                {
                    "source": left,
                    "target": right,
                    "type": "similar",
                    "score": item.get("score", 0),
                }
            )

        for card in knowledge_cards[:6]:
            source_file = card.get("source_file") or {}
            source = add_file_node(source_file)
            card_id = f"card:{source_file.get('id', 0)}:{card.get('title', '')[:32]}"
            add_node(card_id, "card", str(card.get("title") or "知识卡片"))
            edges.append({"source": card_id, "target": source, "type": "card_source"})

        active_node = f"file:{active_file_id}" if active_file_id else ""
        for link in backlinks[:6]:
            linked = add_file_node(link.get("file") or {})
            if active_node:
                edges.append({"source": linked, "target": active_node, "type": "backlink"})
        for link in outbound_links[:6]:
            linked = add_file_node(link.get("file") or {})
            if active_node:
                edges.append({"source": active_node, "target": linked, "type": "source_link"})

        return {
            "nodes": list(nodes.values())[:30],
            "edges": edges[:48],
            "stats": {
                "nodes": min(len(nodes), 30),
                "edges": min(len(edges), 48),
            },
        }

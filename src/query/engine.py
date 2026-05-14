"""
QueryEngine — 串联 HybridRetriever + AnswerGenerator。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.embedding_backend import embedding_backend_config_from_dict
from src.ingest.store import DocStore
from src.query.answer_quality import (
    grounded_quality,
    insufficient_evidence_quality,
    local_model_unavailable_quality,
    retrieval_quality_from_chunks,
)
from src.query.generator import Answer, AnswerGenerator, citation_from_chunk
from src.query.retriever import HybridRetriever

TABLE_KEYWORDS = {
    "表格",
    "数据",
    "统计",
    "总计",
    "合计",
    "金额",
    "数量",
    "比例",
    "table",
    "data",
    "total",
    "sum",
    "amount",
    "count",
    "ratio",
    "percent",
}

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "在现有文档中未找到足够可靠的信息。请扩大提问范围、换个问法，或确认相关文件已经完成入库。"
)
MIN_RERANK_SCORE = 0.12
MIN_VECTOR_SCORE = 0.40
RELATED_NOTES_LIMIT = 4
DEFAULT_ANSWER_CHUNKS = 5
MIN_ANSWER_CHUNKS = 3
MAX_RESEARCH_STEPS = 4


@dataclass(frozen=True)
class QuerySettings:
    min_rerank_score: float = MIN_RERANK_SCORE
    min_vector_score: float = MIN_VECTOR_SCORE
    related_notes_limit: int = RELATED_NOTES_LIMIT
    default_answer_chunks: int = DEFAULT_ANSWER_CHUNKS
    min_answer_chunks: int = MIN_ANSWER_CHUNKS
    max_research_steps: int = MAX_RESEARCH_STEPS
    table_keywords: frozenset[str] = field(default_factory=lambda: frozenset(TABLE_KEYWORDS))

    @classmethod
    def from_config(cls, cfg: dict) -> QuerySettings:
        query_cfg = cfg.get("query", {})
        table_keywords = query_cfg.get("table_keywords")
        return cls(
            min_rerank_score=float(query_cfg.get("min_rerank_score", MIN_RERANK_SCORE)),
            min_vector_score=float(query_cfg.get("min_vector_score", MIN_VECTOR_SCORE)),
            related_notes_limit=int(query_cfg.get("related_notes_limit", RELATED_NOTES_LIMIT)),
            default_answer_chunks=int(
                query_cfg.get("default_answer_chunks", DEFAULT_ANSWER_CHUNKS)
            ),
            min_answer_chunks=int(query_cfg.get("min_answer_chunks", MIN_ANSWER_CHUNKS)),
            max_research_steps=int(query_cfg.get("max_research_steps", MAX_RESEARCH_STEPS)),
            table_keywords=frozenset(table_keywords or TABLE_KEYWORDS),
        )


class QueryEngine:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        settings: QuerySettings | None = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.settings = settings or QuerySettings()

    @classmethod
    def from_config(cls, config_path: str | Path, store: DocStore | None = None) -> QueryEngine:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        db_path = Path(cfg["paths"]["db_path"]).expanduser()
        reranker_cfg = cfg.get("reranker", {})
        privacy_cfg = cfg.get("privacy", {})
        allow_model_download = bool(privacy_cfg.get("allow_model_download", False))
        embedding_config = embedding_backend_config_from_dict(cfg, config_path)
        retriever = HybridRetriever(
            qdrant_host=cfg["qdrant"]["host"],
            qdrant_port=cfg["qdrant"]["port"],
            collection_name=cfg["qdrant"].get("collection", "docflow"),
            reranker_model=reranker_cfg.get("model", "Qwen/Qwen3-Reranker-0.6B"),
            reranker_instruction=reranker_cfg.get("instruction", ""),
            db_path=db_path,
            store=store,
            embedding_config=embedding_config,
            allow_model_download=allow_model_download,
        )
        llm_cfg = cfg.get("llm", {})
        query_cfg = cfg.get("query", {})
        generator = AnswerGenerator(
            backend=llm_cfg.get("backend", cfg.get("llm_backend", "local")),
            ollama_base_url=cfg["ollama"]["base_url"],
            ollama_model=llm_cfg.get("ollama_model", cfg["ollama"]["llm_model"]),
            mlx_model_name=llm_cfg.get("mlx_model", "mlx-community/Qwen3-4B-4bit"),
            mlx_model_enhanced=llm_cfg.get("mlx_model_enhanced", "mlx-community/Qwen3-8B-4bit"),
            claude_model=llm_cfg.get("claude_model", "claude-sonnet-4-6"),
            claude_api_key=llm_cfg.get("claude_api_key", os.getenv("ANTHROPIC_API_KEY", "")),
            seed=query_cfg.get("seed", 42),
            temperature=float(query_cfg.get("temperature", 0.0)),
            top_p=float(query_cfg.get("top_p", 1.0)),
            max_tokens=int(query_cfg.get("max_tokens", 2048)),
            allow_model_download=allow_model_download,
        )
        return cls(retriever, generator, settings=QuerySettings.from_config(cfg))

    def query(
        self,
        question: str,
        file_filter: list[str] | None = None,
        retrieval_mode: str = "hybrid",
        conversation_context: list[dict] | None = None,
        retrieval_query: str | None = None,
    ) -> Answer:
        effective_query = retrieval_query or question
        prefer_tables = self._is_table_query(effective_query, self.settings)
        chunks = self.retriever.retrieve(
            query=effective_query,
            file_filter=file_filter,
            retrieval_mode=retrieval_mode,
            prefer_tables=prefer_tables,
            related_k=self.settings.related_notes_limit,
        )
        answer_chunks, related_notes = self._split_answer_and_related(chunks)
        if not self._has_sufficient_evidence(answer_chunks, self.settings):
            return Answer(
                text=INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=[],
                related_notes=related_notes,
                quality=insufficient_evidence_quality(),
            )
        try:
            answer = self.generator.generate(
                question,
                answer_chunks,
                conversation_context=conversation_context,
            )
            answer.related_notes = related_notes
            answer.quality = retrieval_quality_from_chunks(answer_chunks)
            return answer
        except Exception as exc:
            logger.warning(
                "[query] answer generation failed; returning retrieved snippets", exc_info=True
            )
            answer = self._fallback_answer(answer_chunks, exc)
            answer.related_notes = related_notes
            return answer

    def query_stream(
        self,
        question: str,
        file_filter: list[str] | None = None,
        retrieval_mode: str = "hybrid",
        cancel_event=None,
        conversation_context: list[dict] | None = None,
        retrieval_query: str | None = None,
        include_related: bool = False,
    ):
        """返回 (chunks, token_generator)，先做检索再流式生成。"""
        effective_query = retrieval_query or question
        prefer_tables = self._is_table_query(effective_query, self.settings)
        chunks = self.retriever.retrieve(
            query=effective_query,
            file_filter=file_filter,
            retrieval_mode=retrieval_mode,
            prefer_tables=prefer_tables,
            cancel_event=cancel_event,
            related_k=self.settings.related_notes_limit if include_related else 0,
        )
        answer_chunks, related_notes = (
            self._split_answer_and_related(chunks) if include_related else (chunks, [])
        )
        if not self._has_sufficient_evidence(answer_chunks, self.settings):
            if include_related:
                return (
                    [],
                    iter([INSUFFICIENT_EVIDENCE_MESSAGE]),
                    [],
                    insufficient_evidence_quality(),
                )
            return [], iter([INSUFFICIENT_EVIDENCE_MESSAGE])
        quality = retrieval_quality_from_chunks(answer_chunks)
        token_gen = self._safe_generate_stream(
            question,
            answer_chunks,
            cancel_event=cancel_event,
            conversation_context=conversation_context,
            quality=quality,
        )
        if include_related:
            return answer_chunks, token_gen, related_notes, quality
        return answer_chunks, token_gen

    def summarize_file(self, file_name: str, qdrant_ids: list[int]) -> str:
        """生成单个文件的摘要（Markdown）。"""
        chunks = self.retriever.fetch_file_chunks(qdrant_ids)
        return self.generator.summarize(file_name, chunks)

    def generate_knowledge_output(self, output_type: str, title: str, source_text: str) -> str:
        """生成可保存为本地 Markdown 的知识产物。"""
        return self.generator.generate_knowledge_output(output_type, title, source_text)

    def close(self) -> None:
        self.retriever.close()

    def deep_research(
        self,
        question: str,
        file_filter: list[str] | None = None,
        retrieval_mode: str = "hybrid",
        max_steps: int = 3,
        conversation_context: list[dict] | None = None,
    ) -> Answer:
        step_queries = self._research_queries(question, max_steps=max_steps)
        all_chunks: list[dict] = []
        steps: list[dict] = []
        seen_keys: set[str] = set()

        for index, step_query in enumerate(step_queries, 1):
            prefer_tables = self._is_table_query(step_query, self.settings)
            chunks = self.retriever.retrieve(
                query=step_query,
                file_filter=file_filter,
                retrieval_mode=retrieval_mode,
                prefer_tables=prefer_tables,
                related_k=0,
            )
            added = 0
            top_files: list[str] = []
            for chunk in chunks:
                key = self._chunk_key(chunk)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_chunks.append(chunk)
                added += 1
                file_name = chunk.get("file_name", "")
                if file_name and file_name not in top_files:
                    top_files.append(file_name)
            steps.append(
                {
                    "step": index,
                    "query": step_query,
                    "result_count": len(chunks),
                    "new_results": added,
                    "top_files": top_files[:5],
                }
            )

        answer_chunks, related_notes = self._split_answer_and_related(all_chunks)
        if not self._has_sufficient_evidence(answer_chunks, self.settings):
            return Answer(
                text=INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=[],
                related_notes=related_notes,
                research_steps=steps,
                quality=insufficient_evidence_quality(),
            )
        try:
            answer = self.generator.generate(
                question,
                answer_chunks,
                conversation_context=conversation_context,
            )
        except Exception as exc:
            logger.warning(
                "[query] deep research generation failed; returning retrieved snippets",
                exc_info=True,
            )
            answer = self._fallback_answer(answer_chunks, exc)
        answer.related_notes = related_notes
        answer.research_steps = steps
        if not answer.quality:
            answer.quality = retrieval_quality_from_chunks(answer_chunks)
        return answer

    def _split_answer_and_related(self, chunks: list[dict]) -> tuple[list[dict], list[dict]]:
        if not chunks:
            return [], []
        related_limit = self.settings.related_notes_limit
        if len(chunks) > related_limit + self.settings.min_answer_chunks:
            answer_limit = len(chunks) - related_limit
        else:
            answer_limit = min(len(chunks), self.settings.default_answer_chunks)
        answer_chunks = chunks[:answer_limit]
        related_chunks = chunks[answer_limit:]
        return answer_chunks, self._related_notes(
            answer_chunks,
            related_chunks,
            limit=related_limit,
        )

    def _research_queries(self, question: str, max_steps: int = 3) -> list[str]:
        clean = " ".join(str(question or "").split())
        if not clean:
            return []
        limit = max(1, min(int(max_steps or 3), self.settings.max_research_steps))
        candidates = [
            clean,
            f"{clean}\n关键事实 证据 背景",
            f"{clean}\n对比 差异 风险 结论",
            f"{clean}\n时间线 原因 影响",
        ]
        result: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _chunk_key(chunk: dict) -> str:
        if chunk.get("qdrant_id") is not None:
            return f"q:{chunk.get('qdrant_id')}"
        return "|".join(
            str(chunk.get(key, ""))
            for key in ("file_path", "file_name", "page_num", "section", "text", "raw_text")
        )

    @staticmethod
    def _related_notes(
        answer_chunks: list[dict],
        related_chunks: list[dict],
        extra_exclude_keys: set[str] | None = None,
        limit: int = RELATED_NOTES_LIMIT,
    ) -> list[dict]:
        cited_keys = {
            chunk.get("file_path") or chunk.get("file_name")
            for chunk in answer_chunks
            if chunk.get("file_path") or chunk.get("file_name")
        }
        cited_keys.update(extra_exclude_keys or set())
        notes: list[dict] = []
        seen: set[str] = set()
        for chunk in related_chunks:
            key = chunk.get("file_path") or chunk.get("file_name")
            if not key or key in cited_keys or key in seen:
                continue
            seen.add(key)
            score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0.0)))
            text = chunk.get("text") or chunk.get("raw_text") or chunk.get("parent_text") or ""
            notes.append(
                {
                    "file_name": chunk.get("file_name", ""),
                    "file_path": chunk.get("file_path", ""),
                    "page_num": chunk.get("page_num", 0),
                    "section": chunk.get("section", ""),
                    "snippet": text[:220],
                    "score": round(float(score or 0.0), 4),
                    "chunk_type": chunk.get("chunk_type", "text"),
                }
            )
            if len(notes) >= limit:
                break
        return notes

    @staticmethod
    def _is_table_query(question: str, settings: QuerySettings | None = None) -> bool:
        settings = settings or QuerySettings()
        q_lower = question.lower()
        return any(kw in q_lower for kw in settings.table_keywords)

    @staticmethod
    def _has_sufficient_evidence(
        chunks: list[dict],
        settings: QuerySettings | None = None,
    ) -> bool:
        settings = settings or QuerySettings()
        if not chunks:
            return False
        top = chunks[0]
        if top.get("rerank_score") is not None and not top.get("rerank_fallback"):
            return float(top.get("rerank_score") or 0) >= settings.min_rerank_score
        vec_score = top.get("vec_score")
        if vec_score is not None and float(vec_score or 0) > 0:
            return float(vec_score) >= settings.min_vector_score
        return True

    @staticmethod
    def _fallback_answer(chunks: list[dict], exc: Exception) -> Answer:
        if not chunks:
            return Answer(
                text=INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=[],
                quality=insufficient_evidence_quality(),
            )

        text = (
            "已找到相关文档片段，但本地回答模型暂时不可用。"
            "请先查看下方引用片段；稍后可重试完整回答。"
        )
        citations = [citation_from_chunk(chunk) for chunk in chunks]
        return Answer(text=text, citations=citations, quality=local_model_unavailable_quality(exc))

    def _safe_generate_stream(
        self,
        question: str,
        chunks: list[dict],
        cancel_event=None,
        conversation_context: list[dict] | None = None,
        quality: dict | None = None,
    ):
        try:
            yield from self.generator.generate_stream(
                question,
                chunks,
                cancel_event=cancel_event,
                conversation_context=conversation_context,
            )
        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                return
            logger.warning(
                "[query] stream generation failed; returning fallback message", exc_info=True
            )
            if quality is not None:
                quality.clear()
                quality.update(local_model_unavailable_quality(exc))
            yield self._fallback_answer(chunks, exc).text

    @staticmethod
    def default_quality() -> dict:
        return grounded_quality()

"""
QueryEngine — 串联 HybridRetriever + AnswerGenerator。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.config import DocFlowSettings
from src.config_query import QuerySettings
from src.embedding_backend import embedding_backend_config_from_dict
from src.ingest.store import DocStore
from src.query.answer_quality import (
    grounded_quality,
    insufficient_evidence_quality,
    local_model_unavailable_quality,
    retrieval_quality_from_chunks,
)
from src.query.engine_helpers import (
    answer_quality as build_answer_quality,
)
from src.query.engine_helpers import (
    chunk_key,
    fallback_answer,
    has_sufficient_evidence,
    is_table_query,
    research_queries,
    split_answer_and_related,
)
from src.query.generator import Answer, AnswerGenerator
from src.query.retriever import HybridRetriever

logger = logging.getLogger(__name__)


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
        settings = DocFlowSettings.from_file(config_path)
        cfg = settings.raw
        embedding_config = embedding_backend_config_from_dict(cfg, config_path)
        retriever = HybridRetriever(
            qdrant_host=settings.qdrant.host,
            qdrant_port=settings.qdrant.port,
            collection_name=settings.qdrant.collection,
            reranker_model=settings.reranker.model,
            reranker_instruction=settings.reranker.instruction,
            db_path=settings.paths.db_path,
            store=store,
            embedding_config=embedding_config,
            allow_model_download=settings.privacy.allow_model_download,
        )
        generator = AnswerGenerator(
            backend=settings.llm.backend,
            ollama_base_url=settings.ollama.base_url,
            ollama_model=settings.llm.ollama_model,
            mlx_model_name=settings.llm.mlx_model,
            mlx_model_enhanced=settings.llm.mlx_model_enhanced,
            claude_model=settings.llm.claude_model,
            claude_api_key=settings.llm.claude_api_key or os.getenv("ANTHROPIC_API_KEY", ""),
            seed=settings.query.seed,
            temperature=settings.query.temperature,
            top_p=settings.query.top_p,
            max_tokens=settings.query.max_tokens,
            allow_model_download=settings.privacy.allow_model_download,
        )
        return cls(retriever, generator, settings=settings.query)

    def query(
        self,
        question: str,
        file_filter: list[str] | None = None,
        retrieval_mode: str = "hybrid",
        conversation_context: list[dict] | None = None,
        retrieval_query: str | None = None,
    ) -> Answer:
        effective_query = retrieval_query or question
        prefer_tables = is_table_query(effective_query, self.settings)
        chunks = self.retriever.retrieve(
            query=effective_query,
            file_filter=file_filter,
            retrieval_mode=retrieval_mode,
            prefer_tables=prefer_tables,
            related_k=self.settings.related_notes_limit,
        )
        answer_chunks, related_notes = split_answer_and_related(chunks, self.settings)
        if not has_sufficient_evidence(answer_chunks, self.settings):
            return Answer(
                text=self.settings.insufficient_evidence_message,
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
            answer.quality = build_answer_quality(answer_chunks, answer.quality)
            return answer
        except Exception as exc:
            logger.warning(
                "[query] answer generation failed; returning retrieved snippets", exc_info=True
            )
            answer = fallback_answer(answer_chunks, exc, self.settings)
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
        prefer_tables = is_table_query(effective_query, self.settings)
        chunks = self.retriever.retrieve(
            query=effective_query,
            file_filter=file_filter,
            retrieval_mode=retrieval_mode,
            prefer_tables=prefer_tables,
            cancel_event=cancel_event,
            related_k=self.settings.related_notes_limit if include_related else 0,
        )
        answer_chunks, related_notes = (
            split_answer_and_related(chunks, self.settings) if include_related else (chunks, [])
        )
        if not has_sufficient_evidence(answer_chunks, self.settings):
            if include_related:
                return (
                    [],
                    iter([self.settings.insufficient_evidence_message]),
                    [],
                    insufficient_evidence_quality(),
                )
            return [], iter([self.settings.insufficient_evidence_message])
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
        step_queries = research_queries(question, self.settings, max_steps=max_steps)
        all_chunks: list[dict] = []
        steps: list[dict] = []
        seen_keys: set[str] = set()

        for index, step_query in enumerate(step_queries, 1):
            prefer_tables = is_table_query(step_query, self.settings)
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
                key = chunk_key(chunk)
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

        answer_chunks, related_notes = split_answer_and_related(all_chunks, self.settings)
        if not has_sufficient_evidence(answer_chunks, self.settings):
            return Answer(
                text=self.settings.insufficient_evidence_message,
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
            answer = fallback_answer(answer_chunks, exc, self.settings)
        answer.related_notes = related_notes
        answer.research_steps = steps
        answer.quality = build_answer_quality(answer_chunks, answer.quality)
        return answer

    @staticmethod
    def _is_table_query(question: str, settings: QuerySettings | None = None) -> bool:
        return is_table_query(question, settings)

    @staticmethod
    def _has_sufficient_evidence(
        chunks: list[dict],
        settings: QuerySettings | None = None,
    ) -> bool:
        return has_sufficient_evidence(chunks, settings)

    @staticmethod
    def _fallback_answer(chunks: list[dict], exc: Exception) -> Answer:
        return fallback_answer(chunks, exc)

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
            yield fallback_answer(chunks, exc, self.settings).text

    @staticmethod
    def default_quality() -> dict:
        return grounded_quality()

    @staticmethod
    def _answer_quality(chunks: list[dict], answer_quality: dict | None) -> dict:
        return build_answer_quality(chunks, answer_quality)

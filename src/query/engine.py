"""
QueryEngine — 串联 HybridRetriever + AnswerGenerator。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from src.embedding_backend import embedding_backend_config_from_dict
from src.query.generator import Answer, AnswerGenerator, Citation
from src.query.retriever import HybridRetriever
from src.ingest.store import DocStore


TABLE_KEYWORDS = {"表格", "数据", "统计", "总计", "合计", "金额", "数量", "比例",
                  "table", "data", "total", "sum", "amount", "count", "ratio", "percent"}

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "在现有文档中未找到足够可靠的信息。"
    "请扩大提问范围、换个问法，或确认相关文件已经完成入库。"
)
MIN_RERANK_SCORE = 0.12
MIN_VECTOR_SCORE = 0.40
RELATED_NOTES_LIMIT = 4
DEFAULT_ANSWER_CHUNKS = 5
MIN_ANSWER_CHUNKS = 3


class QueryEngine:
    def __init__(self, retriever: HybridRetriever, generator: AnswerGenerator):
        self.retriever = retriever
        self.generator = generator

    @classmethod
    def from_config(cls, config_path: str | Path, store: DocStore | None = None) -> "QueryEngine":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        db_path = Path(cfg["paths"]["db_path"]).expanduser()
        reranker_cfg = cfg.get("reranker", {})
        embedding_config = embedding_backend_config_from_dict(cfg, config_path)
        retriever = HybridRetriever(
            qdrant_host=cfg["qdrant"]["host"],
            qdrant_port=cfg["qdrant"]["port"],
            reranker_model=reranker_cfg.get("model", "Qwen/Qwen3-Reranker-0.6B"),
            reranker_instruction=reranker_cfg.get("instruction", ""),
            db_path=db_path,
            store=store,
            embedding_config=embedding_config,
        )
        llm_cfg = cfg.get("llm", {})
        generator = AnswerGenerator(
            backend=llm_cfg.get("backend", cfg.get("llm_backend", "local")),
            ollama_base_url=cfg["ollama"]["base_url"],
            ollama_model=llm_cfg.get("ollama_model", cfg["ollama"]["llm_model"]),
            mlx_model_name=llm_cfg.get("mlx_model", "mlx-community/Qwen3-4B-4bit"),
            mlx_model_enhanced=llm_cfg.get("mlx_model_enhanced", "mlx-community/Qwen3-8B-4bit"),
            claude_model=llm_cfg.get("claude_model", "claude-sonnet-4-6"),
            claude_api_key=llm_cfg.get("claude_api_key", os.getenv("ANTHROPIC_API_KEY", "")),
        )
        return cls(retriever, generator)

    def query(
        self,
        question: str,
        file_filter: list[str] | None = None,
        retrieval_mode: str = "hybrid",
        conversation_context: list[dict] | None = None,
        retrieval_query: str | None = None,
    ) -> Answer:
        effective_query = retrieval_query or question
        prefer_tables = self._is_table_query(effective_query)
        chunks = self.retriever.retrieve(
            query=effective_query,
            file_filter=file_filter,
            retrieval_mode=retrieval_mode,
            prefer_tables=prefer_tables,
            related_k=RELATED_NOTES_LIMIT,
        )
        answer_chunks, related_notes = self._split_answer_and_related(chunks)
        if not self._has_sufficient_evidence(answer_chunks):
            return Answer(text=INSUFFICIENT_EVIDENCE_MESSAGE, citations=[])
        try:
            answer = self.generator.generate(
                question,
                answer_chunks,
                conversation_context=conversation_context,
            )
            answer.related_notes = related_notes
            return answer
        except Exception as exc:
            logger.warning("[query] answer generation failed; returning retrieved snippets", exc_info=True)
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
        prefer_tables = self._is_table_query(effective_query)
        chunks = self.retriever.retrieve(
            query=effective_query,
            file_filter=file_filter,
            retrieval_mode=retrieval_mode,
            prefer_tables=prefer_tables,
            cancel_event=cancel_event,
            related_k=RELATED_NOTES_LIMIT if include_related else 0,
        )
        answer_chunks, related_notes = (
            self._split_answer_and_related(chunks) if include_related else (chunks, [])
        )
        if not self._has_sufficient_evidence(answer_chunks):
            if include_related:
                return [], iter([INSUFFICIENT_EVIDENCE_MESSAGE]), []
            return [], iter([INSUFFICIENT_EVIDENCE_MESSAGE])
        token_gen = self._safe_generate_stream(
            question,
            answer_chunks,
            cancel_event=cancel_event,
            conversation_context=conversation_context,
        )
        if include_related:
            return answer_chunks, token_gen, related_notes
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

    @classmethod
    def _split_answer_and_related(cls, chunks: list[dict]) -> tuple[list[dict], list[dict]]:
        if not chunks:
            return [], []
        if len(chunks) > RELATED_NOTES_LIMIT + MIN_ANSWER_CHUNKS:
            answer_limit = len(chunks) - RELATED_NOTES_LIMIT
        else:
            answer_limit = min(len(chunks), DEFAULT_ANSWER_CHUNKS)
        answer_chunks = chunks[:answer_limit]
        related_chunks = chunks[answer_limit:]
        return answer_chunks, cls._related_notes(answer_chunks, related_chunks)

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
    def _is_table_query(question: str) -> bool:
        q_lower = question.lower()
        return any(kw in q_lower for kw in TABLE_KEYWORDS)

    @staticmethod
    def _has_sufficient_evidence(chunks: list[dict]) -> bool:
        if not chunks:
            return False
        top = chunks[0]
        if top.get("rerank_score") is not None and not top.get("rerank_fallback"):
            return float(top.get("rerank_score") or 0) >= MIN_RERANK_SCORE
        vec_score = top.get("vec_score")
        if vec_score is not None and float(vec_score or 0) > 0:
            return float(vec_score) >= MIN_VECTOR_SCORE
        return True

    @staticmethod
    def _fallback_answer(chunks: list[dict], exc: Exception) -> Answer:
        if not chunks:
            return Answer(text=INSUFFICIENT_EVIDENCE_MESSAGE, citations=[])

        text = (
            "已找到相关文档片段，但回答模型暂时不可用。"
            "请先查看下方引用片段；稍后可重试完整回答。"
            f"\n\n错误类型：{exc.__class__.__name__}"
        )
        citations = [
            Citation(
                file_name=c["file_name"],
                file_path=c.get("file_path", ""),
                page_num=c["page_num"],
                snippet=c["text"][:200],
                score=c.get("rerank_score", c.get("rrf_score", 0.0)),
                section=c.get("section", ""),
            )
            for c in chunks
        ]
        return Answer(text=text, citations=citations)

    def _safe_generate_stream(
        self,
        question: str,
        chunks: list[dict],
        cancel_event=None,
        conversation_context: list[dict] | None = None,
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
            logger.warning("[query] stream generation failed; returning fallback message", exc_info=True)
            yield self._fallback_answer(chunks, exc).text

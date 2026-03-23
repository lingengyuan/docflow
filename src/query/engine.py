"""
QueryEngine — 串联 HybridRetriever + AnswerGenerator。
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from src.embedding_backend import embedding_backend_config_from_dict
from src.query.generator import Answer, AnswerGenerator
from src.query.retriever import HybridRetriever
from src.ingest.store import DocStore


TABLE_KEYWORDS = {"表格", "数据", "统计", "总计", "合计", "金额", "数量", "比例",
                  "table", "data", "total", "sum", "amount", "count", "ratio", "percent"}


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
    ) -> Answer:
        prefer_tables = self._is_table_query(question)
        chunks = self.retriever.retrieve(
            query=question,
            file_filter=file_filter,
            prefer_tables=prefer_tables,
        )
        return self.generator.generate(question, chunks)

    def query_stream(
        self,
        question: str,
        file_filter: list[str] | None = None,
        cancel_event=None,
    ):
        """返回 (chunks, token_generator)，先做检索再流式生成。"""
        prefer_tables = self._is_table_query(question)
        chunks = self.retriever.retrieve(
            query=question,
            file_filter=file_filter,
            prefer_tables=prefer_tables,
            cancel_event=cancel_event,
        )
        token_gen = self.generator.generate_stream(
            question,
            chunks,
            cancel_event=cancel_event,
        )
        return chunks, token_gen

    def summarize_file(self, file_name: str, qdrant_ids: list[int]) -> str:
        """生成单个文件的摘要（Markdown）。"""
        chunks = self.retriever.fetch_file_chunks(qdrant_ids)
        return self.generator.summarize(file_name, chunks)

    @staticmethod
    def _is_table_query(question: str) -> bool:
        q_lower = question.lower()
        return any(kw in q_lower for kw in TABLE_KEYWORDS)

"""
AnswerGenerator — 基于检索结果生成带引用的答案。

支持两种后端：
  - local: Qwen2.5-7B via Ollama
  - claude: Claude API (claude-sonnet-4-6)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.knowledge_outputs import get_knowledge_output_type
from src.query import generator_backends as backend_calls
from src.query.citations import (
    Citation,
    apply_structured_citations,
    citation_from_chunk,
    sanitize_inline_citations,
    validate_citations,
)
from src.query.claim_support import audit_answer_claim_support
from src.query.generator_context import (
    build_context,
    build_conversation_context,
    build_user_message,
)
from src.query.generator_prompts import (
    KNOWLEDGE_OUTPUT_SYSTEM_PROMPT,
    SUMMARIZE_PROMPT,
    SYSTEM_PROMPT,
)


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    related_notes: list[dict] = field(default_factory=list)
    research_steps: list[dict] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    reproducible: bool = True


class AnswerGenerator:
    def __init__(
        self,
        backend: str = "local",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:7b",
        mlx_model_name: str = "mlx-community/Qwen3-4B-4bit",
        mlx_model_enhanced: str = "mlx-community/Qwen3-8B-4bit",
        claude_model: str = "claude-sonnet-4-6",
        claude_api_key: str = "",
        seed: int | None = 42,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        allow_model_download: bool = False,
    ):
        self.backend = backend
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.mlx_model_name = mlx_model_name
        self.mlx_model_enhanced = mlx_model_enhanced
        self.claude_model = claude_model
        self.claude_api_key = claude_api_key
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.allow_model_download = allow_model_download
        self._anthropic_client: Any | None = None
        # MLX model instance (loaded lazily via _load_mlx_model)
        self._mlx_model: Any | None = None
        self._mlx_tokenizer: Any | None = None

    @property
    def current_model(self) -> str:
        """当前使用的模型名（用于 /api/llm 端点展示）。"""
        if self.backend == "mlx":
            return self.mlx_model_name
        if self.backend == "claude":
            return self.claude_model
        return self.ollama_model

    def summarize(self, file_name: str, chunks: list[dict]) -> str:
        """为单个文件生成结构化摘要（Markdown 格式）。"""
        if not chunks:
            return f"## {file_name}\n\n无法获取文档内容。"
        context = build_context(chunks)
        user_msg = f"文件名：{file_name}\n\n文档内容片段：\n{context}"
        if self.backend == "claude":
            text = self._call_with_system(SUMMARIZE_PROMPT, user_msg)
        elif self.backend == "mlx":
            text = self._call_mlx(SUMMARIZE_PROMPT, user_msg)
        else:
            text = self._call_ollama_with_system(SUMMARIZE_PROMPT, user_msg)
        return f"## {file_name}\n\n{text}"

    def generate_knowledge_output(self, output_type: str, title: str, source_text: str) -> str:
        """基于手动输入或文件片段生成可入库的 Markdown 知识产物。"""
        output = get_knowledge_output_type(output_type)
        source = source_text.strip()
        if not source:
            raise ValueError("Knowledge output source is empty")
        system_prompt = f"{KNOWLEDGE_OUTPUT_SYSTEM_PROMPT}\n\n产物要求：{output.instruction}"
        user_msg = f"产物标题：{title}\n产物类型：{output.label}\n\n资料：\n{source}"
        if self.backend == "claude":
            return self._call_with_system(system_prompt, user_msg)
        if self.backend == "mlx":
            return self._call_mlx(system_prompt, user_msg)
        return self._call_ollama_with_system(system_prompt, user_msg)

    def generate(
        self,
        query: str,
        chunks: list[dict],
        conversation_context: list[dict] | None = None,
    ) -> Answer:
        """
        chunks: list of retriever result dicts with keys:
            text, file_name, page_num, rerank_score
        """
        if not chunks:
            return Answer(text="在现有文档中未找到相关信息。", citations=[])

        context = build_context(chunks)
        user_msg = build_user_message(query, context, conversation_context)

        if self.backend == "claude":
            answer_text = self._call_with_system(SYSTEM_PROMPT, user_msg)
        elif self.backend == "mlx":
            answer_text = self._call_mlx(SYSTEM_PROMPT, user_msg)
        else:
            answer_text = self._call_ollama_with_system(SYSTEM_PROMPT, user_msg)

        citations = validate_citations(
            [citation_from_chunk(chunk) for chunk in chunks],
            chunks,
        )
        answer_text, citations = apply_structured_citations(answer_text, citations)
        answer_text = sanitize_inline_citations(answer_text, citations)
        quality = {"claim_support": audit_answer_claim_support(answer_text, citations)}
        return Answer(
            text=answer_text,
            citations=citations,
            quality=quality,
            reproducible=self.is_reproducible,
        )

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        return build_context(chunks)

    @classmethod
    def _build_user_message(
        cls,
        query: str,
        context: str,
        conversation_context: list[dict] | None = None,
    ) -> str:
        return build_user_message(query, context, conversation_context)

    @staticmethod
    def _build_conversation_context(messages: list[dict]) -> str:
        return build_conversation_context(messages)

    # ------------------------------------------------------------------
    # Ollama (local)
    # ------------------------------------------------------------------

    def _call_ollama_with_system(self, system_prompt: str, user_msg: str) -> str:
        return backend_calls.call_ollama_with_system(self, system_prompt, user_msg)

    def _stream_ollama_with_system(self, system_prompt: str, user_msg: str, cancel_event=None):
        """Yield token strings as they arrive from Ollama."""
        yield from backend_calls.stream_ollama_with_system(
            self,
            system_prompt,
            user_msg,
            cancel_event=cancel_event,
        )

    def generate_stream(
        self,
        query: str,
        chunks: list[dict],
        cancel_event=None,
        conversation_context: list[dict] | None = None,
    ):
        """Yield token strings; caller is responsible for building Answer."""
        if cancel_event is not None and cancel_event.is_set():
            return
        if not chunks:
            yield "在现有文档中未找到相关信息。"
            return
        context = build_context(chunks)
        user_msg = build_user_message(query, context, conversation_context)
        if self.backend == "mlx":
            yield from self._stream_mlx(SYSTEM_PROMPT, user_msg, cancel_event=cancel_event)
        elif self.backend == "claude":
            if cancel_event is not None and cancel_event.is_set():
                return
            yield self._call_with_system(SYSTEM_PROMPT, user_msg)
        else:
            yield from self._stream_ollama_with_system(
                SYSTEM_PROMPT,
                user_msg,
                cancel_event=cancel_event,
            )

    # ------------------------------------------------------------------
    # MLX (in-process, Apple Silicon)
    # ------------------------------------------------------------------

    @property
    def is_reproducible(self) -> bool:
        return self.backend in {"local", "ollama", "mlx"} and (
            self.temperature == 0.0 or self.seed is not None
        )

    def _ollama_options(self) -> dict:
        return backend_calls.ollama_options(self)

    def _mlx_generation_kwargs(self) -> dict:
        return backend_calls.mlx_generation_kwargs(self)

    def _load_mlx_model(self, model_name: str | None = None) -> None:
        """加载（或切换）MLX LLM 模型。必须在 ml_executor 线程内调用。"""
        backend_calls.load_mlx_model(self, model_name)

    def _build_prompt_nothink(self, system: str, user: str) -> str:
        """构建 enable_thinking=False prompt，注入空 think 块，跳过推理过程。"""
        return backend_calls.build_prompt_nothink(self, system, user)

    def _stream_mlx(self, system: str, user: str, cancel_event=None):
        """通过 mlx_lm.stream_generate 逐 token yield。"""
        yield from backend_calls.stream_mlx(self, system, user, cancel_event=cancel_event)

    def _call_mlx(self, system: str, user: str) -> str:
        """非流式 MLX 生成（用于 summarize / generate）。"""
        return backend_calls.call_mlx(self, system, user)

    # ------------------------------------------------------------------
    # Claude API
    # ------------------------------------------------------------------

    def _call_with_system(self, system_prompt: str, user_msg: str) -> str:
        return backend_calls.call_claude_with_system(self, system_prompt, user_msg)

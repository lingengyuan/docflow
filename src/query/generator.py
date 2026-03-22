"""
AnswerGenerator — 基于检索结果生成带引用的答案。

支持两种后端：
  - local: Qwen2.5-7B via Ollama
  - claude: Claude API (claude-sonnet-4-6)
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field

SYSTEM_PROMPT = """你是一个专业的文档问答助手。请严格基于提供的文档片段回答问题。

规则：
1. 每个关键事实必须附带引用：[来源: 文件名, 第N页]
2. 若文档中未找到答案，明确回答"在现有文档中未找到相关信息"，不要编造
3. 回答使用中文，简洁专业
4. 若多个文档有相关信息，综合后回答并分别标注来源"""

SUMMARIZE_PROMPT = """你是一个专业的文档摘要助手。请基于提供的文档片段生成结构化摘要。

输出格式（Markdown）：
1. **文档主题**：1-2句话概括文档核心内容
2. **核心要点**：3-5个要点，每点一行（使用 - 列表）
3. **关键数据/结论**：提取重要数字、日期、结论（如无则省略此节）

规则：使用中文，简洁专业，严格基于文档内容，不要编造。"""


@dataclass
class Citation:
    file_name: str
    page_num: int
    snippet: str
    score: float


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)


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
    ):
        self.backend = backend
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.mlx_model_name = mlx_model_name
        self.mlx_model_enhanced = mlx_model_enhanced
        self.claude_model = claude_model
        self.claude_api_key = claude_api_key
        # MLX model instance (loaded lazily via _load_mlx_model)
        self._mlx_model = None
        self._mlx_tokenizer = None

    @property
    def current_model(self) -> str:
        """当前使用的模型名（用于 /api/llm 端点展示）。"""
        if self.backend == "mlx":
            return self.mlx_model_name
        return self.ollama_model

    def summarize(self, file_name: str, chunks: list[dict]) -> str:
        """为单个文件生成结构化摘要（Markdown 格式）。"""
        if not chunks:
            return f"## {file_name}\n\n无法获取文档内容。"
        context = self._build_context(chunks)
        user_msg = f"文件名：{file_name}\n\n文档内容片段：\n{context}"
        if self.backend == "claude":
            text = self._call_with_system(SUMMARIZE_PROMPT, user_msg)
        elif self.backend == "mlx":
            text = self._call_mlx(SUMMARIZE_PROMPT, user_msg)
        else:
            text = self._call_ollama_with_system(SUMMARIZE_PROMPT, user_msg)
        return f"## {file_name}\n\n{text}"

    def generate(self, query: str, chunks: list[dict]) -> Answer:
        """
        chunks: list of retriever result dicts with keys:
            text, file_name, page_num, rerank_score
        """
        if not chunks:
            return Answer(text="在现有文档中未找到相关信息。", citations=[])

        context = self._build_context(chunks)
        user_msg = f"问题：{query}\n\n文档片段：\n{context}"

        if self.backend == "claude":
            answer_text = self._call_with_system(SYSTEM_PROMPT, user_msg)
        elif self.backend == "mlx":
            answer_text = self._call_mlx(SYSTEM_PROMPT, user_msg)
        else:
            answer_text = self._call_ollama_with_system(SYSTEM_PROMPT, user_msg)

        citations = [
            Citation(
                file_name=c["file_name"],
                page_num=c["page_num"],
                snippet=c["text"][:200],
                score=c.get("rerank_score", c.get("rrf_score", 0.0)),
            )
            for c in chunks
        ]
        return Answer(text=answer_text, citations=citations)

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            section = f" > {c['section']}" if c.get("section") else ""
            parts.append(
                f"[片段{i}] 来源: {c['file_name']}, 第{c['page_num']}页{section}\n{c['text']}"
            )
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Ollama (local)
    # ------------------------------------------------------------------

    def _call_ollama_with_system(self, system_prompt: str, user_msg: str) -> str:
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {"think": False},  # Qwen3 thinking mode off：RAG 不需要思考过程
        }
        req = urllib.request.Request(
            f"{self.ollama_base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.load(resp)
        return result["message"]["content"].strip()

    def _stream_ollama_with_system(self, system_prompt: str, user_msg: str):
        """Yield token strings as they arrive from Ollama."""
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "stream": True,
            "options": {"think": False},  # Qwen3 thinking mode off：RAG 不需要思考过程
        }
        req = urllib.request.Request(
            f"{self.ollama_base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token

    def generate_stream(self, query: str, chunks: list[dict]):
        """Yield token strings; caller is responsible for building Answer."""
        if not chunks:
            yield "在现有文档中未找到相关信息。"
            return
        context = self._build_context(chunks)
        user_msg = f"问题：{query}\n\n文档片段：\n{context}"
        if self.backend == "mlx":
            yield from self._stream_mlx(SYSTEM_PROMPT, user_msg)
        else:
            yield from self._stream_ollama_with_system(SYSTEM_PROMPT, user_msg)

    # ------------------------------------------------------------------
    # MLX (in-process, Apple Silicon)
    # ------------------------------------------------------------------

    def _load_mlx_model(self, model_name: str | None = None) -> None:
        """加载（或切换）MLX LLM 模型。必须在 ml_executor 线程内调用。"""
        from mlx_lm import load
        target = model_name or self.mlx_model_name
        if self._mlx_model is None or target != self.mlx_model_name:
            self._mlx_model, self._mlx_tokenizer = load(target)
            self.mlx_model_name = target

    def _build_prompt_nothink(self, system: str, user: str) -> str:
        """构建 enable_thinking=False prompt，注入空 think 块，跳过推理过程。"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._mlx_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # 注入 <think>\n\n</think>\n\n 前缀
        )

    def _stream_mlx(self, system: str, user: str):
        """通过 mlx_lm.stream_generate 逐 token yield。"""
        from mlx_lm import stream_generate
        prompt = self._build_prompt_nothink(system, user)
        for response in stream_generate(
            self._mlx_model, self._mlx_tokenizer,
            prompt=prompt, max_tokens=2048,
        ):
            if response.text:
                yield response.text

    def _call_mlx(self, system: str, user: str) -> str:
        """非流式 MLX 生成（用于 summarize / generate）。"""
        from mlx_lm import generate as mlx_generate
        prompt = self._build_prompt_nothink(system, user)
        return mlx_generate(
            self._mlx_model, self._mlx_tokenizer,
            prompt=prompt, max_tokens=2048, verbose=False,
        )

    # ------------------------------------------------------------------
    # Claude API
    # ------------------------------------------------------------------

    def _call_with_system(self, system_prompt: str, user_msg: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.claude_api_key)
        message = client.messages.create(
            model=self.claude_model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        return message.content[0].text.strip()

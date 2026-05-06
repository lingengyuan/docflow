"""
测试 AnswerGenerator 的 context 构建和 API 调用（mock）。
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.query.generator import Answer, AnswerGenerator, Citation


CHUNKS = [
    {
        "text": "Q3 total sales: 7,460,000 RMB, up 15.3% YoY",
        "file_name": "Q3报告.pdf",
        "file_path": "/docs/Q3报告.pdf",
        "page_num": 12,
        "section": "第三章 > 销售数据",
        "chunk_type": "text",
        "rerank_score": 0.95,
        "rrf_score": 0.03,
    },
    {
        "text": "East China: 2,450,000 RMB",
        "file_name": "Q3报告.pdf",
        "file_path": "/docs/Q3报告.pdf",
        "page_num": 13,
        "section": "",
        "chunk_type": "text",
        "rerank_score": 0.80,
        "rrf_score": 0.02,
    },
]


class TestContextBuilder:
    def test_includes_file_and_page(self):
        ctx = AnswerGenerator._build_context(CHUNKS)
        assert "Q3报告.pdf" in ctx
        assert "第12页" in ctx
        assert "第13页" in ctx

    def test_includes_section(self):
        ctx = AnswerGenerator._build_context(CHUNKS)
        assert "第三章 > 销售数据" in ctx

    def test_includes_text(self):
        ctx = AnswerGenerator._build_context(CHUNKS)
        assert "7,460,000" in ctx

    def test_empty_chunks_context(self):
        ctx = AnswerGenerator._build_context([])
        assert ctx == ""

    def test_builds_conversation_context(self):
        user_msg = AnswerGenerator._build_user_message(
            "展开第二点",
            "文档内容",
            [
                {"role": "user", "content": "总结这份文档"},
                {"role": "assistant", "content": "第一点是本地优先。第二点是可验证。"},
            ],
        )

        assert "最近对话" in user_msg
        assert "用户：总结这份文档" in user_msg
        assert "DocFlow：第一点是本地优先。第二点是可验证。" in user_msg
        assert "当前问题：展开第二点" in user_msg


class TestOllamaGenerate:
    def _mock_ollama_response(self, text: str):
        body = json.dumps({"message": {"content": text}}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_returns_answer_with_citations(self):
        gen = AnswerGenerator(backend="local")
        mock_resp = self._mock_ollama_response("Q3总销售额为7,460,000元 [来源: Q3报告.pdf, 第12页]")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            answer = gen.generate("Q3销售额是多少", CHUNKS)

        assert isinstance(answer, Answer)
        assert "7,460,000" in answer.text
        assert len(answer.citations) == 2
        assert answer.citations[0].file_name == "Q3报告.pdf"
        assert answer.citations[0].file_path == "/docs/Q3报告.pdf"
        assert answer.citations[0].page_num == 12
        assert answer.citations[0].section == "第三章 > 销售数据"

    def test_empty_chunks_returns_no_info(self):
        gen = AnswerGenerator(backend="local")
        answer = gen.generate("任何问题", [])
        assert "未找到" in answer.text
        assert answer.citations == []

    def test_ollama_payload_format(self):
        gen = AnswerGenerator(backend="local", ollama_model="qwen2.5:7b")
        body = json.dumps({"message": {"content": "answer"}}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            gen.generate(
                "test question",
                CHUNKS,
                conversation_context=[{"role": "user", "content": "previous question"}],
            )

        req = mock_url.call_args[0][0]
        payload = json.loads(req.data.decode())
        assert payload["model"] == "qwen2.5:7b"
        assert payload["stream"] is False
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert "test question" in payload["messages"][1]["content"]
        assert "previous question" in payload["messages"][1]["content"]

    def test_knowledge_output_payload_uses_template_instruction(self):
        gen = AnswerGenerator(backend="local", ollama_model="qwen2.5:7b")
        body = json.dumps({"message": {"content": "## 当前状态\n\n完成"}}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            output = gen.generate_knowledge_output(
                "project_brief",
                "Phase Brief",
                "Phase 17 source text",
            )

        req = mock_url.call_args[0][0]
        payload = json.loads(req.data.decode())
        assert output.startswith("## 当前状态")
        assert "项目简报" in payload["messages"][1]["content"]
        assert "Phase 17 source text" in payload["messages"][1]["content"]
        assert "## 背景" in payload["messages"][0]["content"]

    def test_claude_stream_falls_back_to_single_chunk(self):
        gen = AnswerGenerator(backend="claude", claude_api_key="test-key")
        with patch.object(gen, "_call_with_system", return_value="claude answer") as mock_call:
            tokens = list(gen.generate_stream("test question", CHUNKS))

        assert tokens == ["claude answer"]
        mock_call.assert_called_once()

    def test_generate_stream_honors_pre_cancelled_event(self):
        import threading

        gen = AnswerGenerator(backend="claude", claude_api_key="test-key")
        cancel = threading.Event()
        cancel.set()

        with patch.object(gen, "_call_with_system", return_value="claude answer") as mock_call:
            tokens = list(gen.generate_stream("test question", CHUNKS, cancel_event=cancel))

        assert tokens == []
        mock_call.assert_not_called()

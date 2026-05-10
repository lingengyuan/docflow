from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _render_markdown(sample: str) -> str:
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    start = html.index("function escHtml")
    end = html.index("function appendUserMessage")
    script = (
        html[start:end]
        + "\n"
        + f"process.stdout.write(renderMarkdown({json.dumps(sample, ensure_ascii=False)}));"
    )
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    return result.stdout


def test_markdown_renderer_supports_common_answer_shapes():
    rendered = _render_markdown(
        "\n".join(
            [
                "# 标题",
                "",
                "- 第一项",
                "- **第二项**",
                "",
                "1. 步骤一",
                "2. 步骤二",
                "",
                "> 引用内容",
                "",
                "[链接](https://example.com)",
                "",
                "```",
                "print('hello')",
                "```",
            ]
        )
    )

    assert "<h3" in rendered
    assert "<ul><li>第一项</li><li><strong>第二项</strong></li></ul>" in rendered
    assert "<ol><li>步骤一</li><li>步骤二</li></ol>" in rendered
    assert "<blockquote" in rendered
    assert '<a href="https://example.com"' in rendered
    assert "<pre" in rendered
    assert "print(&#39;hello&#39;)" in rendered


def test_markdown_renderer_keeps_tables_and_escapes_html():
    rendered = _render_markdown(
        "\n".join(
            [
                "| 文件 | 说明 |",
                "| --- | --- |",
                "| README.md | `<script>` |",
            ]
        )
    )

    assert "<table" in rendered
    assert "<th>文件</th>" in rendered
    assert "&lt;script&gt;" in rendered
    assert "<script>" not in rendered

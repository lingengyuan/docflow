"""
StructuredChunker — 三层切块策略。

| 内容类型   | 策略                                                         |
|----------|--------------------------------------------------------------|
| 普通段落   | 递归切块，512 token，10% 重叠                                |
| 标题/章节  | 按 # / ## 边界切，chunk 携带面包屑路径                       |
| 表格       | 整表作为独立 chunk + 规则生成文字摘要作为检索入口             |
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    chunk_type: str          # "text" | "table" | "table_summary"
    file_name: str
    file_path: str
    page_num: int
    section: str = ""        # 面包屑路径，如 "章节一 > 1.2 背景"
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)


# ---------------------------------------------------------------------------
# StructuredChunker
# ---------------------------------------------------------------------------

class StructuredChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 51,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_page(
        self,
        text: str,
        file_name: str,
        file_path: str,
        page_num: int,
        is_ocr: bool = False,
    ) -> list[Chunk]:
        """
        对单页文字进行切块。OCR 输出的 Markdown 和原生文字均支持。
        返回的 Chunk 列表包含文字块、表格块和表格摘要块。
        """
        chunks: list[Chunk] = []
        breadcrumbs: list[str] = []  # 当前标题堆栈

        # 1. 分离表格（Markdown table）
        sections = self._split_tables(text)

        for segment_type, content in sections:
            content = content.strip()
            if not content:
                continue

            if segment_type == "table":
                # 整表作为独立 chunk
                chunks.append(Chunk(
                    text=content,
                    chunk_type="table",
                    file_name=file_name,
                    file_path=file_path,
                    page_num=page_num,
                    section=" > ".join(breadcrumbs),
                ))
                # 规则生成摘要
                summary = self._table_summary(content, breadcrumbs)
                if summary:
                    chunks.append(Chunk(
                        text=summary,
                        chunk_type="table_summary",
                        file_name=file_name,
                        file_path=file_path,
                        page_num=page_num,
                        section=" > ".join(breadcrumbs),
                    ))
            else:
                # 文字段：按标题边界分组，再递归切块
                sub_sections = self._split_by_headers(content)
                for header, body in sub_sections:
                    if header:
                        # 更新面包屑
                        level = self._header_level(header)
                        breadcrumbs = breadcrumbs[:level - 1] + [header.lstrip("#").strip()]

                    if not body.strip():
                        continue

                    section_path = " > ".join(breadcrumbs)
                    for sub_chunk in self._recursive_split(body.strip()):
                        chunks.append(Chunk(
                            text=sub_chunk,
                            chunk_type="text",
                            file_name=file_name,
                            file_path=file_path,
                            page_num=page_num,
                            section=section_path,
                        ))

        return chunks

    # ------------------------------------------------------------------
    # Table detection
    # ------------------------------------------------------------------

    def _split_tables(self, text: str) -> list[tuple[str, str]]:
        """
        将文本分割为 (type, content) 列表，type 为 "text" 或 "table"。
        检测 Markdown 表格（含 | 分隔符的连续行）。
        """
        lines = text.splitlines()
        segments: list[tuple[str, str]] = []
        current_type = "text"
        current_lines: list[str] = []

        def flush():
            if current_lines:
                segments.append((current_type, "\n".join(current_lines)))
                current_lines.clear()

        i = 0
        while i < len(lines):
            line = lines[i]
            in_table = self._is_table_line(line)

            if in_table and current_type == "text":
                flush()
                current_type = "table"
                current_lines.append(line)
            elif not in_table and current_type == "table":
                flush()
                current_type = "text"
                current_lines.append(line)
            else:
                current_lines.append(line)
            i += 1

        flush()
        return segments

    @staticmethod
    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        # Markdown table: starts and/or ends with |, or is a separator row (|---|)
        return bool(re.match(r"^\|.+\|", stripped) or re.match(r"^\|[-:| ]+\|", stripped))

    # ------------------------------------------------------------------
    # Header splitting
    # ------------------------------------------------------------------

    def _split_by_headers(self, text: str) -> list[tuple[str, str]]:
        """
        按 Markdown 标题行（# / ##  / ###）分割文本。
        返回 [(header_line, body_text), ...]，首段 header_line 为空。
        """
        header_re = re.compile(r"^(#{1,6})\s+.+", re.MULTILINE)
        result: list[tuple[str, str]] = []
        last_end = 0
        last_header = ""

        for m in header_re.finditer(text):
            body = text[last_end:m.start()].strip()
            result.append((last_header, body))
            last_header = m.group(0)
            last_end = m.end()

        result.append((last_header, text[last_end:].strip()))
        return [(h, b) for h, b in result if h or b]

    @staticmethod
    def _header_level(header_line: str) -> int:
        m = re.match(r"^(#{1,6})", header_line.strip())
        return len(m.group(1)) if m else 1

    # ------------------------------------------------------------------
    # Recursive text splitting
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str) -> list[str]:
        """
        递归按优先级分割：\n\n → \n → 句号 → 空格。
        chunk_size 以字符数估算（中文 1char ≈ 1.5 token，英文 4char ≈ 1 token）。
        保守估算：chunk_size * 2 个字符 ≈ chunk_size tokens。
        """
        max_chars = self.chunk_size * 2
        overlap_chars = self.chunk_overlap * 2

        if len(text) <= max_chars:
            return [text] if text.strip() else []

        separators = ["\n\n", "\n", "。", ".", " "]
        return self._split_recursive(text, separators, max_chars, overlap_chars)

    def _split_recursive(
        self, text: str, separators: list[str], max_chars: int, overlap: int
    ) -> list[str]:
        if len(text) <= max_chars:
            return [text] if text.strip() else []

        sep = next((s for s in separators if s in text), None)
        if sep is None:
            # No separator found, hard split
            return self._hard_split(text, max_chars, overlap)

        parts = text.split(sep)
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).lstrip(sep) if current else part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current.strip():
                    # Recursively split current if still too large
                    sub = self._split_recursive(current, separators[1:], max_chars, overlap)
                    chunks.extend(sub)
                    # Overlap: prepend last chunk's tail to new current
                    if sub:
                        tail = sub[-1][-overlap:] if len(sub[-1]) > overlap else sub[-1]
                        current = tail + sep + part if tail else part
                    else:
                        current = part
                else:
                    current = part

        if current.strip():
            sub = self._split_recursive(current, separators[1:], max_chars, overlap)
            chunks.extend(sub)

        return chunks

    @staticmethod
    def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunks.append(text[start:end])
            start = end - overlap
        return [c for c in chunks if c.strip()]

    # ------------------------------------------------------------------
    # Table summary (rule-based, no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _table_summary(table_text: str, breadcrumbs: list[str]) -> str:
        """
        规则生成表格摘要，作为检索入口。
        格式：[表格] {上下文} 包含 {N 行} 数据，列：{col1, col2, ...}
        """
        lines = [l.strip() for l in table_text.splitlines() if l.strip()]
        # Find header row (first non-separator row with |)
        header_row = ""
        data_rows = 0
        for line in lines:
            if re.match(r"^\|[-:| ]+\|", line):
                continue  # separator
            if not header_row:
                header_row = line
            else:
                data_rows += 1

        if not header_row:
            return ""

        # Extract column names
        cols = [c.strip() for c in header_row.strip("|").split("|") if c.strip()]
        context = " > ".join(breadcrumbs) if breadcrumbs else "文档"
        col_str = "、".join(cols[:6])  # max 6 cols shown
        if len(cols) > 6:
            col_str += f"等{len(cols)}列"

        return f"[表格] {context} 包含 {data_rows} 行数据，列：{col_str}"

"""MarkdownParser — UTF-8 读取 .md 文件，作为单页文档返回。"""

from __future__ import annotations

from pathlib import Path

from src.ingest.pdf_analyzer import PageContent, ParsedDocument


class MarkdownParser:
    def parse(self, file_path: Path) -> ParsedDocument:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        page = PageContent(page_num=1, text=text, headers=[], is_ocr=False)
        return ParsedDocument(
            file_path=file_path,
            file_name=file_path.name,
            total_pages=1,
            is_scanned=False,
            pages=[page],
        )

"""
FileParser Protocol — 所有文件解析器的统一接口。

所有 Parser 必须实现 parse(file_path) → ParsedDocument。
ParsedDocument / PageContent 定义复用 pdf_analyzer.py 中的数据类。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.ingest.pdf_analyzer import ParsedDocument  # noqa: F401 (re-export)


class FileParser(Protocol):
    def parse(self, file_path: Path) -> ParsedDocument:
        ...

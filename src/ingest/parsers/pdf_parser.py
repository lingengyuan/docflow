"""PDFParser — 薄包装 PDFAnalyzer，统一 FileParser 接口。"""

from __future__ import annotations

from pathlib import Path

from src.ingest.pdf_analyzer import ParsedDocument, PDFAnalyzer


class PDFParser:
    def __init__(self, ollama_base_url: str = "http://localhost:11434", ocr_model: str = "glm-ocr"):
        self._analyzer = PDFAnalyzer(ollama_base_url=ollama_base_url, ocr_model=ocr_model)

    def parse(self, file_path: Path) -> ParsedDocument:
        return self._analyzer.analyze(file_path)

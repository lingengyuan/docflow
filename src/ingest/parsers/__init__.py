"""
ParserRegistry — 根据文件扩展名选择对应的 FileParser。

用法：
    registry = ParserRegistry.from_config(cfg)
    doc = registry.resolve(path).parse(path)
"""

from __future__ import annotations

from pathlib import Path

from src.ingest.parsers.docx_parser import DocxParser
from src.ingest.parsers.image_parser import ImageParser
from src.ingest.parsers.markdown_parser import MarkdownParser
from src.ingest.parsers.pdf_parser import PDFParser
from src.ingest.parsers.txt_parser import TxtParser


class ParserRegistry:
    def __init__(self):
        self._parsers: dict[str, object] = {}

    def register(self, ext: str, parser: object) -> None:
        self._parsers[ext.lower()] = parser

    def resolve(self, file_path: Path) -> object:
        ext = file_path.suffix.lower()
        parser = self._parsers.get(ext)
        if parser is None:
            raise ValueError(f"No parser registered for extension: {ext}")
        return parser

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self._parsers

    @property
    def supported_extensions(self) -> list[str]:
        return list(self._parsers.keys())

    @classmethod
    def from_config(cls, cfg: dict) -> "ParserRegistry":
        registry = cls()
        ollama_url = cfg["ollama"]["base_url"]
        ocr_model = cfg["ollama"]["ocr_model"]
        registry.register(".pdf", PDFParser(ollama_base_url=ollama_url, ocr_model=ocr_model))
        registry.register(".md", MarkdownParser())
        registry.register(".markdown", MarkdownParser())
        registry.register(".txt", TxtParser())
        registry.register(".docx", DocxParser())

        vlm_cfg = cfg.get("vlm", {})
        if vlm_cfg.get("enabled", True):
            image_parser = ImageParser(
                vlm_model=vlm_cfg.get("model", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"),
                prompt=vlm_cfg.get("prompt", ""),
                max_tokens=vlm_cfg.get("max_tokens", 512),
            )
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"):
                registry.register(ext, image_parser)

        return registry

"""
PDFAnalyzer — 检测 PDF 类型并路由到对应解析路径。

双路策略：
  - 原生文字 PDF（文字覆盖率 ≥ 10%）→ PyMuPDF 快速提取
  - 扫描件 / 图片型 PDF（覆盖率 < 10%）→ GLM-OCR 兜底
"""

from __future__ import annotations

import base64
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PageContent:
    page_num: int          # 1-based
    text: str
    headers: list[str]     # lines identified as headers (large font / bold)
    is_ocr: bool = False   # True if this page was processed by GLM-OCR


@dataclass
class ParsedDocument:
    file_path: Path
    file_name: str
    total_pages: int
    is_scanned: bool
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # parser 可附加元数据，如 {"tags": [...]}

    def all_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


# ---------------------------------------------------------------------------
# PDFAnalyzer
# ---------------------------------------------------------------------------

class PDFAnalyzer:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ocr_model: str = "glm-ocr",
        text_coverage_threshold: float = 0.10,
        ocr_detection_pages: int = 3,
    ):
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ocr_model = ocr_model
        self.text_coverage_threshold = text_coverage_threshold
        self.ocr_detection_pages = ocr_detection_pages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, pdf_path: str | Path) -> ParsedDocument:
        """解析 PDF，自动选择路径，返回结构化文档。"""
        path = Path(pdf_path)
        doc = fitz.open(str(path))
        is_scanned = self._is_scanned(doc)
        result = ParsedDocument(
            file_path=path,
            file_name=path.name,
            total_pages=len(doc),
            is_scanned=is_scanned,
        )
        for page_content in self._parse_pages(doc, is_scanned):
            result.pages.append(page_content)
        doc.close()
        return result

    # ------------------------------------------------------------------
    # Type detection
    # ------------------------------------------------------------------

    def _is_scanned(self, doc: fitz.Document) -> bool:
        """
        取前 N 页，统计可提取的非空字符数。
        原生 PDF 至少有若干可选文字；扫描件提取结果为空或极少。
        阈值：sample 页总字符数 < 20 视为扫描件。
        （text_coverage_threshold 字段保留供未来密度模式使用）
        """
        sample_pages = min(self.ocr_detection_pages, len(doc))
        total_chars = 0
        for i in range(sample_pages):
            text = doc[i].get_text("text")
            total_chars += sum(1 for c in text if c.strip())
        return total_chars < 20

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_pages(
        self, doc: fitz.Document, is_scanned: bool
    ) -> Iterator[PageContent]:
        for i, page in enumerate(doc):
            page_num = i + 1
            if is_scanned:
                yield self._parse_page_ocr(page, page_num)
            else:
                yield self._parse_page_native(page, page_num)

    def _parse_page_native(self, page: fitz.Page, page_num: int) -> PageContent:
        """PyMuPDF 快速路径：提取文字 + 识别标题（按字体大小）。"""
        text = page.get_text("text").strip()
        headers = self._extract_headers(page)
        return PageContent(page_num=page_num, text=text, headers=headers)

    def _extract_headers(self, page: fitz.Page) -> list[str]:
        """
        通过字体大小识别标题行：
        字体 >= body_size * 1.2 或 flags & bold 的行视为标题。
        """
        blocks = page.get_text("dict")["blocks"]
        font_sizes: list[float] = []
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_sizes.append(span["size"])

        if not font_sizes:
            return []

        body_size = sorted(font_sizes)[len(font_sizes) // 2]  # median
        headers: list[str] = []
        seen: set[str] = set()

        for block in blocks:
            for line in block.get("lines", []):
                line_text = "".join(s["text"] for s in line.get("spans", []))
                line_text = line_text.strip()
                if not line_text or line_text in seen:
                    continue
                for span in line.get("spans", []):
                    is_large = span["size"] >= body_size * 1.2
                    is_bold = bool(span["flags"] & 2**4)
                    if is_large or is_bold:
                        headers.append(line_text)
                        seen.add(line_text)
                        break

        return headers

    def _parse_page_ocr(self, page: fitz.Page, page_num: int) -> PageContent:
        """
        GLM-OCR 路径：将页面渲染为图片，调用 Ollama GLM-OCR API。
        返回 Markdown 格式的文字。
        """
        # Render page to PNG (2x zoom for better OCR accuracy)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()

        markdown_text = self._call_glm_ocr(img_b64)

        return PageContent(
            page_num=page_num,
            text=markdown_text,
            headers=self._extract_headers_from_markdown(markdown_text),
            is_ocr=True,
        )

    def _call_glm_ocr(self, img_b64: str) -> str:
        """调用 Ollama GLM-OCR，返回识别的 Markdown 文字。"""
        payload = {
            "model": self.ocr_model,
            "prompt": (
                "Please perform OCR on this document image. "
                "Output the complete text content in markdown format, "
                "preserving the document structure including headings, "
                "tables, and paragraphs."
            ),
            "images": [img_b64],
            "stream": False,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.ollama_base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.load(resp)
        return result.get("response", "").strip()

    @staticmethod
    def _extract_headers_from_markdown(text: str) -> list[str]:
        """从 Markdown 文本中提取 # / ## / ### 标题。"""
        headers = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                header_text = stripped.lstrip("#").strip()
                if header_text:
                    headers.append(header_text)
        return headers

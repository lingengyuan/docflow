"""
测试 PDFAnalyzer 的类型检测和双路解析。

运行：cd ~/Projects/docflow && .venv/bin/python -m pytest tests/test_pdf_analyzer.py -v
"""

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest

from src.ingest.pdf_analyzer import PDFAnalyzer, ParsedDocument, PageContent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_native_pdf(tmp_path: Path) -> Path:
    """生成一个含中英文的原生文字 PDF。"""
    out = tmp_path / "native.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Use ASCII-safe text (Chinese fonts not embedded in test env)
    page.insert_text((50, 72), "Sales Report Q3 2024", fontsize=18)
    page.insert_text((50, 120), "East China: 2,450,000 RMB", fontsize=11)
    page.insert_text((50, 140), "North China: 1,890,000 RMB", fontsize=11)
    page.insert_text((50, 160), "South China: 3,120,000 RMB", fontsize=11)
    page.insert_text((50, 200), "Q3 total: 7,460,000 RMB (+15.3% YoY)", fontsize=11)
    doc.save(str(out))
    doc.close()
    return out


def make_blank_pdf(tmp_path: Path) -> Path:
    """生成一个完全空白（无文字）的 PDF，模拟扫描件。"""
    out = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(out))
    doc.close()
    return out


# ---------------------------------------------------------------------------
# Type detection tests
# ---------------------------------------------------------------------------

class TestTypeDetection:
    def test_native_pdf_is_not_scanned(self, tmp_path):
        pdf = make_native_pdf(tmp_path)
        analyzer = PDFAnalyzer()
        doc = fitz.open(str(pdf))
        assert analyzer._is_scanned(doc) is False
        doc.close()

    def test_blank_pdf_is_scanned(self, tmp_path):
        pdf = make_blank_pdf(tmp_path)
        analyzer = PDFAnalyzer()
        doc = fitz.open(str(pdf))
        assert analyzer._is_scanned(doc) is True
        doc.close()


# ---------------------------------------------------------------------------
# Native path tests
# ---------------------------------------------------------------------------

class TestNativeParsing:
    def test_extracts_text(self, tmp_path):
        pdf = make_native_pdf(tmp_path)
        analyzer = PDFAnalyzer()
        result = analyzer.analyze(pdf)

        assert isinstance(result, ParsedDocument)
        assert result.total_pages == 1
        assert result.is_scanned is False
        assert len(result.pages) == 1

        page = result.pages[0]
        assert page.page_num == 1
        assert "2,450,000" in page.text
        assert "7,460,000" in page.text
        assert page.is_ocr is False

    def test_extracts_headers(self, tmp_path):
        pdf = make_native_pdf(tmp_path)
        analyzer = PDFAnalyzer()
        result = analyzer.analyze(pdf)
        # The 18pt title should be detected as a header (body ~11pt, 18 >= 11*1.2)
        assert len(result.pages[0].headers) >= 1
        assert any("Sales Report" in h or "Q3" in h for h in result.pages[0].headers)

    def test_all_text_joins_pages(self, tmp_path):
        pdf = make_native_pdf(tmp_path)
        analyzer = PDFAnalyzer()
        result = analyzer.analyze(pdf)
        full = result.all_text()
        assert "Q3" in full


# ---------------------------------------------------------------------------
# OCR path tests (mocked)
# ---------------------------------------------------------------------------

class TestOCRParsing:
    MOCK_OCR_RESPONSE = "# Sales Report Q3\n\nEast China: 2,450,000 RMB\n\n## Summary\n\nTotal: 7,460,000 RMB"

    def test_ocr_called_for_scanned_pdf(self, tmp_path):
        pdf = make_blank_pdf(tmp_path)
        analyzer = PDFAnalyzer()

        with patch.object(analyzer, "_call_glm_ocr", return_value=self.MOCK_OCR_RESPONSE) as mock_ocr:
            result = analyzer.analyze(pdf)

        assert result.is_scanned is True
        mock_ocr.assert_called_once()
        page = result.pages[0]
        assert page.is_ocr is True
        assert "7,460,000" in page.text

    def test_ocr_headers_extracted_from_markdown(self, tmp_path):
        pdf = make_blank_pdf(tmp_path)
        analyzer = PDFAnalyzer()

        with patch.object(analyzer, "_call_glm_ocr", return_value=self.MOCK_OCR_RESPONSE):
            result = analyzer.analyze(pdf)

        headers = result.pages[0].headers
        assert "Sales Report Q3" in headers
        assert "Summary" in headers

    def test_call_glm_ocr_sends_correct_payload(self):
        analyzer = PDFAnalyzer(ollama_base_url="http://localhost:11434", ocr_model="glm-ocr")
        fake_response = json.dumps({"response": "# Test OCR"}).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            result = analyzer._call_glm_ocr("base64imagedata")

        assert result == "# Test OCR"
        call_args = mock_url.call_args[0][0]
        payload = json.loads(call_args.data.decode())
        assert payload["model"] == "glm-ocr"
        assert payload["images"] == ["base64imagedata"]
        assert payload["stream"] is False

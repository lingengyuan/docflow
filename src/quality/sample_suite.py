from __future__ import annotations

import shutil
import sys
import types
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import fitz
from fastapi.testclient import TestClient

from src.ingest.chunker import StructuredChunker
from src.ingest.imports import build_knowledge_output_markdown
from src.ingest.parsers.image_parser import ImageParser
from src.ingest.pdf_analyzer import PDFAnalyzer
from src.ingest.store import DocStore
from src.ingest.watcher import WatchDir

DEFAULT_SAMPLE_DIR = Path("/tmp/docflow-phase21-samples")


def run_sample_suite(output_dir: str | Path = DEFAULT_SAMPLE_DIR) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    samples = generate_samples(root)

    checks: list[dict[str, Any]] = []
    _run_check(checks, "scanned_pdf_ocr", lambda: _check_scanned_pdf_ocr(samples["scanned_pdf"]))
    _run_check(checks, "vlm_image_parse", lambda: _check_vlm_image_parse(samples["screenshot_png"]))
    _run_check(checks, "table_chunking", lambda: _check_table_chunking(samples["table_markdown"]))
    _run_check(
        checks, "source_preview_api", lambda: _check_source_preview(samples["table_markdown"], root)
    )
    _run_check(
        checks,
        "knowledge_output_api",
        lambda: _check_knowledge_output(samples["table_markdown"], root),
    )

    passed = sum(1 for check in checks if check["passed"])
    return {
        "schema": "docflow.sample_suite.v1",
        "mode": "deterministic_local_samples",
        "output_dir": str(root),
        "samples": {key: str(path) for key, path in samples.items()},
        "checks": checks,
        "passed": passed,
        "failed": len(checks) - passed,
    }


def generate_samples(root: Path) -> dict[str, Path]:
    table_markdown = root / "phase21-table.md"
    table_markdown.write_text(
        "\n".join(
            [
                "# Phase21 Revenue Sample",
                "",
                "This sample validates markdown table chunking and source preview.",
                "",
                "| Region | Q3 Revenue | Growth |",
                "|---|---:|---:|",
                "| East | 2450000 | 15% |",
                "| North | 1890000 | 8% |",
                "| South | 3120000 | 17% |",
                "",
                "The table total is 7460000.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    screenshot_png = root / "phase21-screenshot.png"
    _render_png(
        screenshot_png,
        [
            "DocFlow Phase21 Screenshot Sample",
            "Status: local assets verified",
            "Chart: OCR, VLM, tables, source preview",
        ],
    )

    scanned_pdf = root / "phase21-scanned.pdf"
    _render_scanned_pdf(scanned_pdf)

    return {
        "table_markdown": table_markdown,
        "screenshot_png": screenshot_png,
        "scanned_pdf": scanned_pdf,
    }


def _run_check(
    checks: list[dict[str, Any]], check_id: str, fn: Callable[[], dict[str, Any]]
) -> None:
    try:
        details = fn()
        checks.append({"id": check_id, "passed": True, "details": details})
    except Exception as exc:
        checks.append(
            {
                "id": check_id,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _render_png(path: Path, lines: list[str]) -> None:
    doc = fitz.open()
    page = doc.new_page(width=720, height=420)
    page.draw_rect(fitz.Rect(24, 24, 696, 396), color=(0.18, 0.25, 0.33), width=1.5)
    y = 78
    for index, line in enumerate(lines):
        page.insert_text(
            (56, y),
            line,
            fontsize=22 if index == 0 else 16,
            color=(0.08, 0.12, 0.16),
        )
        y += 48
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    pix.save(path)
    doc.close()


def _render_scanned_pdf(path: Path) -> None:
    image_path = path.with_suffix(".png")
    _render_png(
        image_path,
        [
            "PHASE21 OCR SAMPLE",
            "Invoice total: 321 USD",
            "Line item: local scanned PDF",
        ],
    )
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(48, 72, 564, 372), filename=str(image_path))
    doc.save(path)
    doc.close()


def _check_scanned_pdf_ocr(pdf_path: Path) -> dict[str, Any]:
    analyzer = PDFAnalyzer()
    calls: list[int] = []

    def fake_ocr(img_b64: str) -> str:
        calls.append(len(img_b64))
        return "# Phase21 OCR Sample\n\nInvoice total: 321 USD\n\nLine item: local scanned PDF"

    analyzer._call_glm_ocr = fake_ocr  # type: ignore[method-assign]
    parsed = analyzer.analyze(pdf_path)
    assert parsed.is_scanned is True
    assert parsed.pages[0].is_ocr is True
    assert "Invoice total: 321 USD" in parsed.pages[0].text
    assert calls and calls[0] > 100
    return {
        "file": pdf_path.name,
        "is_scanned": parsed.is_scanned,
        "page_is_ocr": parsed.pages[0].is_ocr,
        "ocr_chars": len(parsed.pages[0].text),
    }


def _check_vlm_image_parse(image_path: Path) -> dict[str, Any]:
    with _fake_mlx_vlm("Phase21 VLM description: screenshot with local assets and table data."):
        parser = ImageParser(vlm_model="phase21/local-vlm", max_tokens=80)
        parsed = parser.parse(image_path)
    assert parsed.is_scanned is True
    assert parsed.pages[0].is_ocr is True
    assert "Phase21 VLM description" in parsed.pages[0].text
    return {
        "file": image_path.name,
        "is_scanned": parsed.is_scanned,
        "page_is_ocr": parsed.pages[0].is_ocr,
        "description_chars": len(parsed.pages[0].text),
    }


def _check_table_chunking(markdown_path: Path) -> dict[str, Any]:
    chunker = StructuredChunker(chunk_size=400, chunk_overlap=40)
    chunks = chunker.chunk_page(
        markdown_path.read_text(encoding="utf-8"),
        file_name=markdown_path.name,
        file_path=str(markdown_path),
        page_num=1,
    )
    chunk_types = [chunk.chunk_type for chunk in chunks]
    assert "table" in chunk_types
    assert "table_summary" in chunk_types
    summary = next(chunk for chunk in chunks if chunk.chunk_type == "table_summary")
    assert "Region" in summary.raw_text or "Region" in summary.text
    return {
        "file": markdown_path.name,
        "chunk_count": len(chunks),
        "chunk_types": chunk_types,
    }


def _check_source_preview(markdown_path: Path, root: Path) -> dict[str, Any]:
    from src.api import app as api_app

    store = DocStore(root / "preview.db")
    file_id = store.upsert_file(
        markdown_path,
        markdown_path.name,
        DocStore.compute_hash(markdown_path),
        status="done",
        total_pages=1,
        mtime_ns=markdown_path.stat().st_mtime_ns,
    )

    old_store = api_app.store
    try:
        api_app.store = store
        client = TestClient(api_app.app)
        head = client.head(f"/api/file/{file_id}/preview")
        body = client.get(f"/api/file/{file_id}/preview")
    finally:
        api_app.store = old_store

    assert head.status_code == 200
    assert body.status_code == 200
    assert "text/markdown" in head.headers["content-type"]
    assert "Phase21 Revenue Sample" in body.text
    return {
        "file_id": file_id,
        "content_type": head.headers["content-type"],
        "content_length": int(head.headers["content-length"]),
    }


def _check_knowledge_output(markdown_path: Path, root: Path) -> dict[str, Any]:
    from src.api import app as api_app

    store = DocStore(root / "knowledge.db")
    file_id = store.upsert_file(
        markdown_path,
        markdown_path.name,
        DocStore.compute_hash(markdown_path),
        status="done",
        total_pages=1,
        mtime_ns=markdown_path.stat().st_mtime_ns,
    )
    store.add_chunks(
        file_id,
        [
            {
                "qdrant_id": 21001,
                "chunk_type": "table",
                "page_num": 1,
                "section": "Phase21 Revenue Sample",
                "char_count": markdown_path.stat().st_size,
                "raw_text": markdown_path.read_text(encoding="utf-8"),
                "tokenized_text": "phase21 revenue sample table total",
            }
        ],
    )
    queued: list[Path] = []
    generated: dict[str, str] = {}

    class FakeQueue:
        def submit(self, path: Path):
            queued.append(path)
            return {"status": "queued", "file": path.name}

    class FakeRetriever:
        def fetch_file_chunks(self, qdrant_ids, max_chunks=12):
            assert qdrant_ids == [21001]
            return [
                {
                    "text": "Phase21 sample table total is 7460000.",
                    "file_name": markdown_path.name,
                    "page_num": 1,
                    "section": "Phase21 Revenue Sample",
                }
            ]

    class FakeQueryEngine:
        retriever = FakeRetriever()

        def generate_knowledge_output(self, output_type, title, source_text):
            generated["output_type"] = output_type
            generated["title"] = title or ""
            generated["source_text"] = source_text
            return "## 一句话结论\n\nPhase21 样本验证通过。\n\n## 核心要点\n\n- 表格总额 7460000。"

    old_store = api_app.store
    old_queue = api_app.ingest_queue
    old_engine = api_app.query_engine
    old_watch_dirs = api_app.watch_dirs
    try:
        api_app.store = store
        api_app.ingest_queue = FakeQueue()
        api_app.query_engine = FakeQueryEngine()
        api_app.watch_dirs = [WatchDir(path=root)]
        client = TestClient(api_app.app)
        response = client.post(
            "/api/knowledge-output",
            json={
                "output_type": "summary",
                "title": "Phase21 Sample Summary",
                "file_ids": [file_id],
                "collection": "Knowledge Outputs",
                "user_tags": ["phase21"],
            },
        )
    finally:
        api_app.store = old_store
        api_app.ingest_queue = old_queue
        api_app.query_engine = old_engine
        api_app.watch_dirs = old_watch_dirs

    assert response.status_code == 200
    body = response.json()
    output_path = Path(body["path"])
    saved = output_path.read_text(encoding="utf-8")
    assert "Phase21 样本验证通过" in saved
    assert "phase21-table.md" in saved
    assert body["source_files"] == [markdown_path.name]
    assert queued == [output_path]
    assert "Phase21 sample table total" in generated["source_text"]

    markdown = build_knowledge_output_markdown(
        "Phase21 Direct Builder",
        "summary",
        "## 一句话结论\n\nBuilder path validated.",
        source_files=[markdown_path.name],
        tags=["phase21"],
    )
    assert "knowledge-output" in markdown.markdown

    return {
        "file": output_path.name,
        "source_files": body["source_files"],
        "queued": [path.name for path in queued],
        "preview": body["preview"],
    }


@contextmanager
def _fake_mlx_vlm(description: str):
    old_modules = {
        name: sys.modules.get(name) for name in ("mlx_vlm", "mlx_vlm.utils", "mlx_vlm.prompt_utils")
    }
    missing = object()
    for name in old_modules:
        if old_modules[name] is None:
            old_modules[name] = missing

    mlx_vlm = types.ModuleType("mlx_vlm")
    utils = types.ModuleType("mlx_vlm.utils")
    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")

    class Result:
        text = description

    def load(model_name: str):
        return object(), object()

    def generate(*args, **kwargs):
        assert kwargs.get("image")
        return Result()

    def load_config(model_name: str):
        return {"model": model_name}

    def apply_chat_template(processor, config, messages, num_images=1):
        assert num_images == 1
        return "phase21 prompt"

    mlx_vlm.load = load
    mlx_vlm.generate = generate
    utils.load_config = load_config
    prompt_utils.apply_chat_template = apply_chat_template

    sys.modules["mlx_vlm"] = mlx_vlm
    sys.modules["mlx_vlm.utils"] = utils
    sys.modules["mlx_vlm.prompt_utils"] = prompt_utils
    try:
        yield
    finally:
        for name, module in old_modules.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

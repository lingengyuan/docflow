from __future__ import annotations

from pathlib import Path

from src.quality.sample_suite import generate_samples, run_sample_suite


def test_generate_samples_creates_real_files(tmp_path):
    samples = generate_samples(tmp_path)

    assert samples["table_markdown"].read_text(encoding="utf-8").startswith("# Phase21")
    assert samples["screenshot_png"].read_bytes().startswith(b"\x89PNG")
    assert samples["scanned_pdf"].read_bytes().startswith(b"%PDF")


def test_phase21_sample_suite_passes(tmp_path):
    report = run_sample_suite(tmp_path / "phase21")

    assert report["schema"] == "docflow.sample_suite.v1"
    assert report["passed"] == 5
    assert report["failed"] == 0
    assert {check["id"] for check in report["checks"]} == {
        "scanned_pdf_ocr",
        "vlm_image_parse",
        "table_chunking",
        "source_preview_api",
        "knowledge_output_api",
    }
    for path in report["samples"].values():
        assert Path(path).exists()

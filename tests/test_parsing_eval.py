from __future__ import annotations

from pathlib import Path

from src.quality.parsing_eval import load_expectations, run_parsing_eval


def test_load_parsing_expectations():
    expectations = load_expectations("eval/parsing_expected/parsing_v1.json")

    assert len(expectations) == 56
    assert [item.id for item in expectations[:2]] == ["markdown_table", "plain_text"]
    assert {item.id for item in expectations} >= {
        "native_text_pdf",
        "pdf_table_report",
        "office_docx",
        "python_sample",
        "markdown_mixed_language",
    }
    assert expectations[0].table_rows_min == 5
    assert any("Local Knowledge Hub" in item.metadata_terms for item in expectations)


def test_run_parsing_eval_passes_committed_corpus():
    report = run_parsing_eval(
        corpus_dir=Path("eval/parsing_corpus"),
        expected_path=Path("eval/parsing_expected/parsing_v1.json"),
        config_path=Path("config.example.yaml"),
    )

    assert report["schema"] == "docflow.parsing_eval.v1"
    assert report["passed"] == report["cases"] == 56
    assert report["failed"] == 0
    assert report["performance"]["total_chunks"] >= 31
    assert {item["id"] for item in report["results"]} >= {
        "markdown_table",
        "plain_text",
        "native_text_pdf",
        "office_docx",
        "markdown_dense_table",
        "obsidian_properties",
        "txt_ocr_noise",
    }

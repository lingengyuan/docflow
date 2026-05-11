from __future__ import annotations

from pathlib import Path

from src.quality.parsing_eval import load_expectations, run_parsing_eval


def test_load_parsing_expectations():
    expectations = load_expectations("eval/parsing_expected/parsing_v1.json")

    assert [item.id for item in expectations] == ["markdown_table", "plain_text"]
    assert expectations[0].table_rows_min == 5


def test_run_parsing_eval_passes_committed_corpus():
    report = run_parsing_eval(
        corpus_dir=Path("eval/parsing_corpus"),
        expected_path=Path("eval/parsing_expected/parsing_v1.json"),
        config_path=Path("config.example.yaml"),
    )

    assert report["schema"] == "docflow.parsing_eval.v1"
    assert report["passed"] == report["cases"] == 2
    assert report["failed"] == 0
    assert {item["id"] for item in report["results"]} == {"markdown_table", "plain_text"}

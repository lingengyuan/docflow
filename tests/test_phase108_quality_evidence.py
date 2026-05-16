from __future__ import annotations

import json
from pathlib import Path

from scripts.run_external_benchmark_status import load_catalog, summary
from scripts.run_external_retrieval_eval import build_scifact_subset, write_scifact_corpus
from scripts.run_faithfulness_eval import evaluate_cases, load_cases
from scripts.run_large_library_benchmark import run_large_library_benchmark


def test_scifact_subset_builds_external_cases_without_source_filter(tmp_path):
    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    _write_jsonl(
        dataset / "corpus.jsonl",
        [
            {"_id": "10", "title": "Alpha study", "text": "Alpha evidence text."},
            {"_id": "20", "title": "Beta study", "text": "Beta distractor text."},
        ],
    )
    _write_jsonl(dataset / "queries.jsonl", [{"_id": "1", "text": "Alpha claim."}])
    (dataset / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n1\t10\t1\n",
        encoding="utf-8",
    )

    subset = build_scifact_subset(
        dataset,
        query_limit=1,
        distractors_per_query=1,
        source_zip_sha256="abc123",
    )
    corpus_paths = write_scifact_corpus(subset, tmp_path / "corpus")

    assert subset.queries[0].id == "beir_scifact_1"
    assert subset.queries[0].expected_files == ["scifact-10.md"]
    assert subset.queries[0].expected_terms == []
    assert {path.name for path in corpus_paths} == {"scifact-10.md", "scifact-20.md"}


def test_faithfulness_fixture_covers_supported_and_failure_modes():
    report = evaluate_cases(load_cases(Path("eval/answer_faithfulness_v1.jsonl")))

    assert report["failed"] == 0
    result_ids = {result["id"] for result in report["results"]}
    assert {
        "supported_claim",
        "uncited_claim",
        "fabricated_source_marker",
        "wrong_source_text",
        "insufficient_evidence_message",
    } <= result_ids


def test_large_library_benchmark_reports_index_and_query_metrics(tmp_path):
    report = run_large_library_benchmark(documents=12, queries=3, tmp_root=tmp_path)

    assert report["schema"] == "docflow.large_library_benchmark.v1"
    assert report["passed"] is True
    assert report["indexing"]["documents"] == 12
    assert report["indexing"]["chunks"] >= 12
    assert report["query"]["cases"] == 3
    assert report["query"]["p95_ms"] >= report["query"]["p50_ms"]
    assert report["query"]["correct_top_file_count"] == 3
    assert all(result["passed"] for result in report["query"]["results"])


def test_external_catalog_tracks_archived_scifact_result():
    report = summary(load_catalog())
    beir = next(item for item in report["benchmarks"] if item["id"] == "beir")

    assert beir["status"] == "archived"
    assert Path(beir["archived_result"]).exists()
    assert beir["claim_note"].startswith("Archived subset only")
    assert report["external_benchmark_scores"] >= 1


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

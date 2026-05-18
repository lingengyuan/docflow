from __future__ import annotations

import json
from pathlib import Path

from scripts.run_external_benchmark_status import load_catalog, summary
from scripts.run_external_retrieval_eval import (
    build_beir_subset,
    build_scifact_subset,
    write_beir_corpus,
    write_scifact_corpus,
)
from scripts.run_faithfulness_eval import evaluate_cases, load_cases
from scripts.run_large_library_benchmark import _query_plan, run_large_library_benchmark


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


def test_generic_beir_subset_uses_dataset_specific_names(tmp_path):
    dataset = tmp_path / "nfcorpus"
    (dataset / "qrels").mkdir(parents=True)
    _write_jsonl(
        dataset / "corpus.jsonl",
        [
            {"_id": "MED/10", "title": "Nutrition", "text": "Diet evidence text."},
            {"_id": "MED/20", "title": "Sleep", "text": "Sleep distractor text."},
        ],
    )
    _write_jsonl(dataset / "queries.jsonl", [{"_id": "plos-1", "text": "Diet claim."}])
    (dataset / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nplos-1\tMED/10\t1\nplos-1\tMED/20\t1\n",
        encoding="utf-8",
    )

    subset = build_beir_subset(
        dataset,
        dataset_slug="nfcorpus",
        dataset_name="NFCorpus",
        category="external_beir_nfcorpus",
        query_limit=1,
        distractors_per_query=1,
        max_relevant_per_query=1,
        source_zip_sha256="def456",
    )
    corpus_paths = write_beir_corpus(subset, tmp_path / "corpus")

    assert subset.queries[0].id == "beir_nfcorpus_plos-1"
    assert subset.queries[0].category == "external_beir_nfcorpus"
    assert subset.queries[0].expected_files == ["nfcorpus-MED_10.md"]
    assert {path.name for path in corpus_paths} == {
        "nfcorpus-MED_10.md",
        "nfcorpus-MED_20.md",
    }


def test_faithfulness_fixture_covers_supported_and_failure_modes():
    report = evaluate_cases(load_cases(Path("eval/answer_faithfulness_v1.jsonl")))

    assert report["failed"] == 0
    assert report["cases"] >= 14
    result_ids = {result["id"] for result in report["results"]}
    assert {
        "supported_claim",
        "uncited_claim",
        "fabricated_source_marker",
        "wrong_source_text",
        "insufficient_evidence_message",
        "no_evidence_answer_without_citations",
        "partial_citation_two_sentences",
        "missing_page_source_marker",
        "file_level_marker_supported",
        "conflicting_sources",
        "stale_source_recommendation",
        "weak_citation_with_verified_marker",
        "multi_cited_claim_supported",
        "alternate_no_answer_marker",
    } <= result_ids
    results = {result["id"]: result for result in report["results"]}
    assert results["conflicting_sources"]["evidence"]["label"] == "存在冲突"
    assert results["stale_source_recommendation"]["evidence"]["recommendations"]
    assert results["no_evidence_answer_without_citations"]["quality"]["label"] == "部分结论缺少来源"


def test_large_library_benchmark_reports_index_and_query_metrics(tmp_path):
    report = run_large_library_benchmark(documents=12, queries=3, tmp_root=tmp_path)

    assert report["schema"] == "docflow.large_library_benchmark.v1"
    assert report["passed"] is True
    assert "embedding" in report["scope"]
    assert report["scope"]["answer_generation"] == "deterministic_local_stub"
    assert report["thresholds"]["min_top_file_accuracy"] == 1.0
    assert report["threshold_failures"] == []
    assert report["indexing"]["documents"] == 12
    assert report["indexing"]["chunks"] >= 12
    assert report["lookup"]["cases"] == 3
    assert report["lookup"]["p95_ms"] >= report["lookup"]["p50_ms"]
    assert report["lookup"]["correct_top_file_count"] == 3
    assert report["retrieval"]["correct_top_file_count"] == 3
    assert report["answer_path"]["correct_top_file_count"] == 3
    assert report["query"] == report["lookup"]
    assert all(result["passed"] for result in report["lookup"]["results"])
    assert all(result["passed"] for result in report["retrieval"]["results"])
    assert all(result["passed"] for result in report["answer_path"]["results"])


def test_large_library_query_plan_spans_the_full_synthetic_library():
    plan = _query_plan(query_count=20, documents=10_000)

    targets = [
        int(item["expected_top_file"].removeprefix("knowledge-note-").removesuffix(".md"))
        for item in plan
    ]
    assert len(targets) == 20
    assert len(set(targets)) == 20
    assert min(targets) <= 250
    assert max(targets) >= 9_750


def test_large_library_benchmark_fails_when_smoke_threshold_is_breached(tmp_path):
    report = run_large_library_benchmark(
        documents=8,
        queries=2,
        tmp_root=tmp_path,
        max_lookup_p95_ms=-1.0,
    )

    assert report["passed"] is False
    assert {
        "metric": "lookup.p95_ms",
        "operator": "<=",
        "threshold": -1.0,
    }.items() <= report["threshold_failures"][0].items()


def test_external_catalog_tracks_archived_beir_results():
    report = summary(load_catalog())
    beir = next(item for item in report["benchmarks"] if item["id"] == "beir")

    assert beir["status"] == "archived"
    assert Path(beir["archived_result"]).exists()
    assert beir["claim_note"].startswith("Archived subset only")
    assert report["external_benchmark_scores"] >= 2
    archived_results = beir["archived_results"]
    archived_names = {item["name"] for item in archived_results}
    assert {"BEIR SciFact-lite", "BEIR NFCorpus-lite"} <= archived_names
    assert all(Path(item["result"]).exists() for item in archived_results)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

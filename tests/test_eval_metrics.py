from __future__ import annotations

from scripts.run_eval import (
    _source_filter_values,
    performance_summary,
    retrieval_metrics,
    write_results,
)


def test_retrieval_metrics_report_rank_quality():
    results = [
        {
            "passed": True,
            "must_find": True,
            "expected_files": ["README.md"],
            "top_sources": [
                {"file_name": "other.md", "file_path": "/tmp/other.md"},
                {"file_name": "README.md", "file_path": "/tmp/README.md"},
            ],
        },
        {
            "passed": False,
            "must_find": True,
            "expected_files": ["docs/privacy.md"],
            "top_sources": [{"file_name": "README.md", "file_path": "/tmp/README.md"}],
        },
    ]

    metrics = retrieval_metrics(results)

    assert metrics["eligible_cases"] == 2
    assert metrics["recall_at_5"] == 0.5
    assert metrics["mrr_at_5"] == 0.25
    assert metrics["ndcg_at_5"] == 0.3155
    assert metrics["pass_rate"] == 0.5


def test_write_results_creates_git_sha_and_latest_files(tmp_path):
    summary = {"schema": "docflow.retrieval_eval.v1", "git_sha": "abc123"}

    output_path = write_results(summary, tmp_path)

    assert output_path == tmp_path / "abc123.json"
    assert output_path.exists()
    assert (tmp_path / "latest.json").exists()


def test_performance_summary_reports_retrieval_latency_percentiles():
    results = [
        {"timings": {"total_ms": 10.0}},
        {"timings": {"total_ms": 20.0}},
        {"timings": {"total_ms": 30.0}},
        {"timings": {"total_ms": 40.0}},
    ]

    summary = performance_summary(results)

    assert summary["cases"] == 4
    assert summary["retrieval_total_ms_p50"] == 25.0
    assert summary["retrieval_total_ms_p95"] == 38.5
    assert summary["retrieval_total_ms_max"] == 40.0
    assert summary["cases_per_second"] == 40.0


def test_source_filter_values_include_project_relative_path():
    values = _source_filter_values(["docs/privacy.md"])

    assert len(values) == 1
    assert values[0].endswith("/docs/privacy.md")

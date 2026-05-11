from __future__ import annotations

from scripts.run_eval import retrieval_metrics, write_results


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

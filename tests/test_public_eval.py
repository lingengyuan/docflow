from __future__ import annotations

from pathlib import Path

from scripts.run_eval import load_cases
from scripts.run_public_eval import build_public_summary, public_benchmark_metadata


def test_public_retrieval_cases_reference_committed_public_corpus():
    cases_path = Path("eval/public_retrieval_v1.jsonl")
    cases = load_cases(cases_path)

    assert len(cases) >= 150
    assert {case.category for case in cases} == {"public_domain_smoke"}
    for case in cases:
        assert case.expected_files
        assert case.expected_terms
        for expected_file in case.expected_files:
            assert Path(expected_file).exists()


def test_public_benchmark_metadata_is_explicitly_not_internal_source_filtered():
    metadata = public_benchmark_metadata(Path("eval/public_retrieval_v1.jsonl"))

    assert metadata["kind"] == "public_reproducible_smoke"
    assert metadata["source_filter"] is False
    assert metadata["corpus_dir"] == "eval/public_corpus"
    assert "not a BEIR, MTEB, or C-MTEB score" in metadata["scope_note"]


def test_public_eval_summary_is_machine_readable():
    summary = build_public_summary(
        cases_path=Path("eval/public_retrieval_v1.jsonl"),
        results=[
            {
                "id": "case-1",
                "question": "Question?",
                "passed": True,
                "must_find": True,
                "expected_files": ["eval/public_corpus/example.txt"],
                "matched_files": ["eval/public_corpus/example.txt"],
                "top_files": ["example.txt"],
                "top_sources": [
                    {
                        "file_name": "example.txt",
                        "file_path": "/tmp/eval/public_corpus/example.txt",
                    }
                ],
                "top_qdrant_ids": ["chunk-1"],
                "timings": {"retrieval": 0.1},
            }
        ],
        include_rerank=True,
        source_refresh={"refreshed": [], "missing": []},
    )

    assert summary["schema"] == "docflow.public_retrieval_eval.v1"
    assert summary["benchmark"]["source_filter"] is False
    assert summary["cases"] == 1
    assert summary["passed"] == 1
    assert summary["metrics"]["recall_at_5"] == 1.0
    assert summary["source_refresh"] == {"refreshed": [], "missing": []}

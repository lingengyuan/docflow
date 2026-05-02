#!/usr/bin/env python3
"""Run DocFlow retrieval eval cases without calling the answer LLM."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.query.engine import QueryEngine


DEFAULT_EVAL_PATH = Path("eval/phase1_questions.jsonl")


@dataclass
class EvalCase:
    id: str
    question: str
    expected_files: list[str]
    expected_terms: list[str]
    must_find: bool


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(
                EvalCase(
                    id=data["id"],
                    question=data["question"],
                    expected_files=list(data.get("expected_files", [])),
                    expected_terms=list(data.get("expected_terms", [])),
                    must_find=bool(data.get("must_find", True)),
                )
            )
    return cases


def contains_term(haystack: str, term: str) -> bool:
    return term.lower() in haystack.lower()


def evaluate_case(engine: QueryEngine, case: EvalCase, include_rerank: bool) -> dict:
    debug = engine.retriever.debug_retrieve(
        case.question,
        prefer_tables=engine._is_table_query(case.question),
        include_rerank=include_rerank,
        max_text_chars=500,
    )
    final_stage = debug["stages"]["reranked"] if include_rerank else debug["stages"]["deduped"]
    combined = "\n".join(
        [
            item.get("file_name", "")
            + "\n"
            + item.get("file_path", "")
            + "\n"
            + item.get("section", "")
            + "\n"
            + item.get("text_preview", "")
            for item in final_stage
        ]
    )
    matched_files = [
        expected
        for expected in case.expected_files
        if any(expected == item.get("file_name") or item.get("file_path", "").endswith(expected) for item in final_stage)
    ]
    matched_terms = [term for term in case.expected_terms if contains_term(combined, term)]

    if case.must_find:
        passed = bool(final_stage) and len(matched_files) == len(case.expected_files) and len(matched_terms) == len(case.expected_terms)
    else:
        passed = not final_stage or not any(contains_term(combined, term) for term in case.expected_terms)

    return {
        "id": case.id,
        "question": case.question,
        "passed": passed,
        "must_find": case.must_find,
        "matched_files": matched_files,
        "missing_files": [f for f in case.expected_files if f not in matched_files],
        "matched_terms": matched_terms,
        "missing_terms": [t for t in case.expected_terms if t not in matched_terms],
        "top_files": [item.get("file_name", "") for item in final_stage[:5]],
        "top_qdrant_ids": [item.get("qdrant_id") for item in final_stage[:5]],
        "timings": debug["timings"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocFlow retrieval eval cases.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--cases", default=str(DEFAULT_EVAL_PATH), help="JSONL eval case file")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranker and evaluate deduped fused results")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    try:
        engine = QueryEngine.from_config(args.config)
        results = [evaluate_case(engine, case, include_rerank=not args.no_rerank) for case in cases]
    except Exception as exc:
        print(
            "Eval failed before completion. Check that Qdrant is running and the expected documents are ingested.",
            file=sys.stderr,
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    passed = sum(1 for result in results if result["passed"])
    summary = {
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "include_rerank": not args.no_rerank,
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"DocFlow retrieval eval: {passed}/{len(results)} passed")
        for result in results:
            mark = "PASS" if result["passed"] else "FAIL"
            print(f"[{mark}] {result['id']} :: {result['question']}")
            if not result["passed"]:
                print(f"  missing_files={result['missing_files']} missing_terms={result['missing_terms']}")
                print(f"  top_files={result['top_files']}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

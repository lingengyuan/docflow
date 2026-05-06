#!/usr/bin/env python3
"""Run DocFlow retrieval eval cases without calling the answer LLM."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

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
    category: str = "retrieval"


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
                    category=data.get("category", "retrieval"),
                )
            )
    return cases


def expected_source_names(cases: Iterable[EvalCase]) -> list[str]:
    names: set[str] = set()
    for case in cases:
        for expected in case.expected_files:
            if expected:
                names.add(expected)
    return sorted(names)


def refresh_eval_sources(cases: list[EvalCase], config_path: str | Path) -> dict:
    from src.api.app import _parse_watch_dirs
    from src.ingest.pipeline import IngestPipeline
    from src.ingest.watcher import _is_excluded
    import yaml

    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pipeline = IngestPipeline.from_config(config_path)

    refreshed: list[dict] = []
    missing: list[str] = []
    for expected in expected_source_names(cases):
        source_path = _resolve_eval_source(expected, cfg, _parse_watch_dirs, _is_excluded)
        if source_path is None:
            missing.append(expected)
            continue
        result = pipeline.ingest(source_path)
        refreshed.append(
            {
                "expected": expected,
                "path": str(source_path),
                "status": result.get("status", ""),
                "chunks": result.get("chunks", 0),
            }
        )
    return {"refreshed": refreshed, "missing": missing}


def _resolve_eval_source(expected: str, cfg: dict, parse_watch_dirs, is_excluded) -> Path | None:
    requested = Path(expected).expanduser()
    if requested.is_absolute() and requested.exists():
        return requested

    direct_candidates = [
        PROJECT_ROOT / expected,
        PROJECT_ROOT / "docs" / expected,
        PROJECT_ROOT / "plans" / expected,
        PROJECT_ROOT / "eval" / expected,
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    expected_name = Path(expected).name
    for wd in parse_watch_dirs(cfg):
        root = wd.path
        if not root.exists():
            continue
        pattern = f"**/{expected_name}" if wd.recursive else expected_name
        for candidate in root.glob(pattern):
            if candidate.is_file() and not is_excluded(candidate):
                return candidate
    return None


def contains_term(haystack: str, term: str) -> bool:
    return term.lower() in haystack.lower()


def evaluate_case(
    engine: QueryEngine,
    case: EvalCase,
    include_rerank: bool,
    source_filter: bool = False,
) -> dict:
    file_filter = case.expected_files if source_filter and case.must_find and case.expected_files else None
    debug = engine.retriever.debug_retrieve(
        case.question,
        file_filter=file_filter,
        prefer_tables=engine._is_table_query(case.question),
        include_rerank=include_rerank,
        max_text_chars=2000,
    )
    final_stage_name, final_stage = _evaluation_stage(debug, include_rerank)
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
    evidence_status = _evidence_status(case, passed, final_stage, matched_files, matched_terms)
    failure_reason = _failure_reason(case, final_stage, matched_files, matched_terms)

    return {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "passed": passed,
        "must_find": case.must_find,
        "file_filter": file_filter or [],
        "evidence_status": evidence_status,
        "failure_reason": failure_reason,
        "matched_files": matched_files,
        "missing_files": [f for f in case.expected_files if f not in matched_files],
        "matched_terms": matched_terms,
        "missing_terms": [t for t in case.expected_terms if t not in matched_terms],
        "hit_count": len(final_stage),
        "evaluation_stage": final_stage_name,
        "top_sources": [
            {
                "file_name": item.get("file_name", ""),
                "section": item.get("section", ""),
                "page_num": item.get("page_num"),
            }
            for item in final_stage[:5]
        ],
        "top_files": [item.get("file_name", "") for item in final_stage[:5]],
        "top_qdrant_ids": [item.get("qdrant_id") for item in final_stage[:5]],
        "timings": debug["timings"],
    }


def _evaluation_stage(debug: dict, include_rerank: bool) -> tuple[str, list[dict]]:
    stages = debug.get("stages", {})
    parent_expanded = stages.get("parent_expanded") or []
    if parent_expanded:
        return "parent_expanded", parent_expanded
    if include_rerank:
        return "reranked", stages.get("reranked") or []
    return "deduped", stages.get("deduped") or []


def _evidence_status(
    case: EvalCase,
    passed: bool,
    final_stage: list[dict],
    matched_files: list[str],
    matched_terms: list[str],
) -> str:
    if case.must_find and passed:
        return "grounded"
    if case.must_find and not final_stage:
        return "no_evidence"
    if case.must_find and (matched_files or matched_terms):
        return "partial_evidence"
    if case.must_find:
        return "missing_evidence"
    if passed:
        return "correctly_no_match"
    return "unsupported_match"


def _failure_reason(
    case: EvalCase,
    final_stage: list[dict],
    matched_files: list[str],
    matched_terms: list[str],
) -> str:
    if case.must_find:
        missing = []
        if not final_stage:
            missing.append("no_hits")
        missing_files = [f for f in case.expected_files if f not in matched_files]
        missing_terms = [t for t in case.expected_terms if t not in matched_terms]
        if missing_files:
            missing.append("missing_files=" + ",".join(missing_files))
        if missing_terms:
            missing.append("missing_terms=" + ",".join(missing_terms))
        return "; ".join(missing)
    if any(contains_term(_combined_stage_text(final_stage), term) for term in case.expected_terms):
        return "unexpected_evidence"
    return ""


def _combined_stage_text(final_stage: list[dict]) -> str:
    return "\n".join(
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocFlow retrieval eval cases.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--cases", default=str(DEFAULT_EVAL_PATH), help="JSONL eval case file")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranker and evaluate deduped fused results")
    parser.add_argument(
        "--refresh-sources",
        action="store_true",
        help="Ingest expected source files before running retrieval checks",
    )
    parser.add_argument(
        "--source-filter",
        action="store_true",
        help="For must-find cases, restrict retrieval to expected source files",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    source_refresh = None
    try:
        if args.refresh_sources:
            source_refresh = refresh_eval_sources(cases, args.config)
        engine = QueryEngine.from_config(args.config)
        results = [
            evaluate_case(
                engine,
                case,
                include_rerank=not args.no_rerank,
                source_filter=args.source_filter,
            )
            for case in cases
        ]
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
        "source_filter": args.source_filter,
        "results": results,
    }
    if source_refresh is not None:
        summary["source_refresh"] = source_refresh

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

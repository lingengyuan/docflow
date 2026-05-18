#!/usr/bin/env python3
"""Run a deterministic desktop large-library benchmark without private data."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain_types import FileStatus  # noqa: E402
from src.ingest.chunker import StructuredChunker  # noqa: E402
from src.ingest.parsers.markdown_parser import MarkdownParser  # noqa: E402
from src.ingest.store import DocStore  # noqa: E402
from src.query.citations import citation_from_chunk, validate_citations  # noqa: E402
from src.query.engine import QueryEngine  # noqa: E402
from src.query.generator import Answer  # noqa: E402
from src.query.keyword_search import tokenize  # noqa: E402
from src.query.retriever import HybridRetriever  # noqa: E402

DEFAULT_RESULTS_DIR = Path("eval/results/large-library")
DEFAULT_DOCUMENTS = 10_000
DEFAULT_QUERIES = 20
DEFAULT_MAX_INDEX_MS_PER_DOCUMENT = 100.0
DEFAULT_MAX_LOOKUP_P95_MS = 500.0
DEFAULT_MAX_RETRIEVAL_P95_MS = 1_500.0
DEFAULT_MAX_ANSWER_P95_MS = 2_000.0
DEFAULT_MIN_TOP_FILE_ACCURACY = 1.0


class BenchmarkQdrantPayloadClient:
    def retrieve(self, *args: Any, **kwargs: Any) -> list[Any]:
        del args, kwargs
        return []


class BenchmarkRetriever(HybridRetriever):
    """Use the normal retriever path while avoiding live model calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._qdrant = BenchmarkQdrantPayloadClient()

    def _rerank(
        self,
        query: str,
        candidates: list[dict],
        cancel_event=None,
        top_k: int | None = None,
    ) -> list[dict]:
        if self._is_cancelled(cancel_event):
            return []
        for index, item in enumerate(candidates):
            text = str(item.get("text") or item.get("raw_text") or "")
            item["rerank_score"] = 1.0 if query in text else max(0.1, 0.8 - (index * 0.01))
        candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
        return candidates[: (top_k or self.top_k_rerank)]


class BenchmarkAnswerGenerator:
    """Deterministic answer generator used only for performance-path timing."""

    def generate(
        self,
        query: str,
        chunks: list[dict],
        conversation_context: list[dict] | None = None,
    ) -> Answer:
        del conversation_context
        citations = validate_citations([citation_from_chunk(chunk) for chunk in chunks], chunks)
        top = chunks[0]
        citation = citations[0] if citations else None
        marker = f" [[cite:{citation.chunk_id}]]" if citation and citation.chunk_id else ""
        return Answer(
            text=(
                f"Benchmark answer for {query}: "
                f"{top.get('file_name', 'unknown source')} contains the requested marker.{marker}"
            ),
            citations=citations[:1],
            reproducible=True,
        )


def current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_large_library_benchmark(
    *,
    documents: int = DEFAULT_DOCUMENTS,
    queries: int = DEFAULT_QUERIES,
    tmp_root: Path | None = None,
    keep_temp: bool = False,
    max_index_ms_per_document: float | None = DEFAULT_MAX_INDEX_MS_PER_DOCUMENT,
    max_lookup_p95_ms: float | None = DEFAULT_MAX_LOOKUP_P95_MS,
    max_retrieval_p95_ms: float | None = DEFAULT_MAX_RETRIEVAL_P95_MS,
    max_answer_p95_ms: float | None = DEFAULT_MAX_ANSWER_P95_MS,
    min_top_file_accuracy: float | None = DEFAULT_MIN_TOP_FILE_ACCURACY,
) -> dict[str, Any]:
    temp_context = None
    if tmp_root is None:
        temp_context = tempfile.TemporaryDirectory(prefix="docflow-large-library-")
        root = Path(temp_context.name)
    else:
        root = tmp_root
        root.mkdir(parents=True, exist_ok=True)
    try:
        library_dir = root / "library"
        db_path = root / "docflow.db"
        _write_library(library_dir, documents)

        store = DocStore(db_path)
        parser = MarkdownParser()
        chunker = StructuredChunker(chunk_size=512, chunk_overlap=51)
        index_started = perf_counter()
        indexed_chunks = _index_library(store, parser, chunker, library_dir)
        index_ms = round((perf_counter() - index_started) * 1000, 2)

        query_plan = _query_plan(queries, documents)
        lookup_results = _run_lookup_queries(store, query_plan)
        retriever = BenchmarkRetriever(store=store, allow_model_download=False)
        retrieval_results = _run_retrieval_queries(retriever, query_plan)
        engine = QueryEngine(retriever, BenchmarkAnswerGenerator())
        answer_results = _run_answer_queries(engine, query_plan)
        retriever.close()
        store.close()
        lookup_summary = _stage_summary(lookup_results)
        retrieval_summary = _stage_summary(retrieval_results)
        answer_summary = _stage_summary(answer_results)
        thresholds = _thresholds(
            max_index_ms_per_document=max_index_ms_per_document,
            max_lookup_p95_ms=max_lookup_p95_ms,
            max_retrieval_p95_ms=max_retrieval_p95_ms,
            max_answer_p95_ms=max_answer_p95_ms,
            min_top_file_accuracy=min_top_file_accuracy,
        )
        report = {
            "schema": "docflow.large_library_benchmark.v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "git_sha": current_git_sha(),
            "source_tree": _source_tree_state(),
            "scope": {
                "documents": documents,
                "queries": queries,
                "mode": (
                    "Synthetic Markdown indexing plus direct lookup, full-text retrieval "
                    "orchestration, and deterministic answer assembly."
                ),
                "embedding": "not_measured",
                "vector_store": "not_measured",
                "mlx_reranker": "not_measured",
                "llm_first_token_latency": "not_measured",
                "answer_generation": "deterministic_local_stub",
                "note": (
                    "This benchmark measures repeatable local desktop performance on "
                    "synthetic Markdown. It exercises the parser/chunker, SQLite store, "
                    "FTS lookup, retrieval orchestration, citation construction, and "
                    "answer assembly path. It does not measure embedding, Qdrant vector "
                    "search, MLX reranking, or live LLM latency."
                ),
            },
            "thresholds": thresholds,
            "indexing": {
                "documents": documents,
                "chunks": indexed_chunks,
                "total_ms": index_ms,
                "ms_per_document": round(index_ms / documents, 4) if documents else 0.0,
                "documents_per_second": round(documents / (index_ms / 1000), 4)
                if index_ms > 0
                else 0.0,
                "db_bytes": db_path.stat().st_size,
                "library_bytes": _dir_size(library_dir),
                "workspace_bytes": _dir_size(root),
            },
            "lookup": lookup_summary,
            "retrieval": retrieval_summary,
            "answer_path": answer_summary,
        }
        report["query"] = lookup_summary
        threshold_failures = _threshold_failures(report, thresholds)
        report["threshold_failures"] = threshold_failures
        report["passed"] = (
            documents > 0
            and indexed_chunks > 0
            and bool(lookup_results)
            and lookup_summary["top_file_accuracy"] >= 1.0
            and retrieval_summary["top_file_accuracy"] >= 1.0
            and answer_summary["top_file_accuracy"] >= 1.0
            and not threshold_failures
        )
        if keep_temp:
            report["temp_dir"] = str(root)
        return report
    finally:
        if temp_context is not None and not keep_temp:
            temp_context.cleanup()


def _write_library(library_dir: Path, documents: int) -> None:
    if library_dir.exists():
        shutil.rmtree(library_dir)
    library_dir.mkdir(parents=True)
    for index in range(1, documents + 1):
        topic = index % 97
        review = index % 31
        source = index % 17
        path = library_dir / f"knowledge-note-{index:05d}.md"
        path.write_text(
            "\n".join(
                [
                    "---",
                    f"title: Synthetic Knowledge Note {index}",
                    f"tags: [topic-{topic}, review-{review}, source-{source}]",
                    "---",
                    f"# Synthetic Knowledge Note {index}",
                    (
                        f"This desktop benchmark document covers topic-{topic}, "
                        f"review-{review}, and source-{source}."
                    ),
                    (
                        "DocFlow should keep many local notes searchable, source-addressable, "
                        "and safe to inspect without network access."
                    ),
                    f"Unique evidence marker: benchmark-evidence-{index:05d}.",
                ]
            ),
            encoding="utf-8",
        )


def _index_library(
    store: DocStore,
    parser: MarkdownParser,
    chunker: StructuredChunker,
    library_dir: Path,
) -> int:
    qdrant_id = 1
    chunk_total = 0
    for path in sorted(library_dir.glob("*.md")):
        file_hash = DocStore.compute_hash(path)
        file_id = store.upsert_file(
            file_path=path,
            file_name=path.name,
            file_hash=file_hash,
            status=FileStatus.DONE,
            mtime_ns=path.stat().st_mtime_ns,
        )
        doc = parser.parse(path)
        records = []
        for page in doc.pages:
            chunks = chunker.chunk_page(
                page.text,
                file_name=doc.file_name,
                file_path=str(doc.file_path),
                page_num=page.page_num,
                is_ocr=page.is_ocr,
            )
            for chunk in chunks:
                raw_text = chunk.raw_text or chunk.text
                records.append(
                    {
                        "qdrant_id": qdrant_id,
                        "chunk_type": chunk.chunk_type,
                        "page_num": chunk.page_num,
                        "section": chunk.section,
                        "char_count": chunk.char_count,
                        "parent_id": chunk.parent_id,
                        "raw_text": raw_text,
                        "embedding_text": chunk.embedding_text or raw_text,
                        "parent_text": chunk.parent_text,
                        "contextual_prefix": chunk.contextual_prefix,
                        "tokenized_text": " ".join(tokenize(raw_text)),
                    }
                )
                qdrant_id += 1
        store.add_chunks(file_id, records)
        store.set_chunk_count(path, len(records))
        chunk_total += len(records)
    return chunk_total


def _query_plan(query_count: int, documents: int) -> list[dict[str, Any]]:
    if query_count <= 0 or documents <= 0:
        return []

    plan = []
    used_targets: set[int] = set()
    step = documents / query_count if query_count <= documents else 1
    for index in range(1, query_count + 1):
        if query_count <= documents:
            target = min(documents, max(1, round((index - 0.5) * step)))
            while target in used_targets and target < documents:
                target += 1
            while target in used_targets and target > 1:
                target -= 1
        else:
            target = ((index - 1) % documents) + 1
        used_targets.add(target)
        plan.append(
            {
                "id": f"large_library_{index:02d}",
                "query": f"benchmark-evidence-{target:05d}",
                "expected_top_file": f"knowledge-note-{target:05d}.md",
            }
        )
    return plan


def _run_lookup_queries(store: DocStore, query_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for item in query_plan:
        query = item["query"]
        fts_query = " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokenize(query))
        started = perf_counter()
        rows = store.search_fts(fts_query, file_filter=None, limit=5) if fts_query else []
        total_ms = round((perf_counter() - started) * 1000, 2)
        top_file = rows[0]["file_name"] if rows else ""
        results.append(
            {
                "id": item["id"],
                "query": query,
                "total_ms": total_ms,
                "hit_count": len(rows),
                "expected_top_file": item["expected_top_file"],
                "top_file": top_file,
                "passed": top_file == item["expected_top_file"],
            }
        )
    return results


def _run_retrieval_queries(
    retriever: BenchmarkRetriever,
    query_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for item in query_plan:
        started = perf_counter()
        debug = retriever.debug_retrieve(
            item["query"],
            retrieval_mode="full_text",
            include_rerank=False,
            max_text_chars=80,
        )
        total_ms = round((perf_counter() - started) * 1000, 2)
        stage = debug["stages"]["parent_expanded"]
        top_file = stage[0]["file_name"] if stage else ""
        timings = debug.get("timings", {})
        results.append(
            {
                "id": item["id"],
                "query": item["query"],
                "total_ms": total_ms,
                "fts_ms": timings.get("fts_ms", 0.0),
                "fusion_ms": round(
                    total_ms
                    - float(timings.get("embed_ms", 0.0))
                    - float(timings.get("vector_ms", 0.0))
                    - float(timings.get("fts_ms", 0.0))
                    - float(timings.get("rerank_ms", 0.0)),
                    2,
                ),
                "hit_count": len(stage),
                "expected_top_file": item["expected_top_file"],
                "top_file": top_file,
                "passed": top_file == item["expected_top_file"],
            }
        )
    return results


def _run_answer_queries(
    engine: QueryEngine,
    query_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for item in query_plan:
        started = perf_counter()
        answer = engine.query(item["query"], retrieval_mode="full_text")
        total_ms = round((perf_counter() - started) * 1000, 2)
        top_file = answer.citations[0].file_name if answer.citations else ""
        results.append(
            {
                "id": item["id"],
                "query": item["query"],
                "total_ms": total_ms,
                "citation_count": len(answer.citations),
                "answer_chars": len(answer.text),
                "quality_status": answer.quality.get("status", ""),
                "expected_top_file": item["expected_top_file"],
                "top_file": top_file,
                "passed": top_file == item["expected_top_file"],
            }
        )
    return results


def _stage_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    timings = sorted(float(item["total_ms"]) for item in results)
    correct_top_file_count = sum(1 for item in results if item["passed"])
    return {
        "cases": len(results),
        "p50_ms": _percentile(timings, 0.50),
        "p95_ms": _percentile(timings, 0.95),
        "max_ms": round(timings[-1], 2) if timings else 0.0,
        "correct_top_file_count": correct_top_file_count,
        "top_file_accuracy": round(correct_top_file_count / len(results), 4) if results else 0.0,
        "results": results,
    }


def _thresholds(
    *,
    max_index_ms_per_document: float | None,
    max_lookup_p95_ms: float | None,
    max_retrieval_p95_ms: float | None,
    max_answer_p95_ms: float | None,
    min_top_file_accuracy: float | None,
) -> dict[str, float]:
    return {
        key: value
        for key, value in {
            "max_index_ms_per_document": max_index_ms_per_document,
            "max_lookup_p95_ms": max_lookup_p95_ms,
            "max_retrieval_p95_ms": max_retrieval_p95_ms,
            "max_answer_p95_ms": max_answer_p95_ms,
            "min_top_file_accuracy": min_top_file_accuracy,
        }.items()
        if value is not None
    }


def _threshold_failures(
    report: dict[str, Any], thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    checks = [
        (
            "indexing.ms_per_document",
            report["indexing"]["ms_per_document"],
            "<=",
            "max_index_ms_per_document",
        ),
        ("lookup.p95_ms", report["lookup"]["p95_ms"], "<=", "max_lookup_p95_ms"),
        ("retrieval.p95_ms", report["retrieval"]["p95_ms"], "<=", "max_retrieval_p95_ms"),
        ("answer_path.p95_ms", report["answer_path"]["p95_ms"], "<=", "max_answer_p95_ms"),
        (
            "lookup.top_file_accuracy",
            report["lookup"]["top_file_accuracy"],
            ">=",
            "min_top_file_accuracy",
        ),
        (
            "retrieval.top_file_accuracy",
            report["retrieval"]["top_file_accuracy"],
            ">=",
            "min_top_file_accuracy",
        ),
        (
            "answer_path.top_file_accuracy",
            report["answer_path"]["top_file_accuracy"],
            ">=",
            "min_top_file_accuracy",
        ),
    ]
    failures = []
    for metric, actual, operator, threshold_key in checks:
        if threshold_key not in thresholds:
            continue
        expected = thresholds[threshold_key]
        failed = actual > expected if operator == "<=" else actual < expected
        if failed:
            failures.append(
                {
                    "metric": metric,
                    "actual": actual,
                    "operator": operator,
                    "threshold": expected,
                }
            )
    return failures


def _dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 2)
    idx = (len(values) - 1) * percentile
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return round(values[int(idx)], 2)
    weight = idx - lower
    return round(values[lower] * (1 - weight) + values[upper] * weight, 2)


def _source_tree_state() -> dict[str, Any]:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"worktree_dirty": None, "note": "Git status unavailable."}
    return {
        "worktree_dirty": bool(status),
        "status_entries": len(status),
        "note": (
            "Result was generated from the current local source tree. "
            "If worktree_dirty is true, the artifact records a pre-commit working-tree run."
        ),
    }


def write_results(report: dict[str, Any], results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"large-library-{report['git_sha']}.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    output_path.write_text(payload, encoding="utf-8")
    (results_dir / "large-library-latest.json").write_text(payload, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocFlow desktop large-library benchmark.")
    parser.add_argument("--documents", type=int, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument(
        "--max-index-ms-per-document",
        type=float,
        default=DEFAULT_MAX_INDEX_MS_PER_DOCUMENT,
        help="Fail when indexing exceeds this many milliseconds per document.",
    )
    parser.add_argument(
        "--max-lookup-p95-ms",
        type=float,
        default=DEFAULT_MAX_LOOKUP_P95_MS,
        help="Fail when direct source lookup P95 exceeds this threshold.",
    )
    parser.add_argument(
        "--max-retrieval-p95-ms",
        type=float,
        default=DEFAULT_MAX_RETRIEVAL_P95_MS,
        help="Fail when full-text retrieval orchestration P95 exceeds this threshold.",
    )
    parser.add_argument(
        "--max-answer-p95-ms",
        type=float,
        default=DEFAULT_MAX_ANSWER_P95_MS,
        help="Fail when deterministic answer assembly P95 exceeds this threshold.",
    )
    parser.add_argument(
        "--min-top-file-accuracy",
        type=float,
        default=DEFAULT_MIN_TOP_FILE_ACCURACY,
        help="Fail when any measured stage falls below this top-file accuracy.",
    )
    parser.add_argument(
        "--no-thresholds",
        action="store_true",
        help="Report metrics without failing on performance threshold breaches.",
    )
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_large_library_benchmark(
        documents=max(1, args.documents),
        queries=max(1, args.queries),
        keep_temp=args.keep_temp,
        max_index_ms_per_document=None
        if args.no_thresholds
        else args.max_index_ms_per_document,
        max_lookup_p95_ms=None if args.no_thresholds else args.max_lookup_p95_ms,
        max_retrieval_p95_ms=None if args.no_thresholds else args.max_retrieval_p95_ms,
        max_answer_p95_ms=None if args.no_thresholds else args.max_answer_p95_ms,
        min_top_file_accuracy=None if args.no_thresholds else args.min_top_file_accuracy,
    )
    if args.write_results:
        report["results_path"] = str(write_results(report, Path(args.results_dir)))

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            "DocFlow large-library benchmark: "
            f"{report['indexing']['documents']} documents, "
            f"index={report['indexing']['total_ms']}ms, "
            f"lookup P95={report['lookup']['p95_ms']}ms, "
            f"retrieval P95={report['retrieval']['p95_ms']}ms, "
            f"answer P95={report['answer_path']['p95_ms']}ms"
        )
        if report["threshold_failures"]:
            print("Threshold failures:")
            for failure in report["threshold_failures"]:
                print(
                    f"- {failure['metric']}: {failure['actual']} "
                    f"{failure['operator']} {failure['threshold']}"
                )
        if report.get("results_path"):
            print(f"Results written: {report['results_path']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

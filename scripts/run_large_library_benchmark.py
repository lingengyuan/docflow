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
from src.query.keyword_search import tokenize  # noqa: E402

DEFAULT_RESULTS_DIR = Path("eval/results/large-library")
DEFAULT_DOCUMENTS = 10_000
DEFAULT_QUERIES = 20


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

        query_results = _run_queries(store, queries, documents)
        store.close()
        query_ms = sorted(float(item["total_ms"]) for item in query_results)
        correct_top_file_count = sum(1 for item in query_results if item["passed"])
        report = {
            "schema": "docflow.large_library_benchmark.v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "git_sha": current_git_sha(),
            "source_tree": _source_tree_state(),
            "scope": {
                "documents": documents,
                "queries": queries,
                "mode": "SQLite FTS parser/chunker benchmark",
                "answer_generation": "not_measured",
                "note": (
                    "This benchmark measures desktop-scale local indexing and source "
                    "lookup on synthetic Markdown. It does not measure embedding, "
                    "reranking, or LLM first-token latency."
                ),
            },
            "indexing": {
                "documents": documents,
                "chunks": indexed_chunks,
                "total_ms": index_ms,
                "documents_per_second": round(documents / (index_ms / 1000), 4)
                if index_ms > 0
                else 0.0,
                "db_bytes": db_path.stat().st_size,
                "library_bytes": _dir_size(library_dir),
                "workspace_bytes": _dir_size(root),
            },
            "query": {
                "cases": len(query_results),
                "p50_ms": _percentile(query_ms, 0.50),
                "p95_ms": _percentile(query_ms, 0.95),
                "max_ms": round(query_ms[-1], 2) if query_ms else 0.0,
                "correct_top_file_count": correct_top_file_count,
                "top_file_accuracy": round(correct_top_file_count / len(query_results), 4)
                if query_results
                else 0.0,
                "results": query_results,
            },
            "passed": documents > 0
            and indexed_chunks > 0
            and bool(query_results)
            and correct_top_file_count == len(query_results),
        }
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


def _run_queries(store: DocStore, query_count: int, documents: int) -> list[dict[str, Any]]:
    results = []
    for index in range(1, query_count + 1):
        target = ((index * 37 - 1) % documents) + 1
        query = f"benchmark-evidence-{target:05d}"
        expected_file = f"knowledge-note-{target:05d}.md"
        fts_query = " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokenize(query))
        started = perf_counter()
        rows = store.search_fts(fts_query, file_filter=None, limit=5) if fts_query else []
        total_ms = round((perf_counter() - started) * 1000, 2)
        top_file = rows[0]["file_name"] if rows else ""
        results.append(
            {
                "id": f"large_library_{index:02d}",
                "query": query,
                "total_ms": total_ms,
                "hit_count": len(rows),
                "expected_top_file": expected_file,
                "top_file": top_file,
                "passed": top_file == expected_file,
            }
        )
    return results


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
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_large_library_benchmark(
        documents=max(1, args.documents),
        queries=max(1, args.queries),
        keep_temp=args.keep_temp,
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
            f"query P50={report['query']['p50_ms']}ms, "
            f"P95={report['query']['p95_ms']}ms"
        )
        if report.get("results_path"):
            print(f"Results written: {report['results_path']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

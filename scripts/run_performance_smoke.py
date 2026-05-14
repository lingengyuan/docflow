#!/usr/bin/env python3
"""Run deterministic parser/chunker performance smoke checks."""

from __future__ import annotations

import argparse
import json
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

from src.ingest.chunker import StructuredChunker  # noqa: E402
from src.ingest.parsers.markdown_parser import MarkdownParser  # noqa: E402

DEFAULT_RESULTS_DIR = Path("eval/results")
DEFAULT_LARGE_SECTIONS = 160
DEFAULT_LIBRARY_FILES = 80
LARGE_FILE_TOTAL_MS_LIMIT = 5000
LIBRARY_TOTAL_MS_LIMIT = 8000


def current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def run_performance_smoke(
    *,
    large_sections: int = DEFAULT_LARGE_SECTIONS,
    library_files: int = DEFAULT_LIBRARY_FILES,
    tmp_root: Path | None = None,
) -> dict[str, Any]:
    if tmp_root is not None:
        return _run_performance_smoke_at(
            tmp_root,
            large_sections=large_sections,
            library_files=library_files,
        )
    with tempfile.TemporaryDirectory(prefix="docflow-performance-smoke-") as temp_dir:
        return _run_performance_smoke_at(
            Path(temp_dir),
            large_sections=large_sections,
            library_files=library_files,
        )


def _run_performance_smoke_at(
    root: Path,
    *,
    large_sections: int,
    library_files: int,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    large_file = root / "large-knowledge-note.md"
    library_dir = root / "library"
    library_dir.mkdir(exist_ok=True)

    large_file.write_text(_large_markdown(large_sections), encoding="utf-8")
    library_paths = []
    for index in range(1, library_files + 1):
        path = library_dir / f"library-note-{index:03d}.md"
        path.write_text(_small_markdown(index), encoding="utf-8")
        library_paths.append(path)

    parser = MarkdownParser()
    chunker = StructuredChunker(chunk_size=512, chunk_overlap=51)
    large = _measure_files([large_file], parser, chunker)
    library = _measure_files(library_paths, parser, chunker)
    thresholds = {
        "large_file_total_ms_max": LARGE_FILE_TOTAL_MS_LIMIT,
        "library_total_ms_max": LIBRARY_TOTAL_MS_LIMIT,
    }
    large_passed = large["total_ms"] <= LARGE_FILE_TOTAL_MS_LIMIT and large["chunks"] > 0
    library_passed = library["total_ms"] <= LIBRARY_TOTAL_MS_LIMIT and library["chunks"] > 0
    return {
        "schema": "docflow.performance_smoke.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": current_git_sha(),
        "thresholds": thresholds,
        "large_file": {**large, "passed": large_passed},
        "library": {**library, "passed": library_passed},
        "passed": large_passed and library_passed,
    }


def _measure_files(
    paths: list[Path],
    parser: MarkdownParser,
    chunker: StructuredChunker,
) -> dict[str, Any]:
    parse_ms = 0.0
    chunk_ms = 0.0
    chunks = 0
    bytes_total = 0
    started = perf_counter()
    for path in paths:
        bytes_total += path.stat().st_size
        parse_started = perf_counter()
        doc = parser.parse(path)
        parse_ms += (perf_counter() - parse_started) * 1000
        for page in doc.pages:
            chunk_started = perf_counter()
            page_chunks = chunker.chunk_page(
                page.text,
                file_name=doc.file_name,
                file_path=str(doc.file_path),
                page_num=page.page_num,
                is_ocr=page.is_ocr,
            )
            chunk_ms += (perf_counter() - chunk_started) * 1000
            chunks += len(page_chunks)
    total_ms = (perf_counter() - started) * 1000
    return {
        "files": len(paths),
        "bytes": bytes_total,
        "chunks": chunks,
        "parse_ms": round(parse_ms, 2),
        "chunk_ms": round(chunk_ms, 2),
        "total_ms": round(total_ms, 2),
    }


def _large_markdown(sections: int) -> str:
    parts = [
        "---",
        'title: "Large Knowledge Note"',
        "tags: [performance, smoke]",
        "---",
        "# Large Knowledge Note",
    ]
    paragraph = (
        "DocFlow should keep long local notes readable, chunkable, and source-addressable. "
        "This synthetic section repeats realistic planning, citation, and review language "
        "without requiring private user data or network access."
    )
    table = "| Area | Signal | Action |\n|---|---|---|\n| Retrieval | cited | review |\n"
    for index in range(1, sections + 1):
        parts.append(f"## Section {index}\n\n{paragraph}\n\n{paragraph}\n\n")
        if index % 10 == 0:
            parts.append(table)
    return "\n".join(parts)


def _small_markdown(index: int) -> str:
    return "\n".join(
        [
            "---",
            f'title: "Library Note {index}"',
            "tags: [library, smoke]",
            "---",
            f"# Library Note {index}",
            "This note checks many-file parsing, chunking, and metadata extraction.",
            f"Topic marker: review-{index % 7}, source-{index % 5}, workflow-{index % 3}.",
        ]
    )


def write_results(report: dict[str, Any], results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    git_sha = report.get("git_sha") or current_git_sha()
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"performance-{git_sha}.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    output_path.write_text(payload, encoding="utf-8")
    (results_dir / "performance-latest.json").write_text(payload, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocFlow performance smoke checks.")
    parser.add_argument("--large-sections", type=int, default=DEFAULT_LARGE_SECTIONS)
    parser.add_argument("--library-files", type=int, default=DEFAULT_LIBRARY_FILES)
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_performance_smoke(
        large_sections=max(1, args.large_sections),
        library_files=max(1, args.library_files),
    )
    if args.write_results:
        report["results_path"] = str(write_results(report, Path(args.results_dir)))

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            "DocFlow performance smoke: "
            f"large={report['large_file']['total_ms']}ms, "
            f"library={report['library']['total_ms']}ms"
        )
        if report.get("results_path"):
            print(f"Results written: {report['results_path']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

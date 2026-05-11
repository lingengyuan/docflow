from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from src.ingest.chunker import Chunk, StructuredChunker
from src.ingest.parsers import ParserRegistry
from src.ingest.pdf_analyzer import ParsedDocument

DEFAULT_CORPUS_DIR = Path("eval/parsing_corpus")
DEFAULT_EXPECTED_PATH = Path("eval/parsing_expected/parsing_v1.json")


@dataclass(frozen=True)
class ParseExpectation:
    id: str
    path: str
    required_phrases: list[str]
    chunk_count_min: int
    chunk_count_max: int
    required_chunk_types: list[str]
    table_rows_min: int = 0


def load_expectations(path: str | Path = DEFAULT_EXPECTED_PATH) -> list[ParseExpectation]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        ParseExpectation(
            id=item["id"],
            path=item["path"],
            required_phrases=list(item.get("required_phrases", [])),
            chunk_count_min=int(item.get("chunk_count_min", 1)),
            chunk_count_max=int(item.get("chunk_count_max", 999)),
            required_chunk_types=list(item.get("required_chunk_types", [])),
            table_rows_min=int(item.get("table_rows_min", 0)),
        )
        for item in data.get("documents", [])
    ]


def run_parsing_eval(
    corpus_dir: str | Path = DEFAULT_CORPUS_DIR,
    expected_path: str | Path = DEFAULT_EXPECTED_PATH,
    config_path: str | Path = "config.example.yaml",
) -> dict[str, Any]:
    corpus = Path(corpus_dir)
    cfg = _load_config(config_path)
    registry = ParserRegistry.from_config(cfg)
    chunker = StructuredChunker(
        chunk_size=int(cfg.get("chunking", {}).get("chunk_size", 512)),
        chunk_overlap=int(cfg.get("chunking", {}).get("chunk_overlap", 51)),
    )
    started = perf_counter()
    results = [
        evaluate_document(corpus / expectation.path, expectation, registry, chunker)
        for expectation in load_expectations(expected_path)
    ]
    total_ms = round((perf_counter() - started) * 1000, 2)
    passed = sum(1 for result in results if result["passed"])
    chunk_count = sum(result["details"].get("chunk_count", 0) for result in results)
    text_chars = sum(result["details"].get("text_chars", 0) for result in results)
    return {
        "schema": "docflow.parsing_eval.v1",
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "performance": {
            "total_ms": total_ms,
            "documents_per_second": round(len(results) / (total_ms / 1000), 4)
            if total_ms > 0
            else 0.0,
            "total_chunks": chunk_count,
            "total_text_chars": text_chars,
        },
        "results": results,
    }


def evaluate_document(
    path: Path,
    expectation: ParseExpectation,
    registry: ParserRegistry,
    chunker: StructuredChunker,
) -> dict[str, Any]:
    failures: list[str] = []
    if not path.exists():
        return _result(expectation, path, False, ["missing_file"], {}, [])

    try:
        doc = registry.resolve(path).parse(path)
    except Exception as exc:
        return _result(
            expectation, path, False, [f"parse_error={type(exc).__name__}: {exc}"], {}, []
        )

    chunks = _chunk_document(doc, chunker)
    text = _document_text(doc)
    chunk_types = _chunk_type_counts(chunks)

    missing_phrases = [
        phrase for phrase in expectation.required_phrases if phrase.lower() not in text.lower()
    ]
    if missing_phrases:
        failures.append("missing_phrases=" + ",".join(missing_phrases))

    if len(chunks) < expectation.chunk_count_min:
        failures.append(f"too_few_chunks={len(chunks)}")
    if len(chunks) > expectation.chunk_count_max:
        failures.append(f"too_many_chunks={len(chunks)}")

    for chunk_type in expectation.required_chunk_types:
        if chunk_types.get(chunk_type, 0) == 0:
            failures.append(f"missing_chunk_type={chunk_type}")

    table_rows = _table_row_count(chunks)
    if table_rows < expectation.table_rows_min:
        failures.append(f"too_few_table_rows={table_rows}")

    details = {
        "total_pages": doc.total_pages,
        "chunk_count": len(chunks),
        "chunk_types": chunk_types,
        "table_rows": table_rows,
        "text_chars": len(text),
    }
    return _result(expectation, path, not failures, failures, details, chunks)


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("vlm", {})["enabled"] = False
    return cfg


def _chunk_document(doc: ParsedDocument, chunker: StructuredChunker) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in doc.pages:
        chunks.extend(
            chunker.chunk_page(
                text=page.text,
                file_name=doc.file_name,
                file_path=str(doc.file_path),
                page_num=page.page_num,
                is_ocr=page.is_ocr,
            )
        )
    return chunks


def _document_text(doc: ParsedDocument) -> str:
    return "\n".join(page.text for page in doc.pages)


def _chunk_type_counts(chunks: list[Chunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.chunk_type] = counts.get(chunk.chunk_type, 0) + 1
    return counts


def _table_row_count(chunks: list[Chunk]) -> int:
    rows = 0
    for chunk in chunks:
        if chunk.chunk_type != "table":
            continue
        rows += sum(1 for line in chunk.raw_text.splitlines() if line.strip().startswith("|"))
    return rows


def _result(
    expectation: ParseExpectation,
    path: Path,
    passed: bool,
    failures: list[str],
    details: dict,
    chunks: list[Chunk],
) -> dict[str, Any]:
    return {
        "id": expectation.id,
        "path": str(path),
        "passed": passed,
        "failures": failures,
        "details": details,
        "sample_chunks": [
            {
                "type": chunk.chunk_type,
                "section": chunk.section,
                "preview": chunk.raw_text[:160],
            }
            for chunk in chunks[:3]
        ],
    }

#!/usr/bin/env python3
"""Run small archived external retrieval benchmarks against DocFlow.

The default task is a deterministic BEIR SciFact test split subset. The script can also
run other configured BEIR-lite subsets. It downloads the public BEIR zip at runtime,
builds an isolated temporary DocFlow library, and writes only metrics and source
identifiers to the result artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_eval import (  # noqa: E402
    EvalCase,
    current_git_sha,
    evaluate_case,
    performance_summary,
    retrieval_metrics,
)
from src.ingest.pipeline import IngestPipeline  # noqa: E402
from src.query.engine import QueryEngine  # noqa: E402

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
NFCORPUS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
DEFAULT_RESULTS_DIR = Path("eval/results/external")
DEFAULT_QUERY_LIMIT = 20
DEFAULT_DISTRACTORS_PER_QUERY = 3
DEFAULT_MAX_RELEVANT_PER_QUERY = 0

BEIR_DATASETS: dict[str, dict[str, str]] = {
    "scifact": {
        "name": "SciFact",
        "url": SCIFACT_URL,
        "category": "external_beir_scifact",
        "artifact_slug": "beir-scifact-lite",
        "claim_scope": "Archived BEIR SciFact subset result; not a full BEIR leaderboard score.",
    },
    "nfcorpus": {
        "name": "NFCorpus",
        "url": NFCORPUS_URL,
        "category": "external_beir_nfcorpus",
        "artifact_slug": "beir-nfcorpus-lite",
        "claim_scope": "Archived BEIR NFCorpus subset result; not a full BEIR leaderboard score.",
    },
}


@dataclass(frozen=True)
class ExternalSubset:
    dataset_dir: Path
    dataset_slug: str
    dataset_name: str
    category: str
    queries: list[EvalCase]
    corpus_ids: list[str]
    corpus_by_id: dict[str, dict[str, Any]]
    source_zip_sha256: str


def download_dataset(url: str, cache_dir: Path, dataset_slug: str = "scifact") -> tuple[Path, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"{dataset_slug}.zip"
    if not zip_path.exists():
        try:
            urllib.request.urlretrieve(url, zip_path)
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Could not download BEIR {dataset_slug} dataset: {exc}") from exc
    return zip_path, _sha256(zip_path)


def extract_dataset(zip_path: Path, cache_dir: Path, dataset_slug: str = "scifact") -> Path:
    dataset_dir = cache_dir / dataset_slug
    required = [
        dataset_dir / "corpus.jsonl",
        dataset_dir / "queries.jsonl",
        dataset_dir / "qrels" / "test.tsv",
    ]
    if all(path.exists() for path in required):
        return dataset_dir
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(cache_dir)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Downloaded {dataset_slug} archive is missing: " + ", ".join(missing))
    return dataset_dir


def build_beir_subset(
    dataset_dir: Path,
    *,
    dataset_slug: str,
    dataset_name: str,
    category: str,
    query_limit: int,
    distractors_per_query: int,
    max_relevant_per_query: int = DEFAULT_MAX_RELEVANT_PER_QUERY,
    source_zip_sha256: str,
) -> ExternalSubset:
    corpus = _load_jsonl_by_id(dataset_dir / "corpus.jsonl")
    queries = _load_jsonl_by_id(dataset_dir / "queries.jsonl")
    qrels = _load_qrels(dataset_dir / "qrels" / "test.tsv")

    selected_queries: list[EvalCase] = []
    selected_corpus_ids: list[str] = []
    for query_id in sorted(qrels, key=_numeric_sort_key):
        if query_id not in queries:
            continue
        relevant = [doc_id for doc_id in qrels[query_id] if doc_id in corpus]
        if max_relevant_per_query > 0:
            relevant = relevant[:max_relevant_per_query]
        if not relevant:
            continue
        for doc_id in relevant:
            if doc_id not in selected_corpus_ids:
                selected_corpus_ids.append(doc_id)
        selected_queries.append(
            EvalCase(
                id=f"beir_{dataset_slug}_{query_id}",
                category=category,
                question=str(queries[query_id].get("text") or ""),
                expected_files=[_doc_file_name(dataset_slug, doc_id) for doc_id in relevant],
                expected_terms=[],
                must_find=True,
            )
        )
        if len(selected_queries) >= query_limit:
            break

    if not selected_queries:
        raise RuntimeError(f"No usable {dataset_name} qrels were found")

    target_doc_count = len(selected_corpus_ids) + max(0, distractors_per_query) * len(
        selected_queries
    )
    for doc_id in sorted(corpus, key=_numeric_sort_key):
        if len(selected_corpus_ids) >= target_doc_count:
            break
        if doc_id not in selected_corpus_ids:
            selected_corpus_ids.append(doc_id)

    return ExternalSubset(
        dataset_dir=dataset_dir,
        dataset_slug=dataset_slug,
        dataset_name=dataset_name,
        category=category,
        queries=selected_queries,
        corpus_ids=selected_corpus_ids,
        corpus_by_id=corpus,
        source_zip_sha256=source_zip_sha256,
    )


def build_scifact_subset(
    dataset_dir: Path,
    *,
    query_limit: int,
    distractors_per_query: int,
    max_relevant_per_query: int = DEFAULT_MAX_RELEVANT_PER_QUERY,
    source_zip_sha256: str,
) -> ExternalSubset:
    spec = BEIR_DATASETS["scifact"]
    return build_beir_subset(
        dataset_dir,
        dataset_slug="scifact",
        dataset_name=spec["name"],
        category=spec["category"],
        query_limit=query_limit,
        distractors_per_query=distractors_per_query,
        max_relevant_per_query=max_relevant_per_query,
        source_zip_sha256=source_zip_sha256,
    )


def write_beir_corpus(subset: ExternalSubset, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for doc_id in subset.corpus_ids:
        doc = subset.corpus_by_id[doc_id]
        title = str(doc.get("title") or "").strip()
        text = str(doc.get("text") or "").strip()
        body = "\n".join(
            [
                "---",
                f"benchmark: BEIR {subset.dataset_name}",
                f"doc_id: {json.dumps(doc_id)}",
                "---",
                f"# {title or doc_id}",
                "",
                text,
                "",
            ]
        )
        path = output_dir / _doc_file_name(subset.dataset_slug, doc_id)
        path.write_text(body, encoding="utf-8")
        paths.append(path)
    return paths


def write_scifact_corpus(subset: ExternalSubset, output_dir: Path) -> list[Path]:
    return write_beir_corpus(subset, output_dir)


def make_eval_config(
    base_config: Path,
    temp_root: Path,
    corpus_dir: Path,
    collection: str,
    dataset_slug: str = "scifact",
) -> Path:
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    cfg.setdefault("paths", {})
    cfg["paths"]["db_path"] = str(temp_root / "docflow.db")
    cfg["paths"]["id_counter"] = str(temp_root / "qdrant_id_counter.txt")
    cfg["paths"]["watch_dirs"] = [
        {
            "path": str(corpus_dir),
            "recursive": True,
            "extensions": [".md"],
        }
    ]
    cfg["paths"]["supported_extensions"] = [".md"]
    cfg.setdefault("qdrant", {})
    cfg["qdrant"]["collection"] = collection
    cfg.setdefault("privacy", {})
    cfg["privacy"]["allow_model_download"] = bool(cfg["privacy"].get("allow_model_download", False))
    cfg.setdefault("llm", {})
    cfg["llm"]["backend"] = "local"

    config_path = temp_root / f"config.external-{dataset_slug}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def run_external_eval(
    *,
    dataset_slug: str,
    config_path: Path,
    dataset_url: str | None,
    cache_dir: Path,
    query_limit: int,
    distractors_per_query: int,
    max_relevant_per_query: int,
    include_rerank: bool,
    collection: str,
    keep_temp: bool = False,
) -> dict[str, Any]:
    spec = BEIR_DATASETS.get(dataset_slug)
    if spec is None:
        raise RuntimeError(f"Unsupported external dataset: {dataset_slug}")
    resolved_url = dataset_url or spec["url"]
    zip_path, zip_sha = download_dataset(resolved_url, cache_dir, dataset_slug)
    dataset_dir = extract_dataset(zip_path, cache_dir, dataset_slug)
    subset = build_beir_subset(
        dataset_dir,
        dataset_slug=dataset_slug,
        dataset_name=spec["name"],
        category=spec["category"],
        query_limit=query_limit,
        distractors_per_query=distractors_per_query,
        max_relevant_per_query=max_relevant_per_query,
        source_zip_sha256=zip_sha,
    )

    temp_context: tempfile.TemporaryDirectory[str] | None = None
    if keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix=f"docflow-external-{dataset_slug}-"))
    else:
        temp_context = tempfile.TemporaryDirectory(prefix=f"docflow-external-{dataset_slug}-")
        temp_root = Path(temp_context.name)
    try:
        corpus_dir = temp_root / "corpus"
        corpus_paths = write_beir_corpus(subset, corpus_dir)
        eval_config = make_eval_config(
            config_path,
            temp_root,
            corpus_dir,
            collection,
            dataset_slug,
        )
        _reset_qdrant_collection(config_path, collection)

        ingest_started = perf_counter()
        pipeline = IngestPipeline.from_config(eval_config)
        ingest_results = [pipeline.ingest(path) for path in corpus_paths]
        pipeline.close()
        ingest_elapsed_ms = round((perf_counter() - ingest_started) * 1000, 2)

        engine = QueryEngine.from_config(eval_config)
        results = [
            _sanitize_result(
                evaluate_case(
                    engine,
                    case,
                    include_rerank=include_rerank,
                    source_filter=False,
                )
            )
            for case in subset.queries
        ]
        engine.close()

        passed = sum(1 for result in results if result["passed"])
        report = {
            "schema": "docflow.external_retrieval_eval.v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "git_sha": current_git_sha(),
            "source_tree": _source_tree_state(),
            "benchmark": {
                "id": f"beir_{dataset_slug}_lite",
                "suite": "BEIR",
                "dataset": spec["name"],
                "split": "test",
                "source_url": resolved_url,
                "source_zip_sha256": subset.source_zip_sha256,
                "query_limit": query_limit,
                "distractors_per_query": distractors_per_query,
                "max_relevant_per_query": max_relevant_per_query,
                "source_filter": False,
                "artifact_slug": spec["artifact_slug"],
                "claim_scope": spec["claim_scope"],
            },
            "cases": len(results),
            "corpus_documents": len(subset.corpus_ids),
            "passed": passed,
            "failed": len(results) - passed,
            "include_rerank": include_rerank,
            "metrics": retrieval_metrics(results),
            "performance": {
                **performance_summary(results),
                "ingest_total_ms": ingest_elapsed_ms,
                "indexed_files": len(corpus_paths),
                "indexed_chunks": sum(int(item.get("chunks") or 0) for item in ingest_results),
                "db_bytes": (temp_root / "docflow.db").stat().st_size,
            },
            "results": results,
        }
        if keep_temp:
            report["temp_dir"] = str(temp_root)
        return report
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def write_results(report: dict[str, Any], results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    artifact_slug = str(report.get("benchmark", {}).get("artifact_slug") or "external-lite")
    output_path = results_dir / f"{artifact_slug}-{report['git_sha']}.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    output_path.write_text(payload, encoding="utf-8")
    (results_dir / f"{artifact_slug}-latest.json").write_text(payload, encoding="utf-8")
    return output_path


def _reset_qdrant_collection(config_path: Path, collection: str) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    qdrant = cfg.get("qdrant", {})
    client = QdrantClient(
        host=str(qdrant.get("host", "localhost")),
        port=int(qdrant.get("port", 6333)),
    )
    try:
        if client.collection_exists(collection):
            client.delete_collection(collection)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _sanitize_result(result: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(result)
    cleaned["top_sources"] = [
        {
            "file_name": source.get("file_name", ""),
            "section": source.get("section", ""),
            "page_num": source.get("page_num"),
        }
        for source in result.get("top_sources", [])
    ]
    return cleaned


def _load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["_id"])] = row
    return rows


def _load_qrels(path: Path) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line_no == 0 and line.lower().startswith("query-id"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            query_id, corpus_id, score = parts[:3]
            if int(score) > 0:
                qrels[query_id].append(corpus_id)
    return dict(qrels)


def _doc_file_name(dataset_slug: str, doc_id: str) -> str:
    safe_doc_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(doc_id)).strip("._") or "doc"
    return f"{dataset_slug}-{safe_doc_id}.md"


def _numeric_sort_key(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (sys.maxsize, str(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BEIR-lite external retrieval eval.")
    parser.add_argument("--dataset", choices=sorted(BEIR_DATASETS), default="scifact")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset-url", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--query-limit", type=int, default=DEFAULT_QUERY_LIMIT)
    parser.add_argument(
        "--distractors-per-query",
        type=int,
        default=DEFAULT_DISTRACTORS_PER_QUERY,
    )
    parser.add_argument(
        "--max-relevant-per-query",
        type=int,
        default=DEFAULT_MAX_RELEVANT_PER_QUERY,
        help="Cap relevant corpus documents per query; 0 keeps all relevant documents.",
    )
    parser.add_argument("--include-rerank", action="store_true")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        dataset_slug = str(args.dataset)
        collection = str(
            args.collection or f"docflow_external_{dataset_slug}_{current_git_sha()}"
        )
        report = run_external_eval(
            dataset_slug=dataset_slug,
            config_path=Path(args.config),
            dataset_url=args.dataset_url,
            cache_dir=Path(args.cache_dir or f".cache/docflow/external/{dataset_slug}"),
            query_limit=max(1, args.query_limit),
            distractors_per_query=max(0, args.distractors_per_query),
            max_relevant_per_query=max(0, args.max_relevant_per_query),
            include_rerank=args.include_rerank,
            collection=collection,
            keep_temp=args.keep_temp,
        )
    except RuntimeError as exc:
        print(f"External retrieval eval failed: {exc}", file=sys.stderr)
        return 2

    if args.write_results:
        report["results_path"] = str(write_results(report, Path(args.results_dir)))

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        metrics = report["metrics"]
        print(
            "DocFlow external retrieval eval: "
            f"{report['benchmark']['dataset']} "
            f"{report['passed']}/{report['cases']} passed, "
            f"Recall@5={metrics['recall_at_5']} "
            f"MRR@5={metrics['mrr_at_5']} "
            f"nDCG@5={metrics['ndcg_at_5']}"
        )
        if report.get("results_path"):
            print(f"Results written: {report['results_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

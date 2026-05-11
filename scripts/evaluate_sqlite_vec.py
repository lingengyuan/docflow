#!/usr/bin/env python3
"""Evaluate sqlite-vec readiness without changing DocFlow storage."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path


def build_report(db_path: str | Path = "docflow.db") -> dict:
    sqlite_vec_available = importlib.util.find_spec("sqlite_vec") is not None
    db_exists = Path(db_path).expanduser().exists()
    return {
        "status": "ready_to_benchmark" if sqlite_vec_available and db_exists else "not_ready",
        "active_vector_store": "qdrant",
        "candidate_vector_store": "sqlite-vec",
        "sqlite_vec_available": sqlite_vec_available,
        "database_exists": db_exists,
        "sqlite_version": sqlite3.sqlite_version,
        "migration_decision": "do_not_migrate_without_benchmark",
        "required_before_migration": [
            "verified backup",
            "verified restore drill",
            "rebuild plan for all vectors",
            "retrieval quality comparison",
            "ingest speed comparison",
            "query latency comparison",
            "disk usage comparison",
        ],
        "benchmark_plan": [
            "export current chunk metadata and embeddings",
            "load a copy into sqlite-vec",
            "run the same benchmark question set against Qdrant and sqlite-vec",
            "compare top-k overlap, citation accuracy, p50/p95 latency, ingest time, and storage size",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate sqlite-vec readiness without migrating storage.")
    parser.add_argument("--db-path", default="docflow.db")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report(args.db_path)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Status: {report['status']}")
        print(f"Active vector store: {report['active_vector_store']}")
        print(f"Candidate: {report['candidate_vector_store']}")
        print(f"Migration decision: {report['migration_decision']}")
        print("Required before migration:")
        for item in report["required_before_migration"]:
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

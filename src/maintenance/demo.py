"""Demo library setup for first-run users."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.maintenance.startup import ensure_config_file

DEMO_FILES: dict[str, str] = {
    "docflow-overview.md": """# DocFlow Demo Overview

DocFlow is a local-first personal knowledge assistant. It indexes local files,
answers questions from those files, and shows source snippets so the user can
check where an answer came from.

The normal workflow is simple: add files, wait for indexing, ask a question,
then save useful answers as notes.
""",
    "local-privacy.md": """# Local Privacy Notes

DocFlow does not include telemetry, analytics, or automatic error reporting.
Documents, metadata, vectors, and generated notes stay on the user's machine
unless the user explicitly configures an external model or imports a webpage.

The offline doctor checks for unexpected outbound connections.
""",
    "knowledge-workflow.md": """# Knowledge Workflow

Useful DocFlow sessions usually follow four steps:

1. Collect local documents.
2. Ask a focused question.
3. Inspect the cited source snippets.
4. Save the final answer as a reusable note.
""",
    "sample-code.py": '''"""Small code sample for DocFlow demo search."""


def normalize_title(value: str) -> str:
    """Return a lowercase, dash-separated title."""
    return "-".join(value.strip().lower().split())
''',
}


def _load_config(config_path: str | Path) -> tuple[dict, Path]:
    path = ensure_config_file(config_path)
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}, path


def _resolve_watch_dir(cfg: dict, config_path: Path) -> Path:
    paths_cfg = cfg.get("paths", {})
    raw_watch_dirs = paths_cfg.get("watch_dirs") or paths_cfg.get("watch_dir") or "data/watch"
    if isinstance(raw_watch_dirs, list) and raw_watch_dirs:
        first = raw_watch_dirs[0]
        raw_path = first.get("path") if isinstance(first, dict) else first
    else:
        raw_path = raw_watch_dirs
    watch_dir = Path(str(raw_path)).expanduser()
    if not watch_dir.is_absolute():
        watch_dir = config_path.resolve().parent / watch_dir
    return watch_dir


def create_demo_files(config_path: str | Path = "config.yaml") -> dict:
    cfg, resolved_config = _load_config(config_path)
    demo_dir = _resolve_watch_dir(cfg, resolved_config) / "DocFlow Demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for name, content in DEMO_FILES.items():
        path = demo_dir / name
        status = "created"
        if path.exists() and path.read_text(encoding="utf-8") == content:
            status = "unchanged"
        else:
            path.write_text(content, encoding="utf-8")
        files.append(
            {"path": str(path), "name": name, "status": status, "bytes": path.stat().st_size}
        )

    return {
        "status": "created",
        "demo_dir": str(demo_dir),
        "file_count": len(files),
        "files": files,
    }


def ingest_demo_files(config_path: str | Path = "config.yaml") -> dict:
    from src.ingest.pipeline import IngestPipeline

    result = create_demo_files(config_path)
    pipeline = IngestPipeline.from_config(config_path)
    ingest_results = []
    for item in result["files"]:
        ingest_results.append(pipeline.ingest(item["path"]))
    return {**result, "status": "ingested", "ingest_results": ingest_results}


def demo_command(args: list[str]) -> int:
    config_path = "config.yaml"
    as_json = "--json" in args
    create_only = "--create-only" in args
    if "--config" in args:
        idx = args.index("--config")
        if idx + 1 < len(args):
            config_path = args[idx + 1]

    try:
        result = create_demo_files(config_path) if create_only else ingest_demo_files(config_path)
        exit_code = 0
    except Exception as exc:
        result = {
            "status": "error",
            "error": str(exc),
            "hint": "Start Qdrant and make sure local embedding dependencies are available.",
        }
        exit_code = 1

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["status"] == "error":
            print(f"Demo setup failed: {result['error']}")
            print(result["hint"])
        else:
            print(f"Demo library {result['status']}: {result['demo_dir']}")
            print(f"Files: {result['file_count']}")
    return exit_code

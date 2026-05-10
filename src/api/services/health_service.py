"""Health, settings, and storage helpers for API handlers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable


class HealthService:
    def build_health(
        self,
        cfg: dict,
        *,
        timed_check: Callable,
        check_sqlite: Callable,
        check_qdrant: Callable,
        check_ollama: Callable,
        check_models: Callable,
        health_capabilities: Callable,
        aggregate_health_status: Callable,
        health_groups: Callable,
        health_actions: Callable,
    ) -> dict:
        checks = {
            "api": {"status": "ok"},
            "sqlite": timed_check(lambda: check_sqlite(cfg)),
            "qdrant": timed_check(lambda: check_qdrant(cfg)),
            "ollama": timed_check(lambda: check_ollama(cfg)),
            "models": check_models(cfg),
        }
        capabilities = health_capabilities(cfg, checks)
        status = aggregate_health_status(checks)
        groups = health_groups(cfg, checks, capabilities)
        return {
            "status": status,
            "checks": checks,
            "capabilities": capabilities,
            "groups": groups,
            "actions": health_actions(checks, capabilities),
        }

    def collect_storage_usage(
        self,
        cfg: dict,
        doc_store,
        *,
        configured_model_cache_paths: Callable[[dict], list[Path]],
        app_data_paths: Callable[[dict], list[Path]],
        disk_usage: Callable = shutil.disk_usage,
    ) -> dict:
        disk = disk_usage(Path.home())
        files = doc_store.list_files()
        source_usage = self.source_file_usage(files)
        model_cache_bytes = sum(self.safe_path_size(path) for path in configured_model_cache_paths(cfg))
        app_data_bytes = sum(self.safe_path_size(path) for path in app_data_paths(cfg))
        known_bytes = source_usage["bytes"] + model_cache_bytes + app_data_bytes
        other_bytes = max(0, int(disk.used) - known_bytes)
        collections = sorted({str(item.get("collection") or "Inbox") for item in files})
        return {
            "disk": {
                "path": str(Path.home()),
                "total_bytes": int(disk.total),
                "used_bytes": int(disk.used),
                "free_bytes": int(disk.free),
                "used_percent": round((disk.used / disk.total) * 100, 1) if disk.total else 0,
            },
            "categories": [
                {
                    "id": "library",
                    "label": "资料库文件",
                    "bytes": source_usage["bytes"],
                    "detail": f"{source_usage['existing_files']} 个本地文件",
                },
                {
                    "id": "models",
                    "label": "模型缓存",
                    "bytes": model_cache_bytes,
                    "detail": "本地问答、检索和图片理解模型",
                },
                {
                    "id": "app_data",
                    "label": "应用数据",
                    "bytes": app_data_bytes,
                    "detail": "索引、数据库和本地记录",
                },
                {
                    "id": "other",
                    "label": "其他本地占用",
                    "bytes": other_bytes,
                    "detail": "系统和其他个人文件",
                },
            ],
            "library": {
                "file_count": len(files),
                "existing_file_count": source_usage["existing_files"],
                "missing_file_count": source_usage["missing_files"],
                "collection_count": len(collections),
            },
        }

    def safe_path_size(self, path: Path, *, max_entries: int = 100_000) -> int:
        path = path.expanduser()
        if not path.exists():
            return 0
        try:
            if path.is_file():
                return path.stat().st_size
        except OSError:
            return 0

        total = 0
        entries_seen = 0
        for root, dirs, files in os.walk(path, followlinks=False):
            dirs[:] = [name for name in dirs if name not in {".git", "__pycache__", ".venv"}]
            for file_name in files:
                entries_seen += 1
                if entries_seen > max_entries:
                    return total
                try:
                    file_path = Path(root) / file_name
                    total += file_path.stat().st_size
                except OSError:
                    continue
        return total

    def source_file_usage(self, files: list[dict]) -> dict:
        total = 0
        existing = 0
        missing = 0
        seen: set[str] = set()
        for item in files:
            file_path = item.get("file_path")
            if not file_path:
                continue
            path = Path(str(file_path)).expanduser()
            try:
                key = str(path.resolve())
            except OSError:
                key = str(path)
            if key in seen:
                continue
            seen.add(key)
            try:
                if path.is_file():
                    total += path.stat().st_size
                    existing += 1
                else:
                    missing += 1
            except OSError:
                missing += 1
        return {"bytes": total, "existing_files": existing, "missing_files": missing}

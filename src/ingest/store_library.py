"""Favorites, collections, tags, and row normalization for DocStore."""

# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import json
import sqlite3

from src.domain_types import FileRecord, FileStatus
from src.ingest.store_shared import DEFAULT_COLLECTION


class StoreLibraryMixin:
    def toggle_favorite(self, file_id: int) -> bool:
        """添加/取消收藏。返回 True=已收藏，False=已取消。"""
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM favorites WHERE file_id = ?", (file_id,)
            ).fetchone()
            if existing:
                conn.execute("DELETE FROM favorites WHERE file_id = ?", (file_id,))
                return False
            else:
                conn.execute("INSERT INTO favorites (file_id) VALUES (?)", (file_id,))
                return True
    def set_favorites(self, file_ids: list[int], favorited: bool) -> list[int]:
        ids = self._unique_ids(file_ids)
        if not ids:
            return []
        with self._conn() as conn:
            placeholders = ",".join("?" * len(ids))
            existing_rows = conn.execute(
                f"SELECT id FROM files WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            existing_ids = [row["id"] for row in existing_rows]
            if not existing_ids:
                return []
            if favorited:
                conn.executemany(
                    "INSERT OR IGNORE INTO favorites (file_id) VALUES (?)",
                    [(file_id,) for file_id in existing_ids],
                )
            else:
                placeholders = ",".join("?" * len(existing_ids))
                conn.execute(
                    f"DELETE FROM favorites WHERE file_id IN ({placeholders})", existing_ids
                )
        return existing_ids
    def is_favorite(self, file_id: int) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM favorites WHERE file_id = ?", (file_id,)).fetchone()
        return row is not None
    def list_favorites(self) -> list[FileRecord]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT f.*, 1 AS favorited FROM files f
                INNER JOIN favorites fav ON f.id = fav.file_id
                ORDER BY fav.created_at DESC
            """).fetchall()
        return [self._file_row_to_dict(r) for r in rows]
    @staticmethod
    def _unique_ids(file_ids: list[int]) -> list[int]:
        return [int(file_id) for file_id in dict.fromkeys(file_ids) if int(file_id) > 0]
    @staticmethod
    def _parse_json_list(value) -> list[str]:
        if isinstance(value, list):
            raw = value
        else:
            try:
                raw = json.loads(value or "[]")
            except (json.JSONDecodeError, TypeError):
                raw = []
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]
    @classmethod
    def _normalize_tags(cls, values: list[str]) -> list[str]:
        seen = set()
        tags: list[str] = []
        for value in values:
            tag = cls._normalize_tag(value)
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
        return tags
    @staticmethod
    def _normalize_tag(value: str | None) -> str:
        return str(value or "").strip().lstrip("#")[:40]
    @staticmethod
    def _normalize_collection(value: str | None) -> str:
        collection = str(value or "").strip()
        return collection[:80] if collection else DEFAULT_COLLECTION
    @staticmethod
    def _normalize_status(status: FileStatus | str) -> str:
        value = status.value if isinstance(status, FileStatus) else str(status)
        return FileStatus(value).value
    @classmethod
    def _file_row_to_dict(cls, row: sqlite3.Row) -> FileRecord:
        data = dict(row)
        data["tags"] = cls._parse_json_list(data.get("tags"))
        data["collection"] = cls._normalize_collection(data.get("collection"))
        data["user_tags"] = cls._parse_json_list(data.get("user_tags"))
        data["favorited"] = bool(data.get("favorited", False))
        return data  # type: ignore[return-value]
    @staticmethod
    def _facet_rows(counts: dict[str, int]) -> list[dict]:
        return [
            {"name": name, "count": counts[name]}
            for name in sorted(counts, key=lambda item: item.lower())
        ]

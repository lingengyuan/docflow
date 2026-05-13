"""Saved-note source link operations for DocStore."""

# mypy: disable-error-code="attr-defined"

from __future__ import annotations


class StoreLinksMixin:
    def replace_note_source_links(
        self,
        note_file_id: int,
        source_file_ids: list[int],
        relation: str = "answer_note",
    ) -> list[int]:
        note_file_id = int(note_file_id)
        relation = str(relation or "answer_note").strip()[:40] or "answer_note"
        source_ids = [
            file_id
            for file_id in self._unique_ids(source_file_ids)
            if int(file_id) != note_file_id
        ]
        with self._conn() as conn:
            note = conn.execute("SELECT id FROM files WHERE id = ?", (note_file_id,)).fetchone()
            if note is None:
                raise KeyError(f"Note file not found: {note_file_id}")
            if source_ids:
                placeholders = ",".join("?" * len(source_ids))
                rows = conn.execute(
                    f"SELECT id FROM files WHERE id IN ({placeholders})",
                    source_ids,
                ).fetchall()
                source_ids = [int(row["id"]) for row in rows]
            conn.execute(
                "DELETE FROM note_source_links WHERE note_file_id = ? AND relation = ?",
                (note_file_id, relation),
            )
            if source_ids:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO note_source_links (
                        note_file_id, source_file_id, relation
                    )
                    VALUES (?, ?, ?)
                    """,
                    [(note_file_id, source_id, relation) for source_id in source_ids],
                )
        return source_ids

    def list_backlinks(self, file_id: int) -> list[dict]:
        return self._list_note_links("source_file_id", int(file_id))

    def list_outbound_links(self, file_id: int) -> list[dict]:
        return self._list_note_links("note_file_id", int(file_id))

    def _list_note_links(self, column: str, file_id: int) -> list[dict]:
        if column == "source_file_id":
            joined_file_column = "note_file_id"
        elif column == "note_file_id":
            joined_file_column = "source_file_id"
        else:
            raise ValueError(f"Unsupported link column: {column}")
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT l.id AS link_id, l.relation, l.created_at,
                       f.*, fav.id IS NOT NULL AS favorited
                FROM note_source_links l
                JOIN files f ON f.id = l.{joined_file_column}
                LEFT JOIN favorites fav ON fav.file_id = f.id
                WHERE l.{column} = ?
                ORDER BY l.created_at DESC, l.id DESC
                """,
                (file_id,),
            ).fetchall()
        return [
            {
                "link_id": int(row["link_id"]),
                "relation": row["relation"],
                "created_at": row["created_at"],
                "file": self._file_row_to_dict(row),
            }
            for row in rows
        ]

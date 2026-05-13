"""History and conversation operations for DocStore."""

# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import sqlite3


class StoreHistoryMixin:
    def add_history(
        self,
        question: str,
        answer: str,
        citations_json: str,
        file_filter_json: str = "[]",
        conversation_id: int = 0,
    ) -> int:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO history (question, answer, citations, file_filter, conversation_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (question, answer, citations_json, file_filter_json, conversation_id),
            )
            row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO history_fts(rowid, question) VALUES (?, ?)",
                (row_id, question),
            )
        return row_id
    def search_history(self, query: str, limit: int = 20) -> list[dict]:
        """全文搜索历史问题，返回匹配的历史记录（含 answer、citations）。"""
        try:
            sql = """
                SELECT h.id, h.question, h.answer, h.citations, h.file_filter,
                       h.conversation_id, h.created_at
                FROM (
                    SELECT rowid, -rank AS score
                    FROM history_fts
                    WHERE history_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                ) fts
                JOIN history h ON h.id = fts.rowid
                ORDER BY fts.score DESC
            """
            with self._conn() as conn:
                rows = conn.execute(sql, [query, limit]).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.DatabaseError:
            return []
    def list_history(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM history ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    def clear_history(self):
        with self._conn() as conn:
            conn.execute("DELETE FROM history_fts")
            conn.execute("DELETE FROM history")
    def create_conversation(self, title: str = "") -> int:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations (title) VALUES (?)",
                (title.strip(),),
            )
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    def get_conversation(self, conversation_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None
    def list_conversations(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT c.*,
                       (
                           SELECT COUNT(*)
                           FROM messages m
                           WHERE m.conversation_id = c.id
                       ) AS message_count,
                       (
                           SELECT m.content
                           FROM messages m
                           WHERE m.conversation_id = c.id
                           ORDER BY m.id DESC
                           LIMIT 1
                       ) AS last_message
                FROM conversations c
                ORDER BY c.updated_at DESC, c.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
    def delete_conversation(self, conversation_id: int) -> bool:
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if existing is None:
                return False
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            return True
    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        citations_json: str = "[]",
        file_filter_json: str = "[]",
    ) -> int:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role}")
        with self._conn() as conn:
            conversation = conn.execute(
                "SELECT title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise ValueError(f"Conversation not found: {conversation_id}")
            conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, citations, file_filter)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, citations_json, file_filter_json),
            )
            message_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            title_update = ""
            if role == "user" and not conversation["title"].strip():
                title_update = ", title = ?"
                params = [self._conversation_title(content), conversation_id]
            else:
                params = [conversation_id]
            conn.execute(
                f"UPDATE conversations SET updated_at = datetime('now'){title_update} WHERE id = ?",
                params,
            )
            return message_id
    def list_messages(self, conversation_id: int, limit: int | None = None) -> list[dict]:
        if limit is None:
            query = "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id"
            params: tuple[int, ...] = (conversation_id,)
        else:
            query = """
                SELECT *
                FROM (
                    SELECT *
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id
            """
            params = (conversation_id, limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    @staticmethod
    def _conversation_title(text: str, max_chars: int = 40) -> str:
        title = " ".join(text.strip().split())
        if len(title) <= max_chars:
            return title
        return title[: max_chars - 1] + "..."

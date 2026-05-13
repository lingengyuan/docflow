"""SQLite connection and migration support for DocStore."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path


class StoreDatabaseMixin:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._local = threading.local()
        self._init_db()
        self._migrate()
    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS files (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path    TEXT    NOT NULL UNIQUE,
                    file_name    TEXT    NOT NULL,
                    file_hash    TEXT    NOT NULL,
                    status       TEXT    NOT NULL DEFAULT 'pending',
                    total_pages  INTEGER NOT NULL DEFAULT 0,
                    is_scanned   INTEGER NOT NULL DEFAULT 0,
                    chunk_count  INTEGER NOT NULL DEFAULT 0,
                    error_msg    TEXT    NOT NULL DEFAULT '',
                    tags         TEXT    NOT NULL DEFAULT '[]',
                    collection   TEXT    NOT NULL DEFAULT 'Inbox',
                    user_tags    TEXT    NOT NULL DEFAULT '[]',
                    mtime_ns     INTEGER NOT NULL DEFAULT 0,
                    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                    updated_at   TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id      INTEGER NOT NULL REFERENCES files(id),
                    qdrant_id    INTEGER NOT NULL,
                    chunk_type   TEXT    NOT NULL,
                    page_num     INTEGER NOT NULL,
                    section      TEXT    NOT NULL DEFAULT '',
                    char_count   INTEGER NOT NULL DEFAULT 0,
                    parent_id    INTEGER NOT NULL DEFAULT 0,
                    raw_text     TEXT    NOT NULL DEFAULT '',
                    embedding_text TEXT  NOT NULL DEFAULT '',
                    parent_text  TEXT    NOT NULL DEFAULT '',
                    contextual_prefix TEXT NOT NULL DEFAULT '',
                    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    question    TEXT    NOT NULL,
                    answer      TEXT    NOT NULL,
                    citations   TEXT    NOT NULL DEFAULT '[]',
                    file_filter TEXT    NOT NULL DEFAULT '[]',
                    conversation_id INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    title      TEXT    NOT NULL DEFAULT '',
                    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                    role            TEXT    NOT NULL,
                    content         TEXT    NOT NULL,
                    citations       TEXT    NOT NULL DEFAULT '[]',
                    file_filter     TEXT    NOT NULL DEFAULT '[]',
                    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS favorites (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id    INTEGER NOT NULL UNIQUE REFERENCES files(id),
                    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS answer_feedback (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    history_id INTEGER NOT NULL UNIQUE REFERENCES history(id),
                    rating     TEXT    NOT NULL,
                    note       TEXT    NOT NULL DEFAULT '',
                    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS note_source_links (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_file_id   INTEGER NOT NULL REFERENCES files(id),
                    source_file_id INTEGER NOT NULL REFERENCES files(id),
                    relation       TEXT    NOT NULL DEFAULT 'answer_note',
                    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(note_file_id, source_file_id, relation)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
                CREATE INDEX IF NOT EXISTS idx_files_hash     ON files(file_hash);
                CREATE INDEX IF NOT EXISTS idx_files_status   ON files(status);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                ON messages(conversation_id, id);
                CREATE INDEX IF NOT EXISTS idx_answer_feedback_history_id
                ON answer_feedback(history_id);
                CREATE INDEX IF NOT EXISTS idx_note_source_links_note_file_id
                ON note_source_links(note_file_id);
                CREATE INDEX IF NOT EXISTS idx_note_source_links_source_file_id
                ON note_source_links(source_file_id);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    tokenized_text,
                    tokenize='unicode61'
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_trigram USING fts5(
                    raw_text,
                    tokenize='trigram'
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5(
                    question,
                    tokenize='trigram'
                );

                CREATE TABLE IF NOT EXISTS embedding_cache (
                    model_name TEXT NOT NULL,
                    text_hash  TEXT NOT NULL,
                    vector     BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (model_name, text_hash)
                );
            """)
    def _migrate(self):
        """增量迁移：为已有 DB 添加新列（幂等）。"""
        migrations = [
            "ALTER TABLE files ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE files ADD COLUMN collection TEXT NOT NULL DEFAULT 'Inbox'",
            "ALTER TABLE files ADD COLUMN user_tags TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE files ADD COLUMN mtime_ns INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE chunks ADD COLUMN parent_id INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE chunks ADD COLUMN raw_text TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE chunks ADD COLUMN embedding_text TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE chunks ADD COLUMN parent_text TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE chunks ADD COLUMN contextual_prefix TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE history ADD COLUMN conversation_id INTEGER NOT NULL DEFAULT 0",
        ]
        with self._conn() as conn:
            for sql in migrations:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    pass  # 列已存在
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(file_id, parent_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_collection ON files(collection)")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS answer_feedback (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    history_id INTEGER NOT NULL UNIQUE REFERENCES history(id),
                    rating     TEXT    NOT NULL,
                    note       TEXT    NOT NULL DEFAULT '',
                    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS note_source_links (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_file_id   INTEGER NOT NULL REFERENCES files(id),
                    source_file_id INTEGER NOT NULL REFERENCES files(id),
                    relation       TEXT    NOT NULL DEFAULT 'answer_note',
                    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(note_file_id, source_file_id, relation)
                );

                CREATE INDEX IF NOT EXISTS idx_answer_feedback_history_id
                ON answer_feedback(history_id);
                CREATE INDEX IF NOT EXISTS idx_note_source_links_note_file_id
                ON note_source_links(note_file_id);
                CREATE INDEX IF NOT EXISTS idx_note_source_links_source_file_id
                ON note_source_links(source_file_id);
            """)
    @contextmanager
    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        try:
            yield conn
            conn.commit()
        except Exception:
            # Any error raised by the caller's DB work must roll back before re-raising.
            conn.rollback()
            raise
    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
    @staticmethod
    def compute_hash(file_path: str | Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    @staticmethod
    def compute_text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

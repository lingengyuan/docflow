"""
测试 DocStore SQLite 元数据存储。
"""

import tempfile
import sqlite3
from pathlib import Path

import fitz
import numpy as np
import pytest

from src.ingest.store import DocStore


def make_pdf(path: Path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Test content for hashing.", fontsize=12)
    doc.save(str(path))
    doc.close()


class TestDocStore:
    @pytest.fixture
    def db(self, tmp_path):
        return DocStore(tmp_path / "test.db")

    @pytest.fixture
    def pdf(self, tmp_path):
        p = tmp_path / "test.pdf"
        make_pdf(p)
        return p

    def test_needs_ingest_new_file(self, db, pdf):
        need, h = db.needs_ingest(pdf)
        assert need is True

    def test_needs_ingest_after_done(self, db, pdf):
        h = DocStore.compute_hash(pdf)
        db.upsert_file(pdf, pdf.name, h, status="done")
        need, _ = db.needs_ingest(pdf)
        assert need is False

    def test_needs_ingest_error_status(self, db, pdf):
        h = DocStore.compute_hash(pdf)
        db.upsert_file(pdf, pdf.name, h, status="error")
        need, _ = db.needs_ingest(pdf)
        assert need is True

    def test_needs_ingest_pending_status(self, db, pdf):
        h = DocStore.compute_hash(pdf)
        db.upsert_file(pdf, pdf.name, h, status="pending", mtime_ns=pdf.stat().st_mtime_ns)
        need, _ = db.needs_ingest(pdf)
        assert need is True

    def test_needs_ingest_changed_file(self, db, pdf):
        # Store with a fake hash
        db.upsert_file(pdf, pdf.name, "fakehash000", status="done")
        # Real hash differs → needs re-ingest, and returns the computed hash
        need, cached_hash = db.needs_ingest(pdf)
        assert need is True
        assert cached_hash is not None

    def test_upsert_idempotent(self, db, pdf):
        h = DocStore.compute_hash(pdf)
        id1 = db.upsert_file(pdf, pdf.name, h)
        id2 = db.upsert_file(pdf, pdf.name, h, status="done")
        assert id1 == id2

    def test_set_status(self, db, pdf):
        h = DocStore.compute_hash(pdf)
        db.upsert_file(pdf, pdf.name, h)
        db.set_status(pdf, "done")
        record = db.get_file_by_path(pdf)
        assert record["status"] == "done"

    def test_set_status_error_message(self, db, pdf):
        h = DocStore.compute_hash(pdf)
        db.upsert_file(pdf, pdf.name, h)
        db.set_status(pdf, "error", error_msg="something broke")
        record = db.get_file_by_path(pdf)
        assert record["error_msg"] == "something broke"

    def test_add_chunks_and_count(self, db, pdf):
        h = DocStore.compute_hash(pdf)
        file_id = db.upsert_file(pdf, pdf.name, h)
        records = [
            {"qdrant_id": i, "chunk_type": "text", "page_num": 1,
             "section": "", "char_count": 100}
            for i in range(5)
        ]
        db.add_chunks(file_id, records)
        db.set_chunk_count(pdf, 5)
        record = db.get_file_by_path(pdf)
        assert record["chunk_count"] == 5

    def test_list_file_chunks_returns_metadata_in_order(self, db, pdf):
        h = DocStore.compute_hash(pdf)
        file_id = db.upsert_file(pdf, pdf.name, h)
        records = [
            {"qdrant_id": 20, "chunk_type": "text", "page_num": 2, "section": "B", "char_count": 20},
            {"qdrant_id": 10, "chunk_type": "table", "page_num": 1, "section": "A", "char_count": 10},
        ]
        db.add_chunks(file_id, records)

        chunks = db.list_file_chunks(file_id)

        assert [c["qdrant_id"] for c in chunks] == [20, 10]
        assert chunks[0]["chunk_type"] == "text"
        assert chunks[1]["section"] == "A"

    def test_add_chunks_maps_fts_to_actual_chunk_ids_when_ids_have_gaps(self, db, pdf):
        h = DocStore.compute_hash(pdf)
        file_id = db.upsert_file(pdf, pdf.name, h)
        with db._conn() as conn:
            conn.executescript(
                """
                CREATE TRIGGER chunk_gap_after_insert
                AFTER INSERT ON chunks
                WHEN NEW.qdrant_id > 0
                BEGIN
                    INSERT INTO chunks (
                        file_id, qdrant_id, chunk_type, page_num, section, char_count
                    )
                    VALUES (-1, -1, 'gap', 0, '', 0);
                    DELETE FROM chunks WHERE id = last_insert_rowid();
                END;
                """
            )

        db.add_chunks(
            file_id,
            [
                {
                    "qdrant_id": 101,
                    "chunk_type": "text",
                    "page_num": 1,
                    "section": "",
                    "char_count": 24,
                    "raw_text": "alpha regression phrase",
                    "tokenized_text": "alpha regression phrase",
                },
                {
                    "qdrant_id": 102,
                    "chunk_type": "text",
                    "page_num": 1,
                    "section": "",
                    "char_count": 23,
                    "raw_text": "beta regression phrase",
                    "tokenized_text": "beta regression phrase",
                },
            ],
        )

        chunks = db.list_file_chunks(file_id)
        chunk_ids = [row["id"] for row in chunks]
        assert chunk_ids[1] > chunk_ids[0] + 1

        with db._conn() as conn:
            fts_ids = [
                row["rowid"]
                for row in conn.execute("SELECT rowid FROM chunks_fts ORDER BY rowid").fetchall()
            ]
            trigram_ids = [
                row["rowid"]
                for row in conn.execute("SELECT rowid FROM chunks_fts_trigram ORDER BY rowid").fetchall()
            ]

        assert fts_ids == chunk_ids
        assert trigram_ids == chunk_ids
        assert db.search_fts('"alpha"', None, 10)[0]["qdrant_id"] == 101
        assert db.search_fts_trigram("beta", None, 10)[0]["qdrant_id"] == 102
        assert db.search_fts_trigram("beta?", None, 10)[0]["qdrant_id"] == 102

    def test_max_qdrant_id_returns_highest_indexed_id(self, db, pdf):
        assert db.max_qdrant_id() == -1
        h = DocStore.compute_hash(pdf)
        file_id = db.upsert_file(pdf, pdf.name, h)
        db.add_chunks(
            file_id,
            [
                {"qdrant_id": 20, "chunk_type": "text", "page_num": 1, "section": "", "char_count": 10},
                {"qdrant_id": 10, "chunk_type": "text", "page_num": 1, "section": "", "char_count": 10},
            ],
        )

        assert db.max_qdrant_id() == 20

    def test_chunk_parent_context_fields_roundtrip(self, db, pdf):
        h = DocStore.compute_hash(pdf)
        file_id = db.upsert_file(pdf, pdf.name, h)
        db.add_chunks(
            file_id,
            [
                {
                    "qdrant_id": 99,
                    "chunk_type": "text",
                    "page_num": 1,
                    "section": "A",
                    "char_count": 10,
                    "parent_id": 7,
                    "raw_text": "child text",
                    "embedding_text": "prefix child text",
                    "parent_text": "parent context",
                    "contextual_prefix": "prefix",
                }
            ],
        )

        chunks = db.list_file_chunks(file_id)
        contexts = db.get_chunk_context_by_qdrant_ids([99])

        assert chunks[0]["parent_id"] == 7
        assert chunks[0]["raw_text"] == "child text"
        assert contexts[99]["parent_text"] == "parent context"
        assert contexts[99]["contextual_prefix"] == "prefix"

    def test_add_chunks_replaces_on_reingest(self, db, pdf):
        h = DocStore.compute_hash(pdf)
        file_id = db.upsert_file(pdf, pdf.name, h)
        records_v1 = [
            {"qdrant_id": i, "chunk_type": "text", "page_num": 1, "section": "", "char_count": 10}
            for i in range(3)
        ]
        db.add_chunks(file_id, records_v1)
        # Re-ingest with different chunks
        records_v2 = [
            {"qdrant_id": i + 100, "chunk_type": "text", "page_num": 1, "section": "", "char_count": 10}
            for i in range(2)
        ]
        db.add_chunks(file_id, records_v2)
        # Only v2 chunks should remain (checked via set_chunk_count)
        db.set_chunk_count(pdf, 2)
        record = db.get_file_by_path(pdf)
        assert record["chunk_count"] == 2

    def test_list_files(self, db, pdf, tmp_path):
        pdf2 = tmp_path / "test2.pdf"
        make_pdf(pdf2)
        db.upsert_file(pdf, pdf.name, "hash1", status="done")
        db.upsert_file(pdf2, pdf2.name, "hash2", status="pending")
        all_files = db.list_files()
        assert len(all_files) == 2
        done_files = db.list_files(status="done")
        assert len(done_files) == 1
        assert done_files[0]["file_name"] == pdf.name

    def test_file_metadata_filters_and_facets(self, db, pdf, tmp_path):
        pdf2 = tmp_path / "test2.pdf"
        make_pdf(pdf2)
        first_id = db.upsert_file(pdf, pdf.name, "hash1", status="done")
        second_id = db.upsert_file(pdf2, pdf2.name, "hash2", status="done")

        db.update_file_metadata(first_id, collection="Research", user_tags=["ai", "#paper", "ai"])
        db.update_file_metadata(second_id, collection="Inbox", user_tags=["todo"])
        db.set_favorites([first_id], favorited=True)

        assert db.get_file_by_id(first_id)["user_tags"] == ["ai", "paper"]
        assert [row["id"] for row in db.list_files(collection="Research")] == [first_id]
        assert [row["id"] for row in db.list_files(tag="paper")] == [first_id]
        assert [row["id"] for row in db.list_files(favorite=True)] == [first_id]
        assert {row["id"] for row in db.list_files(kind="pdf")} == {first_id, second_id}
        assert [row["id"] for row in db.list_files(kind="markdown")] == []
        assert {row["id"] for row in db.list_files(recent=True)} == {first_id, second_id}

        facets = db.list_library_facets()
        assert {"name": "Research", "count": 1} in facets["collections"]
        assert {"name": "paper", "count": 1} in facets["user_tags"]
        assert facets["favorites"] == 1
        assert facets["types"]["pdf"] == 2

    def test_recent_file_filter_keeps_tag_matches_before_limit(self, db, tmp_path):
        target = tmp_path / "target.md"
        target_id = db.upsert_file(target, target.name, "target-hash", status="done")
        db.update_file_metadata(target_id, user_tags=["keep"])
        for idx in range(35):
            path = tmp_path / f"decoy-{idx}.md"
            db.upsert_file(path, path.name, f"decoy-hash-{idx}", status="done")

        with db._conn() as conn:
            conn.execute(
                "UPDATE files SET updated_at = '2025-01-01 00:00:00' WHERE id = ?",
                (target_id,),
            )
            conn.execute(
                "UPDATE files SET updated_at = '2025-02-01 00:00:00' WHERE id != ?",
                (target_id,),
            )

        assert [row["id"] for row in db.list_files(tag="keep", recent=True)] == [target_id]

    def test_batch_favorites_ignore_missing_files(self, db, pdf):
        file_id = db.upsert_file(pdf, pdf.name, "hash1", status="done")

        changed = db.set_favorites([file_id, 999], favorited=True)

        assert changed == [file_id]
        assert [row["id"] for row in db.list_favorites()] == [file_id]

    def test_reingest_preserves_user_metadata(self, db, pdf):
        file_id = db.upsert_file(pdf, pdf.name, "hash1", status="done")
        db.update_file_metadata(file_id, collection="Research", user_tags=["paper"])

        same_id = db.upsert_file(pdf, pdf.name, "hash2", status="done")
        record = db.get_file_by_id(same_id)

        assert same_id == file_id
        assert record["collection"] == "Research"
        assert record["user_tags"] == ["paper"]

    def test_embedding_cache_roundtrip(self, db):
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        db.put_cached_embeddings("test-model", {"hash-a": vector})
        cached = db.get_cached_embeddings("test-model", ["hash-a", "hash-b"])
        assert set(cached) == {"hash-a"}
        np.testing.assert_allclose(cached["hash-a"], vector)

    def test_conversation_messages_persist_and_delete(self, tmp_path):
        db_path = tmp_path / "conversation.db"
        db = DocStore(db_path)
        conversation_id = db.create_conversation()

        db.add_message(conversation_id, "user", "请总结第一点")
        db.add_message(
            conversation_id,
            "assistant",
            "第一点是本地优先。",
            citations_json='[{"file_name":"README.md"}]',
        )

        reopened = DocStore(db_path)
        conversation = reopened.get_conversation(conversation_id)
        messages = reopened.list_messages(conversation_id)
        conversations = reopened.list_conversations()

        assert conversation["title"] == "请总结第一点"
        assert [message["role"] for message in messages] == ["user", "assistant"]
        assert conversations[0]["message_count"] == 2
        assert conversations[0]["last_message"] == "第一点是本地优先。"

        assert reopened.delete_conversation(conversation_id) is True
        assert reopened.get_conversation(conversation_id) is None
        assert reopened.list_messages(conversation_id) == []

    def test_migrates_existing_chunks_table_before_parent_index(self, tmp_path):
        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                qdrant_id INTEGER NOT NULL,
                chunk_type TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                section TEXT NOT NULL DEFAULT '',
                char_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        conn.close()

        store = DocStore(db_path)
        with store._conn() as migrated:
            columns = [row["name"] for row in migrated.execute("PRAGMA table_info(chunks)").fetchall()]

        assert "parent_id" in columns
        assert "parent_text" in columns

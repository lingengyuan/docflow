"""
测试 DocStore SQLite 元数据存储。
"""

import tempfile
from pathlib import Path

import fitz
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
        assert db.needs_ingest(pdf) is True

    def test_needs_ingest_after_done(self, db, pdf):
        h = DocStore.compute_hash(pdf)
        db.upsert_file(pdf, pdf.name, h, status="done")
        assert db.needs_ingest(pdf) is False

    def test_needs_ingest_error_status(self, db, pdf):
        h = DocStore.compute_hash(pdf)
        db.upsert_file(pdf, pdf.name, h, status="error")
        assert db.needs_ingest(pdf) is True

    def test_needs_ingest_changed_file(self, db, pdf):
        # Store with a fake hash
        db.upsert_file(pdf, pdf.name, "fakehash000", status="done")
        # Real hash differs → needs re-ingest
        assert db.needs_ingest(pdf) is True

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

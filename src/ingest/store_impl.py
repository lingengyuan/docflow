"""DocStore public implementation assembled from focused mixins."""

from __future__ import annotations

from src.ingest.store_db import StoreDatabaseMixin
from src.ingest.store_files import StoreFileMixin
from src.ingest.store_history import StoreHistoryMixin
from src.ingest.store_library import StoreLibraryMixin
from src.ingest.store_vectors import StoreVectorMixin


class DocStore(
    StoreDatabaseMixin,
    StoreFileMixin,
    StoreVectorMixin,
    StoreHistoryMixin,
    StoreLibraryMixin,
):
    """SQLite-backed metadata store for files, chunks, history, and library state."""

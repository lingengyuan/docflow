"""Shared helpers for DocStore mixins."""

from __future__ import annotations

import re

DEFAULT_COLLECTION = "Inbox"


def _fts5_phrase(query: str) -> str:
    without_quotes = str(query or "").replace('"', " ")
    cleaned = " ".join(re.sub(r"[^\w\s\u4e00-\u9fff-]", " ", without_quotes).split())
    return f'"{cleaned}"'

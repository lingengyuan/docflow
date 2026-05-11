"""Public storage facade.

`DocStore` remains importable from `src.ingest.store`; implementation details
live in `src.ingest.store_impl`.
"""

from __future__ import annotations

from src.ingest.store_impl import *  # noqa: F403

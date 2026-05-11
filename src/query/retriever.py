"""Public retrieval facade.

`HybridRetriever` and `QueryRouter` remain importable from `src.query.retriever`;
implementation details live in `src.query.retriever_impl`.
"""

from __future__ import annotations

from src.query.retriever_impl import *  # noqa: F403

"""Vector store adapters.

Phase40 introduces this boundary without changing the production backend.
Qdrant remains the active vector store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointIdsList,
    PointStruct,
    VectorParams,
)


@dataclass(frozen=True)
class VectorPoint:
    id: int
    vector: list[float]
    payload: dict


@dataclass(frozen=True)
class VectorSearchHit:
    id: int | str
    score: float
    payload: dict


class VectorStore(Protocol):
    def ensure_collection(self, collection_name: str, vector_dim: int) -> None: ...

    def upsert_points(self, collection_name: str, points: list[VectorPoint]) -> None: ...

    def search(
        self,
        collection_name: str,
        query: list[float],
        file_filter: list[str] | None,
        limit: int,
    ) -> list[VectorSearchHit]: ...

    def delete_points(self, collection_name: str, point_ids: list[int]) -> None: ...

    def max_point_id(self, collection_name: str) -> int: ...

    def close(self) -> None: ...


class QdrantVectorStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        client: QdrantClient | None = None,
    ):
        self.client = client or QdrantClient(host=host, port=port)

    def ensure_collection(self, collection_name: str, vector_dim: int) -> None:
        if self.client.collection_exists(collection_name):
            info = self.client.get_collection(collection_name)
            vectors = info.config.params.vectors
            if isinstance(vectors, VectorParams):
                existing_dim = vectors.size
            elif isinstance(vectors, dict) and vectors:
                existing_dim = next(iter(vectors.values())).size
            else:
                existing_dim = None
            if existing_dim == vector_dim:
                return
            self.client.delete_collection(collection_name)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    def upsert_points(self, collection_name: str, points: list[VectorPoint]) -> None:
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in points
            ],
        )

    def search(
        self,
        collection_name: str,
        query: list[float],
        file_filter: list[str] | None,
        limit: int,
    ) -> list[VectorSearchHit]:
        search_filter = None
        if file_filter:
            search_filter = Filter(
                must=[FieldCondition(key="file_name", match=MatchAny(any=file_filter))]
            )
        results = self.client.query_points(
            collection_name=collection_name,
            query=query,
            query_filter=search_filter,
            limit=limit,
        )
        return [
            VectorSearchHit(
                id=point.id if isinstance(point.id, (int, str)) else str(point.id),
                score=point.score,
                payload=point.payload or {},
            )
            for point in results.points
        ]

    def delete_points(self, collection_name: str, point_ids: list[int]) -> None:
        if not point_ids:
            return
        self.client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=cast(Any, point_ids)),
        )

    def max_point_id(self, collection_name: str) -> int:
        if not self.client.collection_exists(collection_name):
            return -1
        max_id = -1
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for record in records:
                max_id = max(max_id, int(record.id))
            if offset is None:
                break
        return max_id

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

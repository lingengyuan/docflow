from types import SimpleNamespace

from src.vector_store import QdrantVectorStore, VectorPoint


class FakeQdrantClient:
    def __init__(self):
        self.created = []
        self.upserts = []
        self.deleted_points = []
        self.closed = False
        self.exists = False
        self.query_filter = None

    def collection_exists(self, collection_name):
        return self.exists

    def get_collection(self, collection_name):
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=3)))
        )

    def create_collection(self, collection_name, vectors_config):
        self.created.append((collection_name, vectors_config.size))
        self.exists = True

    def delete_collection(self, collection_name):
        self.exists = False

    def upsert(self, collection_name, points):
        self.upserts.append((collection_name, points))

    def query_points(self, collection_name, query, query_filter, limit):
        self.query_filter = query_filter
        return SimpleNamespace(
            points=[
                SimpleNamespace(id=7, score=0.91, payload={"file_name": "note.md"}),
            ]
        )

    def delete(self, collection_name, points_selector):
        self.deleted_points.append((collection_name, points_selector.points))

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        if offset is None:
            return [SimpleNamespace(id=3), SimpleNamespace(id=9)], "done"
        return [], None

    def close(self):
        self.closed = True


def test_qdrant_vector_store_maps_points_and_search_hits():
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client)

    store.ensure_collection("docflow", 3)
    store.upsert_points(
        "docflow", [VectorPoint(id=1, vector=[0.1, 0.2, 0.3], payload={"file_name": "a.md"})]
    )
    hits = store.search("docflow", [0.1, 0.2, 0.3], ["a.md"], 5)

    assert client.created == [("docflow", 3)]
    assert client.upserts[0][1][0].id == 1
    assert client.query_filter.should[0].key == "file_name"
    assert client.query_filter.should[1].key == "file_path"
    assert hits[0].id == 7
    assert hits[0].payload["file_name"] == "note.md"


def test_qdrant_vector_store_lifecycle_helpers():
    client = FakeQdrantClient()
    client.exists = True
    store = QdrantVectorStore(client=client)

    assert store.max_point_id("docflow") == 9
    store.delete_points("docflow", [1, 2])
    store.close()

    assert client.deleted_points == [("docflow", [1, 2])]
    assert client.closed is True

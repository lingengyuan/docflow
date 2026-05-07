from concurrent.futures import ThreadPoolExecutor

from src.ingest.embedder import Embedder


def test_reserve_ids_advances_stale_counter_to_floor(tmp_path):
    counter_path = tmp_path / "qdrant_id_counter.txt"
    counter_path.write_text("5", encoding="utf-8")
    embedder = Embedder(id_counter_path=counter_path)

    ids = embedder._reserve_ids(3, min_next_id=10)

    assert ids == [10, 11, 12]
    assert counter_path.read_text(encoding="utf-8") == "13"


def test_reserve_ids_uses_file_lock_across_instances(tmp_path):
    counter_path = tmp_path / "qdrant_id_counter.txt"
    counter_path.write_text("0", encoding="utf-8")
    embedders = [Embedder(id_counter_path=counter_path) for _ in range(4)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        ranges = list(executor.map(lambda item: item._reserve_ids(5), embedders))

    ids = [qid for batch in ranges for qid in batch]
    assert sorted(ids) == list(range(20))
    assert counter_path.read_text(encoding="utf-8") == "20"


def test_sync_id_counter_does_not_move_backwards(tmp_path):
    counter_path = tmp_path / "qdrant_id_counter.txt"
    counter_path.write_text("42", encoding="utf-8")
    embedder = Embedder(id_counter_path=counter_path)

    result = embedder.sync_id_counter(min_next_id=10)

    assert result["value"] == 42
    assert result["advanced"] is False
    assert counter_path.read_text(encoding="utf-8") == "42"

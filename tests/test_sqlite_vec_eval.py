from scripts.evaluate_sqlite_vec import build_report


def test_sqlite_vec_evaluation_report_keeps_qdrant_as_active_store(tmp_path):
    db_path = tmp_path / "docflow.db"
    db_path.write_text("", encoding="utf-8")

    report = build_report(db_path)

    assert report["active_vector_store"] == "qdrant"
    assert report["candidate_vector_store"] == "sqlite-vec"
    assert report["migration_decision"] == "do_not_migrate_without_benchmark"
    assert "verified backup" in report["required_before_migration"]
    assert "query latency comparison" in report["required_before_migration"]

from __future__ import annotations

from pathlib import Path

from scripts.run_performance_smoke import run_performance_smoke


def test_performance_smoke_reports_large_file_and_library(tmp_path):
    report = run_performance_smoke(large_sections=6, library_files=4, tmp_root=tmp_path)

    assert report["schema"] == "docflow.performance_smoke.v1"
    assert report["passed"] is True
    assert report["large_file"]["files"] == 1
    assert report["large_file"]["chunks"] > 0
    assert report["library"]["files"] == 4
    assert report["library"]["chunks"] >= 4
    assert "large_file_total_ms_max" in report["thresholds"]


def test_performance_smoke_can_write_under_eval_results(tmp_path):
    from scripts.run_performance_smoke import write_results

    report = run_performance_smoke(large_sections=2, library_files=2, tmp_root=tmp_path)
    output = write_results(report, Path(tmp_path / "results"))

    assert output.exists()
    assert (tmp_path / "results" / "performance-latest.json").exists()

from pathlib import Path
import subprocess

from src.maintenance.local_install import build_install_plan, install_local


def test_install_plan_defaults_to_safe_preview(tmp_path):
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")

    plan = build_install_plan(root=tmp_path, python_bin=Path("/usr/bin/python3"))

    assert plan["schema"] == "docflow.local_install_plan.v1"
    assert [step["id"] for step in plan["steps"]] == [
        "create_venv",
        "install_python_deps",
        "startup_check",
        "restore_drill",
        "repair_ids_preview",
        "service_dry_run",
    ]
    assert "--dry-run" in plan["steps"][-1]["command"]


def test_install_local_dry_run_does_not_execute(tmp_path):
    calls = []

    def runner(command, cwd):
        calls.append((command, cwd))
        return subprocess.CompletedProcess(command, 0, "ok", "")

    result = install_local(
        root=tmp_path,
        python_bin=Path("/usr/bin/python3"),
        dry_run=True,
        runner=runner,
    )

    assert result["status"] == "dry_run"
    assert calls == []


def test_install_local_apply_runs_plan_steps(tmp_path):
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    calls = []

    def runner(command, cwd):
        calls.append((command, cwd))
        return subprocess.CompletedProcess(command, 0, "ok", "")

    result = install_local(
        root=tmp_path,
        python_bin=Path("/usr/bin/python3"),
        dry_run=False,
        with_service=True,
        skip_deps=True,
        runner=runner,
    )

    assert result["status"] == "ok"
    assert [call[0][2:4] for call in calls[:3]] == [
        ["start", "--check-only"],
        ["restore-drill", "--json"],
        ["repair-ids", "--dry-run"],
    ]
    assert calls[-1][0][2:4] == ["service", "install"]

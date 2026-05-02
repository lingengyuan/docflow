from __future__ import annotations

from pathlib import Path

from src.maintenance import launchd


def test_build_plist_uses_start_command(tmp_path):
    root = tmp_path / "docflow"
    python_bin = root / ".venv" / "bin" / "python"

    plist = launchd.build_plist(root=root, python_bin=python_bin, host="127.0.0.1", port=8010, log_dir=tmp_path)

    assert plist["Label"] == "com.docflow.local"
    assert plist["ProgramArguments"] == [
        str(python_bin),
        "main.py",
        "start",
        "--host",
        "127.0.0.1",
        "--port",
        "8010",
    ]
    assert plist["WorkingDirectory"] == str(root)
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is True


def test_install_service_dry_run_does_not_write_plist(monkeypatch, tmp_path):
    plist_path = tmp_path / "LaunchAgents" / "com.docflow.local.plist"
    monkeypatch.setattr(launchd, "plist_path", lambda home=None: plist_path)
    monkeypatch.setattr(launchd, "logs_dir", lambda home=None: tmp_path / "Logs")
    monkeypatch.setattr(launchd, "service_domain", lambda: "gui/501")
    monkeypatch.setattr(launchd, "service_target", lambda: "gui/501/com.docflow.local")

    result = launchd.install_service(
        root=tmp_path / "repo",
        python_bin=Path("/usr/bin/python3"),
        port=8011,
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert not plist_path.exists()
    assert result["commands"][1] == ["launchctl", "bootstrap", "gui/501", str(plist_path)]


def test_uninstall_service_dry_run_does_not_remove_plist(monkeypatch, tmp_path):
    plist_path = tmp_path / "com.docflow.local.plist"
    plist_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(launchd, "plist_path", lambda home=None: plist_path)
    monkeypatch.setattr(launchd, "service_domain", lambda: "gui/501")

    result = launchd.uninstall_service(dry_run=True)

    assert result["status"] == "dry_run"
    assert plist_path.exists()
    assert result["commands"] == [["launchctl", "bootout", "gui/501", str(plist_path)]]

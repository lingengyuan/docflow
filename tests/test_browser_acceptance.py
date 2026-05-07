from __future__ import annotations

from src.quality.browser_acceptance import (
    build_browser_acceptance_plan,
    format_browser_acceptance_report,
    run_browser_acceptance,
)


def test_browser_acceptance_plan_covers_main_views_and_commands():
    plan = build_browser_acceptance_plan()

    assert [view.id for view in plan] == ["chat", "library", "notes", "settings"]
    settings = plan[-1]
    assert "#settings-actions-list" in settings.visible_selectors
    assert "python main.py install-local" in settings.expected_text
    assert "python main.py restore-drill" in settings.expected_text
    assert "python main.py repair-ids --dry-run" in settings.expected_text


def test_browser_acceptance_report_formats_failures():
    report = {
        "base_url": "http://127.0.0.1:8000",
        "screenshots": [],
        "checks": [
            {"id": "server_reachable", "passed": True},
            {"id": "settings_text_install_local", "passed": False, "error": "missing text"},
        ],
        "passed": 1,
    }

    text = format_browser_acceptance_report(report)

    assert "DocFlow browser acceptance: 1/2 passed" in text
    assert "[FAIL] settings_text_install_local :: missing text" in text


def test_browser_acceptance_fails_fast_when_server_is_unreachable():
    report = run_browser_acceptance(
        base_url="http://127.0.0.1:9",
        screenshot_dir=None,
        timeout_ms=100,
    )

    assert report["status"] == "failed"
    assert report["checks"][0]["id"] == "server_reachable"
    assert report["checks"][0]["passed"] is False
    assert report["screenshots"] == []

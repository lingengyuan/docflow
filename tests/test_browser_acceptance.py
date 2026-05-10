from __future__ import annotations

import inspect

from src.quality.browser_acceptance import (
    build_browser_acceptance_plan,
    format_browser_acceptance_report,
    run_browser_acceptance,
)


def test_browser_acceptance_plan_covers_main_views_without_developer_language():
    plan = build_browser_acceptance_plan()

    assert [view.id for view in plan] == ["chat", "library", "source", "notes", "settings"]
    settings = plan[-1]
    assert "#settings-insights-list" in settings.visible_selectors
    assert "#settings-storage-list" in settings.visible_selectors
    assert "状态提示" in settings.expected_text
    assert "存储使用" in settings.expected_text
    assert "使用偏好" in settings.expected_text
    assert all("python main.py" not in text for text in settings.expected_text)


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


def test_browser_acceptance_supports_optional_mutation_flow():
    signature = inspect.signature(run_browser_acceptance)

    assert "include_mutation_flow" in signature.parameters
    assert signature.parameters["include_mutation_flow"].default is False

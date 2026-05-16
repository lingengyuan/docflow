from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.quality.browser_acceptance_a11y import (
    check_answer_quality_states_render,
    check_basic_accessibility_contract,
    check_keyboard_focus,
    check_status_messages_are_announced,
)
from src.quality.browser_acceptance_checks import (
    assert_browser_console_clean,
    check_desktop_viewports,
    check_server,
    goto_page,
    run_view_checks,
    wait_for_view_ready,
)
from src.quality.browser_acceptance_mutation import check_note_ingest_query_cleanup_flow
from src.quality.browser_acceptance_plan import (
    DEFAULT_BASE_URL,
    DEFAULT_SCREENSHOT_DIR,
    build_browser_acceptance_plan,
)
from src.quality.browser_acceptance_plan import (
    ViewAcceptance as ViewAcceptance,
)
from src.quality.browser_acceptance_report import (
    build_report,
    run_check,
)
from src.quality.browser_acceptance_report import (
    format_browser_acceptance_report as format_browser_acceptance_report,
)


def run_browser_acceptance(
    base_url: str = DEFAULT_BASE_URL,
    screenshot_dir: str | Path | None = DEFAULT_SCREENSHOT_DIR,
    headless: bool = True,
    timeout_ms: int = 8000,
    include_mutation_flow: bool = False,
) -> dict[str, Any]:
    start = time.perf_counter()
    checks: list[dict[str, Any]] = []
    screenshots: list[str] = []

    run_check(checks, "server_reachable", lambda: check_server(base_url, timeout_ms))
    if checks[-1]["passed"] is False:
        return build_report(base_url, checks, screenshots, start)

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        checks.append(
            {
                "id": "playwright_available",
                "passed": False,
                "error": (
                    f"{type(exc).__name__}: {exc}. Install Playwright and Chromium, "
                    "then run `.venv/bin/python -m playwright install chromium`."
                ),
            }
        )
        return build_report(base_url, checks, screenshots, start)

    output_dir = Path(screenshot_dir).expanduser().resolve() if screenshot_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    console_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": 1440, "height": 960})
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
        )
        run_check(checks, "page_loads", lambda: goto_page(page, base_url, timeout_ms))
        for index, view in enumerate(build_browser_acceptance_plan(), start=1):
            run_view_checks(page, view, checks, timeout_ms)
            run_check(
                checks,
                f"{view.id}_ready_for_screenshot",
                lambda view=view: wait_for_view_ready(page, view, timeout_ms),
            )
            if output_dir:
                screenshot = output_dir / f"{index:02d}-{view.id}.png"
                page.screenshot(path=str(screenshot), full_page=True)
                screenshots.append(str(screenshot))
        if include_mutation_flow:
            run_check(
                checks,
                "mutation_note_ingest_query_cleanup",
                lambda: check_note_ingest_query_cleanup_flow(page, base_url, timeout_ms),
            )
        run_check(
            checks,
            "desktop_viewports_stay_usable",
            lambda: check_desktop_viewports(page, timeout_ms),
        )
        run_check(
            checks,
            "basic_accessibility_contract",
            lambda: check_basic_accessibility_contract(page),
        )
        run_check(
            checks,
            "desktop_status_messages_are_announced",
            lambda: check_status_messages_are_announced(page),
        )
        run_check(
            checks,
            "answer_quality_states_render",
            lambda: check_answer_quality_states_render(page),
        )
        run_check(
            checks,
            "keyboard_focus_reaches_controls",
            lambda: check_keyboard_focus(page, timeout_ms),
        )
        browser.close()

    run_check(
        checks,
        "browser_console_errors",
        lambda: assert_browser_console_clean(console_errors),
    )
    return build_report(base_url, checks, screenshots, start)

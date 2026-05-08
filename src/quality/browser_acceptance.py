from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_SCREENSHOT_DIR = Path("output/playwright/phase25-browser-acceptance")


@dataclass(frozen=True)
class ViewAcceptance:
    id: str
    label: str
    nav_selector: str
    view_selector: str
    visible_selectors: tuple[str, ...]
    expected_text: tuple[str, ...]


def build_browser_acceptance_plan() -> list[ViewAcceptance]:
    return [
        ViewAcceptance(
            id="chat",
            label="Chat",
            nav_selector="#nav-chat",
            view_selector="#view-chat",
            visible_selectors=("#global-search-input", "#input", "#send-btn", "#query-scope-mode", "#chat-context-sources"),
            expected_text=("DocFlow", "全部知识库"),
        ),
        ViewAcceptance(
            id="library",
            label="Library",
            nav_selector="#nav-library",
            view_selector="#view-library",
            visible_selectors=(
                "#refresh-files-btn",
                "#scan-folders-btn",
                "#upload-zone",
                "#library-group-controls",
                "#library-status-filter",
                "#library-collection-filter",
                "#file-tbody",
                "#library-context-panel",
            ),
            expected_text=("文件库", "刷新列表", "扫描文件夹", "全部文件", "最近导入", "拖拽文件至此或点击上传"),
        ),
        ViewAcceptance(
            id="source",
            label="Source Preview",
            nav_selector="#nav-source",
            view_selector="#view-source",
            visible_selectors=(
                "#source-result-list",
                "#source-document-viewer",
                "#source-detail-panel",
                "#source-preview-count",
            ),
            expected_text=("来源预览", "搜索与引用", "引用详情"),
        ),
        ViewAcceptance(
            id="notes",
            label="Notes",
            nav_selector="#nav-notes",
            view_selector="#view-notes",
            visible_selectors=(
                "#notes-title-input",
                "#notes-content-input",
                "#notes-url-input",
                "#knowledge-output-panel",
                "#knowledge-submit-btn",
                "#notes-list",
            ),
            expected_text=("新建 Markdown 笔记", "导入网页", "生成知识产物", "最近采集"),
        ),
        ViewAcceptance(
            id="settings",
            label="Settings",
            nav_selector="#nav-settings",
            view_selector="#view-settings",
            visible_selectors=(
                "#health-btn",
                "#settings-model-list",
                "#settings-sources-list",
                "#settings-insights-list",
                "#top-local-status",
            ),
            expected_text=(
                "系统状态",
                "本地模型",
                "监控目录",
                "使用偏好",
                "状态提示",
                "资料范围",
            ),
        ),
    ]


def run_browser_acceptance(
    base_url: str = DEFAULT_BASE_URL,
    screenshot_dir: str | Path | None = DEFAULT_SCREENSHOT_DIR,
    headless: bool = True,
    timeout_ms: int = 8000,
) -> dict[str, Any]:
    start = time.perf_counter()
    checks: list[dict[str, Any]] = []
    screenshots: list[str] = []

    _run_check(checks, "server_reachable", lambda: _check_server(base_url, timeout_ms))
    if checks[-1]["passed"] is False:
        return _report(base_url, checks, screenshots, start)

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
        return _report(base_url, checks, screenshots, start)

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
        _run_check(
            checks,
            "page_loads",
            lambda: _goto_page(page, base_url, timeout_ms),
        )
        for index, view in enumerate(build_browser_acceptance_plan(), start=1):
            _run_view_checks(page, view, checks, timeout_ms)
            if output_dir:
                screenshot = output_dir / f"{index:02d}-{view.id}.png"
                page.screenshot(path=str(screenshot), full_page=True)
                screenshots.append(str(screenshot))
        browser.close()

    _run_check(
        checks,
        "browser_console_errors",
        lambda: _assert(not console_errors, "; ".join(console_errors[:3])),
    )
    return _report(base_url, checks, screenshots, start)


def format_browser_acceptance_report(report: dict[str, Any]) -> str:
    lines = [
        f"DocFlow browser acceptance: {report['passed']}/{len(report['checks'])} passed",
        f"Base URL: {report['base_url']}",
    ]
    if report.get("screenshots"):
        lines.append(f"Screenshots: {report['screenshots_dir']}")
    for check in report["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        reason = "" if check["passed"] else f" :: {check.get('error', '')}"
        lines.append(f"[{mark}] {check['id']}{reason}")
    return "\n".join(lines)


def _run_view_checks(page: Any, view: ViewAcceptance, checks: list[dict[str, Any]], timeout_ms: int) -> None:
    _run_check(
        checks,
        f"{view.id}_nav",
        lambda: page.locator(view.nav_selector).click(timeout=timeout_ms),
    )
    _run_check(
        checks,
        f"{view.id}_view_visible",
        lambda: page.locator(view.view_selector).wait_for(state="visible", timeout=timeout_ms),
    )
    for selector in view.visible_selectors:
        _run_check(
            checks,
            f"{view.id}_visible_{_selector_id(selector)}",
            lambda selector=selector: page.locator(selector).first.wait_for(
                state="visible",
                timeout=timeout_ms,
            ),
        )
    for text in view.expected_text:
        _run_check(
            checks,
            f"{view.id}_text_{_text_id(text)}",
            lambda text=text: _assert_text(page, text, timeout_ms),
        )
    if view.id == "settings":
        _run_check(checks, "settings_has_no_developer_language", lambda: _assert_no_settings_developer_language(page, timeout_ms))
    if view.id == "library":
        _run_check(checks, "library_group_filter_clicks", lambda: _check_library_groups(page, timeout_ms))
        _run_check(checks, "library_source_review_opens", lambda: _check_library_source_review(page, timeout_ms))


def _check_server(base_url: str, timeout_ms: int) -> dict[str, Any]:
    timeout_seconds = max(timeout_ms / 1000, 1)
    request = urllib.request.Request(base_url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = getattr(response, "status", 0)
        if status < 200 or status >= 400:
            raise AssertionError(f"HTTP {status}")
        return {"status": status}


def _goto_page(page: Any, base_url: str, timeout_ms: int) -> dict[str, str]:
    page.goto(base_url, wait_until="domcontentloaded", timeout=timeout_ms)
    return {"url": page.url}


def _assert_text(page: Any, text: str, timeout_ms: int) -> None:
    body = page.locator("body").inner_text(timeout=timeout_ms)
    if text not in body:
        raise AssertionError(f"missing text: {text}")


def _assert_no_settings_developer_language(page: Any, timeout_ms: int) -> None:
    body = page.locator("#view-settings").inner_text(timeout=timeout_ms)
    forbidden_terms = (
        "python main.py",
        "install-local",
        "restore-drill",
        "repair-ids",
        "dry-run",
        "browser-acceptance",
        "doctor",
        "维护",
        "恢复建议",
        "维护命令",
        "复制命令",
    )
    found = [term for term in forbidden_terms if term in body]
    if found:
        raise AssertionError(f"settings exposes developer language: {', '.join(found)}")


def _check_library_groups(page: Any, timeout_ms: int) -> dict[str, str]:
    page.locator("#library-group-pdf").click(timeout=timeout_ms)
    page.locator("#library-group-all").click(timeout=timeout_ms)
    return {"clicked": "pdf,all"}


def _check_library_source_review(page: Any, timeout_ms: int) -> dict[str, Any]:
    row = page.locator("#file-tbody tr[data-file-id]").first
    if row.count() == 0:
        return {"skipped": "no files"}
    row.click(timeout=timeout_ms)
    button = page.locator("#library-source-review-btn")
    if button.count() == 0:
        return {"skipped": "no active file"}
    button.click(timeout=timeout_ms)
    body = page.locator("#library-source-review")
    body.wait_for(state="visible", timeout=timeout_ms)
    page.wait_for_function(
        """
        () => {
            const el = document.querySelector('#library-source-review');
            if (!el) return false;
            const text = el.innerText || '';
            return ['为什么引用这些片段', '还没有可预览', '读取失败'].some(token => text.includes(token));
        }
        """,
        timeout=timeout_ms,
    )
    text = body.inner_text(timeout=timeout_ms)
    if not any(token in text for token in ("为什么引用这些片段", "还没有可预览", "读取失败")):
        raise AssertionError("source review did not render a known state")
    return {"state": text[:80]}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message or "assertion failed")


def _run_check(checks: list[dict[str, Any]], check_id: str, fn: Any) -> None:
    try:
        details = fn()
        checks.append({"id": check_id, "passed": True, "details": details or {}})
    except (AssertionError, urllib.error.URLError, TimeoutError, Exception) as exc:
        checks.append(
            {
                "id": check_id,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _report(
    base_url: str,
    checks: list[dict[str, Any]],
    screenshots: list[str],
    start: float,
) -> dict[str, Any]:
    passed = sum(1 for check in checks if check["passed"])
    screenshots_dir = str(Path(screenshots[0]).parent) if screenshots else ""
    return {
        "schema": "docflow.browser_acceptance.v1",
        "status": "ok" if passed == len(checks) else "failed",
        "base_url": base_url,
        "duration_ms": round((time.perf_counter() - start) * 1000),
        "screenshots_dir": screenshots_dir,
        "screenshots": screenshots,
        "checks": checks,
        "passed": passed,
        "failed": len(checks) - passed,
    }


def _selector_id(selector: str) -> str:
    return selector.strip("#.").replace("-", "_").replace(" ", "_")


def _text_id(text: str) -> str:
    tokens = {
        "DocFlow": "docflow",
        "全部知识库": "all_scope",
        "文件库": "library_title",
        "刷新列表": "refresh_files",
        "扫描文件夹": "scan_folders",
        "全部文件": "all_files_group",
        "最近导入": "recent_group",
        "拖拽文件至此或点击上传": "upload_zone",
        "来源预览": "source_preview",
        "搜索与引用": "source_search",
        "引用详情": "source_detail",
        "新建 Markdown 笔记": "new_markdown_note",
        "导入网页": "import_url",
        "生成知识产物": "knowledge_output",
        "最近采集": "recent_notes",
        "系统状态": "health",
        "本地模型": "local_models",
        "监控目录": "sources",
        "使用偏好": "preferences",
        "状态提示": "status_insights",
        "资料范围": "source_scope",
    }
    return tokens.get(text, text[:24].replace(" ", "_").replace("/", "_"))

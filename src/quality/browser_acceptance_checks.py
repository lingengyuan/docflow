"""Read-only browser acceptance checks."""

from __future__ import annotations

from typing import Any

from src import net
from src.quality.browser_acceptance_plan import ViewAcceptance
from src.quality.browser_acceptance_report import (
    assert_condition,
    run_check,
    selector_id,
    text_id,
)


def run_view_checks(
    page: Any, view: ViewAcceptance, checks: list[dict[str, Any]], timeout_ms: int
) -> None:
    run_check(
        checks,
        f"{view.id}_nav",
        lambda: page.locator(view.nav_selector).click(timeout=timeout_ms),
    )
    run_check(
        checks,
        f"{view.id}_view_visible",
        lambda: page.locator(view.view_selector).wait_for(state="visible", timeout=timeout_ms),
    )
    for selector in view.visible_selectors:
        run_check(
            checks,
            f"{view.id}_visible_{selector_id(selector)}",
            lambda selector=selector: page.locator(selector).first.wait_for(
                state="visible",
                timeout=timeout_ms,
            ),
        )
    for text in view.expected_text:
        run_check(
            checks,
            f"{view.id}_text_{text_id(text)}",
            lambda text=text: assert_text(page, text, timeout_ms),
        )
    if view.id == "settings":
        run_check(
            checks,
            "settings_has_no_developer_language",
            lambda: assert_no_settings_developer_language(page, timeout_ms),
        )
        run_check(
            checks,
            "settings_theme_toggle",
            lambda: check_theme_toggle(page, timeout_ms),
        )
    if view.id == "library":
        run_check(
            checks, "library_group_filter_clicks", lambda: check_library_groups(page, timeout_ms)
        )
        run_check(
            checks,
            "library_source_review_opens",
            lambda: check_library_source_review(page, timeout_ms),
        )
    if view.id == "source":
        run_check(
            checks, "source_preview_loaded", lambda: check_source_preview_loaded(page, timeout_ms)
        )
    if view.id == "notes":
        run_check(
            checks,
            "notes_knowledge_loop_visible",
            lambda: check_notes_knowledge_loop(page, timeout_ms),
        )


def check_server(base_url: str, timeout_ms: int) -> dict[str, Any]:
    timeout_seconds = max(timeout_ms / 1000, 1)
    response = net.get(base_url, timeout=net.Timeout(timeout_seconds, connect=1.0))
    response.raise_for_status()
    return {"status": response.status_code}


def goto_page(page: Any, base_url: str, timeout_ms: int) -> dict[str, str]:
    page.goto(base_url, wait_until="domcontentloaded", timeout=timeout_ms)
    return {"url": page.url}


def assert_text(page: Any, text: str, timeout_ms: int) -> None:
    body = page.locator("body").inner_text(timeout=timeout_ms)
    if text not in body:
        raise AssertionError(f"missing text: {text}")


def assert_no_settings_developer_language(page: Any, timeout_ms: int) -> None:
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


def check_theme_toggle(page: Any, timeout_ms: int) -> dict[str, str]:
    before = page.evaluate("document.documentElement.dataset.theme || 'light'")
    page.locator("#theme-toggle-btn").click(timeout=timeout_ms)
    page.wait_for_function(
        """theme => (document.documentElement.dataset.theme || 'light') !== theme""",
        arg=before,
        timeout=timeout_ms,
    )
    after = page.evaluate("document.documentElement.dataset.theme || 'light'")
    page.locator("#theme-toggle-btn").click(timeout=timeout_ms)
    page.wait_for_function(
        """theme => (document.documentElement.dataset.theme || 'light') === theme""",
        arg=before,
        timeout=timeout_ms,
    )
    return {"before": before, "after": after}


def check_desktop_viewports(page: Any, timeout_ms: int) -> dict[str, Any]:
    checked: list[int] = []
    for width in (1280, 1440, 1728):
        page.set_viewport_size({"width": width, "height": 900})
        page.locator("#nav-chat").click(timeout=timeout_ms)
        page.locator("#input").wait_for(state="visible", timeout=timeout_ms)
        overflow = page.evaluate(
            """
            () => Math.max(
              document.documentElement.scrollWidth,
              document.body.scrollWidth
            ) - window.innerWidth
            """
        )
        if overflow > 4:
            raise AssertionError(f"horizontal overflow at {width}px: {overflow}px")
        checked.append(width)
    page.set_viewport_size({"width": 1440, "height": 960})
    return {"widths": checked}


def check_library_groups(page: Any, timeout_ms: int) -> dict[str, str]:
    page.locator("#library-group-pdf").click(timeout=timeout_ms)
    wait_for_library_loaded(page, timeout_ms)
    page.locator("#library-group-all").click(timeout=timeout_ms)
    wait_for_library_loaded(page, timeout_ms, expect_rows=True)
    return {"clicked": "pdf,all"}


def check_library_source_review(page: Any, timeout_ms: int) -> dict[str, Any]:
    wait_for_library_loaded(page, timeout_ms, expect_rows=True)
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
            return ['为什么引用这些片段', '还没有可预览', '读取失败']
              .some(token => text.includes(token));
        }
        """,
        timeout=timeout_ms,
    )
    text = body.inner_text(timeout=timeout_ms)
    if not any(token in text for token in ("为什么引用这些片段", "还没有可预览", "读取失败")):
        raise AssertionError("source review did not render a known state")
    return {"state": text[:80]}


def check_source_preview_loaded(page: Any, timeout_ms: int) -> dict[str, Any]:
    page.wait_for_function(
        """
        () => {
            const detail = document.querySelector('#source-detail-panel');
            const viewer = document.querySelector('#source-document-viewer');
            if (!detail || !viewer) return false;
            const text = `${detail.innerText || ''} ${viewer.innerText || ''}`;
            return !text.includes('正在载入') && !text.includes('正在读取') && (
                text.includes('引用关系') ||
                text.includes('从对话引用') ||
                text.includes('没有片段')
            );
        }
        """,
        timeout=timeout_ms,
    )
    return {"state": page.locator("#source-detail-panel").inner_text(timeout=timeout_ms)[:80]}


def check_notes_knowledge_loop(page: Any, timeout_ms: int) -> dict[str, Any]:
    page.wait_for_function(
        """
        () => {
            const panel = document.querySelector('#knowledge-review-panel');
            if (!panel) return false;
            const text = panel.innerText || '';
            return ['知识闭环', '资料', '提问', '来源', '沉淀', '关联', '回顾', '反馈']
              .every(token => text.includes(token));
        }
        """,
        timeout=timeout_ms,
    )
    return {"state": page.locator("#knowledge-review-panel").inner_text(timeout=timeout_ms)[:120]}


def wait_for_view_ready(page: Any, view: ViewAcceptance, timeout_ms: int) -> dict[str, Any]:
    if view.id == "library":
        return wait_for_library_loaded(page, timeout_ms, expect_rows=True)
    if view.id == "source":
        return check_source_preview_loaded(page, timeout_ms)
    if view.id == "settings":
        page.wait_for_function(
            """
            () => {
                const storage = document.querySelector('#settings-storage-list');
                const insights = document.querySelector('#settings-insights-list');
                if (!storage || !insights) return false;
                const text = `${storage.innerText || ''} ${insights.innerText || ''}`;
                return !text.includes('正在读取') && !text.includes('正在检查');
            }
            """,
            timeout=timeout_ms,
        )
        return {"ready": "settings data loaded"}
    if view.id == "notes":
        page.wait_for_function(
            """
            () => {
                const list = document.querySelector('#notes-list');
                const outputs = document.querySelector('#knowledge-output-panel');
                const review = document.querySelector('#knowledge-review-panel');
                return Boolean(
                  list && outputs && review &&
                  (list.innerText || '').trim() &&
                  (review.innerText || '').trim()
                );
            }
            """,
            timeout=timeout_ms,
        )
        return {"ready": "notes data loaded"}
    return {"ready": "static view"}


def wait_for_library_loaded(
    page: Any, timeout_ms: int, *, expect_rows: bool = False
) -> dict[str, Any]:
    page.wait_for_function(
        """
        ({ expectRows }) => {
            const tbody = document.querySelector('#file-tbody');
            if (!tbody) return false;
            const text = tbody.innerText || '';
            const rows = tbody.querySelectorAll('tr[data-file-id]').length;
            const totalText = document.querySelector('#library-group-all-count')?.innerText || '0';
            const total = Number(totalText.trim()) || 0;
            if (expectRows && total > 0) return rows > 0;
            return rows > 0 || text.includes('暂无匹配文件') || text.includes('加载失败');
        }
        """,
        arg={"expectRows": expect_rows},
        timeout=timeout_ms,
    )
    return {
        "rows": page.locator("#file-tbody tr[data-file-id]").count(),
        "count": page.locator("#library-count").inner_text(timeout=timeout_ms),
    }


def assert_browser_console_clean(console_errors: list[str]) -> None:
    assert_condition(not console_errors, "; ".join(console_errors[:3]))

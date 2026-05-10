from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
                "#settings-storage-list",
                "#top-local-status",
            ),
            expected_text=(
                "系统状态",
                "本地模型",
                "监控目录",
                "存储使用",
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
    include_mutation_flow: bool = False,
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
            _run_check(checks, f"{view.id}_ready_for_screenshot", lambda view=view: _wait_for_view_ready(page, view, timeout_ms))
            if output_dir:
                screenshot = output_dir / f"{index:02d}-{view.id}.png"
                page.screenshot(path=str(screenshot), full_page=True)
                screenshots.append(str(screenshot))
        if include_mutation_flow:
            _run_check(
                checks,
                "mutation_note_ingest_query_cleanup",
                lambda: _check_note_ingest_query_cleanup_flow(page, base_url, timeout_ms),
            )
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
    if view.id == "source":
        _run_check(checks, "source_preview_loaded", lambda: _check_source_preview_loaded(page, timeout_ms))


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
    _wait_for_library_loaded(page, timeout_ms)
    page.locator("#library-group-all").click(timeout=timeout_ms)
    _wait_for_library_loaded(page, timeout_ms, expect_rows=True)
    return {"clicked": "pdf,all"}


def _check_library_source_review(page: Any, timeout_ms: int) -> dict[str, Any]:
    _wait_for_library_loaded(page, timeout_ms, expect_rows=True)
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


def _check_source_preview_loaded(page: Any, timeout_ms: int) -> dict[str, Any]:
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


def _check_note_ingest_query_cleanup_flow(page: Any, base_url: str, timeout_ms: int) -> dict[str, Any]:
    stamp = str(int(time.time()))
    title = f"phase32-acceptance-{stamp}"
    token = f"phase32-token-{stamp}"
    question = "这条临时笔记里的验收标记是什么？请只回答标记。"
    file_record: dict[str, Any] | None = None
    cleanup_details: dict[str, Any] = {}
    conversation_id: int | None = None
    try:
        page.locator("#nav-notes").click(timeout=timeout_ms)
        page.locator("#notes-title-input").fill(title, timeout=timeout_ms)
        page.locator("#notes-collection-input").fill("Phase32 Acceptance", timeout=timeout_ms)
        page.locator("#notes-tags-input").fill("phase32,temp", timeout=timeout_ms)
        page.locator("#notes-content-input").fill(
            f"# {title}\n\n验收标记：{token}\n\n这是一条 Phase32 临时笔记。",
            timeout=timeout_ms,
        )
        page.locator("#notes-submit-btn").click(timeout=timeout_ms)
        page.wait_for_function(
            "() => (document.querySelector('#notes-status')?.innerText || '').includes('已加入入库队列')",
            timeout=timeout_ms,
        )
        file_record = _wait_for_file_by_title(base_url, title, max(timeout_ms, 60_000))
        file_record = _wait_for_file_status(base_url, int(file_record["id"]), "done", max(timeout_ms, 120_000))
        query_details = _query_temporary_note(page, file_record, token, question, timeout_ms)
        conversation_id = query_details.get("conversation_id")
        return {
            "created": file_record["file_name"],
            "status": file_record["status"],
            "queried": query_details,
            "cleanup": cleanup_details,
        }
    finally:
        if file_record:
            cleanup_details.update(_cleanup_mutation_file(file_record, conversation_id=conversation_id, question=question))


def _query_temporary_note(page: Any, file_record: dict[str, Any], token: str, question: str, timeout_ms: int) -> dict[str, Any]:
    file_id = int(file_record["id"])
    file_name = str(file_record["file_name"])
    page.locator("#nav-chat").click(timeout=timeout_ms)
    page.wait_for_function(
        """
        ({ fileId }) => {
            const select = document.querySelector('#query-scope-file');
            return Boolean(select && [...select.options].some(option => option.value === String(fileId)));
        }
        """,
        arg={"fileId": file_id},
        timeout=timeout_ms,
    )
    page.locator("#query-scope-mode").select_option("file", timeout=timeout_ms)
    page.locator("#query-scope-file").select_option(str(file_id), timeout=timeout_ms)
    page.locator("#input").fill(question, timeout=timeout_ms)
    page.locator("#send-btn").click(timeout=timeout_ms)
    page.wait_for_function(
        """
        ({ fileName }) => {
            const send = document.querySelector('#send-btn');
            const thinking = document.querySelector('#thinking-indicator');
            const messages = document.querySelector('#messages');
            const text = messages?.innerText || '';
            return Boolean(send && !send.disabled && !thinking && (
                text.includes('耗时') ||
                text.includes(fileName) ||
                text.includes('回答失败') ||
                text.includes('连接中断')
            ));
        }
        """,
        arg={"fileName": file_name},
        timeout=max(timeout_ms, 120_000),
    )
    text = page.locator("#messages").inner_text(timeout=timeout_ms)
    failure_terms = ("本次查询失败", "本次回答失败", "回答失败", "连接中断", "耗时太久", "暂时连接不上")
    matched_failures = [term for term in failure_terms if term in text]
    if matched_failures:
        raise AssertionError(f"temporary note query failed: {', '.join(matched_failures)}")
    answer_visible = bool(text.strip())
    if not answer_visible:
        raise AssertionError("temporary note query did not render an answer")
    conversation_id = page.evaluate("typeof currentConversationId === 'number' ? currentConversationId : null")
    return {
        "answer_visible": answer_visible,
        "citation_visible": file_name in text,
        "token_mentioned": token in text,
        "conversation_id": conversation_id,
    }


def _api_json(base_url: str, path: str, timeout_ms: int, *, method: str = "GET", payload: dict | None = None) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=max(timeout_ms / 1000, 1)) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else None


def _wait_for_file_by_title(base_url: str, title: str, timeout_ms: int) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_ms / 1000
    safe_title = title.lower()
    while time.perf_counter() < deadline:
        files = _api_json(base_url, "/api/files", timeout_ms)
        for item in files:
            name = str(item.get("file_name", "")).lower()
            if safe_title in name:
                return item
        time.sleep(0.5)
    raise AssertionError(f"temporary note was not listed: {title}")


def _wait_for_file_status(base_url: str, file_id: int, status: str, timeout_ms: int) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_ms / 1000
    last_status = ""
    while time.perf_counter() < deadline:
        files = _api_json(base_url, "/api/files", timeout_ms)
        for item in files:
            if int(item.get("id", 0)) != file_id:
                continue
            last_status = str(item.get("status", ""))
            if last_status == status:
                return item
            if last_status == "error":
                raise AssertionError(f"temporary note ingest failed: {item.get('error_msg', '')}")
        time.sleep(1)
    raise AssertionError(f"temporary note did not reach {status}; last status: {last_status or 'missing'}")


def _cleanup_mutation_file(
    file_record: dict[str, Any],
    *,
    conversation_id: int | None = None,
    question: str = "",
) -> dict[str, Any]:
    file_path = Path(str(file_record.get("file_path", ""))).expanduser()
    details: dict[str, Any] = {
        "file_deleted": False,
        "record_deleted": False,
        "vectors_deleted": 0,
        "history_deleted": 0,
        "conversation_deleted": False,
    }
    if file_path.exists():
        file_path.unlink()
        details["file_deleted"] = True
    cfg = _load_project_config()
    qdrant_ids = _delete_file_record(cfg, file_path)
    details["record_deleted"] = bool(qdrant_ids is not None)
    if qdrant_ids:
        _delete_qdrant_points(cfg, qdrant_ids)
        details["vectors_deleted"] = len(qdrant_ids)
    history_deleted, conversation_deleted = _delete_acceptance_history(
        cfg,
        conversation_id=conversation_id,
        question=question,
    )
    details["history_deleted"] = history_deleted
    details["conversation_deleted"] = conversation_deleted
    return details


def _load_project_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _expand_config_path(value: str | Path) -> Path:
    return Path(os.path.expanduser(str(value))).resolve()


def _delete_file_record(cfg: dict[str, Any], file_path: Path) -> list[int] | None:
    db_path = _expand_config_path(cfg.get("paths", {}).get("db_path", "docflow.db"))
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT id FROM files WHERE file_path = ?", (str(file_path),)).fetchone()
        if row is None:
            return None
        file_id = int(row[0])
        chunk_rows = conn.execute("SELECT id, qdrant_id FROM chunks WHERE file_id = ?", (file_id,)).fetchall()
        chunk_ids = [int(row[0]) for row in chunk_rows]
        qdrant_ids = [int(row[1]) for row in chunk_rows]
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})", chunk_ids)
            conn.execute(f"DELETE FROM chunks_fts_trigram WHERE rowid IN ({placeholders})", chunk_ids)
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM favorites WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
    return qdrant_ids


def _delete_acceptance_history(
    cfg: dict[str, Any],
    *,
    conversation_id: int | None,
    question: str,
) -> tuple[int, bool]:
    db_path = _expand_config_path(cfg.get("paths", {}).get("db_path", "docflow.db"))
    deleted_history = 0
    deleted_conversation = False
    with sqlite3.connect(db_path) as conn:
        if question:
            result = conn.execute("DELETE FROM history WHERE question = ?", (question,))
            deleted_history += result.rowcount
        if conversation_id:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            result = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            deleted_conversation = result.rowcount > 0
        elif question:
            rows = conn.execute(
                "SELECT DISTINCT conversation_id FROM messages WHERE content = ?",
                (question,),
            ).fetchall()
            for row in rows:
                cid = int(row[0])
                conn.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
                result = conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))
                deleted_conversation = deleted_conversation or result.rowcount > 0
    return deleted_history, deleted_conversation


def _delete_qdrant_points(cfg: dict[str, Any], qdrant_ids: list[int]) -> None:
    from qdrant_client import QdrantClient

    qdrant_cfg = cfg.get("qdrant", {})
    client = QdrantClient(
        host=qdrant_cfg.get("host", "localhost"),
        port=int(qdrant_cfg.get("port", 6333)),
        timeout=5,
    )
    client.delete(
        collection_name=qdrant_cfg.get("collection", "docflow"),
        points_selector=qdrant_ids,
    )


def _wait_for_view_ready(page: Any, view: ViewAcceptance, timeout_ms: int) -> dict[str, Any]:
    if view.id == "library":
        return _wait_for_library_loaded(page, timeout_ms, expect_rows=True)
    if view.id == "source":
        return _check_source_preview_loaded(page, timeout_ms)
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
                return Boolean(list && outputs && (list.innerText || '').trim());
            }
            """,
            timeout=timeout_ms,
        )
        return {"ready": "notes data loaded"}
    return {"ready": "static view"}


def _wait_for_library_loaded(page: Any, timeout_ms: int, *, expect_rows: bool = False) -> dict[str, Any]:
    page.wait_for_function(
        """
        ({ expectRows }) => {
            const tbody = document.querySelector('#file-tbody');
            if (!tbody) return false;
            const text = tbody.innerText || '';
            const rows = tbody.querySelectorAll('tr[data-file-id]').length;
            const total = Number((document.querySelector('#library-group-all-count')?.innerText || '0').trim()) || 0;
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

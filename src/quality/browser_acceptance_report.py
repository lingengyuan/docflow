"""Browser acceptance reporting helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src import net


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


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message or "assertion failed")


def run_check(checks: list[dict[str, Any]], check_id: str, fn: Any) -> None:
    try:
        details = fn()
        checks.append({"id": check_id, "passed": True, "details": details or {}})
    except (AssertionError, net.HTTPError, TimeoutError, Exception) as exc:
        checks.append(
            {
                "id": check_id,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def build_report(
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


def selector_id(selector: str) -> str:
    return selector.strip("#.").replace("-", "_").replace(" ", "_")


def text_id(text: str) -> str:
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

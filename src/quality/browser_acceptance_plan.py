"""Browser acceptance view plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
            visible_selectors=(
                "#global-search-input",
                "#input",
                "#send-btn",
                "#query-scope-mode",
                "#chat-context-sources",
            ),
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
            expected_text=(
                "文件库",
                "刷新列表",
                "扫描文件夹",
                "全部文件",
                "最近导入",
                "拖拽文件至此或点击上传",
            ),
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
                "#knowledge-review-panel",
            ),
            expected_text=(
                "新建 Markdown 笔记",
                "导入网页",
                "生成知识产物",
                "主动回顾",
                "最近采集",
            ),
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

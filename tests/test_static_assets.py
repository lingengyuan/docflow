from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app as api_app


def test_favicon_svg_is_served():
    client = TestClient(api_app.app)

    response = client.get("/favicon.svg")

    assert response.status_code == 200
    assert "image/svg+xml" in response.headers["content-type"]
    assert response.text.startswith("<svg")


def test_favicon_ico_is_served_without_404():
    client = TestClient(api_app.app)

    response = client.get("/favicon.ico")

    assert response.status_code == 200
    assert "image/svg+xml" in response.headers["content-type"]
    assert response.text.startswith("<svg")


def test_destructive_actions_use_in_app_confirmation():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert 'id="confirm-modal"' in html
    assert "function showConfirmDialog" in html
    assert "function closeConfirmDialog" in html
    assert "清空所有历史记录？" in html
    assert "删除这个对话？" in html
    assert "if (!confirm(" not in html
    assert "text-on-error bg-error" in html
    assert "bg-error/5 text-error hover:bg-error" in html


def test_health_panel_shows_core_and_optional_groups():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert "核心可用" in html
    assert "healthSnapshot.groups" in html
    assert "['core', 'optional']" in html
    assert "group.label" in html
    assert "optional_unavailable" in html
    assert "未安装" in html
    assert "renderHealthGroup" in html


def test_files_actions_have_clear_feedback():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert 'id="refresh-files-btn"' in html
    assert "刷新列表" in html
    assert "刷新中…" in html
    assert 'id="scan-folders-btn"' in html
    assert "folder_sync" in html
    assert "扫描中…" in html
    assert "已加入队列" in html
    assert "setRefreshButtonLoading" in html
    assert "setScanButtonState" in html


def test_library_view_has_filters_and_batch_actions():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert "library-collection-filter" in html
    assert "library-tag-filter" in html
    assert "library-favorite-filter" in html
    assert "batch-collection-input" in html
    assert "batch-tags-input" in html
    assert "favoriteSelected" in html
    assert "applyBatchMetadata" in html
    assert "rebuildSelected" in html


def test_phase13_import_and_note_actions_are_visible():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert "导入网页" in html
    assert "新建笔记" in html
    assert "library-workflow-panel" in html
    assert "runLibraryWorkflow" in html
    assert "/api/import/url" in html
    assert "/api/notes/from-answer" in html
    assert "保存为笔记" in html


def test_phase14_chat_scope_controls_are_visible():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert "query-scope-controls" in html
    assert "全部知识库" in html
    assert "指定集合" in html
    assert "指定文件" in html
    assert "全文模式" in html
    assert "buildQueryScopePayload" in html
    assert "scope_mode" in html
    assert "full_text" in html


def test_clickable_icon_actions_are_labeled_and_keyboard_accessible():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert 'aria-label="添加文件"' in html
    assert 'aria-label="发送问题"' in html
    assert 'aria-label="上传文件"' in html
    assert 'aria-label="复制答案"' in html
    assert 'aria-label="导出 Markdown"' in html
    assert 'aria-label="${escHtml(favoriteLabel)}"' in html
    assert 'role="button" tabindex="0"' in html
    assert "handleUploadZoneKey" in html
    assert "handleConversationKey" in html
    assert "handleSourceKey" in html
    assert "handleHistoryKey" in html

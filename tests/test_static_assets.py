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


def test_favicon_ico_head_is_served_without_404():
    client = TestClient(api_app.app)

    response = client.head("/favicon.ico")

    assert response.status_code == 200
    assert "image/svg+xml" in response.headers["content-type"]
    assert int(response.headers["content-length"]) > 0


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


def test_health_panel_shows_core_runtime_and_optional_groups():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert "核心可用" in html
    assert "healthSnapshot.groups" in html
    assert "['core', 'runtime', 'optional']" in html
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


def test_phase15_app_shell_has_notes_and_settings():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert 'id="nav-chat"' in html
    assert 'id="nav-library"' in html
    assert 'id="nav-notes"' in html
    assert 'id="nav-settings"' in html
    assert 'id="view-notes"' in html
    assert 'id="view-settings"' in html
    assert "createNoteFromNotesView" in html
    assert "importUrlFromNotesView" in html
    assert "refreshSettings" in html
    assert "settings-sources-list" in html
    assert "settings-model-list" in html


def test_phase16_settings_exposes_runtime_without_developer_guidance():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert "settings-insights-list" in html
    assert "renderSettingsInsights" in html
    assert "settingsSafeDetail" in html
    assert "'runtime'" in html
    assert "状态提示" in html
    assert "使用偏好" in html
    assert "恢复建议" not in html
    assert "维护命令" not in html
    assert "复制命令" not in html


def test_phase17_notes_exposes_knowledge_outputs():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert 'id="knowledge-output-panel"' in html
    assert 'id="knowledge-submit-btn"' in html
    assert "Knowledge Outputs" in html
    assert "/api/knowledge-output" in html
    assert "openKnowledgeFromSelectedFiles" in html
    assert "learning_cards" in html
    assert "project_brief" in html


def test_phase18_frontend_uses_local_tailwind_build():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert 'href="/styles.css"' in html
    assert "cdn.tailwindcss.com" not in html
    assert 'id="tailwind-config"' not in html
    assert Path("frontend/styles.css").exists()
    assert Path("tailwind.config.js").exists()
    assert "build:css" in Path("package.json").read_text(encoding="utf-8")
    assert ".py,.rs,.ts,.css,.sh" in html
    assert "'.py','.rs','.ts','.css','.sh'" in html


def test_phase20_frontend_uses_local_icons():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert "fonts.googleapis.com" not in html
    assert "fonts.gstatic.com" not in html
    assert "Material+Symbols" not in html
    assert "LOCAL_ICON_PATHS" in html
    assert "function setIcon" in html
    assert "initLocalIcons();" in html
    assert "icon.textContent =" not in html


def test_phase28_settings_hides_local_install_and_recovery_commands():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    for forbidden in [
        "python main.py install-local",
        "python main.py restore-drill",
        "python main.py repair-ids --dry-run",
        "copyLiteralCommand",
        "dry-run",
        "browser-acceptance",
    ]:
        assert forbidden not in html


def test_phase25_browser_acceptance_command_is_documented():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert Path("scripts/run_browser_acceptance.py").exists()
    assert "python main.py browser-acceptance" in readme


def test_phase26_ui_redesign_shell_has_real_context_panels():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert 'class="app-topbar flex-shrink-0"' in html
    assert 'id="global-search-input"' in html
    assert 'id="chat-context-sources"' in html
    assert 'id="library-context-panel"' in html
    assert 'id="settings-insights-list"' in html
    assert "renderLibraryContext" in html
    assert "renderChatContextSources" in html
    assert "handleGlobalSearchKey" in html
    assert "fonts.googleapis.com" not in html


def test_phase18_release_docs_are_linked_from_readme():
    readme = Path("README.md").read_text(encoding="utf-8")

    for path in [
        "LICENSE",
        "CHANGELOG.md",
        "docs/LOCAL_DEPLOYMENT.md",
        "docs/phase18-final-acceptance.md",
        "docs/phase18-handoff.md",
        "docs/phase26-chat-desktop.png",
        "docs/phase26-library-desktop.png",
    ]:
        assert Path(path).exists()

    assert "docs/LOCAL_DEPLOYMENT.md" in readme
    assert "docs/phase18-final-acceptance.md" in readme
    assert "docs/phase26-chat-desktop.png" in readme
    assert "MIT. See `LICENSE`." in readme
    assert "MIT。见 `LICENSE`。" in readme


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

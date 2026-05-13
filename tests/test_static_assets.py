from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app as api_app


def frontend_source_text() -> str:
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    scripts = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(Path("frontend/js").glob("*.js"))
    )
    return f"{html}\n{scripts}"


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
    html = frontend_source_text()

    assert 'id="confirm-modal"' in html
    assert "function showConfirmDialog" in html
    assert "function closeConfirmDialog" in html
    assert "清空所有历史记录？" in html
    assert "删除这个对话？" in html
    assert "if (!confirm(" not in html
    assert "text-on-error bg-error" in html
    assert "bg-error/5 text-error hover:bg-error" in html


def test_health_panel_shows_core_runtime_and_optional_groups():
    html = frontend_source_text()

    assert "核心可用" in html
    assert "healthSnapshot.groups" in html
    assert "['core', 'runtime', 'optional']" in html
    assert "group.label" in html
    assert "optional_unavailable" in html
    assert "未安装" in html
    assert "renderHealthGroup" in html


def test_files_actions_have_clear_feedback():
    html = frontend_source_text()

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
    html = frontend_source_text()

    assert "library-collection-filter" in html
    assert "library-tag-filter" in html
    assert "library-favorite-filter" in html
    assert "batch-collection-input" in html
    assert "batch-tags-input" in html
    assert "favoriteSelected" in html
    assert "applyBatchMetadata" in html
    assert "rebuildSelected" in html
    assert "refreshFilesRequestId" in html
    assert "requestId !== refreshFilesRequestId" in html


def test_phase13_import_and_note_actions_are_visible():
    html = frontend_source_text()

    assert "导入网页" in html
    assert "新建笔记" in html
    assert "library-workflow-panel" in html
    assert "runLibraryWorkflow" in html
    assert "/api/import/url" in html
    assert "/api/notes/from-answer" in html
    assert "保存为笔记" in html


def test_phase14_chat_scope_controls_are_visible():
    html = frontend_source_text()

    assert "query-scope-controls" in html
    assert "全部知识库" in html
    assert "指定集合" in html
    assert "指定文件" in html
    assert "全文模式" in html
    assert "buildQueryScopePayload" in html
    assert "scope_mode" in html
    assert "full_text" in html


def test_phase37_related_notes_are_visible():
    html = frontend_source_text()

    assert "相关笔记" in html
    assert "chat-related-notes" in html
    assert "chat-related-count" in html
    assert "related_notes" in html
    assert "renderRelatedNotes" in html
    assert "relatedNotesMarkup" in html


def test_phase15_app_shell_has_notes_and_settings():
    html = frontend_source_text()

    assert 'id="nav-chat"' in html
    assert 'id="nav-library"' in html
    assert 'id="nav-source"' in html
    assert 'id="nav-notes"' in html
    assert 'id="nav-settings"' in html
    assert 'id="view-source"' in html
    assert 'id="view-notes"' in html
    assert 'id="view-settings"' in html
    assert "createNoteFromNotesView" in html
    assert "importUrlFromNotesView" in html
    assert "refreshSettings" in html
    assert "settings-sources-list" in html
    assert "settings-model-list" in html


def test_phase16_settings_exposes_runtime_without_developer_guidance():
    html = frontend_source_text()

    assert "settings-insights-list" in html
    assert "renderSettingsInsights" in html
    assert "settingsSafeDetail" in html
    assert "'runtime'" in html
    assert "状态提示" in html
    assert "使用偏好" in html
    assert "恢复建议" not in html
    assert "维护命令" not in html
    assert "复制命令" not in html


def test_settings_storage_uses_real_local_usage():
    html = frontend_source_text()

    assert "存储使用" in html
    assert "/api/storage/usage" in html
    assert 'id="sidebar-storage-meter"' in html
    assert "storageUsage" in html
    assert "本地存储" in html
    assert "模型缓存" in html


def test_phase31_supplemental_polish_stays_in_place():
    html = frontend_source_text()

    assert 'id="chat-context-source-metric"' in html
    assert "sourcePreviewListTitle" in html
    assert 'id="health-details" class="grid grid-cols-2 xl:grid-cols-3 gap-2' in html
    assert 'id="notes-content-input" rows="10"' in html
    assert 'id="knowledge-source-input" rows="2"' in html
    assert "border-outline-variant/35 rounded-lg px-3 py-2" in html


def test_phase32_queue_polling_refreshes_only_on_state_change():
    html = frontend_source_text()
    start = html.index("async function pollQueueOnce")
    end = html.index("async function triggerIngest")
    poll_body = html[start:end]

    assert "function maybeRefreshFilesForQueue" in html
    assert "function queueFilesRefreshKey" in html
    assert "lastQueueFilesRefreshKey" in html
    assert "maybeRefreshFilesForQueue(q)" in poll_body
    assert "refreshFiles();" not in poll_body


def test_phase32_chat_errors_are_user_facing():
    html = frontend_source_text()

    assert "function userFacingErrorMessage" in html
    assert "这次回答耗时太久" in html
    assert "暂时连接不上本地服务" in html
    assert "本地服务还在准备" in html
    assert "错误: ${e.message}" not in html
    assert "JSON.parse(eventData))}</span>" not in html


def test_phase33_frontend_scripts_are_split_by_domain():
    index = Path("frontend/index.html").read_text(encoding="utf-8")
    expected_scripts = [
        "state.js",
        "icons.js",
        "shared-ui.js",
        "i18n.js",
        "app-shell.js",
        "settings.js",
        "chat.js",
        "notes.js",
        "chat-stream.js",
        "source-preview.js",
        "library.js",
        "history.js",
        "queue-upload.js",
        "pwa.js",
    ]

    assert "<script>\nconst API" not in index
    for script in expected_scripts:
        assert f'src="/js/{script}"' in index
        assert Path("frontend/js", script).exists()

    state = Path("frontend/js/state.js").read_text(encoding="utf-8")
    assert "window.DocFlowState" in state
    assert "chat:" in state
    assert "library:" in state
    assert "notes:" in state
    assert "settings:" in state
    assert "Object.defineProperties(window" in state
    assert "locale:" in state

    tailwind_config = Path("tailwind.config.js").read_text(encoding="utf-8")
    assert "./frontend/js/**/*.js" in tailwind_config


def test_phase63_frontend_has_i18n_accessibility_and_pwa_shell():
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    i18n = Path("frontend/js/i18n.js").read_text(encoding="utf-8")
    app_shell = Path("frontend/js/app-shell.js").read_text(encoding="utf-8")
    pwa = Path("frontend/js/pwa.js").read_text(encoding="utf-8")
    sw = Path("frontend/sw.js").read_text(encoding="utf-8")
    manifest = Path("frontend/manifest.webmanifest").read_text(encoding="utf-8")

    assert 'href="/manifest.webmanifest"' in html
    assert 'class="skip-link"' in html
    assert 'data-i18n="nav.chat"' in html
    assert 'data-i18n-placeholder="search.placeholder"' in html
    assert 'id="locale-toggle-btn"' in html
    assert 'aria-current="page"' in html
    assert 'role="region" aria-labelledby="chat-title"' in html
    assert "DOCFLOW_I18N" in i18n
    assert "'zh-CN'" in i18n
    assert "en:" in i18n
    assert "function toggleLocale" in i18n
    assert "setAttribute('aria-current', 'page')" in app_shell
    assert "serviceWorker" in pwa
    assert "caches.open(DOCFLOW_CACHE)" in sw
    assert '"display": "standalone"' in manifest


def test_phase17_notes_exposes_knowledge_outputs():
    html = frontend_source_text()

    assert 'id="knowledge-output-panel"' in html
    assert 'id="knowledge-submit-btn"' in html
    assert "知识产物" in html
    assert "knowledgeOutputCollectionName" in html
    assert "/api/knowledge-output" in html
    assert "openKnowledgeFromSelectedFiles" in html
    assert "learning_cards" in html
    assert "project_brief" in html


def test_phase56_user_ui_uses_product_language_for_models_and_collections():
    html = frontend_source_text()

    assert "本地回答模型" in html
    assert "增强回答模型" in html
    assert "friendlyModelLabel" in html
    assert "collectionLabel" in html
    assert "可粘贴资料，也可从资料库选中文件带入" in html
    assert "仅当前文件夹" in html
    assert "source.path || ''" not in html
    for forbidden in [
        "Embedding（嵌入）",
        "Reranker（重排）",
        "LLM（大语言模型）",
        "Library 中",
        "Source Note -",
    ]:
        assert forbidden not in html


def test_phase57_library_exposes_real_knowledge_views():
    html = frontend_source_text()

    assert "/api/knowledge/overview" in html
    assert "library-knowledge-overview" in html
    assert "主题视图" in html
    assert "相似资料" in html
    assert "知识卡片" in html
    assert "knowledgeOverviewMarkup" in html


def test_phase58_public_maintenance_surface_is_complete():
    release_doc = Path("docs/release.md").read_text(encoding="utf-8")
    pr_template = Path(".github/PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
    status_doc = Path("docs/status.md").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert Path(".github/dependabot.yml").exists()
    assert Path(".github/workflows/codeql.yml").exists()
    assert Path(".github/ISSUE_TEMPLATE/bug.md").exists()
    assert Path(".github/ISSUE_TEMPLATE/feature.md").exists()
    assert Path(".github/ISSUE_TEMPLATE/question.md").exists()
    assert Path(".github/ISSUE_TEMPLATE/config.yml").exists()
    assert Path("docs/release.md").exists()
    assert "version = \"0.58.0\"" in pyproject
    assert "scripts/run_ci.sh" in release_doc
    assert "docflow doctor --offline" in release_doc
    assert "Known limitations" in release_doc
    assert "Privacy impact reviewed" in pr_template
    assert "Status Update Rule" in status_doc


def test_phase59_public_claims_are_precise():
    readme = Path("README.md").read_text(encoding="utf-8")
    readme_zh = Path("README.zh-CN.md").read_text(encoding="utf-8")
    status_doc = Path("docs/status.md").read_text(encoding="utf-8")
    evaluation_doc = Path("docs/evaluation.md").read_text(encoding="utf-8")
    main_source = Path("main.py").read_text(encoding="utf-8")

    combined_public_text = "\n".join([readme, readme_zh, status_doc, evaluation_doc, main_source])
    for forbidden in [
        "100% offline",
        "Recall@5 = 1.0",
        "Phase55",
        "Phase 55",
        "9 分成熟度",
        "docflow-phase22",
    ]:
        assert forbidden not in combined_public_text

    assert "source-filtered" in readme
    assert "不等同于大规模公开 benchmark" in readme_zh
    assert "do not present it as an external benchmark" in status_doc
    assert "internal planning aid only" in evaluation_doc


def test_phase18_frontend_uses_local_tailwind_build():
    html = frontend_source_text()

    assert 'href="/styles.css"' in html
    assert "cdn.tailwindcss.com" not in html
    assert 'id="tailwind-config"' not in html
    assert Path("frontend/styles.css").exists()
    assert Path("tailwind.config.js").exists()
    assert "build:css" in Path("package.json").read_text(encoding="utf-8")
    assert ".py,.rs,.ts,.css,.sh" in html
    assert "'.py','.rs','.ts','.css','.sh'" in html


def test_phase20_frontend_uses_local_icons():
    html = frontend_source_text()

    assert "fonts.googleapis.com" not in html
    assert "fonts.gstatic.com" not in html
    assert "Material+Symbols" not in html
    assert "LOCAL_ICON_PATHS" in html
    assert "function setIcon" in html
    assert "initLocalIcons();" in html
    assert "icon.textContent =" not in html


def test_runtime_http_calls_use_central_network_module():
    offenders = []
    forbidden_tokens = [
        "import httpx",
        "from httpx",
        "import requests",
        "from requests",
        "urllib.request",
        "urllib3",
    ]
    for path in Path("src").rglob("*.py"):
        if path == Path("src/net.py") or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        matches = [token for token in forbidden_tokens if token in text]
        if matches:
            offenders.append(f"{path}: {', '.join(matches)}")

    assert offenders == []


def test_settings_warns_plainly_when_cloud_answers_are_enabled():
    html = frontend_source_text()

    assert "云端回答已启用" in html
    assert "提问内容会发送到你配置的外部模型服务" in html
    assert "network_mode" in html


def test_chat_citations_render_chunk_identity():
    text = Path("frontend/js/chat-stream.js").read_text(encoding="utf-8")

    assert "data-chunk-id" in text
    assert "citation.chunk_id" in text
    assert "char_start" in text
    assert "char_end" in text


def test_source_preview_highlights_exact_citation_range():
    text = Path("frontend/js/source-preview.js").read_text(encoding="utf-8")

    assert "qdrantIdFromChunkId" in text
    assert "highlightRangeApplies" in text
    assert 'data-citation-hit="true"' in text
    assert "已定位到回答引用的原文范围" in text


def test_phase28_settings_hides_local_install_and_recovery_commands():
    html = frontend_source_text()

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
    assert "docflow browser-acceptance" in readme


def test_phase26_ui_redesign_shell_has_real_context_panels():
    html = frontend_source_text()

    assert 'class="app-topbar flex-shrink-0"' in html
    assert 'id="global-search-input"' in html
    assert 'id="chat-context-sources"' in html
    assert 'id="library-context-panel"' in html
    assert 'id="source-result-list"' in html
    assert 'id="source-document-viewer"' in html
    assert "openSourcePreview" in html
    assert 'id="settings-insights-list"' in html
    assert "renderLibraryContext" in html
    assert "renderChatContextSources" in html
    assert "handleGlobalSearchKey" in html
    assert "fonts.googleapis.com" not in html


def test_phase18_release_docs_are_linked_from_readme():
    for path in [
        "LICENSE",
        "CHANGELOG.md",
        "README.md",
        "README.zh-CN.md",
        "docs/features.md",
        "docs/architecture.md",
        "docs/privacy.md",
        "docs/cli.md",
        "docs/development.md",
        "docs/evaluation.md",
        "docs/release.md",
        "docs/assets/chat.png",
        "docs/assets/library.png",
        "docs/assets/notes.png",
        "docs/assets/settings.png",
    ]:
        assert Path(path).exists()

    readme_en = Path("README.md").read_text(encoding="utf-8")
    readme_zh = Path("README.zh-CN.md").read_text(encoding="utf-8")
    readme = readme_en + "\n" + readme_zh

    # Both READMEs should cross-link to each other so users can switch languages.
    assert "README.zh-CN.md" in readme_en
    assert "README.md" in readme_zh

    # Release documentation links should be reachable from at least one README.
    for link in [
        "docs/features.md",
        "docs/architecture.md",
        "docs/privacy.md",
        "docs/cli.md",
        "docs/development.md",
        "docs/evaluation.md",
        "docs/release.md",
        "docs/assets/chat.png",
    ]:
        assert link in readme

    # License wording lives in the English README; the Chinese README keeps the
    # localized variant so neither language version regresses after the split.
    assert "MIT." in readme_en
    assert "MIT。" in readme_zh


def test_clickable_icon_actions_are_labeled_and_keyboard_accessible():
    html = frontend_source_text()

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

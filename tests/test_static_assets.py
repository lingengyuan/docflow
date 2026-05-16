from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app as api_app


def frontend_source_text() -> str:
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    partial_paths = sorted(Path("frontend/partials").glob("*.html"))
    partials = "\n".join(
        path.read_text(encoding="utf-8") for path in partial_paths
    )
    scripts = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(Path("frontend/js").rglob("*.js"))
    )
    return f"{html}\n{partials}\n{scripts}"


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


def test_phase95_frontend_errors_do_not_expose_raw_details():
    html = frontend_source_text()
    raw_error_patterns = [
        "throw new Error(await r.text())",
        "失败：${e.message}",
        "失败: ${e.message}",
        "读取失败：${escHtml(e.message)}",
        "加载失败：${e.message}",
        "加载失败: ${e.message}",
        "保存失败：${escHtml(e.message)}",
        "保存笔记失败：${e.message}",
        "alert(`批量收藏失败: ${e.message}`)",
        "alert(`批量更新失败: ${e.message}`)",
        "alert(`整理失败: ${e.message}`)",
        "alert(`摘要生成失败: ${e.message}`)",
        "renderLLMStatus({ state: 'error', message: e.message })",
        "error: e.message",
    ]

    for pattern in raw_error_patterns:
        assert pattern not in html
    assert html.count("responseUserMessage(") >= 12
    assert html.count("userFacingErrorMessage(") >= 20


def test_phase33_frontend_scripts_are_split_by_domain():
    index = Path("frontend/index.html").read_text(encoding="utf-8")
    bootstrap = Path("frontend/js/bootstrap.js").read_text(encoding="utf-8")
    expected_scripts = [
        "state.js",
        "icons.js",
        "shared-ui.js",
        "i18n.js",
        "theme.js",
        "app-shell.js",
        "settings.js",
        "settings-models.js",
        "settings-data.js",
        "chat.js",
        "notes.js",
        "notes-review.js",
        "chat-evidence.js",
        "chat-stream.js",
        "chat-actions.js",
        "source-preview.js",
        "source-preview-actions.js",
        "library.js",
        "library-render.js",
        "library-knowledge.js",
        "library-actions.js",
        "history.js",
        "queue-upload.js",
        "settings-bootstrap.js",
    ]

    assert "<script>\nconst API" not in index
    for script in expected_scripts:
        assert f"'/js/{script}'" in bootstrap
        assert Path("frontend/js", script).exists()

    state = Path("frontend/js/state.js").read_text(encoding="utf-8")
    assert "window.DocFlowState" in state
    assert "chat:" in state
    assert "library:" in state
    assert "notes:" in state
    assert "settings:" in state
    assert "Object.defineProperties(window" in state
    assert "locale:" in state


def test_phase69_frontend_shell_is_split_and_testable():
    index = Path("frontend/index.html")
    assert len(index.read_text(encoding="utf-8").splitlines()) < 300
    assert Path("frontend/partials/app.html").exists()
    assert Path("frontend/app.css").exists()
    assert Path("frontend/src/stream-parser.ts").exists()
    assert Path("frontend/tests/stream-parser.test.ts").exists()
    assert "consumeSseBuffer" in Path("frontend/js/generated/stream-parser.js").read_text(
        encoding="utf-8"
    )

    limits = {
        "frontend/js/chat-stream.js": 350,
        "frontend/js/chat-actions.js": 350,
        "frontend/js/library.js": 350,
        "frontend/js/library-render.js": 350,
        "frontend/js/library-knowledge.js": 350,
        "frontend/js/library-actions.js": 350,
        "frontend/js/settings.js": 350,
        "frontend/js/settings-models.js": 350,
        "frontend/js/settings-data.js": 350,
        "frontend/js/source-preview.js": 350,
        "frontend/js/source-preview-actions.js": 350,
    }
    for path, limit in limits.items():
        assert len(Path(path).read_text(encoding="utf-8").splitlines()) < limit

    tailwind_config = Path("tailwind.config.js").read_text(encoding="utf-8")
    assert "./frontend/js/**/*.js" in tailwind_config


def test_phase70_theme_tokens_focus_and_live_regions_exist():
    html = frontend_source_text()
    app_css = Path("frontend/app.css").read_text(encoding="utf-8")
    tailwind_config = Path("tailwind.config.js").read_text(encoding="utf-8")

    assert ':root[data-theme="dark"]' in app_css
    assert "--color-primary:" in app_css
    assert 'button:focus-visible' in app_css
    assert "colorVar" in tailwind_config
    assert 'id="theme-toggle-btn"' in html
    assert "function toggleTheme" in html
    assert 'aria-live="polite"' in html
    assert "desktop_viewports_stay_usable" in Path("src/quality/browser_acceptance.py").read_text(
        encoding="utf-8"
    )


def test_phase71_knowledge_graph_is_visible_and_data_backed():
    html = frontend_source_text()
    service = Path("src/api/services/knowledge_service.py").read_text(encoding="utf-8")

    assert "关系图谱" in html
    assert "knowledgeGraphMarkup" in html
    assert "knowledge_graph" in service
    assert "topic_file" in service
    assert "backlink" in service


def test_phase72_active_review_is_visible_and_data_backed():
    html = frontend_source_text()
    service = Path("src/api/services/knowledge_review.py").read_text(encoding="utf-8")
    routes = Path("src/api/routes/knowledge.py").read_text(encoding="utf-8")

    assert "主动回顾" in html
    assert 'id="knowledge-review-panel"' in html
    assert "renderKnowledgeReview" in html
    assert "/api/knowledge/review" in html
    assert "review_queue" in service
    assert "topic_activity" in service
    assert "citation_counts" in service
    assert "/api/knowledge/review" in routes


def test_phase73_trusted_answer_evidence_is_visible_and_data_backed():
    html = frontend_source_text()
    evidence_service = Path("src/api/services/evidence_service.py").read_text(encoding="utf-8")
    schemas = Path("src/api/schemas.py").read_text(encoding="utf-8")
    query_stream_handlers = Path("src/api/handlers/query_stream_handlers.py").read_text(
        encoding="utf-8"
    )

    assert "citationEvidencePill" in html
    assert "renderEvidenceSummary" in html
    assert 'id="stream-evidence"' in html
    assert "强来源" in evidence_service
    assert "存在冲突" in evidence_service
    assert "source_age_days" in evidence_service
    assert "detect_conflicts" in evidence_service
    assert "evidence: dict" in schemas
    assert 'q.put(("evidence"' in query_stream_handlers


def test_phase74_public_eval_is_separate_from_internal_regression():
    readme = Path("README.md").read_text(encoding="utf-8")
    readme_zh = Path("README.zh-CN.md").read_text(encoding="utf-8")
    evaluation_doc = Path("docs/evaluation.md").read_text(encoding="utf-8")
    status_doc = Path("docs/status.md").read_text(encoding="utf-8")
    cli_doc = Path("docs/cli.md").read_text(encoding="utf-8")
    main_source = Path("main.py").read_text(encoding="utf-8")

    assert Path("scripts/run_public_eval.py").exists()
    assert Path("eval/public_retrieval_v1.jsonl").exists()
    assert Path("eval/public_corpus/README.md").exists()
    assert "docflow dev eval public" in cli_doc
    assert 'args[0] == "public"' in main_source
    assert "public-domain regression" in readme
    assert "公开可复现检索评估" in readme_zh
    assert "Public Reproducible Retrieval Benchmark" in evaluation_doc
    assert "Internal Source-Filtered Regression" in evaluation_doc
    assert "not a BEIR, MTEB, or C-MTEB score" in evaluation_doc
    assert "Public eval:" in status_doc


def test_phase88_performance_smoke_is_documented_and_in_ci():
    evaluation_doc = Path("docs/evaluation.md").read_text(encoding="utf-8")
    cli_doc = Path("docs/cli.md").read_text(encoding="utf-8")
    main_source = Path("main.py").read_text(encoding="utf-8")
    ci_script = Path("scripts/run_ci.sh").read_text(encoding="utf-8")

    assert Path("scripts/run_performance_smoke.py").exists()
    assert "docflow dev eval performance" in evaluation_doc
    assert "docflow dev eval performance" in cli_doc
    assert "Performance Smoke" in evaluation_doc
    assert 'args[0] == "performance"' in main_source
    assert "run_performance_smoke.py --json" in ci_script


def test_phase97_github_ci_runs_release_and_eval_gates():
    ci_workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    eval_workflow = Path(".github/workflows/evaluation.yml").read_text(encoding="utf-8")
    release_check = Path("scripts/run_release_surface_check.py").read_text(encoding="utf-8")
    evaluation_doc = Path("docs/evaluation.md").read_text(encoding="utf-8")
    release_doc = Path("docs/release.md").read_text(encoding="utf-8")
    status_doc = Path("docs/status.md").read_text(encoding="utf-8")

    for snippet in [
        "scripts/run_performance_smoke.py --json",
        "main.py dev eval parsing --json",
        "scripts/run_release_surface_check.py",
        "scripts/package_smoke.py",
    ]:
        assert snippet in ci_workflow

    for snippet in [
        "workflow_dispatch",
        "schedule:",
        "qdrant/qdrant",
        "allow_model_download: true",
        "main.py dev eval public",
        "--no-rerank",
        "actions/upload-artifact",
    ]:
        assert snippet in eval_workflow
        assert snippet in release_check or snippet in eval_workflow

    assert ".github/workflows/evaluation.yml" in release_check
    assert "GitHub CI" in evaluation_doc
    assert "weekly evaluation workflow" in evaluation_doc
    assert "release surface" in release_doc
    assert "scheduled evaluation workflow" in release_doc
    assert "GitHub CI now runs" in status_doc


def test_phase99_external_benchmark_claims_are_explicitly_unclaimed():
    catalog = Path("eval/external_benchmarks.json").read_text(encoding="utf-8")
    evaluation_doc = Path("docs/evaluation.md").read_text(encoding="utf-8")
    status_doc = Path("docs/status.md").read_text(encoding="utf-8")
    main_source = Path("main.py").read_text(encoding="utf-8")
    run_ci = Path("scripts/run_ci.sh").read_text(encoding="utf-8")
    ci_workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    release_check = Path("scripts/run_release_surface_check.py").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert Path("scripts/run_external_benchmark_status.py").exists()
    assert '"schema": "docflow.external_benchmark_catalog.v1"' in catalog
    for benchmark in ["BEIR", "MTEB", "C-MTEB"]:
        assert benchmark in catalog
        assert benchmark in evaluation_doc
    assert '"docflow_status": "not_run"' in catalog
    assert "No external benchmark score has been archived yet" in evaluation_doc
    assert "No external benchmark score has been archived yet" in status_doc
    assert "External benchmark catalog: valid; 0 archived external scores." in status_doc
    assert "docflow dev eval external --json" in evaluation_doc
    assert 'args[0] == "external"' in main_source
    assert "run_external_benchmark_status.py --json" in run_ci
    assert "run_external_benchmark_status.py --json" in ci_workflow
    assert "eval/external_benchmarks.json" in pyproject
    assert "No external benchmark score has been archived yet" in release_check


def test_phase75_release_install_surface_is_documented():
    readme = Path("README.md").read_text(encoding="utf-8")
    readme_zh = Path("README.zh-CN.md").read_text(encoding="utf-8")
    development_doc = Path("docs/development.md").read_text(encoding="utf-8")
    architecture_doc = Path("docs/architecture.md").read_text(encoding="utf-8")
    release_doc = Path("docs/release.md").read_text(encoding="utf-8")
    status_doc = Path("docs/status.md").read_text(encoding="utf-8")

    assert Path("docker-compose.image.yml").exists()
    assert Path(".github/workflows/docker-image.yml").exists()
    assert Path(".github/workflows/python-package.yml").exists()
    assert "ghcr.io/lingengyuan/docflow" in Path("docker-compose.image.yml").read_text(
        encoding="utf-8"
    )
    assert "ghcr.io/lingengyuan/docflow" in readme
    assert "ghcr.io/lingengyuan/docflow" in readme_zh
    assert "Docker image release" in development_doc
    assert "DocFlow is not on PyPI yet" in development_doc
    assert "First-run storage expectations" in development_doc
    assert "Failure Modes" in architecture_doc
    assert "Upgrade Boundaries" in architecture_doc
    assert "Tagged releases build" in release_doc
    assert "GHCR Docker images" in release_doc
    assert "not published to PyPI yet" in status_doc


def test_phase78_package_artifacts_include_runtime_resources():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    run_ci = Path("scripts/run_ci.sh").read_text(encoding="utf-8")
    package_smoke = Path("scripts/package_smoke.py").read_text(encoding="utf-8")
    app_impl = Path("src/api/app_impl.py").read_text(encoding="utf-8")
    app_static = Path("src/api/app_static.py").read_text(encoding="utf-8")
    startup = Path("src/maintenance/startup.py").read_text(encoding="utf-8")
    development_doc = Path("docs/development.md").read_text(encoding="utf-8")
    release_doc = Path("docs/release.md").read_text(encoding="utf-8")

    assert Path("src/resources.py").exists()
    assert '"share/docflow/frontend"' in pyproject
    assert '"frontend/partials/app.html"' in pyproject
    assert '"config.example.yaml"' in pyproject
    assert '"docs/status.md"' in pyproject
    assert '"eval/public_retrieval_v1.jsonl"' in pyproject
    assert '"eval/public_corpus/wizard_oz_excerpt.txt"' in pyproject
    assert "scripts/package_smoke.py" in run_ci
    assert "resource_path(\"frontend\")" in app_static
    assert 'os.getenv("DOCFLOW_CONFIG", "config.yaml")' in app_impl
    assert 'resource_path("config.example.yaml")' in startup
    assert "installed-wheel smoke test" in development_doc
    assert "scripts/package_smoke.py" in release_doc
    assert "public_cases" in package_smoke
    assert '"install"' in package_smoke
    assert '"--target"' in package_smoke


def test_phase80_desktop_ui_has_labels_focus_and_status_contract():
    html = frontend_source_text()
    app_html = Path("frontend/partials/app.html").read_text(encoding="utf-8")
    app_css = Path("frontend/app.css").read_text(encoding="utf-8")
    browser_acceptance = Path("src/quality/browser_acceptance.py").read_text(encoding="utf-8")
    browser_acceptance_a11y = Path("src/quality/browser_acceptance_a11y.py").read_text(
        encoding="utf-8"
    )

    labeled_controls = [
        "query-scope-mode",
        "input",
        "workflow-title-input",
        "workflow-url-input",
        "workflow-collection-input",
        "workflow-tags-input",
        "workflow-content-input",
        "library-status-filter",
        "library-collection-filter",
        "library-tag-filter",
        "batch-collection-input",
        "batch-tags-input",
        "notes-title-input",
        "notes-collection-input",
        "notes-tags-input",
        "notes-content-input",
        "notes-url-input",
        "notes-url-title-input",
        "notes-url-collection-input",
        "notes-url-tags-input",
        "knowledge-title-input",
        "knowledge-collection-input",
        "knowledge-tags-input",
        "knowledge-source-input",
        "knowledge-model-select",
    ]
    for control_id in labeled_controls:
        assert f'for="{control_id}"' in app_html

    for status_id in [
        "query-scope-status",
        "queue-banner",
        "workflow-status",
        "notes-status",
        "notes-url-status",
        "knowledge-status",
        "settings-insights-list",
        "settings-storage-list",
        "chat-context-queue",
    ]:
        assert f'id="{status_id}"' in html

    assert "desktop_status_messages_are_announced" in browser_acceptance
    assert "hasAssociatedLabel" in browser_acceptance_a11y
    assert "document.querySelector(`label[for=\"" in browser_acceptance_a11y
    assert "notes-url-status" in browser_acceptance_a11y
    assert "knowledge-status" in browser_acceptance_a11y
    assert "button:focus-visible" in app_css
    assert "outline: 2px solid rgb(var(--color-primary))" in app_css


def test_phase81_degraded_answer_quality_is_visible():
    html = frontend_source_text()
    engine = Path("src/query/engine.py").read_text(encoding="utf-8")
    quality = Path("src/query/answer_quality.py").read_text(encoding="utf-8")
    schemas = Path("src/api/schemas.py").read_text(encoding="utf-8")
    query_handlers = Path("src/api/handlers/query_handlers.py").read_text(encoding="utf-8")
    query_stream_handlers = Path("src/api/handlers/query_stream_handlers.py").read_text(
        encoding="utf-8"
    )

    assert "answerQualityMarkup" in html
    assert "renderAnswerQuality" in html
    assert 'id="stream-quality"' in html
    assert "data-answer-quality" in html
    for status in [
        "grounded",
        "insufficient_evidence",
        "local_model_unavailable",
        "vector_store_unavailable",
        "snippet_fallback",
    ]:
        assert status in quality + html

    assert "quality: dict" in schemas
    assert 'q.put(("quality"' in query_stream_handlers
    assert '"quality"' in query_handlers
    assert "answer_quality_states_render" in Path("src/quality/browser_acceptance.py").read_text(
        encoding="utf-8"
    )
    assert "insufficient_evidence_quality" in engine
    assert "local_model_unavailable_quality" in engine
    assert "retrieval_quality_from_chunks" in engine


def test_phase63_frontend_has_i18n_and_accessibility_shell():
    html = frontend_source_text()
    i18n = Path("frontend/js/i18n.js").read_text(encoding="utf-8")
    app_shell = Path("frontend/js/app-shell.js").read_text(encoding="utf-8")

    assert 'href="/manifest.webmanifest"' not in html
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
    assert not Path("frontend/js/pwa.js").exists()
    assert not Path("frontend/sw.js").exists()
    assert not Path("frontend/manifest.webmanifest").exists()


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
    assert "可连接资料" in html
    assert "knowledgeRelationshipOpportunityMarkup" in html
    assert "relationship_opportunities" in html


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


def test_streaming_answer_hides_internal_citation_markers_until_finalized():
    text = Path("frontend/js/chat-actions.js").read_text(encoding="utf-8")

    assert "function streamDisplayAnswer" in text
    assert "streamDisplayAnswer(answerText)" in text
    assert r"\[\[cite:" in text


def test_notes_review_surfaces_answer_source_relationships():
    text = Path("frontend/js/notes-review.js").read_text(encoding="utf-8")

    assert "relationship_timeline" in text
    assert "knowledgeRelationshipMarkup" in text
    assert "account_tree" in text
    assert "来源：" in text


def test_phase94_knowledge_depth_surfaces_usage_loops():
    service = Path("src/api/services/knowledge_depth.py").read_text(encoding="utf-8")
    review_service = Path("src/api/services/knowledge_review.py").read_text(encoding="utf-8")
    ui = Path("frontend/js/notes-review.js").read_text(encoding="utf-8")

    assert "KnowledgeDepthService" in service
    assert "source_trails" in service
    assert "coverage_gaps" in service
    assert "concepts" in service
    assert "knowledge_depth" in review_service
    assert "knowledgeSourceTrailMarkup" in ui
    assert "knowledgeCoverageGapMarkup" in ui
    assert "knowledgeConceptMarkup" in ui


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
    cli_doc = Path("docs/cli.md").read_text(encoding="utf-8")

    assert Path("scripts/run_browser_acceptance.py").exists()
    assert "docflow dev browser-acceptance" in cli_doc


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


def test_phase64_feedback_and_backlink_ui_are_visible():
    html = frontend_source_text()

    assert "/api/answers/feedback" in html
    assert "feedbackControlsMarkup" in html
    assert "反向关联" in html
    assert "回答反馈" in html
    assert "引用来源" in html


def test_phase66_public_surface_hides_internal_planning_artifacts():
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
    status_doc = Path("docs/status.md").read_text(encoding="utf-8")

    assert not Path("plans").exists()
    assert "no longer ships a public `plans/` directory" in status_doc
    for forbidden in [
        "Phase 29 handoff",
        "phase25-browser-acceptance",
        "python main.py browser-acceptance",
        "python main.py repair-ids",
        "python main.py restore-drill",
    ]:
        assert forbidden not in changelog


def test_phase103_cli_surface_is_grouped_and_audited():
    main_source = Path("main.py").read_text(encoding="utf-8")
    cli_doc = Path("docs/cli.md").read_text(encoding="utf-8")
    run_ci = Path("scripts/run_ci.sh").read_text(encoding="utf-8")
    ci_workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    public_text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in [
            "README.md",
            "README.zh-CN.md",
            "CONTRIBUTING.md",
            ".github/PULL_REQUEST_TEMPLATE.md",
            "docs/cli.md",
            "docs/evaluation.md",
            "docs/release.md",
        ]
    )

    assert Path("scripts/run_dead_code_audit.py").exists()
    assert "scripts/run_dead_code_audit.py --json" in run_ci
    assert "scripts/run_dead_code_audit.py --json" in ci_workflow
    assert "docflow admin platform" in cli_doc
    assert "docflow admin check" in cli_doc
    assert "docflow admin rebuild --dry-run" in cli_doc
    assert "docflow dev eval public" in cli_doc
    assert "docflow dev browser-acceptance" in cli_doc
    assert "docflow dev dead-code-audit" in cli_doc
    assert "RETIRED_TOP_LEVEL_COMMANDS" in main_source
    for command in [
        'cmd == "browser-acceptance"',
        'cmd == "sample-suite"',
        'cmd == "maturity-eval"',
        'cmd == "restore-drill"',
        'cmd == "repair-ids"',
        'cmd == "backup"',
        'cmd == "install-local"',
        'cmd == "platform"',
        'cmd == "eval"',
        'cmd == "benchmark"',
    ]:
        assert command not in main_source
    for retired in [
        "docflow eval",
        "docflow platform",
        "docflow browser-acceptance",
        "docflow restore-drill",
        "docflow repair-ids",
        "docflow rebuild",
        "docflow sample-suite",
        "docflow maturity-eval",
        "python main.py browser-acceptance",
        "python main.py eval",
        "python main.py platform",
        "python main.py repair-ids",
        "python main.py restore-drill",
    ]:
        assert retired not in public_text

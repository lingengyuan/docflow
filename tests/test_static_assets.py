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

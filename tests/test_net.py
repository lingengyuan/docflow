from __future__ import annotations

import pytest

from src import net


def test_net_allows_local_urls():
    net.assert_allowed_url("http://localhost:11434/api/tags")
    net.assert_allowed_url("http://127.0.0.1:6333/collections")


def test_net_blocks_external_urls_by_default():
    with pytest.raises(net.UnexpectedNetworkAccess):
        net.assert_allowed_url("https://example.com")


def test_net_allows_explicit_external_user_actions():
    net.assert_allowed_url("https://example.com", allow_external=True)


def test_network_registry_documents_all_allowed_runtime_cases():
    registry = {item["id"]: item for item in net.network_access_registry()}

    assert set(registry) == {
        "local_services",
        "user_web_import",
        "model_download",
        "cloud_llm",
    }
    assert registry["model_download"]["default"].startswith("blocked")


def test_configured_allowed_hosts_normalizes_privacy_hosts():
    hosts = net.configured_allowed_hosts(
        {"privacy": {"allowed_hosts": ["Example.com", "[::1]", ""]}}
    )

    assert hosts == {"example.com", "::1"}

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

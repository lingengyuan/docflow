"""Central outbound HTTP helpers for DocFlow runtime code."""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, cast
from urllib.parse import urlparse

import httpx

Timeout = httpx.Timeout
HTTPError = httpx.HTTPError
HTTPStatusError = httpx.HTTPStatusError
ConnectTimeout = httpx.ConnectTimeout
ReadTimeout = httpx.ReadTimeout
RequestError = httpx.RequestError

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

NETWORK_ACCESS_REGISTRY = (
    {
        "id": "local_services",
        "scope": "localhost, 127.0.0.1, ::1",
        "when": "Qdrant, Ollama, and the local DocFlow web app are contacted.",
        "default": "allowed",
    },
    {
        "id": "user_web_import",
        "scope": "user supplied webpage URLs",
        "when": "A user explicitly imports a webpage into the library.",
        "default": "user initiated",
    },
    {
        "id": "model_download",
        "scope": "model hosting providers",
        "when": "A configured model is missing from the local cache.",
        "default": "blocked unless privacy.allow_model_download is true",
    },
    {
        "id": "cloud_llm",
        "scope": "explicit cloud model providers",
        "when": "A user configures a cloud LLM backend and key.",
        "default": "off",
    },
)


class UnexpectedNetworkAccess(RuntimeError):
    """Raised when runtime code attempts an unapproved outbound connection."""


def network_access_registry() -> list[dict[str, str]]:
    return [dict(item) for item in NETWORK_ACCESS_REGISTRY]


def configured_allowed_hosts(cfg: dict) -> set[str]:
    privacy_cfg = cfg.get("privacy", {}) if isinstance(cfg, dict) else {}
    raw_hosts = privacy_cfg.get("allowed_hosts", [])
    if isinstance(raw_hosts, str):
        raw_hosts = [raw_hosts]
    return {str(host).strip("[]").lower() for host in raw_hosts if str(host).strip()}


def is_local_host(host: str | None) -> bool:
    if not host:
        return False
    normalized = host.strip("[]").lower()
    if normalized in LOCAL_HOSTS:
        return True
    try:
        ip = socket.gethostbyname(normalized)
    except OSError:
        return False
    return ip.startswith("127.")


def assert_allowed_url(url: str, *, allow_external: bool = False) -> None:
    if allow_external:
        return
    parsed = urlparse(str(url))
    if parsed.scheme not in {"http", "https"}:
        raise UnexpectedNetworkAccess(f"Unsupported outbound URL scheme: {url}")
    if not is_local_host(parsed.hostname):
        raise UnexpectedNetworkAccess(f"Unexpected outbound host: {parsed.hostname or url}")


def get(url: str, *, allow_external: bool = False, **kwargs: Any) -> httpx.Response:
    assert_allowed_url(url, allow_external=allow_external)
    return httpx.get(url, **kwargs)


def post(url: str, *, allow_external: bool = False, **kwargs: Any) -> httpx.Response:
    assert_allowed_url(url, allow_external=allow_external)
    return httpx.post(url, **kwargs)


def request(
    method: str,
    url: str,
    *,
    allow_external: bool = False,
    **kwargs: Any,
) -> httpx.Response:
    assert_allowed_url(url, allow_external=allow_external)
    return httpx.request(method, url, **kwargs)


def stream(
    method: str,
    url: str,
    *,
    allow_external: bool = False,
    **kwargs: Any,
):
    assert_allowed_url(url, allow_external=allow_external)
    return httpx.stream(method, url, **kwargs)


class Client:
    def __init__(self, *args: Any, allow_external: bool = False, **kwargs: Any):
        self._allow_external = allow_external
        self._client = httpx.Client(*args, **kwargs)

    def __enter__(self) -> Client:
        self._client.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self._client.__exit__(exc_type, exc, traceback)
        return None

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        assert_allowed_url(url, allow_external=self._allow_external)
        return self._client.get(url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        assert_allowed_url(url, allow_external=self._allow_external)
        return self._client.post(url, **kwargs)


@dataclass
class NetworkGuard:
    """Temporary guard that records unexpected socket connections."""

    allowed_hosts: set[str] = field(default_factory=set)
    unexpected_hosts: set[str] = field(default_factory=set)
    _original_create_connection: Any = None

    def __enter__(self) -> NetworkGuard:
        self._original_create_connection = socket.create_connection

        def guarded_create_connection(
            address,
            timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
            source_address=None,
        ):
            host = address[0] if isinstance(address, tuple) and address else str(address)
            if not self._is_allowed(host):
                self.unexpected_hosts.add(host)
                raise UnexpectedNetworkAccess(f"Unexpected outbound host: {host}")
            return self._original_create_connection(
                address,
                timeout=timeout,
                source_address=source_address,
            )

        socket.create_connection = cast(Any, guarded_create_connection)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        socket.create_connection = cast(Any, self._original_create_connection)
        return isinstance(exc, UnexpectedNetworkAccess)

    def _is_allowed(self, host: str | None) -> bool:
        if is_local_host(host):
            return True
        normalized = str(host or "").strip("[]").lower()
        return normalized in {item.lower() for item in self.allowed_hosts}

"""Settings and health route registration."""

from __future__ import annotations

from fastapi import APIRouter


def create_router(handlers) -> APIRouter:
    router = APIRouter()
    router.add_api_route("/api/llm", handlers["get_llm"], methods=["GET"])
    router.add_api_route("/api/llm", handlers["set_llm"], methods=["POST"])
    router.add_api_route("/api/sources", handlers["list_sources"], methods=["GET"])
    router.add_api_route("/api/health", handlers["health"], methods=["GET"])
    return router


__all__ = ["create_router"]

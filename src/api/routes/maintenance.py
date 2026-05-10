"""Local maintenance and diagnostics route registration."""

from __future__ import annotations

from fastapi import APIRouter


def create_router(handlers) -> APIRouter:
    router = APIRouter()
    router.add_api_route("/api/debug/retrieve", handlers["debug_retrieve"], methods=["POST"])
    return router


__all__ = ["create_router"]

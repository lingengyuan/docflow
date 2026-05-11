"""Obsidian plugin route registration."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import ObsidianRelatedRequest


def create_router(handlers) -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/api/obsidian/related", handlers["obsidian_related_notes"], methods=["POST"]
    )
    return router


__all__ = ["ObsidianRelatedRequest", "create_router"]

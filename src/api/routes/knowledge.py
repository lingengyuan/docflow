"""Knowledge workspace route registration."""

from __future__ import annotations

from fastapi import APIRouter


def create_router(handlers: dict) -> APIRouter:
    router = APIRouter()
    router.add_api_route("/api/knowledge/overview", handlers["knowledge_overview"], methods=["GET"])
    router.add_api_route("/api/knowledge/review", handlers["knowledge_review"], methods=["GET"])
    return router

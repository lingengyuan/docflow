"""Import, upload, and note route registration."""

from __future__ import annotations

from fastapi import APIRouter


def create_router(handlers) -> APIRouter:
    router = APIRouter()
    router.add_api_route("/api/import/url", handlers["import_url"], methods=["POST"])
    router.add_api_route("/api/notes", handlers["create_note"], methods=["POST"])
    router.add_api_route("/api/notes/from-answer", handlers["save_answer_note"], methods=["POST"])
    router.add_api_route("/api/knowledge-output", handlers["create_knowledge_output"], methods=["POST"])
    router.add_api_route("/api/upload", handlers["upload_file"], methods=["POST"])
    return router


__all__ = ["create_router"]

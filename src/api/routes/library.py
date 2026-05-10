"""Library, file, history, and source-preview route registration."""

from __future__ import annotations

from fastapi import APIRouter


def create_router(handlers) -> APIRouter:
    router = APIRouter()
    router.add_api_route("/api/ingest", handlers["trigger_ingest"], methods=["POST"])
    router.add_api_route("/api/queue", handlers["queue_status"], methods=["GET"])
    router.add_api_route("/api/files", handlers["list_files"], methods=["GET"])
    router.add_api_route("/api/library/meta", handlers["library_meta"], methods=["GET"])
    router.add_api_route("/api/storage/usage", handlers["storage_usage"], methods=["GET"])
    router.add_api_route("/api/files/{file_id}/metadata", handlers["update_file_metadata"], methods=["PATCH"])
    router.add_api_route("/api/files/batch/favorite", handlers["batch_favorite"], methods=["POST"])
    router.add_api_route("/api/files/batch/metadata", handlers["batch_update_file_metadata"], methods=["POST"])
    router.add_api_route("/api/files/batch/rebuild", handlers["batch_rebuild_files"], methods=["POST"])
    router.add_api_route("/api/file/{file_id}/preview", handlers["preview_file"], methods=["GET"])
    router.add_api_route("/api/file/{file_id}/preview", handlers["preview_file_head"], methods=["HEAD"])
    router.add_api_route("/api/file/{file_id}/chunks", handlers["list_file_chunks"], methods=["GET"])
    router.add_api_route("/api/history", handlers["list_history"], methods=["GET"])
    router.add_api_route("/api/history/search", handlers["search_history"], methods=["GET"])
    router.add_api_route("/api/history", handlers["clear_history"], methods=["DELETE"])
    router.add_api_route("/api/favorites", handlers["list_favorites"], methods=["GET"])
    router.add_api_route("/api/favorites/{file_id}", handlers["toggle_favorite"], methods=["POST"])
    router.add_api_route("/api/summarize", handlers["summarize_files"], methods=["POST"])
    return router


__all__ = ["create_router"]

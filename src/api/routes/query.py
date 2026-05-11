"""Query and conversation route registration."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import ConversationCreateRequest, QueryRequest, QueryResponse, ResearchRequest, ResearchResponse


def create_router(handlers) -> APIRouter:
    router = APIRouter()
    router.add_api_route("/api/query", handlers["query"], methods=["POST"], response_model=QueryResponse)
    router.add_api_route("/api/research", handlers["research"], methods=["POST"], response_model=ResearchResponse)
    router.add_api_route("/api/query/stream", handlers["query_stream"], methods=["POST"])
    router.add_api_route("/api/conversations", handlers["list_conversations"], methods=["GET"])
    router.add_api_route(
        "/api/conversations",
        handlers["create_conversation"],
        methods=["POST"],
        response_model=None,
    )
    router.add_api_route(
        "/api/conversations/{conversation_id}/messages",
        handlers["list_conversation_messages"],
        methods=["GET"],
    )
    router.add_api_route(
        "/api/conversations/{conversation_id}",
        handlers["delete_conversation"],
        methods=["DELETE"],
    )
    return router


__all__ = ["ConversationCreateRequest", "QueryRequest", "ResearchRequest", "create_router"]

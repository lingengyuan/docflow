"""FastAPI route registration for DocFlow."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.handlers.import_handlers import (
    create_demo_library,
    create_knowledge_output,
    create_note,
    import_url,
    save_answer_note,
    upload_file,
)
from src.api.handlers.library_handlers import (
    batch_favorite,
    batch_rebuild_files,
    batch_update_file_metadata,
    clear_history,
    knowledge_overview,
    knowledge_review,
    library_meta,
    list_favorites,
    list_file_chunks,
    list_files,
    list_history,
    preview_file,
    preview_file_head,
    queue_status,
    search_history,
    storage_usage,
    summarize_files,
    toggle_favorite,
    trigger_ingest,
    update_file_metadata,
)
from src.api.handlers.maintenance_handlers import debug_retrieve
from src.api.handlers.query_handlers import (
    answer_feedback,
    create_conversation,
    delete_conversation,
    list_conversation_messages,
    list_conversations,
    query,
    research,
)
from src.api.handlers.query_stream_handlers import query_stream
from src.api.handlers.settings_handlers import get_llm, health, list_sources, set_llm
from src.api.routes import imports as imports_routes
from src.api.routes import knowledge as knowledge_routes
from src.api.routes import library as library_routes
from src.api.routes import maintenance as maintenance_routes
from src.api.routes import query as query_routes
from src.api.routes import settings as settings_routes


def register_api_routes(app: FastAPI) -> None:
    app.include_router(
        query_routes.create_router(
            {
                "query": query,
                "research": research,
                "query_stream": query_stream,
                "answer_feedback": answer_feedback,
                "list_conversations": list_conversations,
                "create_conversation": create_conversation,
                "list_conversation_messages": list_conversation_messages,
                "delete_conversation": delete_conversation,
            }
        )
    )
    app.include_router(
        library_routes.create_router(
            {
                "trigger_ingest": trigger_ingest,
                "queue_status": queue_status,
                "list_files": list_files,
                "library_meta": library_meta,
                "storage_usage": storage_usage,
                "update_file_metadata": update_file_metadata,
                "batch_favorite": batch_favorite,
                "batch_update_file_metadata": batch_update_file_metadata,
                "batch_rebuild_files": batch_rebuild_files,
                "preview_file": preview_file,
                "preview_file_head": preview_file_head,
                "list_file_chunks": list_file_chunks,
                "list_history": list_history,
                "search_history": search_history,
                "clear_history": clear_history,
                "list_favorites": list_favorites,
                "toggle_favorite": toggle_favorite,
                "summarize_files": summarize_files,
            }
        )
    )
    app.include_router(
        imports_routes.create_router(
            {
                "import_url": import_url,
                "create_note": create_note,
                "save_answer_note": save_answer_note,
                "create_knowledge_output": create_knowledge_output,
                "upload_file": upload_file,
                "create_demo_library": create_demo_library,
            }
        )
    )
    app.include_router(
        knowledge_routes.create_router(
            {
                "knowledge_overview": knowledge_overview,
                "knowledge_review": knowledge_review,
            }
        )
    )
    app.include_router(
        settings_routes.create_router(
            {
                "get_llm": get_llm,
                "set_llm": set_llm,
                "list_sources": list_sources,
                "health": health,
            }
        )
    )
    app.include_router(
        maintenance_routes.create_router(
            {
                "debug_retrieve": debug_retrieve,
            }
        )
    )

"""Request and response schemas for the DocFlow API."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    file_filter: list[str] | None = None
    scope_mode: str | None = "all"
    collection: str | None = None
    file_id: int | None = None
    file_name: str | None = None
    retrieval_mode: str | None = "hybrid"
    conversation_id: int | None = None


class DebugRetrieveRequest(BaseModel):
    question: str
    file_filter: list[str] | None = None
    scope_mode: str | None = "all"
    collection: str | None = None
    file_id: int | None = None
    file_name: str | None = None
    retrieval_mode: str | None = "hybrid"
    include_rerank: bool = True
    max_text_chars: int = 300


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    related_notes: list[dict] = Field(default_factory=list)
    conversation_id: int | None = None
    scope: dict | None = None


class ConversationCreateRequest(BaseModel):
    title: str = ""


class FileMetadataRequest(BaseModel):
    collection: str | None = None
    user_tags: list[str] | None = None


class BatchFavoriteRequest(BaseModel):
    file_ids: list[int]
    favorited: bool = True


class BatchMetadataRequest(BaseModel):
    file_ids: list[int]
    collection: str | None = None
    user_tags: list[str] | None = None


class BatchRebuildRequest(BaseModel):
    file_ids: list[int]


class WebImportRequest(BaseModel):
    url: str
    title: str | None = None
    collection: str | None = None
    user_tags: list[str] | None = None


class NoteCreateRequest(BaseModel):
    title: str
    content: str
    collection: str | None = None
    user_tags: list[str] | None = None


class AnswerNoteRequest(BaseModel):
    title: str | None = None
    question: str | None = None
    answer: str
    citations: list[dict] | None = None
    collection: str | None = None
    user_tags: list[str] | None = None


class KnowledgeOutputRequest(BaseModel):
    output_type: str
    title: str | None = None
    source_text: str | None = None
    file_ids: list[int] = Field(default_factory=list)
    collection: str | None = None
    user_tags: list[str] | None = None


class SummarizeRequest(BaseModel):
    file_ids: list[int]


class LLMSwitchRequest(BaseModel):
    model: str


@dataclass(frozen=True)
class QueryOptions:
    file_filter: list[str]
    retrieval_mode: str
    scope: dict

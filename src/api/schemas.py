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
    history_id: int | None = None
    conversation_id: int | None = None
    scope: dict | None = None
    reproducible: bool = True


class ResearchRequest(QueryRequest):
    max_steps: int = 3


class ResearchResponse(QueryResponse):
    research_steps: list[dict] = Field(default_factory=list)


class ConversationCreateRequest(BaseModel):
    title: str = ""


class AnswerFeedbackRequest(BaseModel):
    history_id: int
    rating: str
    note: str | None = ""


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


class ObsidianRelatedRequest(BaseModel):
    note_title: str | None = None
    note_path: str | None = None
    note_content: str | None = None
    selection: str | None = None
    retrieval_mode: str | None = "hybrid"
    limit: int = 6


class SummarizeRequest(BaseModel):
    file_ids: list[int]


class LLMSwitchRequest(BaseModel):
    model: str


@dataclass(frozen=True)
class QueryOptions:
    file_filter: list[str]
    retrieval_mode: str
    scope: dict

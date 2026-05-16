from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app as api_app
from src.api.services.knowledge_service import KnowledgeService
from src.domain_types import FileStatus
from src.ingest.store import DocStore


def _add_file(store: DocStore, root: Path, name: str, text: str, collection: str = "Inbox") -> int:
    path = root / name
    path.write_text(text, encoding="utf-8")
    file_id = store.upsert_file(
        path,
        name,
        DocStore.compute_hash(path),
        status=FileStatus.DONE,
        total_pages=1,
        mtime_ns=path.stat().st_mtime_ns,
    )
    store.update_file_metadata(file_id, collection=collection, user_tags=["phase57"])
    store.add_chunks(
        file_id,
        [
            {
                "qdrant_id": file_id,
                "chunk_type": "text",
                "page_num": 1,
                "section": "Overview",
                "char_count": len(text),
                "raw_text": text,
                "embedding_text": text,
                "tokenized_text": text.lower(),
            }
        ],
    )
    return file_id


def test_knowledge_service_derives_topics_similar_documents_and_cards(tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    try:
        _add_file(
            store,
            tmp_path,
            "privacy-roadmap.md",
            "privacy local offline model cache source control",
        )
        _add_file(
            store,
            tmp_path,
            "privacy-check.md",
            "privacy local offline checks source preview",
        )
        history_id = store.add_history("Is privacy local?", "Yes", citations_json="[]")
        store.set_answer_feedback(history_id, "useful")

        overview = KnowledgeService().overview(store)

        assert overview["topics"]
        assert overview["similar_documents"]
        assert overview["knowledge_cards"]
        assert overview["knowledge_cards"][0]["source_file"]["file_name"]
        assert overview["knowledge_graph"]["nodes"]
        assert overview["knowledge_graph"]["edges"]
        assert overview["feedback"]["useful"] == 1
    finally:
        store.close()


def test_knowledge_service_builds_active_review_from_usage_signals(tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    try:
        source_id = _add_file(
            store,
            tmp_path,
            "research-notes.md",
            "privacy review citations backlinks useful answers",
        )
        note_id = _add_file(
            store,
            tmp_path,
            "saved-answer.md",
            "saved answer connected to research notes",
            collection="Saved Answers",
        )
        store.replace_note_source_links(note_id, [source_id])
        history_id = store.add_history(
            "How should I review privacy notes?",
            "Review source notes.",
            citations_json=json.dumps([{"file_name": "research-notes.md"}]),
        )
        store.set_answer_feedback(history_id, "useful")

        review = KnowledgeService().review(store)

        assert review["signals"]["files"] == 2
        assert review["signals"]["questions"] == 1
        assert review["signals"]["saved_answers"] == 1
        assert review["signals"]["backlinks"] >= 1
        assert review["topic_activity"]
        assert review["recommendations"]
        assert review["review_queue"][0]["file"]["id"] == source_id
        assert review["review_queue"][0]["signals"]["citations"] == 1
        assert review["relationship_timeline"][0]["note"]["id"] == note_id
        assert review["relationship_timeline"][0]["source"]["id"] == source_id
        assert review["relationship_timeline"][0]["label"] == "保存回答引用了来源"
        depth = review["knowledge_depth"]
        assert depth["concepts"]
        assert depth["relationship_opportunities"] == []
        assert depth["source_trails"][0]["question"] == "How should I review privacy notes?"
        assert depth["source_trails"][0]["files"][0]["id"] == source_id
        assert depth["source_trails"][0]["feedback"]["rating"] == "useful"
        assert depth["next_actions"]
    finally:
        store.close()


def test_knowledge_depth_flags_cited_sources_without_saved_notes(tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    try:
        source_id = _add_file(
            store,
            tmp_path,
            "citation-gap.md",
            "privacy citation source review needs durable note coverage",
        )
        store.add_history(
            "Which privacy source should become a note?",
            "Use the cited source.",
            citations_json=json.dumps([{"file_name": "citation-gap.md"}]),
        )

        review = KnowledgeService().review(store)
        depth = review["knowledge_depth"]

        assert depth["coverage_gaps"]
        assert depth["coverage_gaps"][0]["type"] == "cited_without_note"
        assert depth["coverage_gaps"][0]["file"]["id"] == source_id
        assert depth["source_trails"][0]["citation_count"] == 1
        assert depth["next_actions"][0]["type"] == "coverage_gap"
    finally:
        store.close()


def test_knowledge_depth_suggests_unlinked_related_sources(tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    try:
        first_id = _add_file(
            store,
            tmp_path,
            "project-review.md",
            "retrieval privacy citation review source grounding",
        )
        second_id = _add_file(
            store,
            tmp_path,
            "privacy-citations.md",
            "privacy citation review evidence source trust",
        )

        review = KnowledgeService().review(store)
        opportunities = review["knowledge_depth"]["relationship_opportunities"]

        assert opportunities
        assert {opportunities[0]["source"]["id"], opportunities[0]["target"]["id"]} == {
            first_id,
            second_id,
        }
        assert "privacy" in opportunities[0]["shared_terms"]
        assert any(
            item["type"] == "relationship"
            for item in review["knowledge_depth"]["next_actions"]
        )

        store.replace_note_source_links(first_id, [second_id])
        linked_review = KnowledgeService().review(store)

        assert linked_review["knowledge_depth"]["relationship_opportunities"] == []
    finally:
        store.close()


def test_confirm_relationship_api_turns_suggestion_into_saved_link(monkeypatch, tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    try:
        first_id = _add_file(
            store,
            tmp_path,
            "project-review.md",
            "retrieval privacy citation review source grounding",
        )
        second_id = _add_file(
            store,
            tmp_path,
            "privacy-citations.md",
            "privacy citation review evidence source trust",
        )
        third_id = _add_file(
            store,
            tmp_path,
            "trust-evidence.md",
            "privacy evidence source trust review notes",
        )
        monkeypatch.setattr(api_app, "store", store)
        client = TestClient(api_app.app)

        before = client.get("/api/knowledge/review").json()
        assert before["knowledge_depth"]["relationship_opportunities"]

        response = client.post(
            "/api/knowledge/relationships",
            json={"source_file_id": first_id, "target_file_id": second_id},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["source_links"] == [second_id]
        assert body["relation"] == "manual_relationship"
        assert store.list_outbound_links(first_id)[0]["file"]["id"] == second_id

        second_response = client.post(
            "/api/knowledge/relationships",
            json={"source_file_id": first_id, "target_file_id": third_id},
        )

        assert second_response.status_code == 200
        saved_targets = {
            int(link["file"]["id"])
            for link in store.list_outbound_links(first_id)
            if link["relation"] == "manual_relationship"
        }
        assert saved_targets == {second_id, third_id}

        after = client.get("/api/knowledge/review").json()
        remaining_pairs = {
            (
                int(item["source"]["id"]),
                int(item["target"]["id"]),
            )
            for item in after["knowledge_depth"]["relationship_opportunities"]
        }
        assert (first_id, second_id) not in remaining_pairs
        assert (first_id, third_id) not in remaining_pairs
        assert after["relationship_timeline"][0]["label"] == "确认了相关资料"
    finally:
        store.close()


def test_knowledge_overview_api_uses_current_store(monkeypatch, tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    try:
        file_id = _add_file(
            store,
            tmp_path,
            "knowledge-card.md",
            "knowledge cards connect answers notes and source files",
            collection="Knowledge Outputs",
        )
        monkeypatch.setattr(api_app, "store", store)

        client = TestClient(api_app.app)
        response = client.get(f"/api/knowledge/overview?file_id={file_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["stats"]["files"] == 1
        assert body["knowledge_cards"][0]["source_file"]["id"] == file_id
        assert body["backlinks"] == []
        assert body["knowledge_graph"]["stats"]["nodes"] >= 1
    finally:
        store.close()


def test_knowledge_review_api_uses_current_store(monkeypatch, tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    try:
        _add_file(
            store,
            tmp_path,
            "active-review.md",
            "active review should be based on local usage signals",
        )
        monkeypatch.setattr(api_app, "store", store)

        client = TestClient(api_app.app)
        response = client.get("/api/knowledge/review")

        assert response.status_code == 200
        body = response.json()
        assert body["signals"]["files"] == 1
        assert body["review_queue"]
        assert body["recommendations"]
    finally:
        store.close()

"""Mutation browser acceptance flow and cleanup helpers."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import yaml

from src import net


def check_note_ingest_query_cleanup_flow(
    page: Any, base_url: str, timeout_ms: int
) -> dict[str, Any]:
    stamp = str(int(time.time()))
    title = f"phase32-acceptance-{stamp}"
    token = f"phase32-token-{stamp}"
    question = "这条临时笔记里的验收标记是什么？请只回答标记。"
    file_record: dict[str, Any] | None = None
    cleanup_details: dict[str, Any] = {}
    conversation_id: int | None = None
    try:
        page.locator("#nav-notes").click(timeout=timeout_ms)
        page.locator("#notes-title-input").fill(title, timeout=timeout_ms)
        page.locator("#notes-collection-input").fill("Phase32 Acceptance", timeout=timeout_ms)
        page.locator("#notes-tags-input").fill("phase32,temp", timeout=timeout_ms)
        page.locator("#notes-content-input").fill(
            f"# {title}\n\n验收标记：{token}\n\n这是一条 Phase32 临时笔记。",
            timeout=timeout_ms,
        )
        page.locator("#notes-submit-btn").click(timeout=timeout_ms)
        page.wait_for_function(
            """
            () => (document.querySelector('#notes-status')?.innerText || '')
              .includes('已加入入库队列')
            """,
            timeout=timeout_ms,
        )
        file_record = wait_for_file_by_title(base_url, title, max(timeout_ms, 60_000))
        file_record = wait_for_file_status(
            base_url, int(file_record["id"]), "done", max(timeout_ms, 120_000)
        )
        query_details = query_temporary_note(page, file_record, token, question, timeout_ms)
        conversation_id = query_details.get("conversation_id")
        return {
            "created": file_record["file_name"],
            "status": file_record["status"],
            "queried": query_details,
            "cleanup": cleanup_details,
        }
    finally:
        if file_record:
            cleanup_details.update(
                cleanup_mutation_file(
                    file_record, conversation_id=conversation_id, question=question
                )
            )


def query_temporary_note(
    page: Any, file_record: dict[str, Any], token: str, question: str, timeout_ms: int
) -> dict[str, Any]:
    file_id = int(file_record["id"])
    file_name = str(file_record["file_name"])
    page.locator("#nav-chat").click(timeout=timeout_ms)
    page.wait_for_function(
        """
        ({ fileId }) => {
            const select = document.querySelector('#query-scope-file');
            return Boolean(
              select && [...select.options].some(option => option.value === String(fileId))
            );
        }
        """,
        arg={"fileId": file_id},
        timeout=timeout_ms,
    )
    page.locator("#query-scope-mode").select_option("file", timeout=timeout_ms)
    page.locator("#query-scope-file").select_option(str(file_id), timeout=timeout_ms)
    page.locator("#input").fill(question, timeout=timeout_ms)
    page.locator("#send-btn").click(timeout=timeout_ms)
    page.wait_for_function(
        """
        ({ fileName }) => {
            const send = document.querySelector('#send-btn');
            const thinking = document.querySelector('#thinking-indicator');
            const messages = document.querySelector('#messages');
            const text = messages?.innerText || '';
            return Boolean(send && !send.disabled && !thinking && (
                text.includes('耗时') ||
                text.includes(fileName) ||
                text.includes('回答失败') ||
                text.includes('连接中断')
            ));
        }
        """,
        arg={"fileName": file_name},
        timeout=max(timeout_ms, 120_000),
    )
    text = page.locator("#messages").inner_text(timeout=timeout_ms)
    failure_terms = (
        "本次查询失败",
        "本次回答失败",
        "回答失败",
        "连接中断",
        "耗时太久",
        "暂时连接不上",
    )
    matched_failures = [term for term in failure_terms if term in text]
    if matched_failures:
        raise AssertionError(f"temporary note query failed: {', '.join(matched_failures)}")
    answer_visible = bool(text.strip())
    if not answer_visible:
        raise AssertionError("temporary note query did not render an answer")
    conversation_id = page.evaluate(
        "typeof currentConversationId === 'number' ? currentConversationId : null"
    )
    return {
        "answer_visible": answer_visible,
        "citation_visible": file_name in text,
        "token_mentioned": token in text,
        "conversation_id": conversation_id,
    }


def api_json(
    base_url: str, path: str, timeout_ms: int, *, method: str = "GET", payload: dict | None = None
) -> Any:
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    response = net.request(
        method,
        f"{base_url.rstrip('/')}{path}",
        json=payload,
        headers=headers,
        timeout=net.Timeout(max(timeout_ms / 1000, 1), connect=1.0),
    )
    response.raise_for_status()
    return response.json() if response.content else None


def wait_for_file_by_title(base_url: str, title: str, timeout_ms: int) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_ms / 1000
    safe_title = title.lower()
    while time.perf_counter() < deadline:
        files = api_json(base_url, "/api/files", timeout_ms)
        for item in files:
            name = str(item.get("file_name", "")).lower()
            if safe_title in name:
                return item
        time.sleep(0.5)
    raise AssertionError(f"temporary note was not listed: {title}")


def wait_for_file_status(
    base_url: str, file_id: int, status: str, timeout_ms: int
) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_ms / 1000
    last_status = ""
    while time.perf_counter() < deadline:
        files = api_json(base_url, "/api/files", timeout_ms)
        for item in files:
            if int(item.get("id", 0)) != file_id:
                continue
            last_status = str(item.get("status", ""))
            if last_status == status:
                return item
            if last_status == "error":
                raise AssertionError(f"temporary note ingest failed: {item.get('error_msg', '')}")
        time.sleep(1)
    raise AssertionError(
        f"temporary note did not reach {status}; last status: {last_status or 'missing'}"
    )


def cleanup_mutation_file(
    file_record: dict[str, Any],
    *,
    conversation_id: int | None = None,
    question: str = "",
) -> dict[str, Any]:
    file_path = Path(str(file_record.get("file_path", ""))).expanduser()
    details: dict[str, Any] = {
        "file_deleted": False,
        "record_deleted": False,
        "vectors_deleted": 0,
        "history_deleted": 0,
        "conversation_deleted": False,
    }
    if file_path.exists():
        file_path.unlink()
        details["file_deleted"] = True
    cfg = load_project_config()
    qdrant_ids = delete_file_record(cfg, file_path)
    details["record_deleted"] = bool(qdrant_ids is not None)
    if qdrant_ids:
        delete_qdrant_points(cfg, qdrant_ids)
        details["vectors_deleted"] = len(qdrant_ids)
    history_deleted, conversation_deleted = delete_acceptance_history(
        cfg,
        conversation_id=conversation_id,
        question=question,
    )
    details["history_deleted"] = history_deleted
    details["conversation_deleted"] = conversation_deleted
    return details


def load_project_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def expand_config_path(value: str | Path) -> Path:
    return Path(os.path.expanduser(str(value))).resolve()


def delete_file_record(cfg: dict[str, Any], file_path: Path) -> list[int] | None:
    db_path = expand_config_path(cfg.get("paths", {}).get("db_path", "docflow.db"))
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT id FROM files WHERE file_path = ?", (str(file_path),)).fetchone()
        if row is None:
            return None
        file_id = int(row[0])
        chunk_rows = conn.execute(
            "SELECT id, qdrant_id FROM chunks WHERE file_id = ?", (file_id,)
        ).fetchall()
        chunk_ids = [int(row[0]) for row in chunk_rows]
        qdrant_ids = [int(row[1]) for row in chunk_rows]
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})", chunk_ids)
            conn.execute(
                f"DELETE FROM chunks_fts_trigram WHERE rowid IN ({placeholders})", chunk_ids
            )
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM favorites WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
    return qdrant_ids


def delete_acceptance_history(
    cfg: dict[str, Any],
    *,
    conversation_id: int | None,
    question: str,
) -> tuple[int, bool]:
    db_path = expand_config_path(cfg.get("paths", {}).get("db_path", "docflow.db"))
    deleted_history = 0
    deleted_conversation = False
    with sqlite3.connect(db_path) as conn:
        if question:
            result = conn.execute("DELETE FROM history WHERE question = ?", (question,))
            deleted_history += result.rowcount
        if conversation_id:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            result = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            deleted_conversation = result.rowcount > 0
        elif question:
            rows = conn.execute(
                "SELECT DISTINCT conversation_id FROM messages WHERE content = ?",
                (question,),
            ).fetchall()
            for row in rows:
                cid = int(row[0])
                conn.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
                result = conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))
                deleted_conversation = deleted_conversation or result.rowcount > 0
    return deleted_history, deleted_conversation


def delete_qdrant_points(cfg: dict[str, Any], qdrant_ids: list[int]) -> None:
    from qdrant_client import QdrantClient

    qdrant_cfg = cfg.get("qdrant", {})
    client = QdrantClient(
        host=qdrant_cfg.get("host", "localhost"),
        port=int(qdrant_cfg.get("port", 6333)),
        timeout=5,
    )
    client.delete(
        collection_name=qdrant_cfg.get("collection", "docflow"),
        points_selector=qdrant_ids,
    )

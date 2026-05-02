# Phase 7 Handoff: Multi-Turn Conversations

Date: 2026-05-02

## Status

Phase 7 is implemented for optimization plan section 2.11: multi-turn conversations.

Completed scope:

- Added persistent `conversations` storage.
- Added persistent `messages` storage.
- Added conversation-aware synchronous query support.
- Added conversation-aware streaming query support.
- Added conversation creation, listing, message listing, and deletion APIs.
- Added recent-message context to answer generation.
- Added deterministic follow-up query rewrite for prompts like "展开第二点".
- Kept the newest question as the answer target while using recent turns only as context.
- Updated the browser chat flow to keep using the same conversation after the first streamed reply.
- Preserved the older `/api/history` behavior for the existing history view.
- Updated README API and feature references in English and Chinese.

## Files Changed

- `src/ingest/store.py`
  - Added `conversations` and `messages` tables.
  - Added conversation/message CRUD helpers.
  - Added `conversation_id` to history rows for compatibility with the existing history view.
- `src/api/app.py`
  - Added `conversation_id` support to query requests and responses.
  - Added `/api/conversations`.
  - Added `/api/conversations/{id}/messages`.
  - Added `DELETE /api/conversations/{id}`.
  - Added recent conversation context lookup and follow-up query rewrite.
- `src/query/engine.py`
  - Added separate retrieval query support.
  - Passes recent conversation context to the answer generator.
- `src/query/generator.py`
  - Adds recent conversation context into the model prompt.
  - Keeps document snippets as the required source of factual answers.
- `frontend/index.html`
  - Stores the current conversation id during streaming chat.
  - Sends the conversation id on later questions in the same chat.
- `tests/test_conversations.py`
  - Added API tests for creation, query persistence, follow-up rewrite, deletion, and streaming persistence.
- `tests/test_store.py`
  - Added persistence and deletion coverage for conversations/messages.
- `tests/test_engine.py`
  - Added retrieval-query and conversation-context coverage.
- `tests/test_generator.py`
  - Added prompt-context coverage.
- `README.md`
  - Added conversation feature and API references.
- `docs/phase7-handoff.md`
  - This handoff.

## API

Create or list conversations:

```bash
GET /api/conversations
POST /api/conversations
```

List messages:

```bash
GET /api/conversations/{conversation_id}/messages
```

Delete a conversation:

```bash
DELETE /api/conversations/{conversation_id}
```

Ask within a conversation:

```json
{
  "question": "展开第二点",
  "conversation_id": 1
}
```

Streaming query now emits an extra SSE event before citations:

```text
event: conversation
data: {"conversation_id": 1}
```

## Validation Results

Commands that passed:

```bash
.venv/bin/python -m pytest tests/test_conversations.py tests/test_engine.py tests/test_generator.py tests/test_store.py
```

Result:

- 33 passed
- 5 warnings from third-party SWIG/PyMuPDF imports

```bash
.venv/bin/python -m pytest
```

Result:

- 105 passed
- 5 warnings from third-party SWIG/PyMuPDF imports

```bash
awk '/<script>/{flag=1;next}/<\\/script>/{flag=0}flag' frontend/index.html | node --check
```

Result:

- Frontend script syntax check passed

```bash
/Users/hughlin/.codex/skills/readme-maintainer/scripts/check_bilingual_readme.sh README.md
```

Result:

- Bilingual README parity check passed

```bash
git diff --check
```

Result:

- No whitespace errors

GitHub check:

```bash
gh issue list --repo lingengyuan/docflow --search "conversation OR message OR phase7" --limit 20
gh pr list --repo lingengyuan/docflow --search "conversation OR message OR phase7" --limit 20
```

Result:

- No matching open issues or PRs were returned.

## Known Limitations

- The browser UI only keeps the current chat conversation alive. It does not yet provide a full conversation sidebar or conversation switching UI.
- Follow-up query rewrite is deterministic and pattern-based. It handles prompts like "展开第二点", but it is not a full LLM-powered query rewrite.
- Conversation deletion removes messages for that conversation, but the legacy history list is still separate.
- The old history view remains a flat list. Full frontend conversation management belongs to Phase 8.

## Next Phase

Start with optimization plan section 2.12: frontend polish.

Exact next tasks:

1. Add a visible conversation list or conversation switcher in the browser UI.
2. Let users create, switch, and delete conversations without using API calls manually.
3. Improve citation display so Markdown references show sections and PDF references show pages clearly.
4. Make source preview/open behavior more useful from citations.
5. Add copy/export affordances for long answers.
6. Improve queue progress, dependency state, and retrieval timing visibility.
7. Ensure failure states are visible in the page, not only in terminal output.
8. Run `.venv/bin/python -m pytest`, frontend script syntax check, relevant browser/API flow checks, README parity check, `git diff --check`, and write the next phase handoff before reporting completion.

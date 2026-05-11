from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.query.generator import AnswerGenerator


CHUNKS = [
    {
        "text": "DocFlow keeps private notes local by default.",
        "file_name": "privacy.md",
        "file_path": "/docs/privacy.md",
        "page_num": 1,
        "section": "Privacy",
        "qdrant_id": 501,
        "rerank_score": 0.9,
    }
]


def _mock_response(text: str) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"message": {"content": text}}
    response.raise_for_status.return_value = None
    return response


def test_same_seed_builds_same_ollama_request_payload():
    generator = AnswerGenerator(
        backend="local",
        ollama_model="qwen2.5:7b",
        seed=77,
        temperature=0.0,
        top_p=1.0,
    )
    payloads: list[str] = []

    def record_payload(*args, **kwargs):
        payloads.append(json.dumps(kwargs["json"], ensure_ascii=False, sort_keys=True))
        return _mock_response("same answer")

    with patch("src.net.post", side_effect=record_payload):
        first = generator.generate("What does DocFlow do?", CHUNKS)
        second = generator.generate("What does DocFlow do?", CHUNKS)

    assert first.text == second.text == "same answer"
    assert first.reproducible is True
    assert second.reproducible is True
    assert payloads[0] == payloads[1]


def test_cloud_backend_is_not_reported_as_reproducible():
    generator = AnswerGenerator(backend="claude", claude_api_key="test-key")

    with patch.object(generator, "_call_with_system", return_value="cloud answer"):
        answer = generator.generate("What does DocFlow do?", CHUNKS)

    assert answer.text == "cloud answer"
    assert answer.reproducible is False

from src.query.engine import QueryEngine
from src.query.generator import Answer


CHUNKS = [
    {
        "text": "Health checks report degraded optional capabilities.",
        "file_name": "README.md",
        "file_path": "/tmp/README.md",
        "page_num": 1,
        "rrf_score": 0.5,
    }
]


class FakeRetriever:
    def retrieve(self, query, file_filter=None, prefer_tables=False, cancel_event=None):
        return CHUNKS


class FailingGenerator:
    def generate(self, question, chunks):
        raise RuntimeError("model unavailable")

    def generate_stream(self, question, chunks, cancel_event=None):
        raise RuntimeError("stream unavailable")
        yield


class WorkingGenerator:
    def generate(self, question, chunks):
        return Answer(text="answer", citations=[])

    def generate_stream(self, question, chunks, cancel_event=None):
        yield "answer"


def test_query_returns_retrieved_snippets_when_llm_fails():
    engine = QueryEngine(FakeRetriever(), FailingGenerator())

    answer = engine.query("health status")

    assert "回答模型暂时不可用" in answer.text
    assert "RuntimeError" in answer.text
    assert len(answer.citations) == 1
    assert answer.citations[0].file_name == "README.md"


def test_query_stream_returns_fallback_message_when_llm_fails():
    engine = QueryEngine(FakeRetriever(), FailingGenerator())

    chunks, token_gen = engine.query_stream("health status")

    assert chunks == CHUNKS
    assert "回答模型暂时不可用" in "".join(token_gen)


def test_query_still_uses_generator_when_available():
    engine = QueryEngine(FakeRetriever(), WorkingGenerator())

    answer = engine.query("health status")

    assert answer.text == "answer"

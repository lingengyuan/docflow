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
    def __init__(self):
        self.queries = []

    def retrieve(self, query, file_filter=None, prefer_tables=False, cancel_event=None):
        self.queries.append(query)
        return CHUNKS


class FailingGenerator:
    def generate(self, question, chunks, conversation_context=None):
        raise RuntimeError("model unavailable")

    def generate_stream(self, question, chunks, cancel_event=None, conversation_context=None):
        raise RuntimeError("stream unavailable")
        yield


class WorkingGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, question, chunks, conversation_context=None):
        self.calls.append((question, conversation_context))
        return Answer(text="answer", citations=[])

    def generate_stream(self, question, chunks, cancel_event=None, conversation_context=None):
        self.calls.append((question, conversation_context))
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
    generator = WorkingGenerator()
    engine = QueryEngine(FakeRetriever(), generator)

    answer = engine.query("health status")

    assert answer.text == "answer"


def test_query_uses_retrieval_query_and_passes_conversation_context():
    retriever = FakeRetriever()
    generator = WorkingGenerator()
    engine = QueryEngine(retriever, generator)
    context = [{"role": "user", "content": "上一问"}]

    answer = engine.query(
        "展开第二点",
        conversation_context=context,
        retrieval_query="上一问\n展开第二点",
    )

    assert answer.text == "answer"
    assert retriever.queries == ["上一问\n展开第二点"]
    assert generator.calls[0] == ("展开第二点", context)


def test_query_stream_uses_conversation_context():
    retriever = FakeRetriever()
    generator = WorkingGenerator()
    engine = QueryEngine(retriever, generator)
    context = [{"role": "assistant", "content": "上一答"}]

    chunks, token_gen = engine.query_stream(
        "继续",
        conversation_context=context,
        retrieval_query="上一问\n继续",
    )

    assert chunks == CHUNKS
    assert "".join(token_gen) == "answer"
    assert retriever.queries == ["上一问\n继续"]
    assert generator.calls[0] == ("继续", context)

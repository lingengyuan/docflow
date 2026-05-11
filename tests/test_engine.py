from src.query.engine import QueryEngine
from src.query.generator import Answer


CHUNKS = [
    {
        "text": "Health checks report degraded optional capabilities.",
        "file_name": "README.md",
        "file_path": "/tmp/README.md",
        "page_num": 1,
        "section": "Health",
        "rrf_score": 0.5,
    }
]


class FakeRetriever:
    def __init__(self):
        self.queries = []
        self.retrieval_modes = []

    def retrieve(
        self,
        query,
        file_filter=None,
        retrieval_mode="hybrid",
        prefer_tables=False,
        cancel_event=None,
        related_k=0,
    ):
        self.queries.append(query)
        self.retrieval_modes.append(retrieval_mode)
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
    assert answer.citations[0].section == "Health"


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


def test_query_can_force_full_text_retrieval_mode():
    retriever = FakeRetriever()
    generator = WorkingGenerator()
    engine = QueryEngine(retriever, generator)

    answer = engine.query("exact terms", retrieval_mode="full_text")

    assert answer.text == "answer"
    assert retriever.retrieval_modes == ["full_text"]


def test_query_returns_related_notes_from_unused_candidates():
    chunks = [
        {
            "text": f"answer chunk {idx}",
            "file_name": f"answer-{idx}.md",
            "file_path": f"/tmp/answer-{idx}.md",
            "page_num": 1,
            "section": "",
            "rrf_score": 0.9 - idx * 0.01,
        }
        for idx in range(5)
    ] + [
        {
            "text": "related alpha",
            "file_name": "related-a.md",
            "file_path": "/tmp/related-a.md",
            "page_num": 2,
            "section": "Alpha",
            "rrf_score": 0.4,
        },
        {
            "text": "related beta",
            "file_name": "related-b.md",
            "file_path": "/tmp/related-b.md",
            "page_num": 3,
            "section": "Beta",
            "rrf_score": 0.3,
        },
    ]

    class RelatedRetriever(FakeRetriever):
        def retrieve(self, *args, **kwargs):
            self.queries.append(kwargs.get("query") or args[0])
            return chunks

    answer = QueryEngine(RelatedRetriever(), WorkingGenerator()).query("find related")

    assert [item["file_name"] for item in answer.related_notes] == [
        "related-a.md",
        "related-b.md",
    ]
    assert answer.related_notes[0]["section"] == "Alpha"


def test_full_text_results_with_zero_vector_score_are_allowed():
    class FullTextRetriever(FakeRetriever):
        def retrieve(self, *args, **kwargs):
            item = dict(CHUNKS[0])
            item["vec_score"] = 0.0
            item["rrf_score"] = 0.02
            return [item]

    generator = WorkingGenerator()
    engine = QueryEngine(FullTextRetriever(), generator)

    answer = engine.query("exact term", retrieval_mode="full_text")

    assert answer.text == "answer"
    assert generator.calls


def test_query_refuses_low_evidence_before_generation():
    class LowEvidenceRetriever(FakeRetriever):
        def retrieve(self, *args, **kwargs):
            item = dict(CHUNKS[0])
            item["rerank_score"] = 0.01
            return [item]

    generator = WorkingGenerator()
    engine = QueryEngine(LowEvidenceRetriever(), generator)

    answer = engine.query("weak match")

    assert "未找到足够可靠的信息" in answer.text
    assert answer.citations == []
    assert generator.calls == []

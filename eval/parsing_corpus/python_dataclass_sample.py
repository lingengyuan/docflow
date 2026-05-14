from dataclasses import dataclass


@dataclass
class KnowledgeRecord:
    title: str
    source_id: str


def dataclass_parser_evidence(record: KnowledgeRecord) -> str:
    return f"{record.title}:{record.source_id}"

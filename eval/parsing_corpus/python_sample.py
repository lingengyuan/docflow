from pathlib import Path


def build_knowledge_card(source: Path) -> dict[str, str]:
    return {"title": source.stem, "evidence": "python parser evidence"}

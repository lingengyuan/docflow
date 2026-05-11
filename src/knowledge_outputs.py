"""Knowledge output templates shared by API, imports, and generation."""

from __future__ import annotations

from dataclasses import dataclass

KNOWLEDGE_OUTPUT_SOURCE_CHAR_LIMIT = 12_000


@dataclass(frozen=True)
class KnowledgeOutputType:
    id: str
    label: str
    instruction: str


KNOWLEDGE_OUTPUT_TYPES: dict[str, KnowledgeOutputType] = {
    "summary": KnowledgeOutputType(
        id="summary",
        label="结构化总结",
        instruction=(
            "把资料整理成 Markdown。必须包含：## 一句话结论、## 核心要点、"
            "## 关键细节、## 可继续追问。核心要点使用短列表，不要加入资料外的信息。"
        ),
    ),
    "learning_cards": KnowledgeOutputType(
        id="learning_cards",
        label="学习卡片",
        instruction=(
            "把资料整理成适合复习的 Markdown 学习卡片。每张卡片包含问题、答案、"
            "适用场景或易错点。只基于资料内容生成。"
        ),
    ),
    "action_items": KnowledgeOutputType(
        id="action_items",
        label="行动项",
        instruction=(
            "把资料整理成 Markdown 行动清单。必须包含：## 待办事项、## 风险或阻塞、"
            "## 建议下一步。每个行动项说明原因，不能凭空指定负责人。"
        ),
    ),
    "project_brief": KnowledgeOutputType(
        id="project_brief",
        label="项目简报",
        instruction=(
            "把资料整理成 Markdown 项目简报。必须包含：## 背景、## 当前状态、"
            "## 已确定事项、## 风险、## 下一步。用简洁中文表达。"
        ),
    ),
}


def normalize_knowledge_output_type(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def get_knowledge_output_type(value: str) -> KnowledgeOutputType:
    output_type = normalize_knowledge_output_type(value)
    try:
        return KNOWLEDGE_OUTPUT_TYPES[output_type]
    except KeyError as exc:
        allowed = ", ".join(sorted(KNOWLEDGE_OUTPUT_TYPES))
        raise ValueError(f"Unknown knowledge output type: {value}. Allowed: {allowed}") from exc


def knowledge_output_tags(output_type: str, user_tags: list[str] | None = None) -> list[str]:
    normalized = get_knowledge_output_type(output_type).id
    tags = ["knowledge-output", normalized]
    for tag in user_tags or []:
        if tag not in tags:
            tags.append(tag)
    return tags

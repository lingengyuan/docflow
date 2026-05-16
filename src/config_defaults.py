"""Default configuration values shared by typed settings and config templates."""

from __future__ import annotations

from typing import Any

TABLE_KEYWORDS = {
    "表格",
    "数据",
    "统计",
    "总计",
    "合计",
    "金额",
    "数量",
    "比例",
    "table",
    "data",
    "total",
    "sum",
    "amount",
    "count",
    "ratio",
    "percent",
}

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "在现有文档中未找到足够可靠的信息。请扩大提问范围、换个问法，或确认相关文件已经完成入库。"
)

DEFAULT_CONFIG_SECTIONS: dict[str, dict[str, Any]] = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "ocr_model": "glm-ocr",
        "llm_model": "qwen2.5:7b",
        "llm_model_enhanced": "qwen3:8b",
    },
    "embedding": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "backend": "torch",
        "batch_size": 32,
    },
    "chunking": {
        "chunk_size": 512,
        "chunk_overlap": 51,
        "ocr_text_threshold": 0.10,
    },
    "ingest": {},
    "query": {},
    "privacy": {"allow_model_download": False, "allowed_hosts": []},
    "llm": {},
    "reranker": {},
}

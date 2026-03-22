"""
MarkdownParser — 解析 .md 文件，清洗 Obsidian 语法后作为单页文档返回。

清洗规则（按 obsidian-markdown 语法规范）：
  - YAML frontmatter: strip from text, 提取 tags/aliases 存入 metadata
  - [[wikilink]]: → 保留文字
  - ![[embed]]: strip
  - %%隐藏注释%%: strip
  - > [!type] callout: 去标记，保留内容
  - #inline-tag（非标题）: 提取到 tags，保留文字
  - ^block-id: strip
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.ingest.pdf_analyzer import PageContent, ParsedDocument

# Pre-compiled patterns
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)---\n?", re.DOTALL)
_COMMENT_RE = re.compile(r"%%.*?%%", re.DOTALL)
_EMBED_RE = re.compile(r"!\[\[.+?\]\]")
_WIKILINK_RE = re.compile(r"\[\[([^|\]]+?)(?:\|([^\]]+?))?\]\]")
_CALLOUT_RE = re.compile(r"^(>\s*)\[!(\w+)\]\s*(.*)", re.MULTILINE)
_BLOCK_ID_RE = re.compile(r"\s+\^[\w-]+\s*$", re.MULTILINE)
_INLINE_TAG_RE = re.compile(r"(?<!\w)#([\w\u4e00-\u9fff][\w\u4e00-\u9fff/-]*)")
_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
# Markdown anchor link: [text](#anchor) or (#anchor)
_MD_ANCHOR_RE = re.compile(r"\(#[\w\u4e00-\u9fff-]+\)")
# Hex color code: #FFF, #F7F5F0 (3/4/6/8 hex digits)
_HEX_COLOR_RE = re.compile(r"^[0-9A-Fa-f]{3,8}$")


def _extract_inline_tags(text: str) -> list[str]:
    """提取行内 #tag（排除标题行、锚链接、hex 颜色码、URL fragment）。"""
    # 先把 markdown 锚链接替换掉，防止 (#anchor) 被匹配为 tag
    cleaned = _MD_ANCHOR_RE.sub("", text)
    tags: list[str] = []
    for line in cleaned.splitlines():
        if _HEADING_RE.match(line):
            continue
        for m in _INLINE_TAG_RE.finditer(line):
            tag = m.group(1)
            end = m.end()
            # 排除 URL fragment: #page=1, #section=foo
            if end < len(line) and line[end] == "=":
                continue
            # 排除 hex 颜色码: #FFF, #F7F5F0
            if _HEX_COLOR_RE.match(tag):
                continue
            if tag not in tags:
                tags.append(tag)
    return tags


class MarkdownParser:
    def parse(self, file_path: Path) -> ParsedDocument:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        tags: list[str] = []
        metadata: dict = {}

        # 1. 分离 YAML frontmatter
        fm_match = _FRONTMATTER_RE.match(raw)
        if fm_match:
            try:
                fm = yaml.safe_load(fm_match.group(1)) or {}
            except yaml.YAMLError:
                fm = {}
            fm_tags = fm.get("tags", [])
            if isinstance(fm_tags, list):
                tags.extend(str(t) for t in fm_tags)
            elif isinstance(fm_tags, str):
                tags.append(fm_tags)
            if fm.get("aliases"):
                metadata["aliases"] = fm["aliases"]
            raw = raw[fm_match.end():]

        # 2. 提取行内 #tags（在清洗前）
        inline_tags = _extract_inline_tags(raw)
        for t in inline_tags:
            if t not in tags:
                tags.append(t)

        # 3. 清洗 Obsidian 语法
        text = raw
        text = _COMMENT_RE.sub("", text)             # %%comments%%
        text = _EMBED_RE.sub("", text)                # ![[embeds]]
        text = _WIKILINK_RE.sub(                      # [[links]]
            lambda m: m.group(2) or m.group(1), text
        )
        text = _CALLOUT_RE.sub(r"\1\3", text)         # > [!type] title
        text = _BLOCK_ID_RE.sub("", text)             # ^block-id

        # 去除多余空行
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        metadata["tags"] = tags

        page = PageContent(page_num=1, text=text, headers=[], is_ocr=False)
        return ParsedDocument(
            file_path=file_path,
            file_name=file_path.name,
            total_pages=1,
            is_scanned=False,
            pages=[page],
            metadata=metadata,
        )

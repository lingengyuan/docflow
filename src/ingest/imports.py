"""Local import helpers for webpages and notes."""

from __future__ import annotations

import html
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

from src import net
from src.knowledge_outputs import get_knowledge_output_type, knowledge_output_tags

MAX_WEBPAGE_BYTES = 2_000_000
REQUEST_TIMEOUT_S = 15


@dataclass(frozen=True)
class MarkdownImport:
    title: str
    markdown: str


class ReadableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._parts: list[str] = []
        self._link_href: str | None = None

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "section", "article", "header", "footer", "br", "li", "tr"}:
            self._parts.append("\n")
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            self._parts.append("\n" + "#" * min(level + 1, 6) + " ")
        if tag == "li":
            self._parts.append("- ")
        if tag == "a":
            attrs_dict = dict(attrs)
            self._link_href = attrs_dict.get("href")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if self._skip_depth and tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag == "a":
            self._link_href = None
        if tag in {
            "p",
            "div",
            "section",
            "article",
            "li",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            self._parts.append("\n")

    def handle_data(self, data: str):
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title = (self.title + " " + text).strip()
            return
        if self._skip_depth:
            return
        if self._link_href:
            self._parts.append(f"[{text}]({self._link_href}) ")
        else:
            self._parts.append(text + " ")

    def markdown_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def fetch_webpage_markdown(url: str, title: str | None = None) -> MarkdownImport:
    normalized_url = _validate_url(url)
    content_type, raw = _fetch_webpage_bytes(normalized_url)
    if len(raw) > MAX_WEBPAGE_BYTES:
        raise ValueError("Webpage is too large to import")

    charset = _charset_from_content_type(content_type)
    decoded = raw.decode(charset, errors="replace")
    if "html" in content_type.lower() or "<html" in decoded[:1000].lower():
        parsed_title, body = html_to_markdown(decoded)
    else:
        parsed_title, body = "", decoded.strip()

    final_title = _clean_title(
        title or parsed_title or urllib.parse.urlparse(normalized_url).netloc
    )
    markdown = (
        _frontmatter(final_title, source_url=normalized_url, tags=["web-import"])
        + f"# {final_title}\n\n"
        + f"Source: {normalized_url}\n\n"
        + body.strip()
        + "\n"
    )
    return MarkdownImport(title=final_title, markdown=markdown)


def _fetch_webpage_bytes(url: str) -> tuple[str, bytes]:
    try:
        with net.Client(
            follow_redirects=True,
            timeout=net.Timeout(REQUEST_TIMEOUT_S, connect=5.0),
            headers={"User-Agent": "DocFlow/1.0 local knowledge import"},
            allow_external=True,
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.headers.get("content-type", ""), response.content[
                : MAX_WEBPAGE_BYTES + 1
            ]
    except net.ConnectTimeout as exc:
        raise TimeoutError("Timed out while connecting to webpage") from exc
    except net.ReadTimeout as exc:
        raise TimeoutError("Timed out while reading webpage") from exc
    except net.HTTPStatusError as exc:
        raise RuntimeError(f"Webpage returned HTTP {exc.response.status_code}") from exc
    except net.RequestError as exc:
        raise RuntimeError(f"Failed to fetch webpage: {exc}") from exc


def html_to_markdown(raw_html: str) -> tuple[str, str]:
    parser = ReadableHTMLParser()
    parser.feed(raw_html)
    title = _clean_title(html.unescape(parser.title))
    body = parser.markdown_text()
    return title, body


def build_quick_note_markdown(
    title: str, content: str, tags: list[str] | None = None
) -> MarkdownImport:
    final_title = _clean_title(title or "Untitled Note")
    body = content.strip()
    if not body:
        raise ValueError("Note content is empty")
    markdown = _frontmatter(final_title, tags=tags or ["note"]) + f"# {final_title}\n\n{body}\n"
    return MarkdownImport(title=final_title, markdown=markdown)


def build_answer_note_markdown(
    title: str | None,
    answer: str,
    question: str | None = None,
    citations: list[dict] | None = None,
    tags: list[str] | None = None,
) -> MarkdownImport:
    final_title = _clean_title(title or "Saved Answer")
    answer_text = answer.strip()
    if not answer_text:
        raise ValueError("Answer content is empty")
    parts = [_frontmatter(final_title, tags=tags or ["answer"]), f"# {final_title}\n"]
    if question and question.strip():
        parts.append(f"## Question\n\n{question.strip()}\n")
    parts.append(f"## Answer\n\n{answer_text}\n")
    citation_rows = _format_citations(citations or [])
    if citation_rows:
        parts.append("## Sources\n\n" + citation_rows + "\n")
    return MarkdownImport(title=final_title, markdown="\n".join(parts))


def build_knowledge_output_markdown(
    title: str | None,
    output_type: str,
    body: str,
    source_files: list[str] | None = None,
    tags: list[str] | None = None,
) -> MarkdownImport:
    output = get_knowledge_output_type(output_type)
    final_title = _clean_title(title or output.label)
    body_text = body.strip()
    if not body_text:
        raise ValueError("Knowledge output content is empty")
    clean_sources = [
        " ".join(str(item).split()) for item in (source_files or []) if str(item).strip()
    ]
    parts = [
        _frontmatter(
            final_title,
            tags=knowledge_output_tags(output.id, tags),
            extra={"output_type": output.id},
        ),
        f"# {final_title}\n",
        f"类型：{output.label}\n",
    ]
    if clean_sources:
        parts.append("## 来源\n\n" + "\n".join(f"- {source}" for source in clean_sources) + "\n")
    parts.append(f"## 内容\n\n{body_text}\n")
    return MarkdownImport(title=final_title, markdown="\n".join(parts))


def write_markdown_import(root_dir: Path, prefix: str, item: MarkdownImport) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(item.title, fallback=prefix)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root_dir / f"{prefix}-{timestamp}-{stem}.md"
    path.write_text(item.markdown, encoding="utf-8")
    return path


def safe_filename(value: str, fallback: str = "note") -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", value.strip(), flags=re.UNICODE)
    text = re.sub(r"-{2,}", "-", text).strip(".-").lower()
    return (text or fallback)[:80]


def _validate_url(url: str) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Only http and https URLs can be imported")
    return urllib.parse.urlunparse(parsed)


def _charset_from_content_type(content_type: str) -> str:
    match = re.search(r"charset=([\w.-]+)", content_type, re.I)
    return match.group(1) if match else "utf-8"


def _clean_title(value: str) -> str:
    title = " ".join(html.unescape(str(value or "")).split())
    return title[:120] if title else "Untitled"


def _frontmatter(
    title: str,
    source_url: str | None = None,
    tags: list[str] | None = None,
    extra: dict[str, str] | None = None,
) -> str:
    lines = [
        "---",
        f"title: {_yaml_string(title)}",
        f"created: {datetime.now().isoformat(timespec='seconds')}",
    ]
    if source_url:
        lines.append(f"source_url: {_yaml_string(source_url)}")
    for key, value in (extra or {}).items():
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            lines.append(f"{key}: {_yaml_string(value)}")
    clean_tags = [safe_filename(tag, fallback="tag") for tag in (tags or []) if str(tag).strip()]
    if clean_tags:
        lines.append("tags:")
        lines.extend(f"  - {tag}" for tag in clean_tags)
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _yaml_string(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _format_citations(citations: list[dict]) -> str:
    rows: list[str] = []
    for citation in citations:
        file_name = citation.get("file_name") or "source"
        section = citation.get("section") or ""
        page = citation.get("page_num")
        suffix = f" p.{page}" if page else ""
        if section:
            suffix += f" / {section}"
        rows.append(f"- {file_name}{suffix}")
    return "\n".join(rows)

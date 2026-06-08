"""
Document cleaning primitives used before chunking.

The first pass is intentionally rule-based and conservative: normalize text,
remove obvious page noise, and keep page/section metadata stable for downstream
chunking and retrieval.
"""

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class CleanedImage:
    image_path: str
    caption: str = ""
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanedSection:
    heading: str
    level: int
    text: str
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanedPage:
    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanedDocument:
    title: str
    sections: list[CleanedSection] = field(default_factory=list)
    pages: list[CleanedPage] = field(default_factory=list)
    images: list[CleanedImage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_PAGE_NUMBER = re.compile(
    r"^(?:[-–—]?\s*)?(?:第\s*)?\d{1,4}\s*(?:页|/|\-|\||of\s+\d{1,4})?\s*$",
    re.IGNORECASE,
)
_PAGE_FRACTION = re.compile(r"^(?:p\.?\s*)?\d{1,4}\s*(?:/|／|共)\s*\d{1,4}\s*(?:页)?$", re.IGNORECASE)
_SPACES = re.compile(r"[ \t\u3000]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_CJK_SENTENCE_END = "。！？；：”’）》】』」〉…"
_MARKDOWN_PREFIX = re.compile(r"^\s{0,3}(?:#{1,6}\s*|[-*+]\s+|\|+|>+\s*)")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EDGE_PAGE_DIGITS = re.compile(r"(?:(?<=\s)|(?<=[/／\-–—|｜]))\d{1,4}$")
_NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*[-_=—–]{3,}\s*$"),
    re.compile(r"^\s*(?:copyright|©|版权所有|责任编辑|编辑|校对|下载|转载请|文章来源|来源[:：])", re.IGNORECASE),
    re.compile(r"(?:https?://|www\.|doi\.org|\.com|\.cn|\.org|\.net)", re.IGNORECASE),
]
_HTML_FOOTER_START_PATTERNS = [
    re.compile(r"^\s*(?:网站地图|友情链接|浏览建议|欢迎关注|网站访问量)\s*[:：]?\s*$"),
    re.compile(r"^\s*(?:上一篇|下一篇)\s*[:：]?\s*"),
    re.compile(r"^\s*[-*]?\s*\[?(?:首页|版权声明|联系我们)\]?", re.IGNORECASE),
]
HEADER_FOOTER_WINDOW = 5
HEADER_FOOTER_MIN_PAGES = 2
HEADER_FOOTER_MIN_COVERAGE = 0.35


def _normalize_line(line: str) -> str:
    line = _CONTROL_CHARS.sub("", line)
    line = _SPACES.sub(" ", line)
    return line.strip()


def _is_page_number(line: str) -> bool:
    stripped = line.strip()
    return bool(_PAGE_NUMBER.match(stripped) or _PAGE_FRACTION.match(stripped))


def normalize_noise_line(line: str) -> str:
    """Normalize a line before repeated header/footer detection."""
    text = _normalize_line(line).lower()
    text = _MARKDOWN_PREFIX.sub("", text)
    text = _URL.sub("", text)
    if _is_page_number(text) or re.search(r"(?:页|page|p\.)", text, re.IGNORECASE):
        text = re.sub(r"\d+", "#", text)
    else:
        text = _EDGE_PAGE_DIGITS.sub("#", text)
    text = re.sub(r"[·•\-–—_=|｜:：,，.。;；()\[\]【】（）<>《》\s]+", "", text)
    return text.strip()


def _is_obvious_noise_line(line: str) -> bool:
    stripped = _normalize_line(line)
    if not stripped:
        return False
    if _is_page_number(stripped):
        return True
    return any(pattern.search(stripped) for pattern in _NOISE_LINE_PATTERNS)


def _is_heading_like(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(("#", "第")) and len(stripped) <= 40:
        return True
    return bool(re.match(r"^\d+(?:\.\d+){0,3}\s+\S+", stripped)) and len(stripped) <= 60


def _should_merge_lines(current: str, nxt: str) -> bool:
    if not current or not nxt:
        return False
    if current.endswith(_CJK_SENTENCE_END):
        return False
    if _is_heading_like(current) or _is_heading_like(nxt):
        return False
    if re.match(r"^[-*•]\s+", nxt):
        return False
    return len(current) + len(nxt) <= 140


def _merge_broken_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines:
        if not line:
            if merged and merged[-1]:
                merged.append("")
            continue
        if merged and merged[-1] and _should_merge_lines(merged[-1], line):
            merged[-1] = f"{merged[-1]}{line}"
        else:
            merged.append(line)
    return merged


def clean_text(text: str) -> str:
    """Clean a single text block while preserving paragraph boundaries."""
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [_normalize_line(line) for line in raw_lines]
    lines = [line for line in lines if not _is_obvious_noise_line(line)]
    lines = _merge_broken_lines(lines)
    cleaned = "\n".join(lines)
    cleaned = _BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()


def _is_html_footer_start(line: str) -> bool:
    stripped = _normalize_line(line)
    return any(pattern.search(stripped) for pattern in _HTML_FOOTER_START_PATTERNS)


def _truncate_html_footer(text: str, metadata: dict[str, Any]) -> str:
    parser = str(metadata.get("parser") or "").lower()
    source_url = str(metadata.get("source_url") or metadata.get("url") or "").lower()
    is_html = parser == "html_fallback" or source_url.endswith((".htm", ".html"))
    if not is_html:
        return text

    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        if _is_html_footer_start(line):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def _candidate_edge_lines(text: str, window: int = HEADER_FOOTER_WINDOW) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    candidates = lines[:window]
    tail = lines[-window:]
    for line in tail:
        if line not in candidates:
            candidates.append(line)
    return candidates


def _find_repeated_page_edges(page_texts: list[str]) -> set[str]:
    non_empty_pages = [
        (page_index, text)
        for page_index, text in enumerate(page_texts)
        if text.strip()
    ]
    page_count = len(non_empty_pages)
    if page_count < HEADER_FOOTER_MIN_PAGES:
        return set()

    occurrences: dict[str, set[int]] = {}
    for page_index, text in non_empty_pages:
        seen_on_page: set[str] = set()
        for line in _candidate_edge_lines(text):
            normalized = normalize_noise_line(line)
            if not normalized or len(normalized) < 4:
                continue
            seen_on_page.add(normalized)
        for normalized in seen_on_page:
            occurrences.setdefault(normalized, set()).add(page_index)

    threshold = max(
        HEADER_FOOTER_MIN_PAGES,
        int(page_count * HEADER_FOOTER_MIN_COVERAGE + 0.999),
    )
    return {
        normalized
        for normalized, pages in occurrences.items()
        if len(pages) >= threshold
    }


def _remove_repeated_edges(text: str, repeated: set[str]) -> str:
    if not repeated:
        return text
    lines = text.splitlines()
    non_empty_indexes = [idx for idx, line in enumerate(lines) if line.strip()]
    edge_indexes = set(non_empty_indexes[:HEADER_FOOTER_WINDOW] + non_empty_indexes[-HEADER_FOOTER_WINDOW:])

    filtered: list[str] = []
    for idx, line in enumerate(lines):
        normalized = normalize_noise_line(line)
        if idx in edge_indexes and normalized in repeated:
            continue
        if _is_obvious_noise_line(line):
            continue
        filtered.append(line)
    lines = filtered
    return "\n".join(lines).strip()


def clean_document_pages(
    pages: list[dict[str, Any]],
    title: str = "",
    metadata: dict[str, Any] | None = None,
) -> CleanedDocument:
    """Clean ordered page text dictionaries into a CleanedDocument."""
    preliminary: list[CleanedPage] = []
    for index, page in enumerate(pages):
        text = clean_text(str(page.get("text") or ""))
        page_number = int(page.get("page_number", index) or 0)
        page_meta = dict(page.get("metadata") or {})
        preliminary.append(CleanedPage(page_number=page_number, text=text, metadata=page_meta))

    repeated = _find_repeated_page_edges([page.text for page in preliminary])
    cleaned_pages: list[CleanedPage] = []
    for page in preliminary:
        text = _remove_repeated_edges(page.text, repeated)
        text = _truncate_html_footer(text, page.metadata)
        if text:
            cleaned_pages.append(CleanedPage(
                page_number=page.page_number,
                text=text,
                metadata={**page.metadata, "cleaner": "rule_based_v1"},
            ))

    sections = [
        CleanedSection(
            heading=f"Page {page.page_number}",
            level=1,
            text=page.text,
            page_number=page.page_number,
            metadata=page.metadata,
        )
        for page in cleaned_pages
    ]
    return CleanedDocument(
        title=title,
        pages=cleaned_pages,
        sections=sections,
        metadata=metadata or {},
    )

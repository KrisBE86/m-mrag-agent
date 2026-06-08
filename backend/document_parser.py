"""
Document parsing layer.

Docling is the primary parser for PDF/Word/HTML sources. It converts messy
documents into Markdown pages before the cleaner and existing L1/L2/L3 chunking
pipeline run.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.url_ingestor import html_file_to_text


class DocumentParseError(RuntimeError):
    """Raised when a document cannot be parsed into page text."""


@dataclass
class ParsedPage:
    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    title: str
    pages: list[ParsedPage]
    markdown: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _call_export_to_markdown(document, **kwargs) -> str:
    try:
        return document.export_to_markdown(**kwargs) or ""
    except TypeError:
        compatible = {
            key: value
            for key, value in kwargs.items()
            if key in {"page_no", "page_break_placeholder"}
        }
        return document.export_to_markdown(**compatible) or ""


def _docling_num_pages(document) -> int:
    value = getattr(document, "num_pages", None)
    if callable(value):
        try:
            return int(value())
        except Exception:
            pass
    if isinstance(value, int):
        return value

    pages = getattr(document, "pages", None)
    if isinstance(pages, dict):
        return len(pages)
    if isinstance(pages, list):
        return len(pages)
    return 1


def parse_document_with_docling(
    file_path: str | Path,
    filename: str | None = None,
    source_url: str | None = None,
) -> ParsedDocument:
    """Parse a local document with Docling and return Markdown by page."""
    path = Path(file_path)
    title = filename or path.name

    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise DocumentParseError(
            "Docling 未安装，请先安装依赖: pip install docling 或 uv sync"
        ) from exc

    try:
        converter = DocumentConverter()
        result = converter.convert(path)
        document = result.document
    except Exception as exc:
        raise DocumentParseError(f"Docling 解析失败: {exc}") from exc

    full_markdown = _call_export_to_markdown(
        document,
        page_break_placeholder="\n\n<!-- page-break -->\n\n",
        traverse_pictures=True,
    ).strip()

    pages: list[ParsedPage] = []
    num_pages = max(_docling_num_pages(document), 1)
    for page_no in range(1, num_pages + 1):
        page_markdown = _call_export_to_markdown(
            document,
            page_no=page_no,
            page_break_placeholder=None,
            traverse_pictures=True,
        ).strip()
        if not page_markdown:
            continue
        pages.append(ParsedPage(
            page_number=page_no,
            text=page_markdown,
            metadata={
                "page": page_no,
                "parser": "docling",
                "source_url": source_url or "",
            },
        ))

    if not pages and full_markdown:
        pages.append(ParsedPage(
            page_number=0,
            text=full_markdown,
            metadata={
                "page": 0,
                "parser": "docling",
                "source_url": source_url or "",
            },
        ))

    return ParsedDocument(
        title=title,
        pages=pages,
        markdown=full_markdown,
        metadata={
            "filename": title,
            "parser": "docling",
            "source_url": source_url or "",
        },
    )


def parse_html_without_docling(
    file_path: str | Path,
    filename: str | None = None,
    source_url: str | None = None,
) -> ParsedDocument:
    """Fallback parser for HTML when Docling is unavailable."""
    path = Path(file_path)
    text, title = html_file_to_text(path)
    display_title = title or filename or path.name
    return ParsedDocument(
        title=display_title,
        pages=[
            ParsedPage(
                page_number=0,
                text=text,
                metadata={
                    "page": 0,
                    "parser": "stdlib_html",
                    "source_url": source_url or "",
                },
            )
        ] if text.strip() else [],
        markdown=text,
        metadata={
            "filename": filename or path.name,
            "parser": "stdlib_html",
            "source_url": source_url or "",
        },
    )


def parse_document(
    file_path: str | Path,
    filename: str | None = None,
    source_url: str | None = None,
    fallback_html: bool = True,
) -> ParsedDocument:
    """Parse a document, using Docling as the primary parser."""
    path = Path(file_path)
    try:
        return parse_document_with_docling(path, filename=filename, source_url=source_url)
    except DocumentParseError:
        if fallback_html and path.suffix.lower() in {".html", ".htm"}:
            return parse_html_without_docling(path, filename=filename, source_url=source_url)
        raise

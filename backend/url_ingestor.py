"""
URL ingestion utilities.

This module only turns a remote URL into a local file plus source metadata. It
does not parse, clean, chunk, embed, or write to vector stores.
"""

from dataclasses import dataclass
from html.parser import HTMLParser
import hashlib
import ipaddress
import os
from pathlib import Path
import re
import socket
from urllib.parse import urljoin, urlparse

import httpx


DEFAULT_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".html", ".htm"}


class URLIngestError(ValueError):
    """Raised when a URL cannot be safely downloaded for ingestion."""


@dataclass
class DownloadedURLDocument:
    source_url: str
    final_url: str
    file_path: str
    filename: str
    content_type: str
    size_bytes: int


def _is_blocked_host(hostname: str) -> bool:
    host = hostname.strip().lower().rstrip(".")
    if host in {"localhost"} or host.endswith(".localhost"):
        return True

    try:
        ip = ipaddress.ip_address(host)
        return not ip.is_global
    except ValueError:
        pass

    try:
        addr_infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise URLIngestError(f"无法解析 URL 主机: {hostname}") from exc

    for info in addr_infos:
        ip_text = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_text)
        except ValueError:
            return True
        if not ip.is_global:
            return True
    return False


def validate_public_http_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise URLIngestError("仅支持 http/https URL")
    if not parsed.hostname:
        raise URLIngestError("URL 缺少主机名")
    if parsed.username or parsed.password:
        raise URLIngestError("URL 不允许包含用户名或密码")
    if _is_blocked_host(parsed.hostname):
        raise URLIngestError("URL 指向本机、内网或非公网地址，已拒绝")
    return parsed.geturl()


def _extension_from_content_type(content_type: str) -> str:
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime == "application/pdf":
        return ".pdf"
    if mime in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/x-docx",
    }:
        return ".docx"
    if mime in {"application/msword", "application/doc"}:
        return ".doc"
    if mime in {"text/html", "application/xhtml+xml"}:
        return ".html"
    return ""


def _limit_text(value: str, max_chars: int) -> str:
    value = (value or "").strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip("._-")


def _safe_filename_from_url(url: str, content_type: str) -> str:
    parsed = urlparse(url)
    raw_name = Path(parsed.path).name
    suffix = Path(raw_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        suffix = _extension_from_content_type(content_type)
    if suffix not in SUPPORTED_EXTENSIONS:
        raise URLIngestError(f"不支持的 URL 内容类型: {content_type or '未知'}")

    stem = Path(raw_name).stem if raw_name else parsed.hostname or "document"
    stem = re.sub(r"[^\w.-]+", "_", stem).strip("._") or "document"
    host = re.sub(r"[^\w.-]+", "_", parsed.hostname or "source")
    stem = _limit_text(stem, 80) or "document"
    host = _limit_text(host, 60) or "source"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"url_{host}_{stem}_{digest}{suffix}"


def _raise_for_large_response(response: httpx.Response, max_bytes: int) -> None:
    content_length = response.headers.get("content-length")
    if not content_length:
        return
    try:
        size = int(content_length)
    except ValueError:
        return
    if size > max_bytes:
        raise URLIngestError(f"URL 文件过大，最大允许 {max_bytes // (1024 * 1024)}MB")


def download_url_document(
    url: str,
    output_dir: str | Path,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = 5,
) -> DownloadedURLDocument:
    """Safely download a public URL to a local file for ingestion."""
    current_url = validate_public_http_url(url)
    original_url = current_url
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "MRagAgent-URLIngestor/1.0",
        "Accept": "text/html,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,*/*;q=0.8",
    }

    with httpx.Client(timeout=timeout, headers=headers) as client:
        for _ in range(max_redirects + 1):
            with client.stream("GET", current_url, follow_redirects=False) as response:
                if 300 <= response.status_code < 400:
                    location = response.headers.get("location")
                    if not location:
                        raise URLIngestError("URL 重定向缺少 Location")
                    current_url = validate_public_http_url(urljoin(current_url, location))
                    continue

                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                filename = _safe_filename_from_url(current_url, content_type)
                file_path = output_dir / filename
                tmp_path = file_path.with_suffix(file_path.suffix + ".download")

                _raise_for_large_response(response, max_bytes)
                size = 0
                with tmp_path.open("wb") as f:
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > max_bytes:
                            tmp_path.unlink(missing_ok=True)
                            raise URLIngestError(f"URL 文件过大，最大允许 {max_bytes // (1024 * 1024)}MB")
                        f.write(chunk)

                os.replace(tmp_path, file_path)
                return DownloadedURLDocument(
                    source_url=original_url,
                    final_url=current_url,
                    file_path=str(file_path),
                    filename=filename,
                    content_type=content_type,
                    size_bytes=size,
                )

    raise URLIngestError("URL 重定向次数过多")


class ReadableHTMLParser(HTMLParser):
    """A small stdlib HTML-to-text parser for fallback HTML ingestion."""

    BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
        "dt", "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5",
        "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
        "section", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
            return
        self._parts.append(text)
        self._parts.append(" ")

    def text(self) -> str:
        raw = "".join(self._parts)
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]{2,}", " ", raw)
        return raw.strip()


def html_file_to_text(file_path: str | Path) -> tuple[str, str]:
    """Extract readable text and title from a downloaded HTML file."""
    path = Path(file_path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    parser = ReadableHTMLParser()
    parser.feed(raw)
    return parser.text(), parser.title

#!/usr/bin/env python
"""
Preview URL document cleaning without writing PostgreSQL, Milvus, or BM25.

Usage:
    uv run python scripts/preview_clean_url.py "https://example.com/file.pdf"
    uv run python scripts/preview_clean_url.py "https://example.com/file.pdf" --with-vlm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.document_cleaner import clean_document_pages
from backend.document_loader import (
    IMAGE_PLACEHOLDER_PATTERN,
    _caption_after_image_placeholder,
    _describe_image_with_vlm,
    _extract_captions,
    _extract_images_from_html,
    _extract_images_from_docx,
    _extract_images_from_pdf,
    _inject_image_descriptions_into_page_text,
    _name_image_from_captions,
    _save_image,
)
from backend.document_parser import parse_document
from backend.url_ingestor import download_url_document


DATA_DIR = ROOT_DIR / "data"
DOCUMENT_DIR = DATA_DIR / "documents"
PREVIEW_DIR = DATA_DIR / "cleaned_previews"
REFERENCE_IMAGE_DIR = DATA_DIR / "reference_images"


def _extract_images(file_path: str, filename: str, source_url: str) -> list[dict]:
    file_lower = filename.lower()
    if file_lower.endswith(".pdf"):
        return _extract_images_from_pdf(file_path)
    if file_lower.endswith((".docx", ".doc")):
        return _extract_images_from_docx(file_path)
    if file_lower.endswith((".html", ".htm")):
        return _extract_images_from_html(file_path, source_url=source_url)
    return []


def _cleanup_preview_reference_images(filename: str) -> None:
    safe_filename = re.sub(r"[^\w\-\.]", "_", filename)
    for path in REFERENCE_IMAGE_DIR.glob(f"{safe_filename}_img*"):
        if path.is_file():
            path.unlink()


def _images_for_page(
    images_by_page: dict[int, list[dict]],
    page_num: int,
    page_count: int,
) -> list[dict]:
    page_images = images_by_page.get(page_num, [])
    if not page_images and page_num == 0:
        page_images = images_by_page.get(0, [])
    if not page_images and page_count == 1:
        page_images = [img for items in images_by_page.values() for img in items]
    return page_images


def _format_image_placeholder(record: dict) -> str:
    label = f"图{int(record.get('image_index', 0)) + 1}"
    name = (record.get("poi_name") or "").strip()
    caption = (record.get("caption") or "").strip()
    path = (record.get("image_path") or "").strip()

    heading = f"[图片占位：{label}"
    if name:
        heading += f"｜{name}"
    heading += "]"

    lines = [heading]
    if caption and caption != name:
        lines.append(f"题注：{caption}")
    if path:
        lines.append(f"图片文件：{path}")
    return "\n".join(lines)


def _inject_image_placeholders_into_page_text(page_text: str, image_records: list[dict]) -> str:
    text = (page_text or "").strip()
    pending_blocks: list[str] = []
    for record in image_records:
        block = _format_image_placeholder(record)
        text, replaced = IMAGE_PLACEHOLDER_PATTERN.subn(block, text, count=1)
        if not replaced:
            pending_blocks.append(block)

    if pending_blocks:
        appendix = "\n\n".join(["## 本页图片占位", *pending_blocks])
        text = f"{text}\n\n{appendix}" if text else appendix
    return text.strip()


def build_preview_text(
    file_path: str,
    filename: str,
    source_url: str,
    with_vlm: bool = False,
) -> str:
    parsed_doc = parse_document(file_path, filename=filename, source_url=source_url)
    cleaned_doc = clean_document_pages(
        [
            {
                "text": page.text,
                "page_number": page.page_number,
                "metadata": page.metadata,
            }
            for page in parsed_doc.pages
        ],
        title=filename,
        metadata={
            "filename": filename,
            "source_url": source_url,
            "parser": parsed_doc.metadata.get("parser", "docling"),
        },
    )
    cleaned_pages_by_number = {page.page_number: page for page in cleaned_doc.pages}

    raw_images = _extract_images(file_path, filename, source_url)
    images_by_page: dict[int, list[dict]] = {}
    for img in raw_images:
        images_by_page.setdefault(int(img.get("page_number", 0)), []).append(img)

    output_parts: list[str] = []
    saved_image_counter = 0
    for page in parsed_doc.pages:
        page_num = page.page_number
        cleaned_page = cleaned_pages_by_number.get(page_num)
        page_text = (cleaned_page.text if cleaned_page else "").strip()
        captions = _extract_captions(page_text)

        image_records: list[dict] = []
        for img_data in _images_for_page(images_by_page, page_num, len(parsed_doc.pages)):
            img_idx = int(img_data["image_index"])
            saved_index = saved_image_counter
            saved_image_counter += 1
            saved_path = _save_image(
                img_data["image_bytes"],
                REFERENCE_IMAGE_DIR,
                filename,
                saved_index,
                img_data.get("format", "png"),
            )
            if not saved_path:
                continue

            name, caption = _name_image_from_captions(img_idx, page_num, page_text, captions)
            html_label = (img_data.get("alt") or img_data.get("title") or "").strip()
            if html_label:
                name = html_label
                caption = html_label
            elif not caption:
                marker_caption = _caption_after_image_placeholder(page_text, len(image_records))
                if marker_caption:
                    name = marker_caption
                    caption = marker_caption
            if not name or name.startswith(f"第{page_num}页图片"):
                name = page_text[:80] if page_text else f"第{page_num}页图片{img_idx + 1}"

            vlm_description = ""
            if with_vlm:
                vlm_description = _describe_image_with_vlm(Path(saved_path).read_bytes())

            image_records.append({
                "image_index": img_idx,
                "image_path": saved_path,
                "poi_name": name,
                "caption": caption,
                "vlm_description": vlm_description,
            })

        if with_vlm:
            page_text = _inject_image_descriptions_into_page_text(page_text, image_records)
        elif image_records:
            page_text = _inject_image_placeholders_into_page_text(page_text, image_records)

        output_parts.append(f"===== Page {page_num} =====\n{page_text.strip()}\n")

    return "\n".join(output_parts).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview URL document cleaning before chunking.")
    parser.add_argument("url", help="Public PDF/Word/HTML URL to download and clean.")
    parser.add_argument(
        "--with-vlm",
        action="store_true",
        help="Call VLM and inject image descriptions into page text.",
    )
    args = parser.parse_args()

    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = download_url_document(args.url, DOCUMENT_DIR)
    _cleanup_preview_reference_images(downloaded.filename)
    preview_text = build_preview_text(
        downloaded.file_path,
        downloaded.filename,
        downloaded.final_url,
        with_vlm=args.with_vlm,
    )

    output_path = PREVIEW_DIR / f"{Path(downloaded.filename).stem}.cleaned.txt"
    output_path.write_text(preview_text, encoding="utf-8")

    print(f"Downloaded: {downloaded.file_path}")
    print(f"Preview: {output_path}")
    print(f"VLM: {'on' if args.with_vlm else 'off'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

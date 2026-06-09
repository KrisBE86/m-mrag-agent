#!/usr/bin/env python
"""
Clean orphan document index data even when the original file is already gone.

Usage:
    uv run python scripts/cleanup_document_index.py "filename.pdf"
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data"
BM25_STATE_PATH = DATA_DIR / "bm25_state.json"
REFERENCE_IMAGE_DIR = DATA_DIR / "reference_images"


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    tokens: list[str] = []
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
    english_pattern = re.compile(r"[a-zA-Z]+")
    i = 0
    while i < len(text):
        char = text[i]
        if chinese_pattern.match(char):
            tokens.append(char)
            i += 1
        elif english_pattern.match(char):
            match = english_pattern.match(text[i:])
            if match:
                token = match.group()
                tokens.append(token)
                i += len(token)
            else:
                i += 1
        else:
            i += 1
    return tokens


def _remove_from_bm25_state(texts: list[str]) -> int:
    if not texts or not BM25_STATE_PATH.is_file():
        return 0

    try:
        state = json.loads(BM25_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return 0

    doc_freq = Counter({str(k): int(v) for k, v in state.get("doc_freq", {}).items()})
    total_docs = int(state.get("total_docs", 0) or 0)
    sum_token_len = int(state.get("sum_token_len", 0) or 0)

    removed = 0
    for text in texts:
        tokens = _tokenize(text)
        if not tokens:
            continue
        total_docs = max(0, total_docs - 1)
        sum_token_len = max(0, sum_token_len - len(tokens))
        for token in set(tokens):
            if token not in doc_freq:
                continue
            doc_freq[token] -= 1
            if doc_freq[token] <= 0:
                del doc_freq[token]
        removed += 1

    state["total_docs"] = total_docs
    state["sum_token_len"] = sum_token_len
    state["doc_freq"] = dict(doc_freq)

    tmp = BM25_STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(BM25_STATE_PATH)
    return removed


def _cleanup_reference_images(filename: str) -> int:
    safe_filename = re.sub(r"[^\w\-\.]", "_", filename)
    removed = 0
    for path in REFERENCE_IMAGE_DIR.glob(f"{safe_filename}_img*"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def cleanup_document_index(filename: str) -> dict:
    from backend.milvus_client import milvus_manager
    from backend.parent_chunk_store import parent_chunk_store

    filename = filename.strip()
    if not filename:
        raise ValueError("filename 不能为空")

    filter_expr = f'filename == "{filename}"'

    text_rows = milvus_manager.query(
        collection=milvus_manager.text_collection,
        filter_expr=filter_expr,
        output_fields=["text"],
        limit=10000,
    )
    texts = [row.get("text") or "" for row in text_rows]
    bm25_removed = _remove_from_bm25_state(texts)

    text_delete = milvus_manager.delete(milvus_manager.text_collection, filter_expr)
    image_delete = milvus_manager.delete(milvus_manager.image_collection, filter_expr)
    parent_deleted = parent_chunk_store.delete_by_filename(filename)
    ref_deleted = _cleanup_reference_images(filename)

    return {
        "filename": filename,
        "milvus_text_rows_found": len(text_rows),
        "bm25_docs_removed": bm25_removed,
        "text_delete_result": text_delete,
        "image_delete_result": image_delete,
        "postgres_parent_chunks_deleted": parent_deleted,
        "reference_images_deleted": ref_deleted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean document index data by filename.")
    parser.add_argument("filename", help="Exact filename stored in metadata.filename")
    args = parser.parse_args()

    result = cleanup_document_index(args.filename)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

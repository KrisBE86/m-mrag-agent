"""
RAG retrieval utilities: hybrid search + auto-merge.

Core pipeline (aligned with SuperMew's rag_utils.py):
  1. Hybrid retrieval: BGE-M3 dense + BM25 sparse → Milvus RRF fusion
  2. Auto-merging: L3 → L2 → L1 when sibling chunks reach threshold
  3. (Optional) External reranker for precision refinement

Module-level singletons ensure BM25 state consistency across writes and reads.
"""

import os
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv

from backend.embedding import bge_embeddings, bm25
from backend.milvus_client import milvus_manager
from backend.parent_chunk_store import parent_chunk_store

load_dotenv()

AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))

# Optional reranker config.
RERANK_MODEL = os.getenv("RERANK_MODEL", "")
RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST", "")
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")


# ═══════════════════════════════════════════════════════════════════
# Auto-Merge (aligned with SuperMew)
# ═══════════════════════════════════════════════════════════════════

def _merge_to_parent_level(docs: list[dict], threshold: int = 2) -> tuple[list[dict], int]:
    """Merge L3→L2 (or L2→L1) when >= threshold siblings share the same parent."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    merge_parent_ids = [
        pid for pid, children in groups.items() if len(children) >= threshold
    ]
    if not merge_parent_ids:
        return docs, 0

    parent_docs = parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: list[dict] = []
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue
        parent_doc = dict(parent_map[parent_id])
        score = doc.get("score")
        if score is not None:
            parent_doc["score"] = max(
                float(parent_doc.get("score", score)), float(score)
            )
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        merged_docs.append(parent_doc)
        merged_count += 1

    # Deduplicate by chunk_id.
    deduped: list[dict] = []
    seen = set()
    for item in merged_docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, merged_count


def _auto_merge_documents(docs: list[dict], top_k: int) -> tuple[list[dict], dict[str, Any]]:
    """Two-stage auto-merge: L3→L2 then L2→L1."""
    if not AUTO_MERGE_ENABLED or not docs:
        return docs[:top_k], {
            "auto_merge_enabled": AUTO_MERGE_ENABLED,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }

    merged_docs, count_l3 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    merged_docs, count_l2 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)

    merged_docs.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    merged_docs = merged_docs[:top_k]

    replaced = count_l3 + count_l2
    return merged_docs, {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": replaced > 0,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": replaced,
        "auto_merge_steps": int(count_l3 > 0) + int(count_l2 > 0),
    }


# ═══════════════════════════════════════════════════════════════════
# Text Retrieval (aligned with SuperMew)
# ═══════════════════════════════════════════════════════════════════

def retrieve_documents(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Hybrid text retrieval: BGE-M3 dense + BM25 sparse → RRF → auto-merge.

    Args:
        query: Chinese text query.
        top_k: Number of results to return.

    Returns:
        {"docs": [...], "meta": {...}}
    """
    candidate_k = max(top_k * 3, top_k)
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"

    try:
        dense_embedding = bge_embeddings.embed_query(query)
        sparse_embedding = bm25.get_sparse_embedding(query)

        retrieved = milvus_manager.hybrid_retrieve(
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=candidate_k,
            filter_expr=filter_expr,
        )
        merged_docs, merge_meta = _auto_merge_documents(docs=retrieved, top_k=top_k)

        return {
            "docs": merged_docs,
            "meta": {
                "retrieval_mode": "hybrid",
                "candidate_k": candidate_k,
                "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                **merge_meta,
            },
        }
    except Exception:
        # Fallback: dense-only retrieval.
        try:
            dense_embedding = bge_embeddings.embed_query(query)
            retrieved = milvus_manager.hybrid_retrieve(
                dense_embedding=dense_embedding,
                sparse_embedding={},  # empty sparse → effectively dense-only
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
            merged_docs, merge_meta = _auto_merge_documents(docs=retrieved, top_k=top_k)
            return {
                "docs": merged_docs,
                "meta": {
                    "retrieval_mode": "dense_fallback",
                    "candidate_k": candidate_k,
                    "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                    **merge_meta,
                },
            }
        except Exception:
            return {
                "docs": [],
                "meta": {
                    "retrieval_mode": "failed",
                    "candidate_k": candidate_k,
                    "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                    "auto_merge_enabled": AUTO_MERGE_ENABLED,
                    "auto_merge_applied": False,
                    "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
                    "auto_merge_replaced_chunks": 0,
                    "auto_merge_steps": 0,
                },
            }


def retrieve_with_context(query: str, top_k: int = 5) -> str:
    """
    Hybrid text retrieval returning formatted context string.
    Suitable for direct consumption by LLM generation.
    """
    result = retrieve_documents(query, top_k=top_k)
    docs = result["docs"]
    if not docs:
        return "【未找到相关内容】"

    lines = []
    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "")
        site = doc.get("site", "")
        cave = doc.get("cave", "")
        poi_name = doc.get("poi_name", "")
        score = doc.get("score", 0)

        header = f"[{i}] "
        if poi_name:
            header += f"{poi_name}"
        if site or cave:
            location = f"{site} {cave}".strip()
            header += f" ({location})"
        if score:
            header += f" [相似度: {score:.4f}]"

        lines.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(lines)

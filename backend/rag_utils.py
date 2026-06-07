"""
RAG 检索工具：混合搜索 + 自动合并。

核心流水线（对齐 SuperMew 的 rag_utils.py）：
  1. 混合检索：BGE-M3 稠密向量 + BM25 稀疏向量 → Milvus RRF 融合
  2. 自动合并：同级块达到阈值时 L3 → L2 → L1
  3. (可选) 外部重排序器用于精度优化

模块级单例保证 BM25 状态在读写间的一致性。
"""

import json
import os
from collections import defaultdict
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from backend.embedding import bge_embeddings, bm25
from backend.milvus_client import milvus_manager
from backend.parent_chunk_store import parent_chunk_store

load_dotenv()

AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))

# 可选的重排序器配置。
RERANK_MODEL = os.getenv("RERANK_MODEL", "")
RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST", "")
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")

_stepback_model = None


def _get_rerank_endpoint() -> str:
    if not RERANK_BINDING_HOST:
        return ""
    host = RERANK_BINDING_HOST.strip().rstrip("/")
    return host if host.endswith("/v1/rerank") else f"{host}/v1/rerank"


# ═══════════════════════════════════════════════════════════════════
# 自动合并（对齐 SuperMew）
# ═══════════════════════════════════════════════════════════════════

def _merge_to_parent_level(docs: list[dict], threshold: int = 2) -> tuple[list[dict], int]:
    """当 >= 阈值个数的同级块共享同一父块时，合并 L3→L2（或 L2→L1）。"""
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

    # 按 chunk_id 去重。
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
    """两阶段自动合并：L3→L2 然后 L2→L1。"""
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


def _rerank_documents(query: str, docs: list[dict], top_k: int) -> tuple[list[dict], dict[str, Any]]:
    """使用外部 rerank API 对候选片段精排；未配置时保持 RRF 顺序。"""
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
    meta: dict[str, Any] = {
        "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
    }
    if not docs_with_rank or not meta["rerank_enabled"]:
        return docs_with_rank[:top_k], meta

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [doc.get("text", "") for doc in docs_with_rank],
        "top_n": min(top_k, len(docs_with_rank)),
        "return_documents": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RERANK_API_KEY}",
    }

    try:
        meta["rerank_applied"] = True
        response = httpx.post(
            meta["rerank_endpoint"],
            headers=headers,
            json=payload,
            timeout=15,
        )
        if response.status_code >= 400:
            meta["rerank_error"] = f"HTTP {response.status_code}: {response.text}"
            return docs_with_rank[:top_k], meta

        items = response.json().get("results", [])
        reranked = []
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs_with_rank):
                doc = dict(docs_with_rank[idx])
                score = item.get("relevance_score")
                if score is not None:
                    doc["rerank_score"] = score
                reranked.append(doc)

        if reranked:
            return reranked[:top_k], meta

        meta["rerank_error"] = "empty_rerank_results"
        return docs_with_rank[:top_k], meta
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        meta["rerank_error"] = str(e)
        return docs_with_rank[:top_k], meta


def _get_stepback_model():
    """Step-back/HyDE 使用轻量低温模型，沿用项目的 OpenAI 兼容配置。"""
    global _stepback_model
    if _stepback_model is None:
        _stepback_model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"),
            temperature=0.2,
            base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
            extra_body={"thinking": {"type": "disabled"}},
        )
    return _stepback_model


def _generate_step_back_question(query: str) -> str:
    prompt = (
        "请将用户的具体问题抽象成更高层次、更概括的退步问题，"
        "用于探寻背后的通用原理、背景或核心概念。只输出退步问题一句话，不要解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (_get_stepback_model().invoke(prompt).content or "").strip()
    except Exception:
        return ""


def _answer_step_back_question(step_back_question: str) -> str:
    if not step_back_question:
        return ""
    prompt = (
        "请简要回答以下退步问题，提供通用原理或背景知识，控制在120字以内。"
        "只输出答案，不要列出推理过程。\n"
        f"退步问题：{step_back_question}"
    )
    try:
        return (_get_stepback_model().invoke(prompt).content or "").strip()
    except Exception:
        return ""


def generate_hypothetical_document(query: str) -> str:
    """HyDE：生成一段像真实资料片段的假设性文档，用于语义检索。"""
    prompt = (
        "请基于用户问题生成一段假设性文档，内容应像文化遗产知识库中的真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (_get_stepback_model().invoke(prompt).content or "").strip()
    except Exception:
        return ""


def step_back_expand(query: str) -> dict:
    """Step-back：生成退步问题和简答，并拼回扩展查询。"""
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    if step_back_question or step_back_answer:
        expanded_query = (
            f"{query}\n\n"
            f"退步问题：{step_back_question}\n"
            f"退步问题答案：{step_back_answer}"
        )
    else:
        expanded_query = query
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }


# ═══════════════════════════════════════════════════════════════════
# 文本检索（对齐 SuperMew）
# ═══════════════════════════════════════════════════════════════════

def retrieve_documents(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    混合文本检索：BGE-M3 稠密向量 + BM25 稀疏向量 → RRF → 自动合并。

    Args:
        query: 中文文本查询。
        top_k: 返回结果数量。

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
        reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
        merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)
        rerank_meta["retrieval_mode"] = "hybrid"
        rerank_meta["candidate_k"] = candidate_k
        rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
        rerank_meta.update(merge_meta)

        return {
            "docs": merged_docs,
            "meta": rerank_meta,
        }
    except Exception:
        # 回退：仅稠密向量检索。
        try:
            dense_embedding = bge_embeddings.embed_query(query)
            retrieved = milvus_manager.hybrid_retrieve(
                dense_embedding=dense_embedding,
                sparse_embedding={},  # 空稀疏向量 → 等效为纯稠密检索
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
            reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
            merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)
            rerank_meta["retrieval_mode"] = "dense_fallback"
            rerank_meta["candidate_k"] = candidate_k
            rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
            rerank_meta.update(merge_meta)
            return {
                "docs": merged_docs,
                "meta": rerank_meta,
            }
        except Exception:
            return {
                "docs": [],
                "meta": {
                    "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
                    "rerank_applied": False,
                    "rerank_model": RERANK_MODEL,
                    "rerank_endpoint": _get_rerank_endpoint(),
                    "rerank_error": "retrieve_failed",
                    "retrieval_mode": "failed",
                    "candidate_k": candidate_k,
                    "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                    "auto_merge_enabled": AUTO_MERGE_ENABLED,
                    "auto_merge_applied": False,
                    "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
                    "auto_merge_replaced_chunks": 0,
                    "auto_merge_steps": 0,
                    "candidate_count": 0,
                },
            }


def retrieve_with_context(query: str, top_k: int = 5) -> str:
    """
    混合文本检索，返回格式化后的上下文字符串。
    适合直接供 LLM 生成使用。
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

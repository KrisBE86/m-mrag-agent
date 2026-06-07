"""
Agentic RAG pipeline for cultural heritage POI retrieval.

Flow:
  retrieve_initial -> grade_documents
    -> END when context is relevant
    -> rewrite_question -> retrieve_expanded -> END when context is weak

The outer LangChain Agent still owns final answer generation. This graph focuses
on retrieval quality control, query repair, trace metadata, and expanded recall.
"""

import os
from typing import Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langsmith import traceable
from pydantic import BaseModel, Field

from backend.rag_utils import (
    generate_hypothetical_document,
    retrieve_documents,
    step_back_expand,
)

load_dotenv()

_grader_model = None
_router_model = None


class GradeDocuments(BaseModel):
    """Binary relevance score for retrieved context."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class RewriteStrategy(BaseModel):
    """Query expansion strategy."""

    strategy: Literal["step_back", "hyde", "complex"]


class RAGState(TypedDict):
    question: str
    query: str
    context: str
    docs: list[dict]
    route: Optional[str]
    expansion_type: Optional[str]
    expanded_query: Optional[str]
    step_back_question: Optional[str]
    step_back_answer: Optional[str]
    hypothetical_doc: Optional[str]
    rag_trace: Optional[dict]


GRADE_PROMPT = (
    "You are a grader assessing relevance of retrieved cultural heritage documents "
    "to a user question.\n\n"
    "Retrieved documents:\n{context}\n\n"
    "User question:\n{question}\n\n"
    "If the documents contain keywords, entities, or semantic meaning that can help "
    "answer the question, grade them as relevant. Return only a binary score."
)


def _create_chat_model(temperature: float = 0.0):
    return ChatOpenAI(
        model=os.getenv("GRADE_MODEL") or os.getenv("OPENAI_MODEL", "deepseek-v4-flash"),
        temperature=temperature,
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        extra_body={"thinking": {"type": "disabled"}},
    )


def _get_grader_model():
    global _grader_model
    if _grader_model is None:
        _grader_model = _create_chat_model(temperature=0.0)
    return _grader_model


def _get_router_model():
    global _router_model
    if _router_model is None:
        _router_model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"),
            temperature=0.0,
            base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
            extra_body={"thinking": {"type": "disabled"}},
        )
    return _router_model


def _format_doc_header(doc: dict, index: int) -> str:
    title = doc.get("poi_name") or doc.get("filename") or "未知来源"
    site = doc.get("site", "")
    cave = doc.get("cave", "")
    location = f"{site} {cave}".strip()
    page = doc.get("page_number")
    score = doc.get("score")
    rerank_score = doc.get("rerank_score")

    parts = [f"[{index}] {title}"]
    if location:
        parts.append(f"位置: {location}")
    if page not in (None, "", 0):
        parts.append(f"页码: {page}")
    if score is not None:
        try:
            parts.append(f"相似度: {float(score):.4f}")
        except (TypeError, ValueError):
            pass
    if rerank_score is not None:
        try:
            parts.append(f"重排分: {float(rerank_score):.4f}")
        except (TypeError, ValueError):
            pass
    if doc.get("merged_from_children"):
        parts.append("自动合并父块")
    return " | ".join(parts)


def _format_docs(docs: list[dict]) -> str:
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "")
        chunks.append(f"{_format_doc_header(doc, i)}\n{text}")
    return "\n\n---\n\n".join(chunks)


def _base_trace(query: str, docs: list[dict], meta: dict) -> dict:
    return {
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": query,
        "expanded_query": query,
        "retrieved_chunks": docs,
        "initial_retrieved_chunks": docs,
        "retrieval_stage": "initial",
        "rerank_enabled": meta.get("rerank_enabled"),
        "rerank_applied": meta.get("rerank_applied"),
        "rerank_model": meta.get("rerank_model"),
        "rerank_endpoint": meta.get("rerank_endpoint"),
        "rerank_error": meta.get("rerank_error"),
        "retrieval_mode": meta.get("retrieval_mode"),
        "candidate_k": meta.get("candidate_k"),
        "leaf_retrieve_level": meta.get("leaf_retrieve_level"),
        "auto_merge_enabled": meta.get("auto_merge_enabled"),
        "auto_merge_applied": meta.get("auto_merge_applied"),
        "auto_merge_threshold": meta.get("auto_merge_threshold"),
        "auto_merge_replaced_chunks": meta.get("auto_merge_replaced_chunks"),
        "auto_merge_steps": meta.get("auto_merge_steps"),
    }


@traceable(name="mrag.retrieve_initial")
def retrieve_initial(state: RAGState) -> dict:
    query = state["question"]
    retrieved = retrieve_documents(query, top_k=5)
    docs = retrieved.get("docs", [])
    meta = retrieved.get("meta", {})
    return {
        "query": query,
        "docs": docs,
        "context": _format_docs(docs),
        "rag_trace": _base_trace(query, docs, meta),
    }


@traceable(name="mrag.grade_documents")
def grade_documents_node(state: RAGState) -> dict:
    docs = state.get("docs", [])
    rag_trace = state.get("rag_trace", {}) or {}
    if not docs:
        rag_trace.update({
            "grade_score": "no",
            "grade_route": "rewrite_question",
            "rewrite_needed": True,
        })
        return {"route": "rewrite_question", "rag_trace": rag_trace}

    prompt = GRADE_PROMPT.format(
        question=state["question"],
        context=state.get("context", ""),
    )
    try:
        response = _get_grader_model().with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        score = (response.binary_score or "").strip().lower()
    except Exception as e:
        score = "unknown"
        rag_trace["grade_error"] = str(e)

    route = "generate_answer" if score == "yes" else "rewrite_question"
    rag_trace.update({
        "grade_score": score,
        "grade_route": route,
        "rewrite_needed": route == "rewrite_question",
    })
    return {"route": route, "rag_trace": rag_trace}


@traceable(name="mrag.rewrite_question")
def rewrite_question_node(state: RAGState) -> dict:
    question = state["question"]
    strategy = "step_back"
    try:
        decision = _get_router_model().with_structured_output(RewriteStrategy).invoke(
            [{
                "role": "user",
                "content": (
                    "请根据用户问题选择最合适的查询扩展策略，仅输出结构化策略名。\n"
                    "- step_back：包含具体名称、日期、位置、文物点等细节，需要先理解通用背景的问题。\n"
                    "- hyde：模糊、概念性、缺少明确关键词、需要补充可能资料片段的问题。\n"
                    "- complex：多步骤、需要分解或综合多种信息的问题。\n"
                    f"用户问题：{question}"
                ),
            }]
        )
        strategy = decision.strategy
    except Exception:
        strategy = "step_back"

    expanded_query = question
    step_back_question = ""
    step_back_answer = ""
    hypothetical_doc = ""

    if strategy in ("step_back", "complex"):
        step_back = step_back_expand(question)
        step_back_question = step_back.get("step_back_question", "")
        step_back_answer = step_back.get("step_back_answer", "")
        expanded_query = step_back.get("expanded_query", question)

    if strategy in ("hyde", "complex"):
        hypothetical_doc = generate_hypothetical_document(question)

    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "rewrite_strategy": strategy,
        "rewrite_query": expanded_query,
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "hypothetical_doc": hypothetical_doc,
        "expansion_type": strategy,
    })

    return {
        "expansion_type": strategy,
        "expanded_query": expanded_query,
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "hypothetical_doc": hypothetical_doc,
        "rag_trace": rag_trace,
    }


def _merge_meta_value(current, new):
    return current if current not in (None, "", False) else new


@traceable(name="mrag.retrieve_expanded")
def retrieve_expanded(state: RAGState) -> dict:
    strategy = state.get("expansion_type") or "step_back"
    all_results: list[dict] = []
    meta_acc: dict = {
        "rerank_applied": False,
        "rerank_enabled": False,
        "auto_merge_applied": False,
        "auto_merge_replaced_chunks": 0,
        "auto_merge_steps": 0,
    }
    rerank_errors = []

    def _retrieve_and_accumulate(query: str, label: str) -> None:
        if not query:
            return
        retrieved = retrieve_documents(query, top_k=5)
        all_results.extend(retrieved.get("docs", []))
        meta = retrieved.get("meta", {})
        meta_acc["rerank_enabled"] = bool(meta_acc["rerank_enabled"] or meta.get("rerank_enabled"))
        meta_acc["rerank_applied"] = bool(meta_acc["rerank_applied"] or meta.get("rerank_applied"))
        meta_acc["auto_merge_applied"] = bool(meta_acc["auto_merge_applied"] or meta.get("auto_merge_applied"))
        meta_acc["rerank_model"] = _merge_meta_value(meta_acc.get("rerank_model"), meta.get("rerank_model"))
        meta_acc["rerank_endpoint"] = _merge_meta_value(meta_acc.get("rerank_endpoint"), meta.get("rerank_endpoint"))
        meta_acc["retrieval_mode"] = _merge_meta_value(meta_acc.get("retrieval_mode"), meta.get("retrieval_mode"))
        meta_acc["candidate_k"] = _merge_meta_value(meta_acc.get("candidate_k"), meta.get("candidate_k"))
        meta_acc["leaf_retrieve_level"] = _merge_meta_value(meta_acc.get("leaf_retrieve_level"), meta.get("leaf_retrieve_level"))
        meta_acc["auto_merge_enabled"] = _merge_meta_value(meta_acc.get("auto_merge_enabled"), meta.get("auto_merge_enabled"))
        meta_acc["auto_merge_threshold"] = _merge_meta_value(meta_acc.get("auto_merge_threshold"), meta.get("auto_merge_threshold"))
        meta_acc["auto_merge_replaced_chunks"] += int(meta.get("auto_merge_replaced_chunks") or 0)
        meta_acc["auto_merge_steps"] += int(meta.get("auto_merge_steps") or 0)
        if meta.get("rerank_error"):
            rerank_errors.append(f"{label}:{meta.get('rerank_error')}")

    if strategy in ("hyde", "complex"):
        _retrieve_and_accumulate(state.get("hypothetical_doc") or "", "hyde")
    if strategy in ("step_back", "complex"):
        _retrieve_and_accumulate(state.get("expanded_query") or state["question"], "step_back")
    if not all_results:
        _retrieve_and_accumulate(state["question"], "fallback")

    deduped = []
    seen = set()
    for item in all_results:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    for idx, item in enumerate(deduped, 1):
        item["rrf_rank"] = idx

    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "expanded_query": state.get("expanded_query") or state["question"],
        "step_back_question": state.get("step_back_question", ""),
        "step_back_answer": state.get("step_back_answer", ""),
        "hypothetical_doc": state.get("hypothetical_doc", ""),
        "expansion_type": strategy,
        "retrieved_chunks": deduped,
        "expanded_retrieved_chunks": deduped,
        "retrieval_stage": "expanded",
        "rerank_enabled": meta_acc.get("rerank_enabled"),
        "rerank_applied": meta_acc.get("rerank_applied"),
        "rerank_model": meta_acc.get("rerank_model"),
        "rerank_endpoint": meta_acc.get("rerank_endpoint"),
        "rerank_error": "; ".join(rerank_errors) if rerank_errors else None,
        "retrieval_mode": meta_acc.get("retrieval_mode"),
        "candidate_k": meta_acc.get("candidate_k"),
        "leaf_retrieve_level": meta_acc.get("leaf_retrieve_level"),
        "auto_merge_enabled": meta_acc.get("auto_merge_enabled"),
        "auto_merge_applied": meta_acc.get("auto_merge_applied"),
        "auto_merge_threshold": meta_acc.get("auto_merge_threshold"),
        "auto_merge_replaced_chunks": meta_acc.get("auto_merge_replaced_chunks"),
        "auto_merge_steps": meta_acc.get("auto_merge_steps"),
    })

    return {
        "docs": deduped,
        "context": _format_docs(deduped),
        "rag_trace": rag_trace,
    }


def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve_initial", retrieve_initial)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("retrieve_expanded", retrieve_expanded)

    graph.set_entry_point("retrieve_initial")
    graph.add_edge("retrieve_initial", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("route"),
        {
            "generate_answer": END,
            "rewrite_question": "rewrite_question",
        },
    )
    graph.add_edge("rewrite_question", "retrieve_expanded")
    graph.add_edge("retrieve_expanded", END)
    return graph.compile()


rag_graph = build_rag_graph()


@traceable(name="mrag.run_agentic_rag")
def run_rag_graph(question: str) -> dict:
    return rag_graph.invoke({
        "question": question,
        "query": question,
        "context": "",
        "docs": [],
        "route": None,
        "expansion_type": None,
        "expanded_query": None,
        "step_back_question": None,
        "step_back_answer": None,
        "hypothetical_doc": None,
        "rag_trace": None,
    })

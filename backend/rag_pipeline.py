"""
LangGraph RAG pipeline for POI identification and scene explanation.

State graph (MVP):
  retrieve → auto_merge → verify → generate

Future expansion (Phase 2, aligned with SuperMew):
  retrieve → grade → [auto_merge → verify → generate]
                   → [rewrite → expand → auto_merge → verify → generate]

This module builds and compiles the LangGraph state machine.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from backend.rag_utils import retrieve_documents, retrieve_with_context
from backend.schemas import CandidateResult, IdentificationResult

load_dotenv()


# ── State definition ────────────────────────────────────────────

from typing import TypedDict


class RAGState(TypedDict):
    """State for the POI identification RAG pipeline."""
    query: str                         # User's question (may include image identification results)
    query_image_path: str              # Original query image path (for context)
    retrieved_docs: list[dict]         # Raw retrieval results
    merged_context: str                # Auto-merged context string
    identification: str                # LLM verification result
    final_answer: str                  # Final generated explanation
    rag_meta: dict                     # Retrieval metadata for tracing


# ── Node functions ───────────────────────────────────────────────

def _retrieve_node(state: RAGState) -> dict:
    """Node 1: Hybrid text retrieval."""
    query = state["query"]
    result = retrieve_documents(query, top_k=5)
    return {
        "retrieved_docs": result["docs"],
        "rag_meta": result["meta"],
    }


def _auto_merge_node(state: RAGState) -> dict:
    """Node 2: Build merged context from retrieved docs."""
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"merged_context": "【未找到相关内容】"}

    lines = []
    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "")
        site = doc.get("site", "")
        cave = doc.get("cave", "")
        poi_name = doc.get("poi_name", "")
        score = doc.get("score", 0)
        merged = doc.get("merged_from_children", False)

        header = f"[{i}] "
        if poi_name:
            header += f"{poi_name}"
        if site or cave:
            location = f"{site} {cave}".strip()
            header += f" ({location})"
        if merged:
            header += " [自动合并父块]"

        lines.append(f"{header}\n{text}")

    return {"merged_context": "\n\n---\n\n".join(lines)}


def _verify_node(state: RAGState, llm: BaseChatModel) -> dict:
    """Node 3: LLM verification / disambiguation."""
    docs = state.get("retrieved_docs", [])
    merged = state.get("merged_context", "")

    if not docs:
        return {"identification": "【无法确定】未在知识库中找到匹配的文物点。"}

    # Build simple verification prompt.
    prompt = (
        "你是一位文化遗产鉴定专家。根据以下检索到的知识库内容，"
        "判断用户查询的内容最可能对应哪一个文物点，并简要说明理由。\n\n"
        f"用户查询: {state['query']}\n\n"
        f"检索上下文:\n{merged}\n\n"
        "请用中文回答，格式:\n"
        "【识别结果】文物点名称\n"
        "【置信度】高/中/低\n"
        "【分析】简要推理过程"
    )

    try:
        response = llm.invoke(prompt)
        return {"identification": response.content if hasattr(response, "content") else str(response)}
    except Exception as e:
        return {"identification": f"【识别异常】LLM 调用失败: {str(e)}"}


def _generate_node(state: RAGState, llm: BaseChatModel) -> dict:
    """Node 4: Generate final scene explanation."""
    query = state.get("query", "")
    identification = state.get("identification", "")
    merged = state.get("merged_context", "")

    prompt = (
        "你是一位文化遗产讲解员，为游客介绍石窟寺和古建筑中的文物点。\n\n"
        f"游客提问: {query}\n\n"
        f"文物识别结果:\n{identification}\n\n"
        f"知识库相关内容:\n{merged}\n\n"
        "请根据以上信息，用生动专业的中文为游客介绍这个文物点，"
        "包括它的历史背景、文化意义、建筑/造像特色等。"
        "如果有不确定的地方，请诚实说明。"
    )

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return {"final_answer": content}
    except Exception as e:
        return {"final_answer": f"讲解生成失败: {str(e)}"}


# ── Graph builder ────────────────────────────────────────────────

def _create_verify_with_llm(llm: BaseChatModel):
    """Factory to inject LLM into verify node."""
    def _verify(state: RAGState) -> dict:
        return _verify_node(state, llm)
    return _verify


def _create_generate_with_llm(llm: BaseChatModel):
    """Factory to inject LLM into generate node."""
    def _generate(state: RAGState) -> dict:
        return _generate_node(state, llm)
    return _generate


def build_rag_graph(llm: Optional[BaseChatModel] = None) -> StateGraph:
    """
    Build the POI identification RAG LangGraph state machine.

    Args:
        llm: LangChain chat model for verification and generation.
             Defaults to the DeepSeek model configured in .env.

    Returns:
        Compiled LangGraph StateGraph.
    """
    if llm is None:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
            base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
            extra_body={"thinking": {"type": "disabled"}},
        )

    graph = StateGraph(RAGState)

    # Add nodes.
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("auto_merge", _auto_merge_node)
    graph.add_node("verify", _create_verify_with_llm(llm))
    graph.add_node("generate", _create_generate_with_llm(llm))

    # Define edges.
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "auto_merge")
    graph.add_edge("auto_merge", "verify")
    graph.add_edge("verify", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# Module-level singleton, aligned with SuperMew pattern.
rag_graph = build_rag_graph()

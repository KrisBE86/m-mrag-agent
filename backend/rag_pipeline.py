"""
LangGraph RAG 流水线，用于 POI 识别和场景解说。

状态图 (MVP):
  retrieve → auto_merge → verify → generate

未来扩展 (Phase 2, 对齐 SuperMew):
  retrieve → grade → [auto_merge → verify → generate]
                   → [rewrite → expand → auto_merge → verify → generate]

本模块负责构建和编译 LangGraph 状态机。
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


# ── 状态定义 ────────────────────────────────────────────────────

from typing import TypedDict


class RAGState(TypedDict):
    """POI 识别 RAG 流水线的状态定义。"""
    query: str                         # 用户问题（可能包含图片识别结果）
    query_image_path: str              # 原始查询图片路径（用于上下文）
    retrieved_docs: list[dict]         # 原始检索结果
    merged_context: str                # 自动合并后的上下文字符串
    identification: str                # LLM 验证结果
    final_answer: str                  # 最终生成的解说
    rag_meta: dict                     # 检索元数据，用于追踪


# ── 节点函数 ───────────────────────────────────────────────

def _retrieve_node(state: RAGState) -> dict:
    """节点 1: 混合文本检索。"""
    query = state["query"]
    result = retrieve_documents(query, top_k=5)
    return {
        "retrieved_docs": result["docs"],
        "rag_meta": result["meta"],
    }


def _auto_merge_node(state: RAGState) -> dict:
    """节点 2: 根据检索到的文档构建合并上下文。"""
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
    """节点 3: LLM 验证 / 消歧。"""
    docs = state.get("retrieved_docs", [])
    merged = state.get("merged_context", "")

    if not docs:
        return {"identification": "【无法确定】未在知识库中找到匹配的文物点。"}

    # 构建简单的验证 prompt。
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
    """节点 4: 生成最终场景解说。"""
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


# ── 图构建器 ────────────────────────────────────────────────

def _create_verify_with_llm(llm: BaseChatModel):
    """工厂函数：将 LLM 注入 verify 节点。"""
    def _verify(state: RAGState) -> dict:
        return _verify_node(state, llm)
    return _verify


def _create_generate_with_llm(llm: BaseChatModel):
    """工厂函数：将 LLM 注入 generate 节点。"""
    def _generate(state: RAGState) -> dict:
        return _generate_node(state, llm)
    return _generate


def build_rag_graph(llm: Optional[BaseChatModel] = None) -> StateGraph:
    """
    构建 POI 识别 RAG LangGraph 状态机。

    Args:
        llm: 用于验证和生成的 LangChain 对话模型。
             默认使用 .env 中配置的 DeepSeek 模型。

    Returns:
        编译后的 LangGraph StateGraph。
    """
    if llm is None:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
            base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
            extra_body={"thinking": {"type": "disabled"}},
        )

    graph = StateGraph(RAGState)

    # 添加节点。
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("auto_merge", _auto_merge_node)
    graph.add_node("verify", _create_verify_with_llm(llm))
    graph.add_node("generate", _create_generate_with_llm(llm))

    # 定义边。
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "auto_merge")
    graph.add_edge("auto_merge", "verify")
    graph.add_edge("verify", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# 模块级单例，对齐 SuperMew 模式。
rag_graph = build_rag_graph()

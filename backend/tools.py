"""
MRagAgent 的 LangChain @tool 函数。

四个工具：
  1. recall_conversation_context — 查询对话上下文，判断是否需要识图
  2. identify_from_image — CLIP 图像搜索 → 文物点文字描述
  3. identify_from_image_vlm — 豆包视觉描述（不直接检索）
  4. search_knowledge_base — 使用后端构造 query 的混合文本 RAG

Agent 上下文优先路由逻辑：
  - 收到消息 → recall_conversation_context 查上下文
    → 上下文已知 → search_knowledge_base（跳过识图）
    → 上下文空白 + 需要识图 → identify_from_image → 写 Redis
      → 置信度低 → identify_from_image_vlm
    → 上下文空白 + 与图无关 → 直接回答
"""

import re
from contextvars import ContextVar

from langchain_core.tools import tool

from backend.image_retriever import identify_from_image as _image_identify
from backend.image_retriever import identify_from_image_vlm as _vlm_identify

# Redis 上下文记忆 key 前缀。实际 key 按会话 thread_id 隔离。
_CONTEXT_KEY_PREFIX = "mragagent:context"
_ACTIVE_THREAD_ID: ContextVar[str] = ContextVar("mragagent_thread_id", default="main-chat")
_ACTIVE_USER_QUESTION: ContextVar[str] = ContextVar("mragagent_user_question", default="")
_PENDING_VISUAL_DESCRIPTION: ContextVar[str] = ContextVar("mragagent_pending_visual_description", default="")
_PENDING_CLIP_CONTEXT: ContextVar[dict] = ContextVar("mragagent_pending_clip_context", default={})
_ACTIVE_THREAD_ID_FALLBACK = "main-chat"
_ACTIVE_USER_QUESTION_FALLBACK = ""
_PENDING_VISUAL_DESCRIPTION_FALLBACK = ""
_PENDING_CLIP_CONTEXT_FALLBACK: dict = {}
_LAST_RAG_CONTEXT: dict | None = None
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0


def _safe_thread_id(thread_id: str | None) -> str:
    raw = (thread_id or "main-chat").strip() or "main-chat"
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", raw)


def set_active_thread_id(thread_id: str | None) -> None:
    """设置当前工具调用使用的会话 ID，用于隔离 Redis 中的 POI 上下文。"""
    global _ACTIVE_THREAD_ID_FALLBACK
    safe_id = _safe_thread_id(thread_id)
    _ACTIVE_THREAD_ID_FALLBACK = safe_id
    _ACTIVE_THREAD_ID.set(safe_id)


def set_active_user_question(user_question: str | None) -> None:
    """设置当前用户原始问题，用于 VLM fallback 后构造统一 RAG 查询。"""
    global _ACTIVE_USER_QUESTION_FALLBACK
    question = (user_question or "").strip()
    _ACTIVE_USER_QUESTION_FALLBACK = question
    _ACTIVE_USER_QUESTION.set(question)


def _active_user_question() -> str:
    question = _ACTIVE_USER_QUESTION.get()
    if not question and _ACTIVE_USER_QUESTION_FALLBACK:
        question = _ACTIVE_USER_QUESTION_FALLBACK
    return question or "请帮我看看这张图片"


def _context_key() -> str:
    thread_id = _ACTIVE_THREAD_ID.get()
    if thread_id == "main-chat" and _ACTIVE_THREAD_ID_FALLBACK != "main-chat":
        thread_id = _ACTIVE_THREAD_ID_FALLBACK
    return f"{_CONTEXT_KEY_PREFIX}:{thread_id}"


def _set_pending_visual_description(description: str) -> None:
    global _PENDING_VISUAL_DESCRIPTION_FALLBACK
    value = (description or "").strip()
    _PENDING_VISUAL_DESCRIPTION_FALLBACK = value
    _PENDING_VISUAL_DESCRIPTION.set(value)


def _pop_pending_visual_description() -> str:
    global _PENDING_VISUAL_DESCRIPTION_FALLBACK
    description = _PENDING_VISUAL_DESCRIPTION.get()
    if not description and _PENDING_VISUAL_DESCRIPTION_FALLBACK:
        description = _PENDING_VISUAL_DESCRIPTION_FALLBACK
    _PENDING_VISUAL_DESCRIPTION_FALLBACK = ""
    _PENDING_VISUAL_DESCRIPTION.set("")
    return description


def _set_pending_clip_context(context: dict | None) -> None:
    global _PENDING_CLIP_CONTEXT_FALLBACK
    value = context or {}
    _PENDING_CLIP_CONTEXT_FALLBACK = value
    _PENDING_CLIP_CONTEXT.set(value)


def _pop_pending_clip_context() -> dict:
    global _PENDING_CLIP_CONTEXT_FALLBACK
    context = _PENDING_CLIP_CONTEXT.get()
    if not context and _PENDING_CLIP_CONTEXT_FALLBACK:
        context = _PENDING_CLIP_CONTEXT_FALLBACK
    _PENDING_CLIP_CONTEXT_FALLBACK = {}
    _PENDING_CLIP_CONTEXT.set({})
    return context or {}


def _build_initial_rag_query() -> tuple[str, str]:
    user_question = _active_user_question()

    visual_description = _pop_pending_visual_description()
    if visual_description:
        return (
            f"用户问题：{user_question}\n"
            f"图片描述：{visual_description}",
            "vlm_visual_description",
        )

    clip_context = _pop_pending_clip_context()
    if clip_context:
        lines = [f"用户问题：{user_question}"]
        name = clip_context.get("name")
        location = clip_context.get("location")
        if name:
            lines.append(f"图片识别结果：{name}")
        if location:
            lines.append(f"位置：{location}")
        return "\n".join(lines), "clip_identification"

    from backend.cache import cache

    context = cache.get_json(_context_key())
    pois = context.get("pois", []) if isinstance(context, dict) else []
    if pois:
        lines = [f"用户问题：{user_question}", "当前上下文："]
        for poi in pois:
            name = poi.get("name", "")
            location = poi.get("location", "")
            if name and location:
                lines.append(f"{name}（{location}）")
            elif name:
                lines.append(name)
        return "\n".join(lines), "conversation_context"

    return f"用户问题：{user_question}", "raw_user_question"


def _set_last_rag_context(context: dict) -> None:
    global _LAST_RAG_CONTEXT
    _LAST_RAG_CONTEXT = context


def get_last_rag_context(clear: bool = True) -> dict | None:
    """获取最近一次 RAG 检索 trace，供 API 层或调试面板读取。"""
    global _LAST_RAG_CONTEXT
    context = _LAST_RAG_CONTEXT
    if clear:
        _LAST_RAG_CONTEXT = None
    return context


def reset_tool_call_guards() -> None:
    """每轮对话开始时重置知识库工具调用限制。"""
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
    _set_pending_visual_description("")
    _set_pending_clip_context({})


def _format_doc_header(doc: dict, index: int) -> str:
    title = doc.get("poi_name") or doc.get("filename") or "未知文物点"
    site = doc.get("site", "")
    cave = doc.get("cave", "")
    score = doc.get("score")
    rerank_score = doc.get("rerank_score")

    parts = [f"[{index}] {title}"]
    location = f"{site} {cave}".strip()
    if location:
        parts.append(f"位置: {location}")
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


def _format_rag_result(rag_result: dict) -> str:
    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        _set_last_rag_context({"rag_trace": rag_trace})

    sections = []
    query_source = rag_trace.get("query_source", "unknown")
    actual_query = rag_trace.get("query", "")
    grade_score = rag_trace.get("grade_score", "unknown")
    retrieval_stage = rag_trace.get("retrieval_stage", "unknown")
    rewrite_needed = bool(rag_trace.get("rewrite_needed"))
    expansion_type = rag_trace.get("expansion_type") or "none"
    sections.append(
        "【RAG流程】"
        f"初检 → 相关性评分={grade_score} → "
        f"{'查询改写/扩展检索' if rewrite_needed else '直接使用初检结果'}；"
        f"最终阶段={retrieval_stage}；扩展策略={expansion_type}"
    )
    sections.append(f"【Query来源】{query_source}")
    if actual_query:
        sections.append(f"【实际初始检索Query】\n{actual_query}")

    if not docs:
        sections.append("【未找到相关内容】知识库中没有检索到足够相关的资料。")
        return "\n\n".join(sections)

    sections.append(
        "【回答约束】如果本次检索用于图片识别，请只在检索结果明确支持同一个具体文物点的名称、"
        "位置和关键视觉特征时给出具体身份。若结果只是类型、风格或题材相似，或证据不足以确认，"
        "请回答知识库没有足够可靠依据，不能强行命名。"
    )

    if grade_score != "yes":
        sections.append(
            "【可靠性警告】相关性评分不是 yes。最终回答必须保守处理，不要把候选检索结果说成已确认事实。"
        )

    if rag_trace.get("rewrite_needed"):
        strategy = rag_trace.get("rewrite_strategy") or rag_trace.get("expansion_type") or "unknown"
        sections.append(f"【检索修正】首次检索相关性不足，已使用 {strategy} 策略扩展查询并重新检索。")
        step_back_question = rag_trace.get("step_back_question")
        step_back_answer = rag_trace.get("step_back_answer")
        if step_back_question or step_back_answer:
            sections.append(
                "【Step-back】\n"
                f"退步问题：{step_back_question or '无'}\n"
                f"退步答案：{step_back_answer or '无'}"
            )

    formatted = []
    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "")
        formatted.append(f"{_format_doc_header(doc, i)}\n{text}")

    sections.append("【知识库检索结果】\n" + "\n\n---\n\n".join(formatted))
    return "\n\n".join(sections)


def _extract_poi_name(result_text: str) -> str | None:
    """从 identify_from_image 返回文本中提取最佳匹配的文物点名称。"""
    match = re.search(r"名称:\s*(.+?)(?:\n|$)", result_text)
    return match.group(1).strip() if match else None


def _cache_identified_poi(result_text: str) -> None:
    """如果识别成功，将文物点信息写入 Redis 上下文缓存。"""
    if "无法识别" in result_text or "错误" in result_text:
        return
    if any(marker in result_text for marker in ("低置信度", "中置信度", "无可靠匹配")):
        return

    name = _extract_poi_name(result_text)
    if not name:
        return

    # 也提取位置信息
    loc_match = re.search(r"位置:\s*(.+?)(?:\n|$)", result_text)
    location = loc_match.group(1).strip() if loc_match else "未知"

    from backend.cache import cache

    existing = cache.get_json(_context_key()) or {"pois": []}
    # 避免重复添加
    if not any(p["name"] == name for p in existing["pois"]):
        existing["pois"].append({"name": name, "location": location})
        cache.set_json(_context_key(), existing, ttl=3600)

    _set_pending_clip_context({"name": name, "location": location})


@tool
def recall_conversation_context(user_question: str) -> str:
    """在调用任何图片识别工具之前，必须先调用此工具来检查对话上下文。

    此工具查询当前对话中是否已经涉及某个文物点。如果对话中已经识别过文物点，
    且用户当前提问与该文物点相关，则返回已知信息，你应该跳过图片识别，
    直接调用 search_knowledge_base 搜索该文物点。

    只有当此工具返回"上下文空白"时，才考虑使用图片识别工具。

    参数:
        user_question: 用户当前的文字提问，用于判断是否与已知上下文相关

    返回:
        对话上下文信息，或"上下文空白"提示。
    """
    from backend.cache import cache

    context = cache.get_json(_context_key())
    if not context or not context.get("pois"):
        return (
            "【上下文空白】当前对话中还没有识别过任何文物点。"
            "如果用户提问需要借助图片来定位文物（比如问'这是哪'、'帮我看看'），"
            "请调用 identify_from_image。"
            "如果用户提问与图片无关（比如问天气、计算），直接回答或使用对应工具。"
        )

    pois = context["pois"]
    lines = [f"【已有上下文】当前对话中已涉及以下文物点："]
    for i, p in enumerate(pois, 1):
        lines.append(f"{i}. {p['name']}（{p.get('location', '未知位置')}）")
    lines.append("")
    lines.append(f"用户当前提问：「{user_question}」")
    lines.append("请判断用户提问是否与上述文物点相关。如果相关，跳过图片识别，直接调用 search_knowledge_base。")
    return "\n".join(lines)


@tool
def identify_from_image(image_path: str, top_k: int = 5, min_score: float = 0.0) -> str:
    """当需要识别用户上传的文物图片时使用。

    这个工具通过图像相似度检索（Chinese-CLIP），返回照片对应的文物点名称、位置、
    描述和区分特征。识别成功后会记住文物点信息，后续提问无需再次识图。

    参数:
        image_path: 图片文件的本地路径
        top_k: 返回候选数量，默认5
        min_score: 最低相似度阈值(0-1)，默认0.0

    返回:
        识别到的文物点文字描述，包含置信度、名称、位置、描述和区分特征。
    """
    result = _image_identify(image_path, top_k=top_k, min_score=min_score)
    _cache_identified_poi(result)
    return result


@tool
def identify_from_image_vlm(image_path: str, top_k: int = 5) -> str:
    """当 CLIP 图像识别返回低置信度结果（多个候选高度相似，差距<0.05）时，
    使用此工具进行更精确的视觉识别。

    此工具只通过豆包视觉大模型（Doubao Vision）生成图片的详细视觉描述
    （不猜测场景名称，只描述看到的视觉特征），不会直接检索知识库。

    调用此工具后，必须继续调用 search_knowledge_base。search_knowledge_base 没有 query
    参数，后端会自动使用用户原始问题和本工具生成的视觉描述拼接检索 query。不要直接基于
    本工具结果回答具体文物点名称。

    相比 CLIP 图像对比，此方法对同一石窟内不同位置的文物点区分能力更强，
    因为它能捕捉到手印、服饰、空间位置等 CLIP 无法区分的细节。

    参数:
        image_path: 图片文件的本地路径
        top_k: 保留兼容旧调用，当前不用于检索

    返回:
        图片视觉描述。
    """
    visual_description = _vlm_identify(image_path, top_k=top_k)
    if "【错误】" in visual_description:
        _set_pending_visual_description("")
        return visual_description

    _set_pending_visual_description(visual_description)
    return visual_description


@tool
def search_knowledge_base() -> str:
    """搜索文化遗产知识库。当用户询问关于文物、石窟、古建筑的具体问题时使用。

    此工具没有 query 参数。Agent 只负责决定是否检索，不能自行编写、改写或猜测检索 query。
    后端会根据真实用户问题、当前对话上下文、CLIP 高置信度识别结果或 VLM 视觉描述构造
    初始检索 query。若初检相关性不足，RAG graph 内部再执行 step-back/rewrite。

    支持的问题类型包括但不限于：
    - 文物历史背景（"云冈石窟是什么时候开凿的"）
    - 艺术风格分析（"第20窟主佛的艺术特点"）
    - 建筑形制（"石窟的洞窟形制有哪些类型"）
    - 考古发现（"云冈石窟的考古新发现"）
    - 文化意义（"云冈石窟为什么被列为世界文化遗产"）

    返回:
        知识库中检索到的相关内容，经过语义搜索和自动合并。
    """
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    if _KNOWLEDGE_TOOL_CALLS_THIS_TURN >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: 本轮已经调用过 search_knowledge_base。"
            "请直接基于已有检索结果回答，不要再次调用工具。"
        )
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN += 1

    from backend.rag_pipeline import run_rag_graph

    query, query_source = _build_initial_rag_query()
    rag_result = run_rag_graph(query)
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        rag_trace["query_source"] = query_source
    return _format_rag_result(rag_result)

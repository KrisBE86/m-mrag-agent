"""
MRagAgent 的 LangChain @tool 函数。

四个工具：
  1. recall_conversation_context — 查询对话上下文，判断是否需要识图
  2. identify_from_image — CLIP 图像搜索 → 文物点文字描述
  3. identify_from_image_vlm — 豆包视觉描述 → 文本检索
  4. search_knowledge_base — 混合文本 RAG（BGE-M3 + BM25 + 自动合并）

Agent 上下文优先路由逻辑：
  - 收到消息 → recall_conversation_context 查上下文
    → 上下文已知 → search_knowledge_base（跳过识图）
    → 上下文空白 + 需要识图 → identify_from_image → 写 Redis
      → 置信度低 → identify_from_image_vlm
    → 上下文空白 + 与图无关 → 直接回答
"""

import re

from langchain_core.tools import tool

from backend.image_retriever import identify_from_image as _image_identify
from backend.image_retriever import identify_from_image_vlm as _vlm_identify
from backend.rag_utils import retrieve_with_context

# Redis 上下文记忆 key（与 agent_api.py 的 CONTEXT_KEY 一致）
_CONTEXT_KEY = "mragagent:context:main-chat"


def _extract_poi_name(result_text: str) -> str | None:
    """从 identify_from_image 返回文本中提取最佳匹配的文物点名称。"""
    match = re.search(r"名称:\s*(.+?)(?:\n|$)", result_text)
    return match.group(1).strip() if match else None


def _cache_identified_poi(result_text: str) -> None:
    """如果识别成功，将文物点信息写入 Redis 上下文缓存。"""
    if "无法识别" in result_text or "错误" in result_text:
        return

    name = _extract_poi_name(result_text)
    if not name:
        return

    # 也提取位置信息
    loc_match = re.search(r"位置:\s*(.+?)(?:\n|$)", result_text)
    location = loc_match.group(1).strip() if loc_match else "未知"

    from backend.cache import cache

    existing = cache.get_json(_CONTEXT_KEY) or {"pois": []}
    # 避免重复添加
    if not any(p["name"] == name for p in existing["pois"]):
        existing["pois"].append({"name": name, "location": location})
        cache.set_json(_CONTEXT_KEY, existing, ttl=3600)


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

    context = cache.get_json(_CONTEXT_KEY)
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

    此工具通过豆包视觉大模型（Doubao Vision）生成图片的详细视觉描述
    （不猜测场景名称，只描述看到的视觉特征），然后用文本语义检索（BGE-M3 + BM25）
    在知识库中匹配相关内容。

    相比 CLIP 图像对比，此方法对同一石窟内不同位置的文物点区分能力更强，
    因为它能捕捉到手印、服饰、空间位置等 CLIP 无法区分的细节。

    参数:
        image_path: 图片文件的本地路径
        top_k: 返回的文本块数量，默认5

    返回:
        视觉描述文字 + 知识库文本匹配结果
    """
    return _vlm_identify(image_path, top_k=top_k)


@tool
def search_knowledge_base(query: str) -> str:
    """搜索文化遗产知识库。当用户询问关于文物、石窟、古建筑的具体问题时使用。

    支持的问题类型包括但不限于：
    - 文物历史背景（"云冈石窟是什么时候开凿的"）
    - 艺术风格分析（"第20窟主佛的艺术特点"）
    - 建筑形制（"石窟的洞窟形制有哪些类型"）
    - 考古发现（"云冈石窟的考古新发现"）
    - 文化意义（"云冈石窟为什么被列为世界文化遗产"）

    参数:
        query: 搜索查询文本，建议包含具体的文物名称、地点等关键词。

    返回:
        知识库中检索到的相关内容，经过语义搜索和自动合并。
    """
    return retrieve_with_context(query)

"""
LangChain @tool functions for the MRagAgent.

Three tools:
  1. identify_from_image — CLIP image search → POI text description
  2. identify_from_image_vlm — Doubao Vision description → text retrieval
  3. search_knowledge_base — Hybrid text RAG (BGE-M3 + BM25 + auto-merge)

Agent routes autonomously:
  - Image + question → identify_from_image (fast)
    → if results ambiguous (low confidence) → identify_from_image_vlm (precise)
    → search_knowledge_base
  - Text-only → search_knowledge_base
"""

from langchain_core.tools import tool

from backend.image_retriever import identify_from_image as _image_identify
from backend.image_retriever import identify_from_image_vlm as _vlm_identify
from backend.rag_utils import retrieve_with_context


@tool
def identify_from_image(image_path: str, top_k: int = 5, min_score: float = 0.0) -> str:
    """当用户上传了一张文物/景点/石窟的照片，需要识别照片中具体是哪个文物点时使用。

    这个工具通过图像相似度检索（Chinese-CLIP），返回照片对应的文物点名称、位置、
    描述和区分特征。识别结果可用于后续的知识库检索。

    当结果不明确时（分数接近的多个候选），工具会自动标注置信度并提示需要更多信息。
    如果首次结果不够精确，可以增大 top_k 获取更多候选，或设置 min_score 过滤低分。

    参数:
        image_path: 图片文件的本地路径，例如 "/Users/xxx/photo.jpg"
        top_k: 返回候选数量，默认5。结果模糊时建议增大到10-15
        min_score: 最低相似度阈值(0-1)，默认0.0不过滤。首次检索后若候选太多可设为0.85

    返回:
        识别到的文物点文字描述，包含置信度标注、候选排名、相似度分数及与第1名的差距、
        名称、位置、描述和区分特征。如果无法识别，返回"无法识别"。
    """
    return _image_identify(image_path, top_k=top_k, min_score=min_score)


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

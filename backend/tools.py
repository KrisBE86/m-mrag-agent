"""
LangChain @tool functions for the MRagAgent.

Two tools:
  1. identify_from_image — CLIP image search → POI text description
  2. search_knowledge_base — Hybrid text RAG (BGE-M3 + BM25 + auto-merge)

Agent routes autonomously:
  - Image + question → identify_from_image → search_knowledge_base
  - Text-only → search_knowledge_base
"""

from langchain_core.tools import tool

from backend.image_retriever import identify_from_image as _image_identify
from backend.rag_utils import retrieve_with_context


@tool
def identify_from_image(image_path: str) -> str:
    """当用户上传了一张文物/景点/石窟的照片，需要识别照片中具体是哪个文物点时使用。

    这个工具通过图像相似度检索（Chinese-CLIP），返回照片对应的文物点名称、位置、
    描述和区分特征。识别结果可用于后续的知识库检索。

    参数:
        image_path: 图片文件的本地路径，例如 "/Users/xxx/photo.jpg"

    返回:
        识别到的文物点文字描述，包括候选排名、相似度分数、名称、位置、描述和区分特征。
        如果无法识别，返回"无法识别"。
    """
    return _image_identify(image_path)


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

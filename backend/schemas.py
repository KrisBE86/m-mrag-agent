"""
POI 识别流水线的 Pydantic 数据模型。
"""

from typing import Optional

from pydantic import BaseModel, Field


class POIMetadata(BaseModel):
    """L3（叶子）级别的单个文物点元数据。"""

    poi_id: str = Field(..., description="唯一 POI 标识符，例如 'yg-c20-e3-n2'")
    name: str = Field(..., description="中文 POI 名称，例如 '交脚菩萨像'")
    site: str = Field(..., description="所属遗址，例如 '云冈石窟'")
    cave: str = Field(default="", description="窟/区域标识符，例如 '第20窟'")
    location: str = Field(default="", description="窟内具体位置，例如 '东壁第三龛'")
    description: str = Field(..., description="POI 的详细中文描述")
    distinguishing_features: str = Field(
        default="",
        description="将此 POI 与其他类似文物区分开来的视觉特征",
    )
    reference_images: list[str] = Field(default_factory=list, description="参考图片的相对路径")
    tags: list[str] = Field(default_factory=list, description="例如 ['菩萨', '交脚', '北魏']")
    dynasty: str = Field(default="", description="例如 '北魏'")
    century: str = Field(default="", description="例如 '5世纪中期'")


class ParentChunkData(BaseModel):
    """存储在 PostgreSQL 中的 L1 或 L2 父级文本块。"""

    chunk_id: str
    text: str
    chunk_level: int
    parent_chunk_id: str = ""
    root_chunk_id: str = ""
    filename: str = ""
    file_type: str = ""
    site: str = ""
    cave: str = ""


class ImageSearchResult(BaseModel):
    """图像到图像向量搜索（集合1）的单条结果。"""

    chunk_id: str = Field(..., description="关联到文本集合2的 chunk_id")
    poi_name: str
    site: str
    cave: str
    poi_description: str
    distinguishing_features: str
    similarity_score: float = Field(..., description="Chinese-CLIP 余弦相似度")
    image_path: str = ""
    tags: list[str] = Field(default_factory=list)


class CandidateResult(BaseModel):
    """第一阶段文本检索的候选结果，供第二阶段 LLM 验证使用。"""

    chunk_id: str
    text: str
    chunk_level: int
    parent_chunk_id: str = ""
    root_chunk_id: str = ""
    similarity_score: float = 0.0
    site: str = ""
    cave: str = ""
    poi_name: str = ""


class IdentificationResult(BaseModel):
    """两阶段识别流水线的最终结果。"""

    identified_poi: Optional[str] = Field(None, description="POI 名称，未识别出则为 None")
    confidence: str = Field("low", description="高 | 中 | 低")
    reasoning: str = Field("", description="LLM 的中文详细推理过程")
    merged_context: str = Field("", description="从父级文本块自动合并的上下文")
    stage1_time_ms: float = 0.0
    stage2_time_ms: float = 0.0
    candidates: list[CandidateResult] = Field(default_factory=list)

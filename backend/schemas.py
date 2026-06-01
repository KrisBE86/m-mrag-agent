"""
Pydantic data models for the POI identification pipeline.
"""

from typing import Optional

from pydantic import BaseModel, Field


class POIMetadata(BaseModel):
    """Metadata for a single Point of Interest at L3 (leaf) level."""

    poi_id: str = Field(..., description="Unique POI identifier, e.g. 'yg-c20-e3-n2'")
    name: str = Field(..., description="POI name in Chinese, e.g. '交脚菩萨像'")
    site: str = Field(..., description="Parent site, e.g. '云冈石窟'")
    cave: str = Field(default="", description="Cave/area identifier, e.g. '第20窟'")
    location: str = Field(default="", description="Precise location within cave, e.g. '东壁第三龛'")
    description: str = Field(..., description="Detailed Chinese description of the POI")
    distinguishing_features: str = Field(
        default="",
        description="Visual features that distinguish this POI from similar ones",
    )
    reference_images: list[str] = Field(default_factory=list, description="Relative paths to reference images")
    tags: list[str] = Field(default_factory=list, description="e.g. ['菩萨', '交脚', '北魏']")
    dynasty: str = Field(default="", description="e.g. '北魏'")
    century: str = Field(default="", description="e.g. '5世纪中期'")


class ParentChunkData(BaseModel):
    """L1 or L2 parent chunk stored in PostgreSQL."""

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
    """Single result from image-to-image vector search (Collection 1)."""

    chunk_id: str = Field(..., description="Links to text Collection 2 chunk_id")
    poi_name: str
    site: str
    cave: str
    poi_description: str
    distinguishing_features: str
    similarity_score: float = Field(..., description="Cosine similarity from Chinese-CLIP")
    image_path: str = ""
    tags: list[str] = Field(default_factory=list)


class CandidateResult(BaseModel):
    """Candidate from Stage 1 text retrieval for Stage 2 LLM verification."""

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
    """Final result from the two-stage identification pipeline."""

    identified_poi: Optional[str] = Field(None, description="POI name, or None if not identified")
    confidence: str = Field("low", description="high | medium | low")
    reasoning: str = Field("", description="LLM's detailed reasoning in Chinese")
    merged_context: str = Field("", description="Auto-merged context from parent chunks")
    stage1_time_ms: float = 0.0
    stage2_time_ms: float = 0.0
    candidates: list[CandidateResult] = Field(default_factory=list)

"""
Image-to-text retrieval: Chinese-CLIP → Image Milvus → POI text description.

This is the image→text conversion layer that feeds into the text RAG pipeline.

Usage:
    from backend.image_retriever import identify_from_image
    result = identify_from_image("/path/to/query.jpg")
    # result is a string: POI name, site, cave, description, etc.
"""

import os
from pathlib import Path
from typing import Optional

from backend.embedding import clip_embeddings
from backend.milvus_client import milvus_manager


def image_search(
    image_path: str | Path,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[dict]:
    """
    Search Image Milvus Collection with a query image.
    Returns top-K POI metadata, each with similarity score.
    """
    milvus_manager.init_image_collection(dense_dim=clip_embeddings.dimension)

    image_vector = clip_embeddings.embed_image(image_path)
    results = milvus_manager.image_search(
        image_vector=image_vector,
        top_k=top_k,
    )

    if min_score > 0:
        results = [r for r in results if r.get("score", 0) >= min_score]

    return results


def format_image_search_results(results: list[dict]) -> str:
    """
    Format image search results into a single text block
    for downstream text RAG consumption.
    """
    if not results:
        return "【无法识别】未在知识库中找到与该图片匹配的文物点。"

    lines = []
    for i, r in enumerate(results, 1):
        poi_name = r.get("poi_name", "未知")
        site = r.get("site", "")
        cave = r.get("cave", "")
        score = r.get("score", 0)
        description = r.get("poi_description", "")
        features = r.get("distinguishing_features", "")

        location = f"{site} {cave}".strip()
        lines.append(
            f"候选{i} (相似度: {score:.4f}):\n"
            f"  名称: {poi_name}\n"
            f"  位置: {location}\n"
            f"  描述: {description}\n"
            f"  区分特征: {features}"
        )

    return "\n\n".join(lines)


def identify_from_image(
    image_path: str | Path,
    top_k: int = 5,
    min_score: float = 0.0,
) -> str:
    """
    Main entry point for image identification.

    Given a query image path, embeds it with Chinese-CLIP,
    searches the Image Milvus Collection, and returns formatted
    POI text descriptions for downstream text RAG.

    Args:
        image_path: Path to the query image file.
        top_k: Number of candidates to return.
        min_score: Minimum cosine similarity threshold (0.0 = no filter).

    Returns:
        Formatted Chinese text with POI names, locations, descriptions,
        and distinguishing features for top matches.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return f"【错误】图片文件不存在: {image_path}"

    try:
        results = image_search(image_path, top_k=top_k, min_score=min_score)
    except Exception as e:
        return f"【错误】图片检索失败: {str(e)}"

    return format_image_search_results(results)

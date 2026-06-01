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


# ═══════════════════════════════════════════════════════════════════
# Score distribution analysis
# ═══════════════════════════════════════════════════════════════════

def _analyze_score_distribution(results: list[dict]) -> dict:
    """Analyze score distribution of CLIP search results for confidence labeling.

    CLIP scores are cosine similarities (IP metric, L2-normalized), range ~[0, 1].
    Within the same cave, different POIs often cluster tightly (0.85-0.95),
    making them indistinguishable by score alone.

    Returns a dict with:
      - confidence: "high" | "medium" | "low" | "none"
      - label: human-readable Chinese confidence description
      - max_score, min_score, score_gap_1_2, score_std, is_clustered
    """
    scores = [r.get("score", 0) for r in results]
    if not scores:
        return {"confidence": "none", "label": "无匹配结果"}

    max_score = max(scores)
    n = len(scores)

    if n == 1:
        gap = 1.0
        std = 0.0
        is_clustered = False
    else:
        gap = scores[0] - scores[1]
        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / n
        std = variance ** 0.5
        is_clustered = (max_score - min(scores)) < 0.05

    # ── confidence classification ──
    if max_score < 0.5:
        confidence = "none"
        label = "无可靠匹配 — 最高相似度过低，图片内容可能不在知识库中"
    elif n == 1 or gap > 0.05:
        confidence = "high"
        label = "高置信度 — 明确匹配"
    elif gap > 0.02:
        confidence = "medium"
        label = "中置信度 — 可能匹配，建议核实"
    else:
        confidence = "low"
        if is_clustered:
            label = "低置信度 — 多个候选高度相似，可能属于同一石窟的不同位置，无法仅靠图像区分"
        else:
            label = "低置信度 — 无法精确区分"

    return {
        "confidence": confidence,
        "label": label,
        "max_score": round(max_score, 4),
        "min_score": round(min(scores), 4),
        "score_gap_1_2": round(gap, 4),
        "score_std": round(std, 4),
        "is_clustered": is_clustered,
    }


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

    Includes confidence analysis from score distribution to help the LLM
    decide when results are ambiguous (e.g., same-cave POI clustering).
    """
    if not results:
        return "【无法识别】未在知识库中找到与该图片匹配的文物点。"

    analysis = _analyze_score_distribution(results)

    lines = []

    # ── confidence banner ──
    confidence = analysis.get("confidence", "medium")
    if confidence == "none":
        lines.append("⚠️ " + analysis["label"])
    elif confidence == "low":
        lines.append(
            f"⚠️ {analysis['label']}\n"
            f"   最高分: {analysis['max_score']}, "
            f"最低分: {analysis['min_score']}, "
            f"第1-2名差距: {analysis['score_gap_1_2']}"
        )
        if analysis.get("is_clustered"):
            lines.append("   💡 建议：请提供更多线索（如拍摄位置、洞窟编号、可见特征）以缩小范围。")
    elif confidence == "medium":
        lines.append(f"ℹ️ {analysis['label']}（第1-2名差距: {analysis['score_gap_1_2']}）")

    # ── candidates ──
    if confidence != "none":
        lines.append("")

    for i, r in enumerate(results, 1):
        poi_name = r.get("poi_name", "未知")
        site = r.get("site", "")
        cave = r.get("cave", "")
        score = r.get("score", 0)
        description = r.get("poi_description", "")
        features = r.get("distinguishing_features", "")

        location = f"{site} {cave}".strip()

        # Show score gap relative to #1
        score_detail = f"{score:.4f}"
        if i > 1 and len(results) > 0:
            top_score = results[0].get("score", 0)
            score_detail += f" (与第1名差距: {top_score - score:.4f})"

        lines.append(
            f"候选{i} (相似度: {score_detail}):\n"
            f"  名称: {poi_name}\n"
            f"  位置: {location or '未知'}\n"
            f"  描述: {description or '无'}\n"
            f"  区分特征: {features or '无'}"
        )

    return "\n".join(lines)


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


def identify_from_image_vlm(
    image_path: str | Path,
    top_k: int = 5,
) -> str:
    """
    Identify image using Doubao Vision description → text retrieval.

    This bypasses CLIP entirely for fine-grained discrimination:
    1. Doubao Vision generates a pure visual description of the image
       (NO site/scene guessing — only observable features).
    2. The description is used as a query for BGE-M3 + BM25 hybrid
       text retrieval against the knowledge base.
    3. Results include retrieved text chunks with site/cave context
       from L1/L2 hierarchy.

    This handles same-cave POI differentiation much better than
    CLIP image→image search, because the VLM captures distinguishing
    visual details (hand gestures, attire, spatial position) that are
    then matched against detailed text descriptions.

    Args:
        image_path: Path to the query image file.
        top_k: Number of text retrieval results to return.

    Returns:
        Formatted text with visual description + knowledge base matches.
    """
    from backend.vlm_client import describe_image_with_vlm
    from backend.rag_utils import retrieve_with_context

    image_path = Path(image_path)
    if not image_path.exists():
        return f"【错误】图片文件不存在: {image_path}"

    # Step 1: Generate pure visual description via Doubao Vision.
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        description = describe_image_with_vlm(img_bytes)
    except Exception as e:
        return f"【错误】读取图片文件失败: {str(e)}"

    if not description:
        return "【错误】视觉描述生成失败，请检查 DOUBAO_API_KEY 配置或火山方舟模型开通状态。"

    # Step 2: Text retrieval with the visual description as query.
    # BGE-M3 + BM25 hybrid search finds matching L3 chunks (including
    # VLM-generated image descriptions from ingestion).
    try:
        context = retrieve_with_context(description, top_k=top_k)
    except Exception as e:
        return f"【错误】文本检索失败: {str(e)}"

    # Step 3: Format combined output.
    lines = [
        f"🔍 视觉描述（豆包 Vision）: {description}",
        "",
        f"📚 知识库匹配结果（BGE-M3 + BM25 文本检索）:",
        context,
    ]
    return "\n".join(lines)

"""
图像到文本检索：Chinese-CLIP → Image Milvus → POI 文本描述。

这是图像→文本转换层，将结果输入文本 RAG 管道。

用法：
    from backend.image_retriever import identify_from_image
    result = identify_from_image("/path/to/query.jpg")
    # result 为字符串：POI 名称、遗址、洞窟、描述等。
"""

import os
from pathlib import Path
from typing import Optional

from backend.embedding import clip_embeddings
from backend.milvus_client import milvus_manager


# ═══════════════════════════════════════════════════════════════════
# 分数分布分析
# ═══════════════════════════════════════════════════════════════════

def _analyze_score_distribution(results: list[dict]) -> dict:
    """分析 CLIP 搜索结果的分数分布，用于置信度标注。

    CLIP 分数为余弦相似度（IP 度量，L2 归一化），范围 ~[0, 1]。
    在同一洞窟内，不同的 POI 往往紧密聚集（0.85-0.95），
    仅凭分数无法区分。

    返回包含以下字段的字典：
      - confidence："high" | "medium" | "low" | "none"
      - label：人类可读的中文置信度描述
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

    # ── 置信度分类 ──
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
    用查询图像搜索 Image Milvus Collection。
    返回 top-K 个 POI 元数据，每个附带相似度分数。
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
    将图像搜索结果格式化为单个文本块，
    供下游文本 RAG 消费。

    包含基于分数分布的置信度分析，帮助 LLM
    判断结果是否模糊（如同一洞窟 POI 聚集的情况）。
    """
    if not results:
        return "【无法识别】未在知识库中找到与该图片匹配的文物点。"

    analysis = _analyze_score_distribution(results)

    lines = []

    # ── 置信度横幅 ──
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

    # ── 候选列表 ──
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

        # 显示与第 1 名的分数差距
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
    图像识别的主入口。

    给定查询图像路径，用 Chinese-CLIP 编码，
    搜索 Image Milvus Collection，返回格式化后的
    POI 文本描述，供下游文本 RAG 使用。

    Args:
        image_path：查询图像文件的路径。
        top_k：返回的候选数量。
        min_score：最小余弦相似度阈值（0.0 = 不过滤）。

    Returns:
        格式化后的中文文本，包含 POI 名称、位置、描述
        和 Top 匹配的区分特征。
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
    使用豆包 Vision 描述 → 文本检索进行图像识别。

    完全绕过 CLIP，实现细粒度区分：
    1. 豆包 Vision 生成图像的纯视觉描述
       （不猜测遗址/场景 — 仅描述可观察特征）。
    2. 将该描述作为查询，在知识库上执行 BGE-M3 + BM25
       混合文本检索。
    3. 结果包含检索到的文本块，附带来自 L1/L2 层级的
       遗址/洞窟上下文。

    此方法处理同一洞窟 POI 区分的效果远优于
    CLIP 图像→图像搜索，因为 VLM 能捕捉区分性
    视觉细节（手势、服饰、空间位置），再与详细的
    文本描述进行匹配。

    Args:
        image_path：查询图像文件的路径。
        top_k：返回的文本检索结果数量。

    Returns:
        格式化文本，包含视觉描述 + 知识库匹配结果。
    """
    from backend.vlm_client import describe_image_with_vlm
    from backend.rag_utils import retrieve_with_context

    image_path = Path(image_path)
    if not image_path.exists():
        return f"【错误】图片文件不存在: {image_path}"

    # Step 1：通过豆包 Vision 生成纯视觉描述。
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        description = describe_image_with_vlm(img_bytes)
    except Exception as e:
        return f"【错误】读取图片文件失败: {str(e)}"

    if not description:
        return "【错误】视觉描述生成失败，请检查 DOUBAO_API_KEY 配置或火山方舟模型开通状态。"

    # Step 2：将视觉描述作为查询进行文本检索。
    # BGE-M3 + BM25 混合搜索匹配 L3 块（包括
    # 摄入时由 VLM 生成的图像描述）。
    try:
        context = retrieve_with_context(description, top_k=top_k)
    except Exception as e:
        return f"【错误】文本检索失败: {str(e)}"

    # Step 3：格式化组合输出。
    lines = [
        f"🔍 视觉描述（豆包 Vision）: {description}",
        "",
        f"📚 知识库匹配结果（BGE-M3 + BM25 文本检索）:",
        context,
    ]
    return "\n".join(lines)

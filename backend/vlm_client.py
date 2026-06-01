"""
Shared Doubao Vision client (Volcengine Ark) for image description.

Used by both document_loader (ingestion) and image_retriever (query).
The prompt instructs the VLM to describe visual features only — never
to guess site/scene names. Site identification is handled by text retrieval.
"""

import base64
import io
import os

from dotenv import load_dotenv

load_dotenv()


def describe_image_with_vlm(image_bytes: bytes) -> str:
    """Generate a pure visual description of an image using Doubao Vision.

    Returns a Chinese description focusing on observable features:
    subject type/count, poses/gestures, attire/decorations, spatial
    layout, colors/materials. Deliberately avoids guessing site names.

    Returns empty string on any failure (graceful degradation).
    """
    from openai import OpenAI

    api_key = os.getenv("DOUBAO_API_KEY", "")
    base_url = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    model = os.getenv("DOUBAO_VISION_MODEL", "doubao-seed-2-0-pro-260215")

    if not api_key:
        return ""

    try:
        # Detect image format.
        fmt = "png"
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            fmt = img.format.lower() if img.format else "png"
        except Exception:
            pass

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:image/{fmt};base64,{b64}"

        client = OpenAI(api_key=api_key, base_url=base_url)

        prompt = (
            "请详细描述这张图片的视觉内容。只描述你看到的内容，不要猜测或识别具体地点、场景名称。"
            "包括：主体类型与数量、姿态与手势、服饰与装饰、空间位置关系、色彩与材质、"
            "以及其他可辨认的视觉细节。200字以内，中文。"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ]
            }],
            max_tokens=300,
        )

        content = resp.choices[0].message.content
        return content.strip() if content else ""

    except Exception:
        return ""

"""
共享的豆包视觉客户端（火山引擎 Ark），用于图片描述。

同时被 document_loader（文档导入）和 image_retriever（查询）使用。
提示词指示 VLM 仅描述视觉特征——绝不
猜测地点/场景名称。地点识别由文本检索处理。
"""

import base64
import io
import os

from dotenv import load_dotenv

load_dotenv()


def _prepare_image_for_vlm(image_bytes: bytes) -> tuple[bytes, str]:
    """压缩 VLM 输入图片，降低传输和视觉模型处理耗时。"""
    max_side = int(os.getenv("VLM_IMAGE_MAX_SIDE", "1280"))
    jpeg_quality = int(os.getenv("VLM_IMAGE_JPEG_QUALITY", "82"))

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        img.load()

        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        width, height = img.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / longest
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        if img.mode == "L":
            img = img.convert("RGB")

        output = io.BytesIO()
        img.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
        return output.getvalue(), "jpeg"
    except Exception:
        return image_bytes, "png"


def describe_image_with_vlm(image_bytes: bytes) -> str:
    """使用豆包视觉模型生成图片的纯视觉描述。

    返回中文描述，聚焦于可观察的特征：
    主体类型与数量、姿态与手势、服饰与装饰、空间
    布局、色彩与材质。刻意避免猜测地点名称。

    任何失败时返回空字符串（优雅降级）。
    """
    from openai import OpenAI

    api_key = os.getenv("DOUBAO_API_KEY", "")
    base_url = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    model = os.getenv("DOUBAO_VISION_MODEL", "doubao-seed-2-0-pro-260215")

    if not api_key:
        return ""

    try:
        image_bytes, fmt = _prepare_image_for_vlm(image_bytes)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:image/{fmt};base64,{b64}"

        client = OpenAI(api_key=api_key, base_url=base_url)

        prompt = (
            "请详细描述这张图片的视觉内容。只描述你看到的内容，不要猜测或识别具体地点、场景名称。"
            "包括：主体类型与数量、姿态与手势、服饰与装饰、空间位置关系、色彩与材质、"
            "以及其他可辨认的关键视觉细节。150字以内，中文。"
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
            max_tokens=180,
        )

        content = resp.choices[0].message.content
        return content.strip() if content else ""

    except Exception as e:
        print(f"[VLM] 豆包视觉模型调用失败: {type(e).__name__}: {e}")
        return ""

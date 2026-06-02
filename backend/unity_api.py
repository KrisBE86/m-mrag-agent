"""
Unity 客户端 API 路由。

提供:
  - POST /unity/chat  — 主对话端点（SSE 流式，音频+图片 → 文字+语音）
  - POST /unity/stt   — 独立语音识别端点（调试/测试用）

Unity 端点无需认证，直接调用即可。
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from backend.unity_agent_api import unity_chat_stream
from backend.unity_schemas import UnityChatRequest, UnitySTTRequest
from backend.stt_service import recognize as stt_recognize

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/unity", tags=["unity"])


@router.post("/chat")
async def unity_chat_endpoint(
    req: UnityChatRequest,
):
    """
    Unity 对话端点（SSE 流式）。

    请求:
      - audio_base64: Base64 编码的语音数据（WAV/PCM）
      - image_base64: 可选，Base64 编码的摄像头图片（JPEG/PNG）

    响应: text/event-stream，事件类型:
      - thinking: Agent 推理/工具调用
      - token:    逐字输出（打字机效果）
      - text:     完整句子（配合 index 与 audio 配对）
      - audio:    TTS 语音 base64（与对应 text 事件同 index）
      - done:     对话结束
    """

    async def event_generator():
        async for sse_msg in unity_chat_stream(
            audio_base64=req.audio_base64,
            image_base64=req.image_base64,
        ):
            yield sse_msg

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/stt")
async def unity_stt_endpoint(
    req: UnitySTTRequest,
):
    """
    Unity STT 独立端点。

    用于单独测试语音识别功能，不经过 Agent 处理。

    请求:
      - audio_base64: Base64 编码的语音数据

    响应:
      - text: 识别出的中文文本
    """
    text = await stt_recognize(req.audio_base64)
    if text is None:
        return JSONResponse(
            status_code=500,
            content={"error": "STT 识别失败，请检查配置或后端日志"},
        )
    return {"text": text}

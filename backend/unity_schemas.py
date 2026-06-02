"""
Unity 客户端接口的数据模型。

定义 Unity 端请求和 SSE 事件的 Pydantic 模型。
SSE 事件采用 TypedDict 以保持与现有 JSON 序列化兼容。
"""

from typing import Optional

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------

class UnityChatRequest(BaseModel):
    """Unity 对话请求（/unity/chat）。

    音频为必填字段，图片为可选项（取决于用户是否开启了摄像头）。
    """

    audio_base64: str
    image_base64: Optional[str] = None

    @field_validator("audio_base64")
    @classmethod
    def audio_base64_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("audio_base64 不能为空")
        return v


class UnitySTTRequest(BaseModel):
    """Unity STT 独立请求（/unity/stt）。

    用于单独调试语音识别功能，不经过 Agent 处理。
    """

    audio_base64: str

    @field_validator("audio_base64")
    @classmethod
    def audio_base64_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("audio_base64 不能为空")
        return v

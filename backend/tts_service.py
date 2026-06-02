"""
TTS（语音合成）服务抽象层 + 火山引擎实现。

设计:
  - BaseTTSService: 抽象基类，定义统一的 synthesize 接口。
  - VolcanoTTSService: 火山引擎 TTS 实现（与原 Unity C# TTS.cs 对应）。
  - 模块级单例 _tts_service，通过 synthesize() 便捷函数调用。
  - 未配置凭证时自动降级，返回 None 并记录警告日志。

环境变量:
  TTS_APPID:         火山引擎 TTS 应用 ID（必填）
  TTS_ACCESS_TOKEN:  火山引擎 TTS 访问令牌（必填）
  TTS_CLUSTER:       集群名（默认 volcano_tts）
  TTS_VOICE_TYPE:    音色类型（默认 zh_female_vv_uranus_bigtts）
  TTS_API_URL:       TTS API 端点（有默认值）
"""

import os
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# TTS 输出音频采样率（Hz），与火山引擎 wav 编码一致
TTS_SAMPLE_RATE: int = 24000


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseTTSService(ABC):
    """TTS 服务抽象基类。所有 TTS 提供商必须实现此接口。"""

    @abstractmethod
    async def synthesize(self, text: str) -> Optional[str]:
        """
        将文字合成为 Base64 编码的 WAV 音频。

        参数:
            text: 要合成的文本（建议以完整句子为单位）。

        返回:
            Base64 编码的 WAV 音频数据，失败或未配置时返回 None。
        """
        ...


# ---------------------------------------------------------------------------
# 火山引擎 TTS 实现
# ---------------------------------------------------------------------------

class VolcanoTTSService(BaseTTSService):
    """
    火山引擎（ByteDance Volcengine）TTS 服务。

    API 参考:
      POST https://openspeech.bytedance.com/api/v1/tts

    对应原 Unity 项目 C# TTS.cs 的 SpeakAsync 逻辑。
    """

    def __init__(
        self,
        appid: str = "",
        access_token: str = "",
        cluster: str = "volcano_tts",
        voice_type: str = "zh_female_vv_uranus_bigtts",
        api_url: str = "https://openspeech.bytedance.com/api/v1/tts",
    ) -> None:
        self._appid = appid
        self._access_token = access_token
        self._cluster = cluster
        self._voice_type = voice_type
        self._api_url = api_url

    @property
    def configured(self) -> bool:
        """是否已配置（appid 和 access_token 均非空）。"""
        return bool(self._appid and self._access_token)

    async def synthesize(self, text: str) -> Optional[str]:
        """
        调用火山引擎 TTS API 合成语音。

        参数:
            text: 要合成的文本。

        返回:
            Base64 编码的 WAV 音频，失败返回 None。
        """
        if not text or not text.strip():
            return None

        if not self.configured:
            logger.warning("[TTS] ⚠️ TTS_APPID 或 TTS_ACCESS_TOKEN 未配置，跳过 TTS")
            return None

        reqid = str(uuid.uuid4())
        payload = {
            "app": {
                "appid": self._appid,
                "token": self._access_token,
                "cluster": self._cluster,
            },
            "user": {"uid": "python_backend"},
            "audio": {
                "voice_type": self._voice_type,
                "encoding": "wav",
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
            },
            "request": {
                "reqid": reqid,
                "text": text,
                "text_type": "plain",
                "operation": "query",
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer;{self._access_token}",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self._api_url, json=payload, headers=headers)

            if response.status_code != 200:
                logger.error("[TTS] HTTP 错误: %s", response.status_code)
                return None

            data = response.json()
            if data.get("code") == 0 or data.get("message") == "Success":
                audio_b64 = data.get("data")
                if audio_b64:
                    logger.info("[TTS] 合成成功，文本长度: %d 字符", len(text))
                return audio_b64
            else:
                logger.error("[TTS] API 业务错误: %s", data.get("message"))
                return None

        except Exception as e:
            logger.error("[TTS] 请求异常: %s", e)
            return None


# ---------------------------------------------------------------------------
# 模块级单例 & 便捷函数
# ---------------------------------------------------------------------------

_tts_service: Optional[BaseTTSService] = None


def _get_tts_service() -> BaseTTSService:
    """
    延迟初始化 TTS 服务单例。

    读取环境变量构造合适的 TTS 实现。
    当前仅支持火山引擎，未来可通过 TTS_PROVIDER 环境变量扩展。
    """
    global _tts_service
    if _tts_service is None:
        provider = os.getenv("TTS_PROVIDER", "volcano")
        if provider == "volcano":
            _tts_service = VolcanoTTSService(
                appid=os.getenv("TTS_APPID", ""),
                access_token=os.getenv("TTS_ACCESS_TOKEN", ""),
                cluster=os.getenv("TTS_CLUSTER", "volcano_tts"),
                voice_type=os.getenv("TTS_VOICE_TYPE", "zh_female_vv_uranus_bigtts"),
                api_url=os.getenv(
                    "TTS_API_URL",
                    "https://openspeech.bytedance.com/api/v1/tts",
                ),
            )
        else:
            logger.warning("[TTS] 未知 TTS 提供商: %s，回退到火山引擎", provider)
            _tts_service = VolcanoTTSService()
    return _tts_service


async def synthesize(text: str) -> Optional[str]:
    """
    语音合成便捷函数。

    用法:
        audio_b64 = await synthesize("你好世界")
        if audio_b64:
            # 发送 base64 WAV 给 Unity 播放
            ...

    参数:
        text: 要合成的文本。

    返回:
        Base64 编码的 WAV 音频（24000 Hz），失败返回 None。
    """
    return await _get_tts_service().synthesize(text)

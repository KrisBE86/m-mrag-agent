"""
STT（语音识别）服务抽象层 + 火山引擎实现。

设计:
  - BaseSTTService: 抽象基类，定义统一的 recognize 接口。
  - VolcanoSTTService: 火山引擎 ASR 实现（与原 Unity C# STT.cs 对应）。
  - 模块级单例 _stt_service，通过 recognize() 便捷函数调用。
  - 未配置凭证时自动降级，返回 None 并记录警告日志。

环境变量:
  STT_APPID:        火山引擎 STT 应用 ID（必填）
  STT_ACCESS_KEY:   火山引擎 STT 访问密钥（必填）
  STT_API_URL:      ASR API 端点（有默认值）
  STT_RESOURCE_ID:  资源类型（默认 volc.bigasr.auc_turbo）
  STT_MODEL_NAME:   模型名（默认 bigmodel）
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


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseSTTService(ABC):
    """STT 服务抽象基类。所有 STT 提供商必须实现此接口。"""

    @abstractmethod
    async def recognize(self, audio_base64: str) -> Optional[str]:
        """
        将 Base64 编码的音频数据转换为文字。

        参数:
            audio_base64: Base64 编码的音频（WAV/PCM）。

        返回:
            识别出的文本，失败或未配置时返回 None。
        """
        ...


# ---------------------------------------------------------------------------
# 火山引擎 ASR 实现
# ---------------------------------------------------------------------------

class VolcanoSTTService(BaseSTTService):
    """
    火山引擎（ByteDance Volcengine）ASR 服务。

    API 参考:
      POST https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash

    对应原 Unity 项目 C# STT.cs 的 RecognizeAsync 逻辑。
    """

    def __init__(
        self,
        appid: str = "",
        access_key: str = "",
        api_url: str = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash",
        resource_id: str = "volc.bigasr.auc_turbo",
        model_name: str = "bigmodel",
    ) -> None:
        self._appid = appid
        self._access_key = access_key
        self._api_url = api_url
        self._resource_id = resource_id
        self._model_name = model_name

    @property
    def configured(self) -> bool:
        """是否已配置（appid 和 access_key 均非空）。"""
        return bool(self._appid and self._access_key)

    async def recognize(self, audio_base64: str) -> Optional[str]:
        """
        调用火山引擎 ASR API 识别语音。

        参数:
            audio_base64: Base64 编码的音频数据（不含 data URI 前缀）。

        返回:
            识别出的中文文本，失败返回 None。
        """
        if not audio_base64:
            logger.warning("[STT] audio_base64 为空，跳过识别")
            return None

        if not self.configured:
            logger.warning("[STT] ⚠️ STT_APPID 或 STT_ACCESS_KEY 未配置，跳过识别")
            return None

        payload = {
            "user": {"uid": self._appid},
            "audio": {"data": audio_base64},
            "request": {"model_name": self._model_name},
        }

        headers = {
            "Content-Type": "application/json",
            "X-Api-App-Key": self._appid,
            "X-Api-Access-Key": self._access_key,
            "X-Api-Resource-Id": self._resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self._api_url, json=payload, headers=headers)

            if response.status_code != 200:
                logger.error(
                    "[STT] API 返回错误，状态码: %s, 响应: %s",
                    response.status_code, response.text,
                )
                return None

            data = response.json()

            # 火山 ASR 返回结构: result 可能是 list 或 dict
            result_node = data.get("result", {})
            if isinstance(result_node, list):
                text = result_node[0].get("text", "") if result_node else ""
            elif isinstance(result_node, dict):
                text = result_node.get("text", "")
            else:
                text = ""

            if not text:
                logger.warning("[STT] 识别结果为空，原始响应: %s", data)
            else:
                logger.info("[STT] 识别成功: %s", text)

            return text

        except Exception as e:
            logger.error("[STT] 请求异常: %s", e)
            return None


# ---------------------------------------------------------------------------
# 模块级单例 & 便捷函数
# ---------------------------------------------------------------------------

_stt_service: Optional[BaseSTTService] = None


def _get_stt_service() -> BaseSTTService:
    """
    延迟初始化 STT 服务单例。

    读取环境变量构造合适的 STT 实现。
    当前仅支持火山引擎，未来可通过 STT_PROVIDER 环境变量扩展。
    """
    global _stt_service
    if _stt_service is None:
        provider = os.getenv("STT_PROVIDER", "volcano")
        if provider == "volcano":
            _stt_service = VolcanoSTTService(
                appid=os.getenv("STT_APPID", ""),
                access_key=os.getenv("STT_ACCESS_KEY", ""),
                api_url=os.getenv(
                    "STT_API_URL",
                    "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash",
                ),
                resource_id=os.getenv("STT_RESOURCE_ID", "volc.bigasr.auc_turbo"),
                model_name=os.getenv("STT_MODEL_NAME", "bigmodel"),
            )
        else:
            logger.warning("[STT] 未知 STT 提供商: %s，回退到火山引擎", provider)
            _stt_service = VolcanoSTTService()
    return _stt_service


async def recognize(audio_base64: str) -> Optional[str]:
    """
    语音识别便捷函数。

    用法:
        text = await recognize(audio_base64)
        if text is None:
            # 识别失败或未配置 STT
            ...

    参数:
        audio_base64: Base64 编码的音频数据。

    返回:
        识别出的文本，失败返回 None。
    """
    return await _get_stt_service().recognize(audio_base64)

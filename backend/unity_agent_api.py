"""
Unity Agent API — STT → Agent → SentenceBuffer → TTS 完整流水线。

将 Unity 客户端发来的语音+图片请求通过以下流水线处理:
  1. STT: 语音 base64 → 中文文本
  2. 图片处理: base64 解码 → 保存为临时文件
  3. Agent: 构造消息 → LangGraph Agent 流式推理
  4. 分句: SentenceBuffer 将最终回复 token 流聚合为完整句子
  5. TTS: 每句异步合成语音 → SSE 事件返回

阶段分离策略（参考现有 agent_api.py）:
  - Agent 所有文本输出被缓冲。当出现 tool_call_chunks 时，
    缓冲内容作为 "thinking" 事件发射（Agent 正在推理调用工具）。
  - 流结束后，剩余缓冲即为最终回复，通过 SentenceBuffer 分句，
    逐 token 发送，每句触发 TTS。

TTS 并发模型:
  - asyncio.Queue 作为事件输出队列
  - 最终回复文本逐 token 写入队列
  - 每完成一句即启动 asyncio.Task 调用 TTS
  - TTS 完成后将 audio 事件写入同一队列
  - "done" 仅在 Agent 流结束且所有 TTS 完成后才发出
"""

import asyncio
import base64
import json
import logging
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

from backend.agent_api import (
    _build_user_message,
    _extract_content,
    _get_async_agent,
)
from backend.sentence_buffer import SentenceBuffer
from backend.stt_service import recognize as stt_recognize
from backend.tts_service import TTS_SAMPLE_RATE, synthesize as tts_synthesize

logger = logging.getLogger(__name__)

# 用户上传图片的保存目录（与 api.py 中的 /images/upload 一致）
USER_UPLOADS_DIR = Path("data/user_uploads")

# Unity 端不传 session_id。按后端进程生命周期生成一个会话：
# 同一次后端启动内保持连续上下文；重启后自动换新会话。
UNITY_PROCESS_THREAD_ID = f"unity-{uuid.uuid4().hex}"


def _unity_default_config() -> dict:
    return {"configurable": {"thread_id": UNITY_PROCESS_THREAD_ID}}


def _sse_named_event(event: str, data: dict) -> str:
    """
    格式化为带 event: 字段的 SSE 数据行。

    输出格式:
        event: <event_name>
        data: <json_payload>

    与 Web 前端使用的纯 data: 格式并存，Unity 端通过 event 字段区分消息类型。
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def unity_chat_stream(
    audio_base64: str,
    image_base64: Optional[str] = None,
    config: Optional[dict] = None,
) -> AsyncGenerator[str, None]:
    """
    Unity 对话流式处理的异步生成器。

    参数:
        audio_base64: Unity 发来的 Base64 音频数据（WAV/PCM）。
        image_base64: 可选，Unity 发来的 Base64 图片数据（JPEG/PNG）。
        config: LangGraph 配置（thread_id 等）。不传时使用本进程 Unity 会话。

    Yields:
        SSE 格式化字符串，事件类型为 thinking / token / text / audio / done。
    """
    cfg = config or _unity_default_config()
    temp_image_path: Optional[Path] = None

    # ---- 第 1 步：STT 语音识别 ----
    user_text = await stt_recognize(audio_base64)
    if user_text is None:
        yield _sse_named_event(
            "done", {"type": "done", "finish_reason": "error"}
        )
        return

    logger.info("[Unity] STT 识别结果: %s", user_text)

    try:
        # ---- 第 2 步：图片处理（可选） ----
        image_path: Optional[str] = None
        if image_base64:
            # 去除可能的 data URI 前缀（如 "data:image/jpeg;base64,"）
            clean_b64 = image_base64
            if "," in image_base64 and image_base64.startswith("data:"):
                clean_b64 = image_base64.split(",", 1)[1]

            try:
                image_bytes = base64.b64decode(clean_b64)
                USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
                temp_image_path = USER_UPLOADS_DIR / f"{uuid.uuid4()}.jpg"
                temp_image_path.write_bytes(image_bytes)
                image_path = str(temp_image_path)
                logger.info("[Unity] 图片已保存: %s", image_path)
            except Exception as e:
                logger.warning("[Unity] 图片解码失败: %s，忽略图片继续对话", e)
                # 图片解码失败不阻断流程，仅忽略图片

        # ---- 第 3 步：构造消息 ----
        content = _build_user_message(user_text, image_path)

        # ---- 第 4 步：Agent 流式推理 ----
        agent = await _get_async_agent()
        from backend.tools import reset_tool_call_guards, set_active_thread_id

        reset_tool_call_guards()
        set_active_thread_id((cfg.get("configurable") or {}).get("thread_id"))

        # 阶段分离：缓冲区累积 Agent 文本
        text_buffer: str = ""

        async for msg, metadata in agent.astream(
            {"messages": [{"role": "user", "content": content}]},
            config=cfg,
            stream_mode="messages",
        ):
            from langchain_core.messages import AIMessageChunk

            if not isinstance(msg, AIMessageChunk):
                continue

            tool_calls = getattr(msg, "tool_call_chunks", None)

            if tool_calls:
                # 工具调用：将之前的缓冲文本作为 thinking 事件发射
                for tc in tool_calls:
                    name = tc.get("name", "")
                    if name:
                        yield _sse_named_event("thinking", {
                            "type": "thinking",
                            "text": text_buffer.strip() or None,
                            "tool_name": name,
                        })
                        text_buffer = ""
                continue

            chunk_text = _extract_content(msg)
            if chunk_text:
                text_buffer += chunk_text

        # ---- 第 5 步：流结束，处理最终回复 ----
        if not text_buffer.strip():
            # Agent 未产出最终回复（可能只调用了工具）
            yield _sse_named_event(
                "done", {"type": "done", "finish_reason": "stop"}
            )
            return

        # 创建输出队列和 TTS 并发控制
        output_queue: asyncio.Queue = asyncio.Queue()
        tts_pending_count = 0

        async def run_tts(text: str, index: int) -> None:
            """后台 TTS 任务：合成一句语音并写入输出队列。"""
            nonlocal tts_pending_count
            audio_b64 = await tts_synthesize(text)
            event_data: dict = {
                "type": "audio",
                "content": text,
                "index": index,
                "sample_rate": TTS_SAMPLE_RATE,
            }
            if audio_b64:
                event_data["audio_base64"] = audio_b64
            await output_queue.put(("audio", event_data))
            tts_pending_count -= 1
            if tts_pending_count == 0:
                await output_queue.put(None)  # 所有 TTS 完成 → 结束信号

        # 将最终回复文本逐 token 喂入 SentenceBuffer
        sentence_buf = SentenceBuffer()
        sentence_index = 0

        for token in text_buffer:
            await output_queue.put(
                ("token", {"type": "token", "content": token})
            )
            for sentence in sentence_buf.add(token):
                await output_queue.put(
                    ("text", {
                        "type": "text",
                        "content": sentence,
                        "index": sentence_index,
                    })
                )
                tts_pending_count += 1
                asyncio.create_task(run_tts(sentence, sentence_index))
                sentence_index += 1

        # 处理末尾残留文字
        remaining = sentence_buf.flush()
        if remaining:
            await output_queue.put(
                ("text", {
                    "type": "text",
                    "content": remaining,
                    "index": sentence_index,
                })
            )
            tts_pending_count += 1
            asyncio.create_task(run_tts(remaining, sentence_index))

        # 如果没有任何 TTS 任务（文本太短无分割），直接结束
        if tts_pending_count == 0:
            await output_queue.put(None)

        # ---- 第 6 步：从输出队列读取并 yield SSE 事件 ----
        while True:
            item = await output_queue.get()
            if item is None:
                yield _sse_named_event(
                    "done", {"type": "done", "finish_reason": "stop"}
                )
                return

            event_type, data = item
            yield _sse_named_event(event_type, data)

    except Exception:
        logger.exception("[Unity] unity_chat_stream 异常")
        yield _sse_named_event(
            "done", {"type": "done", "finish_reason": "error"}
        )

    finally:
        # ---- 清理临时图片文件 ----
        if temp_image_path and temp_image_path.exists():
            try:
                temp_image_path.unlink()
                logger.debug("[Unity] 临时图片已删除: %s", temp_image_path)
            except OSError:
                pass

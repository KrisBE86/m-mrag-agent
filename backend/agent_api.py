"""
Agent API — 同步对话和 SSE 流式输出包装器。

将现有 MRagAgent (agent.py) 包装为 FastAPI 可用。
支持:
  - 同步对话: 调用 agent 并返回完整响应。
  - SSE 流式输出: 使用 agent.astream() 的异步生成器，遵循
    SuperMew 的模式——无需线程，全部在同一事件循环中运行。

流式输出阶段设计:
  - "thinking" 阶段: agent 的内部推理文本 + 工具调用。
    在前端以可折叠区域展示。
  - "answering" 阶段: 所有工具调用完成后的最终回复。
    这是主要可见的回答内容。

  阶段检测策略:
    - 所有 agent 文本被缓冲。当出现 tool_call_chunks 时，缓冲区
      内容作为 "thinking" 刷新（agent 正在推理调用哪个工具）。
    - 流结束时，剩余缓冲文本即为最终回答，
      作为 "content" 刷新。
    - 此方案简单，不依赖 LangGraph 内部的
      元数据结构，在不同版本间具有鲁棒性。
"""

import json
from typing import AsyncGenerator, Optional

from agent import build_agent, build_agent_async

# 模块级单例 agent（带同步 checkpointer，用于同步场景）。
_sync_agent = build_agent()

# 延迟加载的异步 agent —— 仅在事件循环内创建。
_async_agent = None

# 当前固定 thread_id（单用户测试）。
DEFAULT_CONFIG = {"configurable": {"thread_id": "main-chat"}}

# Redis 上下文记忆 key 前缀
CONTEXT_KEY = "mragagent:context:main-chat"


async def _get_async_agent():
    """延迟初始化异步 agent。必须在事件循环内调用。"""
    global _async_agent
    if _async_agent is None:
        _async_agent = await build_agent_async()
    return _async_agent


def _extract_content(msg) -> str:
    """从 AIMessageChunk 中提取文本内容，处理 str 和 list 两种格式。"""
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        parts = []
        for block in msg.content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _sse_event(data: dict) -> str:
    """将字典格式化为 SSE data 事件。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_user_message(text: str, image_path: str | None = None) -> str:
    """构造发送给 Agent 的用户消息。

    以中性标记传递图片路径（纯上下文），不给 Agent 下指令。
    用户的文字提问是唯一的行动依据——Agent 根据文字内容自行判断是否需要识图。
    """
    if not image_path:
        return text or "你好"
    if text:
        return f"[用户上传了图片: {image_path}]\n\n{text}"
    return f"[用户上传了图片: {image_path}]\n\n请帮我看看这张图片"


def chat_sync(user_message: str, image_path: str | None = None, config: Optional[dict] = None) -> str:
    """
    同步对话: 调用 agent 并返回响应文本。
    """
    cfg = config or DEFAULT_CONFIG
    content = _build_user_message(user_message, image_path)
    result = _sync_agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config=cfg,
    )
    return result["messages"][-1].content


async def chat_stream(
    user_message: str,
    image_path: str | None = None,
    config: Optional[dict] = None,
) -> AsyncGenerator[str, None]:
    """
    SSE 流式对话，带思考/回答阶段分离。

    策略:
      - 缓冲所有 agent 文本。当出现 tool_call_chunks 时，缓冲的
        文本作为 "thinking" 刷新（agent 正在推理调用
        哪个工具）。
      - 工具调用以 "tool_call" 事件发出。
      - 流结束时，剩余缓冲文本即为最终回答，
        作为 "content" 发出。
    """
    agent = await _get_async_agent()
    cfg = config or DEFAULT_CONFIG

    content = _build_user_message(user_message, image_path)

    text_buffer: str = ""

    def _flush_as_thinking():
        """如果缓冲区非空，将当前缓冲区作为 thinking 事件刷新。"""
        nonlocal text_buffer
        if text_buffer:
            yield _sse_event({"type": "thinking", "text": text_buffer})
            text_buffer = ""

    try:
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
                # 工具调用之前的任何文本都是 agent 的内部推理
                for event in _flush_as_thinking():
                    yield event
                # 发出工具调用事件
                for tc in tool_calls:
                    name = tc.get("name", "")
                    if name:
                        yield _sse_event({"type": "tool_call", "name": name})
                continue

            content = _extract_content(msg)
            if content:
                text_buffer += content

        # 流结束 —— 剩余内容即为最终回答
        if text_buffer:
            yield _sse_event({"type": "content", "text": text_buffer})

    except Exception as e:
        yield _sse_event({"type": "error", "text": str(e)})

    yield "data: [DONE]\n\n"

"""
Agent API — synchronous chat and SSE streaming wrappers.

Wraps the existing MRagAgent (agent.py) for use with FastAPI.
Supports:
  - sync chat: invoke the agent and return the full response.
  - SSE streaming: async generator using agent.astream(), following
    SuperMew's pattern — no threads, everything in the same event loop.

Streaming phase design:
  - "thinking" phase: agent's internal reasoning text + tool calls.
    These are displayed in a collapsible section in the frontend.
  - "answering" phase: the final response after all tool calls complete.
    This is the main visible answer.

  Phase detection strategy:
    - All agent text is buffered. When tool_call_chunks appear, the buffer
      is flushed as "thinking" (the agent was reasoning about what tool to call).
    - When the stream ends, any remaining buffered text is the final answer
      and is flushed as "content".
    - This approach is simple and does not depend on LangGraph's internal
      metadata structure, making it robust across versions.
"""

import json
from typing import AsyncGenerator, Optional

from agent import build_agent, build_agent_async

# Module-level singleton agent (sync checkpointer for synchronous use).
_sync_agent = build_agent()

# Lazy async agent — only created inside an event loop.
_async_agent = None

# Fixed thread_id for now (single-user testing).
DEFAULT_CONFIG = {"configurable": {"thread_id": "main-chat"}}


def _get_async_agent():
    """Lazy-init the async agent. Must be called from within an event loop."""
    global _async_agent
    if _async_agent is None:
        _async_agent = build_agent_async()
    return _async_agent


def _extract_content(msg) -> str:
    """Extract text content from an AIMessageChunk, handling str and list formats."""
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
    """Format a dict as an SSE data event."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def chat_sync(user_message: str, config: Optional[dict] = None) -> str:
    """
    Synchronous chat: invoke the agent and return the response text.
    """
    cfg = config or DEFAULT_CONFIG
    result = _sync_agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=cfg,
    )
    return result["messages"][-1].content


async def chat_stream(
    user_message: str,
    config: Optional[dict] = None,
) -> AsyncGenerator[str, None]:
    """
    SSE streaming chat with thinking/answering phase separation.

    Strategy:
      - Buffer all agent text. When tool_call_chunks appear, the buffered
        text is flushed as "thinking" (the agent was reasoning about which
        tool to invoke).
      - Tool invocations are emitted as "tool_call" events.
      - When the stream ends, any remaining buffered text is the final answer
        and is emitted as "content".
    """
    agent = _get_async_agent()
    cfg = config or DEFAULT_CONFIG

    text_buffer: str = ""

    def _flush_as_thinking():
        """Flush the current buffer as a thinking event if non-empty."""
        nonlocal text_buffer
        if text_buffer:
            yield _sse_event({"type": "thinking", "text": text_buffer})
            text_buffer = ""

    try:
        async for msg, metadata in agent.astream(
            {"messages": [{"role": "user", "content": user_message}]},
            config=cfg,
            stream_mode="messages",
        ):
            from langchain_core.messages import AIMessageChunk

            if not isinstance(msg, AIMessageChunk):
                continue

            tool_calls = getattr(msg, "tool_call_chunks", None)

            if tool_calls:
                # Any text before a tool call is the agent's internal reasoning
                for event in _flush_as_thinking():
                    yield event
                # Emit tool call events
                for tc in tool_calls:
                    name = tc.get("name", "")
                    if name:
                        yield _sse_event({"type": "tool_call", "name": name})
                continue

            content = _extract_content(msg)
            if content:
                text_buffer += content

        # Stream ended — whatever is left is the final answer
        if text_buffer:
            yield _sse_event({"type": "content", "text": text_buffer})

    except Exception as e:
        yield _sse_event({"type": "error", "text": str(e)})

    yield "data: [DONE]\n\n"

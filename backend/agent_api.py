"""
Agent API — synchronous chat and SSE streaming wrappers.

Wraps the existing MRagAgent (agent.py) for use with FastAPI.
Supports:
  - sync chat: invoke the agent and return the full response.
  - SSE streaming: async generator using agent.astream(), following
    SuperMew's pattern — no threads, everything in the same event loop.
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
    SSE streaming chat, following SuperMew's pattern.

    Uses async agent with AsyncSqliteSaver + agent.astream().
    """
    agent = _get_async_agent()
    cfg = config or DEFAULT_CONFIG

    try:
        async for msg, metadata in agent.astream(
            {"messages": [{"role": "user", "content": user_message}]},
            config=cfg,
            stream_mode="messages",
        ):
            from langchain_core.messages import AIMessageChunk

            if not isinstance(msg, AIMessageChunk):
                continue
            if getattr(msg, "tool_call_chunks", None):
                continue

            content = ""
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, str):
                        content += block
                    elif isinstance(block, dict) and block.get("type") == "text":
                        content += block.get("text", "")

            if content:
                yield f"data: {json.dumps({'type': 'content', 'text': content}, ensure_ascii=False)}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'text': str(e)}, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"

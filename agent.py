"""
MRagAgent — Multimodal RAG Agent for Cultural Heritage POI Identification.

- LLM: DeepSeek via OpenAI-compatible API (ChatOpenAI).
- Tools: calculator, get_current_weather, identify_from_image, search_knowledge_base.
- Agent framework: LangChain create_agent with LangGraph SqliteSaver for conversation memory.
- Image → text conversion → text RAG pipeline (two tools, serial orchestration).
"""

import os
import sqlite3

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()


def _init_langsmith() -> None:
    """Initialize LangSmith tracing if configured.

    When LANGSMITH_TRACING_V2=true, LangChain auto-registers the
    LangSmith tracer as a callback handler — no manual wiring needed.
    All LLM calls, tool invocations, and agent steps are traced
    automatically, including search_knowledge_base inputs/outputs.
    """
    if os.getenv("LANGSMITH_TRACING_V2", "").lower() != "true":
        return

    project = os.getenv("LANGSMITH_PROJECT", "mragagent")
    api_key = os.getenv("LANGSMITH_API_KEY", "")

    if not api_key or "your-langsmith-key-here" in api_key:
        print("  ⚠ LANGSMITH_TRACING_V2=true 但 API key 未设置，跳过 LangSmith 初始化")
        return

    try:
        # LangChain auto-instruments when env vars are set, but we
        # explicitly configure the tracer for metadata control.
        from langsmith import Client

        client = Client(api_key=api_key)
        # Verify connectivity.
        client.list_projects()
        print(f"  ✓ LangSmith 追踪已启用 (项目: {project})")
        print(f"    查看 trace: https://smith.langchain.com/o/{client._get_tenant_id()}/projects/{project}")
    except Exception as e:
        print(f"  ⚠ LangSmith 连接失败: {e}")
        print(f"    追踪将继续尝试在后台工作")

SYSTEM_PROMPT = """你是一个文化遗产识别助手，名为 MRagAgent。

## 你的能力
1. **图片识别**: 当用户上传照片时，你可以调用 identify_from_image 工具识别照片中的具体文物点
2. **知识检索**: 你可以调用 search_knowledge_base 工具搜索文化遗产知识库，回答专业问题
3. **计算器**: 支持数学计算
4. **天气查询**: 查询国内城市的天气信息

## 工具使用策略
- 用户上传图片并询问相关内容 → 首先调用 identify_from_image 识别图片，拿到文物点信息后，如果用户还有进一步问题，再调用 search_knowledge_base 获取详细知识
- 用户只发图片不提问 → 先调用 identify_from_image 识别，然后根据识别结果主动介绍
- 用户只提问不传图 → 直接调用 search_knowledge_base 搜索知识库
- 用户问数学问题 → 调用 calculator
- 用户问天气 → 调用 get_current_weather

## 回答风格
- 所有回答使用中文
- 对文物点的描述要专业、准确、生动
- 如果涉及历史年代、艺术风格、建筑形制等专业知识，要给出可靠的来源信息
- 如果无法确定某个细节，诚实说明而非编造
"""


def _create_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        extra_body={"thinking": {"type": "disabled"}},
    )


# ── Built-in tools ───────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，支持加减乘除、幂运算等。调用前把文字描述转成 Python 表达式。"""
    allowed = set("0123456789+-*/.() **")
    cleaned = "".join(c for c in expression if c in allowed)
    try:
        result = eval(cleaned)
    except Exception as e:
        return f"计算错误: {str(e)}"
    return f"计算结果: {result}"


@tool
def get_current_weather(city: str) -> str:
    """查询指定城市的当前天气。"""
    weather_data = {
        "北京": "晴天，25°C，湿度 40%",
        "上海": "多云，28°C，湿度 65%",
        "深圳": "雷阵雨，30°C，湿度 80%",
        "杭州": "小雨，22°C，湿度 75%",
    }
    return weather_data.get(city, f"{city}：晴天，23°C，湿度 50%（模拟数据）")


# ── Agent builders ───────────────────────────────────────────────

def build_agent():
    """Build and return the MRagAgent (sync checkpointer)."""

    _init_langsmith()

    from backend.tools import identify_from_image, search_knowledge_base

    llm = _create_llm()
    tools = [
        calculator,
        get_current_weather,
        identify_from_image,
        search_knowledge_base,
    ]
    conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=SqliteSaver(conn),
    )


def build_agent_async():
    """Build and return the MRagAgent without checkpointer (for astream).

    Note: AsyncSqliteSaver requires aiosqlite + async context manager;
    for now streaming uses no persistence. SuperMew handles memory
    externally via PostgreSQL instead of checkpointer.
    """

    _init_langsmith()

    from backend.tools import identify_from_image, search_knowledge_base

    llm = _create_llm()
    tools = [
        calculator,
        get_current_weather,
        identify_from_image,
        search_knowledge_base,
    ]
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

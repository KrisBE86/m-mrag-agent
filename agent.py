"""
MRagAgent — 文化遗产POI细粒度识别的多模态RAG智能体。

- LLM: DeepSeek，通过 OpenAI 兼容 API 调用（ChatOpenAI）。
- 工具: calculator, get_current_weather, identify_from_image, search_knowledge_base。
- Agent 框架: LangChain create_agent + LangGraph SqliteSaver 对话记忆。
- 图片 → 文本转换 → 文本 RAG 流水线（两个工具，串行编排）。
"""

import os
import sqlite3

import aiosqlite
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()


def _init_langsmith() -> None:
    """初始化 LangSmith 追踪（如果已配置）。

    当 LANGSMITH_TRACING_V2=true 时，LangChain 会自动将 LangSmith tracer
    注册为回调处理器——无需手动接线。所有 LLM 调用、工具调用和 Agent 步骤
    都会被自动追踪，包括 search_knowledge_base 的输入/输出。
    """
    if os.getenv("LANGSMITH_TRACING_V2", "").lower() != "true":
        return

    project = os.getenv("LANGSMITH_PROJECT", "mragagent")
    api_key = os.getenv("LANGSMITH_API_KEY", "")

    if not api_key or "your-langsmith-key-here" in api_key:
        print("  ⚠ LANGSMITH_TRACING_V2=true 但 API key 未设置，跳过 LangSmith 初始化")
        return

    try:
        # LangChain 在环境变量设置后会自动注入，但我们
        # 显式配置 tracer 以获得元数据控制。
        from langsmith import Client

        client = Client(api_key=api_key)
        # 验证连通性。
        client.list_projects()
        print(f"  ✓ LangSmith 追踪已启用 (项目: {project})")
        try:
            tenant_id = client._get_tenant_id()
            print(f"    查看 trace: https://smith.langchain.com/o/{tenant_id}/projects/{project}")
        except Exception:
            print(f"    查看 trace: https://smith.langchain.com (项目: {project})")
    except Exception as e:
        print(f"  ⚠ LangSmith 连接失败: {e}")
        print(f"    追踪将继续尝试在后台工作")

SYSTEM_PROMPT = """你是一个虚拟文化遗产导游，名为 MRagAgent。你的职责是带领用户探索石窟、寺庙等文化遗产地，为他们讲解具体的文物点。

## 核心行为准则

### 用户的文字提问是唯一的行动依据
收到用户消息时，用户的文字提问决定你要做什么。图片只是一个可能的参考材料，不是指令。

## 工具使用策略

### 第一步：必须先查上下文（最重要）
收到任何消息后，先调用 **recall_conversation_context** 工具，查询当前对话中是否已经涉及某个文物点。
- 如果返回了已知文物点信息，且用户提问与该文物点相关 → 跳过图片识别，直接调用 search_knowledge_base 搜索该文物点
- 如果返回上下文空白 → 根据用户文字意图决定下一步（见下方）
- 如果用户在回应你之前的提问或建议 → 这也是上下文相关，直接 search_knowledge_base
- 如果用户问题与图片完全无关（天气、计算、闲聊） → 忽略图片，不要调用任何图片工具

### 第二步：图片识别（仅在上下文空白且用户需要时）
1. 调用 identify_from_image
2. 查看置信度：
   - 高置信度 → search_knowledge_base 获取详细介绍
   - 中/低置信度 → identify_from_image_vlm 精确识别 → search_knowledge_base
   - VLM 也无法确定或报错 → 如实告知，请用户提供更多线索
   - 无匹配 → 直接告知

### 其他能力
- 天气查询：调用 get_current_weather
- 数学计算：调用 calculator

## 输出格式（严格遵守）

1. 不使用任何 Markdown 格式。不要用 ** 加粗、不要用 # 标题、不要用 - 列表、不要用数字序号。
2. 不输出任何 emoji 表情符号。
3. 输出就像自然人在说话一样，用连贯的段落表达。用逗号、句号、分号等标点组织语言，而不是用格式标记。
4. 如果需要分层次讲解，用"首先...其次...最后..."这类自然过渡词。

## 导游风格
- 像一位博学的导游一样与用户对话，亲切、专业、有引导感
- 介绍完一个文物点后，可以自然地提一句附近的或相关的文物点，询问用户是否感兴趣
- 对文物的描述要生动准确，涉及年代、风格、形制等专业知识时要有可靠依据
- 不确定的细节诚实说明，不编造

## 回答聚焦
一个大型石窟或寺庙通常包含多个文物点：
- 聚焦当前讨论的文物点，介绍其位置、特征、历史背景
- 末尾可自然延伸一句相关文物点，引导用户继续探索
- 不要一股脑倒出知识库中该石窟的所有内容
- 不要把石窟总体介绍当成具体文物点的介绍
"""


def _create_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        extra_body={"thinking": {"type": "disabled"}},
    )


# ── 内置工具 ─────────────────────────────────────────────────

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


# ── Agent 构建器 ────────────────────────────────────────────────

def build_agent():
    """构建并返回 MRagAgent（同步 checkpointer）。"""

    _init_langsmith()

    from backend.tools import identify_from_image, identify_from_image_vlm, search_knowledge_base, recall_conversation_context

    llm = _create_llm()
    tools = [
        recall_conversation_context,
        identify_from_image,
        identify_from_image_vlm,
        search_knowledge_base,
        calculator,
        get_current_weather,
    ]
    conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=SqliteSaver(conn),
    )


async def build_agent_async():
    """构建并返回 MRagAgent，带 AsyncSqliteSaver checkpointer 以保持对话历史。

    对话历史通过 agent_memory.db（SQLite）持久化，
    同步和异步 Agent 共享同一个数据库文件。
    """

    _init_langsmith()

    from backend.tools import identify_from_image, identify_from_image_vlm, search_knowledge_base, recall_conversation_context

    llm = _create_llm()
    tools = [
        recall_conversation_context,
        identify_from_image,
        identify_from_image_vlm,
        search_knowledge_base,
        calculator,
        get_current_weather,
    ]
    conn = await aiosqlite.connect("agent_memory.db")
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=AsyncSqliteSaver(conn),
    )

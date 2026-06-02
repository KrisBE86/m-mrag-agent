"""
MRagAgent — 文化遗产POI细粒度识别的多模态RAG智能体。

- LLM: DeepSeek，通过 OpenAI 兼容 API 调用（ChatOpenAI）。
- 工具: calculator, get_current_weather, identify_from_image, search_knowledge_base。
- Agent 框架: LangChain create_agent + LangGraph SqliteSaver 对话记忆。
- 图片 → 文本转换 → 文本 RAG 流水线（两个工具，串行编排）。
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

SYSTEM_PROMPT = """你是一个文化遗产识别助手，名为 MRagAgent。

## 输出规则
1. **不确定时禁止猜测**：如果你无法确定图片对应哪个文物点，只需说"抱歉，我暂时无法确定"，然后说明可能原因，建议用户重试。不要提及任何具体的文物点名称。
2. **确定时聚焦回答**：如果能确定文物点，只介绍这一个，不要一股脑把所有相关内容倒出来。

## 你的能力
1. **图片识别**: identify_from_image（CLIP 图像检索）、identify_from_image_vlm（视觉大模型精确识别）
2. **知识检索**: search_knowledge_base 搜索文化遗产知识库
3. **计算器**: 支持数学计算
4. **天气查询**: 查询国内城市的天气信息

## 工具使用策略

### 图片识别流程
1. 用户上传图片 → 首先调用 **identify_from_image**
2. 查看返回结果的置信度：
   - **高置信度** → 调用 search_knowledge_base 获取详细介绍
   - **中/低置信度** → 调用 **identify_from_image_vlm** 进行精确视觉识别
     - VLM 返回有效匹配 → 调用 search_knowledge_base
     - VLM 也无法确定 → 反问用户，不要调用 search_knowledge_base
   - **无匹配** → 直接告知用户

### 其他场景
- 用户只提问不传图 → 直接调用 search_knowledge_base
- 用户问数学问题 → 调用 calculator
- 用户问天气 → 调用 get_current_weather

## 回答风格
- 所有回答使用中文
- 对文物点的描述要专业、准确、生动
- 如果涉及历史年代、艺术风格、建筑形制等专业知识，要给出可靠的来源信息
- 如果无法确定某个细节，诚实说明而非编造

## 回答聚焦原则
一个大型石窟/寺庙通常包含多个具体文物点。当你识别出某个具体文物点时：
- ✅ 只介绍识别到的那个具体文物点，聚焦其位置、特征、历史背景
- ✅ 回答末尾可以自然地提一句相关文物点作为延伸
- ❌ 不要把知识库中和该石窟相关的所有内容全部倒出来
- ❌ 不要把石窟的总体介绍当成具体文物点的回答
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

    from backend.tools import identify_from_image, identify_from_image_vlm, search_knowledge_base

    llm = _create_llm()
    tools = [
        calculator,
        get_current_weather,
        identify_from_image,
        identify_from_image_vlm,
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
    """构建并返回 MRagAgent，不带 checkpointer（用于 astream）。

    注意: AsyncSqliteSaver 需要 aiosqlite + 异步上下文管理器；
    目前流式输出不保存状态。SuperMew 通过 PostgreSQL
    在外部管理记忆，而非通过 checkpointer。
    """

    _init_langsmith()

    from backend.tools import identify_from_image, identify_from_image_vlm, search_knowledge_base

    llm = _create_llm()
    tools = [
        calculator,
        get_current_weather,
        identify_from_image,
        identify_from_image_vlm,
        search_knowledge_base,
    ]
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

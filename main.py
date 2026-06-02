"""
MRagAgent CLI — 文化遗产POI细粒度识别的多模态RAG智能体命令行界面。

启动流程:
  1. 初始化 PostgreSQL 表（parent_chunks）。
  2. 初始化 Milvus 双集合（image_poi + text_chunk）。
  3. 从持久化存储加载 BM25 状态。
  4. 启动交互式 REPL（Agent 对话循环）。

用法:
  python main.py

Agent 可处理:
  - 纯文本提问 → search_knowledge_base
  - 图片提问 → identify_from_image → search_knowledge_base
  - 计算器和天气查询
"""

from agent import build_agent
from backend.database import init_db


def init_rag_modules() -> None:
    """启动时初始化 RAG 基础设施。"""
    print("正在初始化 MRagAgent 后端...")

    # 1. PostgreSQL 表。
    try:
        init_db()
        print("  ✓ PostgreSQL 表初始化完成")
    except Exception as e:
        print(f"  ⚠ PostgreSQL 初始化失败 (Docker 是否已启动?): {e}")

    # 2. Milvus 集合。
    try:
        from backend.milvus_client import milvus_manager
        from backend.embedding import clip_embeddings, bge_embeddings

        milvus_manager.init_image_collection(dense_dim=clip_embeddings.dimension)
        print(f"  ✓ Image Milvus 集合已就绪 ({clip_embeddings.dimension}d)")

        milvus_manager.init_text_collection(dense_dim=bge_embeddings.dimension)
        print(f"  ✓ Text Milvus 集合已就绪 ({bge_embeddings.dimension}d)")
    except Exception as e:
        print(f"  ⚠ Milvus 初始化失败 (Docker 是否已启动?): {e}")

    # 3. BM25 状态在首次导入时由 ChineseBM25 单例自动加载。
    print("  ✓ BM25 状态加载完成")

    print("MRagAgent 后端初始化完成\n")


def main():
    """入口函数：初始化模块、构建 agent、启动 REPL。"""
    init_rag_modules()

    agent = build_agent()
    config = {"configurable": {"thread_id": "main-chat"}}

    print("=" * 60)
    print("  MRagAgent — 文化遗产多模态识别助手")
    print("=" * 60)
    print("  支持功能:")
    print("    • 拍照识文物: 输入图片路径，识别具体文物点")
    print("    • 知识问答: 询问文化遗产相关问题")
    print("    • 计算器 & 天气查询")
    print("  输入 quit 或 exit 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("再见！")
            break

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            print(f"\nMRagAgent: {result['messages'][-1].content}")
        except Exception as e:
            print(f"\nMRagAgent: 【错误】{str(e)}")


if __name__ == "__main__":
    main()

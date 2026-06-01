"""
MRagAgent CLI — Multimodal RAG Agent for Cultural Heritage POI Identification.

Startup:
  1. Initialize PostgreSQL tables (parent_chunks).
  2. Initialize Milvus dual collections (image_poi + text_chunk).
  3. Load BM25 state from persistence.
  4. Start interactive REPL with Agent.

Usage:
  python main.py

The agent handles:
  - Text-only questions → search_knowledge_base
  - Image questions → identify_from_image → search_knowledge_base
  - Calculator and weather queries
"""

from agent import build_agent
from backend.database import init_db


def init_rag_modules() -> None:
    """Initialize RAG infrastructure on startup."""
    print("正在初始化 MRagAgent 后端...")

    # 1. PostgreSQL tables.
    try:
        init_db()
        print("  ✓ PostgreSQL 表初始化完成")
    except Exception as e:
        print(f"  ⚠ PostgreSQL 初始化失败 (Docker 是否已启动?): {e}")

    # 2. Milvus collections.
    try:
        from backend.milvus_client import milvus_manager
        from backend.embedding import clip_embeddings, bge_embeddings

        milvus_manager.init_image_collection(dense_dim=clip_embeddings.dimension)
        print(f"  ✓ Image Milvus 集合已就绪 ({clip_embeddings.dimension}d)")

        milvus_manager.init_text_collection(dense_dim=bge_embeddings.dimension)
        print(f"  ✓ Text Milvus 集合已就绪 ({bge_embeddings.dimension}d)")
    except Exception as e:
        print(f"  ⚠ Milvus 初始化失败 (Docker 是否已启动?): {e}")

    # 3. BM25 state is auto-loaded by ChineseBM25 singleton on first import.
    print("  ✓ BM25 状态加载完成")

    print("MRagAgent 后端初始化完成\n")


def main():
    """Entry point: init modules, build agent, start REPL."""
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

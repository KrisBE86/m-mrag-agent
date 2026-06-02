"""
MRagAgent 的 FastAPI 应用入口。

- 开发环境下对所有来源启用 CORS。
- 对前端文件禁用缓存中间件。
- 启动时初始化 PostgreSQL + Milvus。
- 为 frontend/ 目录提供静态文件服务。
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import router as api_router
from backend.database import init_db

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


def create_app() -> FastAPI:
    app = FastAPI(title="MRagAgent API")

    @app.on_event("startup")
    async def _startup_init():
        # 初始化 PostgreSQL 表。
        try:
            init_db()
        except Exception:
            pass  # 非致命：数据库可能已经初始化过。

        # 初始化 Milvus 集合。
        try:
            from backend.milvus_client import milvus_manager
            from backend.embedding import clip_embeddings, bge_embeddings

            milvus_manager.init_image_collection(dense_dim=clip_embeddings.dimension)
            milvus_manager.init_text_collection(dense_dim=bge_embeddings.dimension)
        except Exception:
            pass  # 非致命：集合可能已经存在。

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 开发环境下禁用缓存中间件。
    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path == "/" or any(path.endswith(ext) for ext in (".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.include_router(api_router)

    # 在根路径提供前端静态文件服务。
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
    )

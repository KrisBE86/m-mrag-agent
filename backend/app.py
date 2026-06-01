"""
FastAPI application entry point for MRagAgent.

- CORS enabled for all origins (development).
- No-cache middleware for frontend files.
- PostgreSQL + Milvus initialization on startup.
- Static file serving for the frontend/ directory.
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
        # Initialize PostgreSQL tables.
        try:
            init_db()
        except Exception:
            pass  # Non-fatal: DB may already be initialized.

        # Initialize Milvus collections.
        try:
            from backend.milvus_client import milvus_manager
            from backend.embedding import clip_embeddings, bge_embeddings

            milvus_manager.init_image_collection(dense_dim=clip_embeddings.dimension)
            milvus_manager.init_text_collection(dense_dim=bge_embeddings.dimension)
        except Exception:
            pass  # Non-fatal: collections may already exist.

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # No-cache middleware for development.
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

    # Serve frontend static files at root.
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

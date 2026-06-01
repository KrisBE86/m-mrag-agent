"""
API routes: chat (sync + SSE streaming) and document management.
"""

import json
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.agent_api import chat_sync, chat_stream
from backend.auth import verify_admin
from backend.document_loader import ingest_document

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "documents"
IMAGE_UPLOAD_DIR = DATA_DIR / "user_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Request/Response models ──────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# ── Chat routes ──────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _: bool = Depends(verify_admin)):
    """Synchronous chat endpoint."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")
    try:
        response = chat_sync(req.message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聊天失败: {str(e)}")


@router.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest, _: bool = Depends(verify_admin)):
    """SSE streaming chat endpoint, following SuperMew's async generator pattern."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    return StreamingResponse(
        chat_stream(req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Document management routes (admin only) ──────────────────────

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    use_llm_naming: bool = False,
    _: bool = Depends(verify_admin),
):
    """Upload and ingest a document (PDF/Word)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    file_lower = file.filename.lower()
    if not (file_lower.endswith(".pdf") or file_lower.endswith((".docx", ".doc"))):
        raise HTTPException(status_code=400, detail="仅支持 PDF 和 Word 文件")

    # Save uploaded file.
    safe_name = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_name
    content = await file.read()

    # Check for duplicate — clean old data first if re-uploading.
    is_reupload = file_path.exists()
    if is_reupload:
        from backend.milvus_client import milvus_manager
        from backend.embedding import bm25
        from backend.parent_chunk_store import parent_chunk_store

        filter_expr = f'filename == "{safe_name}"'
        try:
            # Remove BM25 stats.
            rows = milvus_manager.query(
                collection=milvus_manager.text_collection,
                filter_expr=filter_expr,
                output_fields=["text"],
                limit=10000,
            )
            texts = [r.get("text") or "" for r in rows]
            if texts:
                bm25.increment_remove_documents(texts)

            milvus_manager.delete(milvus_manager.text_collection, filter_expr)
            milvus_manager.delete(milvus_manager.image_collection, filter_expr)
            parent_chunk_store.delete_by_filename(safe_name)
            print(f"  ♻ 已清理旧数据: {safe_name}")
        except Exception as e:
            print(f"  ⚠ 清理旧数据时出错: {e}")

    file_path.write_bytes(content)

    # Ingest into the system.
    try:
        ingest_document(str(file_path), use_llm_naming=use_llm_naming)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

    return {
        "status": "ok",
        "filename": safe_name,
        "message": f"文档 {safe_name} 已成功处理",
    }


@router.get("/documents")
async def list_documents(_: bool = Depends(verify_admin)):
    """List uploaded documents."""
    docs = []
    if UPLOAD_DIR.exists():
        for f in sorted(UPLOAD_DIR.iterdir()):
            if f.is_file():
                docs.append({
                    "filename": f.name,
                    "size": f.stat().st_size,
                    "type": f.suffix.upper(),
                })
    return {"documents": docs}


@router.delete("/documents/{filename}")
async def delete_document(filename: str, _: bool = Depends(verify_admin)):
    """Delete an uploaded document and its chunks from Milvus + PostgreSQL."""
    from backend.milvus_client import milvus_manager
    from backend.embedding import bm25
    from backend.parent_chunk_store import parent_chunk_store

    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    # Delete from Milvus (both collections).
    filter_expr = f'filename == "{filename}"'
    try:
        # Remove BM25 stats for text chunks.
        rows = milvus_manager.query(
            collection=milvus_manager.text_collection,
            filter_expr=filter_expr,
            output_fields=["text"],
            limit=10000,
        )
        texts = [r.get("text") or "" for r in rows]
        if texts:
            bm25.increment_remove_documents(texts)

        milvus_manager.delete(milvus_manager.text_collection, filter_expr)
        milvus_manager.delete(milvus_manager.image_collection, filter_expr)

        # Delete from PostgreSQL.
        parent_chunk_store.delete_by_filename(filename)

        # Delete file.
        file_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

    return {"status": "ok", "filename": filename}


# ── Image upload for chat (saves to disk, returns real path) ──────

@router.post("/images/upload")
async def upload_image(
    file: UploadFile = File(...),
    _: bool = Depends(verify_admin),
):
    """Upload an image file for agent identification. Returns the server path."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    file_lower = file.filename.lower()
    if not file_lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")):
        raise HTTPException(status_code=400, detail="仅支持 JPG/PNG/GIF/BMP/WEBP 图片")

    # Use UUID to avoid name collisions.
    suffix = Path(file.filename).suffix.lower()
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    file_path = IMAGE_UPLOAD_DIR / safe_name
    content = await file.read()
    file_path.write_bytes(content)

    return {
        "status": "ok",
        "image_path": str(file_path),
        "original_name": file.filename,
    }


@router.post("/auth/verify")
async def verify_token(_: bool = Depends(verify_admin)):
    """Verify admin token is valid."""
    return {"status": "ok", "role": "admin", "username": "admin"}

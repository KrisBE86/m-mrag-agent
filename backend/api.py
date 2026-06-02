"""
API 路由：聊天（同步 + SSE 流式）和文档管理。
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


def _cleanup_old_document_data(filename: str) -> None:
    """清理文档在 Milvus、BM25 和 PostgreSQL 中的旧数据。

    用于重新上传和删除文档时，确保旧索引数据被完全清除。
    """
    from backend.milvus_client import milvus_manager
    from backend.embedding import bm25
    from backend.parent_chunk_store import parent_chunk_store

    filter_expr = f'filename == "{filename}"'
    try:
        # 移除 BM25 统计信息。
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
        parent_chunk_store.delete_by_filename(filename)
        print(f"  ♻ 已清理旧数据: {filename}")
    except Exception as e:
        print(f"  ⚠ 清理旧数据时出错: {e}")


# ── 请求/响应模型 ──────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    image_path: str | None = None


class ChatResponse(BaseModel):
    response: str


# ── 聊天路由 ──────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _: bool = Depends(verify_admin)):
    """同步聊天接口。"""
    if not req.message.strip() and not req.image_path:
        raise HTTPException(status_code=400, detail="消息不能为空")
    try:
        response = chat_sync(req.message, req.image_path)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聊天失败: {str(e)}")


@router.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest, _: bool = Depends(verify_admin)):
    """SSE 流式聊天接口，遵循 SuperMew 的异步生成器模式。"""
    if not req.message.strip() and not req.image_path:
        raise HTTPException(status_code=400, detail="消息不能为空")

    return StreamingResponse(
        chat_stream(req.message, req.image_path),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── 文档管理路由（仅管理员） ──────────────────────

@router.post("/documents/upload")
async def upload_document(
    files: list[UploadFile] = File(...),
    use_llm_naming: bool = False,
    _: bool = Depends(verify_admin),
):
    """上传并录入文档（PDF/Word），支持批量上传。"""
    if not files:
        raise HTTPException(status_code=400, detail="未选择任何文件")

    results = []

    for file in files:
        filename = file.filename
        if not filename:
            results.append({"filename": "(未知)", "status": "error", "message": "文件名为空"})
            continue

        # 验证文件类型。
        file_lower = filename.lower()
        if not (file_lower.endswith(".pdf") or file_lower.endswith((".docx", ".doc"))):
            results.append({
                "filename": filename,
                "status": "error",
                "message": f"不支持的文件类型: {Path(filename).suffix}",
            })
            continue

        # 保存上传的文件。
        safe_name = Path(filename).name
        file_path = UPLOAD_DIR / safe_name
        content = await file.read()

        # 检查重复 — 如果是重新上传，先清理旧数据。
        if file_path.exists():
            _cleanup_old_document_data(safe_name)

        file_path.write_bytes(content)

        # 录入系统。
        try:
            ingest_document(str(file_path), use_llm_naming=use_llm_naming)
            results.append({
                "filename": safe_name,
                "status": "success",
                "message": f"文档 {safe_name} 已成功处理",
            })
        except Exception as e:
            results.append({
                "filename": safe_name,
                "status": "error",
                "message": f"文档处理失败: {str(e)}",
            })

    # 汇总结果。
    all_success = all(r["status"] == "success" for r in results)
    success_count = sum(1 for r in results if r["status"] == "success")
    fail_count = len(results) - success_count

    return {
        "status": "ok" if all_success else "partial",
        "results": results,
        "summary": f"成功 {success_count}/{len(results)}" + (f"，失败 {fail_count}" if fail_count else ""),
    }


@router.get("/documents")
async def list_documents(_: bool = Depends(verify_admin)):
    """列出已上传的文档。"""
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
    """删除已上传的文档及其在 Milvus 和 PostgreSQL 中的所有数据块。"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    try:
        _cleanup_old_document_data(filename)
        file_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

    return {"status": "ok", "filename": filename}


# ── 聊天图片上传（保存到磁盘，返回真实路径） ──────

@router.post("/images/upload")
async def upload_image(
    file: UploadFile = File(...),
    _: bool = Depends(verify_admin),
):
    """上传图片供 Agent 识别。返回服务器端文件路径。"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    file_lower = file.filename.lower()
    if not file_lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")):
        raise HTTPException(status_code=400, detail="仅支持 JPG/PNG/GIF/BMP/WEBP 图片")

    # 使用 UUID 避免文件名冲突。
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
    """验证管理员 token 是否有效。"""
    return {"status": "ok", "role": "admin", "username": "admin"}

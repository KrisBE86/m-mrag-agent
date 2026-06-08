"""
API 路由：聊天（同步 + SSE 流式）和文档管理。
"""

import json
import os
import hashlib
import re
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.agent_api import chat_sync, chat_stream
from backend.auth import verify_admin
from backend.document_loader import ingest_document
from backend.tts_service import TTS_SAMPLE_RATE, synthesize as tts_synthesize
from backend.url_ingestor import URLIngestError, download_url_document

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "documents"
IMAGE_UPLOAD_DIR = DATA_DIR / "user_uploads"
REFERENCE_IMAGE_DIR = DATA_DIR / "reference_images"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_reference_images(filename: str) -> None:
    """删除某个文档旧抽图，避免重传后 reference_images 中堆积重复图。"""
    safe_filename = re.sub(r"[^\w\-\.]", "_", filename)
    removed = 0
    for path in REFERENCE_IMAGE_DIR.glob(f"{safe_filename}_img*"):
        if not path.is_file():
            continue
        try:
            path.unlink()
            removed += 1
        except Exception as e:
            print(f"  ⚠ 清理旧参考图失败 {path.name}: {e}")
    if removed:
        print(f"  ♻ 已清理旧参考图: {removed} 张")


def _cleanup_old_document_data(filename: str) -> None:
    """清理文档在 Milvus、BM25 和 PostgreSQL 中的旧数据。

    用于重新上传和删除文档时，确保旧索引数据被完全清除。
    """
    from backend.milvus_client import milvus_manager
    from backend.embedding import bm25
    from backend.parent_chunk_store import parent_chunk_store

    filter_expr = f'filename == "{filename}"'
    # 移除 BM25 统计信息。
    try:
        rows = milvus_manager.query(
            collection=milvus_manager.text_collection,
            filter_expr=filter_expr,
            output_fields=["text"],
            limit=10000,
        )
        texts = [r.get("text") or "" for r in rows]
        if texts:
            bm25.increment_remove_documents(texts)
    except Exception as e:
        print(f"  ⚠ 清理 BM25 旧数据时出错: {e}")

    try:
        milvus_manager.delete(milvus_manager.text_collection, filter_expr)
    except Exception as e:
        print(f"  ⚠ 清理 Text Milvus 旧数据时出错: {e}")

    try:
        milvus_manager.delete(milvus_manager.image_collection, filter_expr)
    except Exception as e:
        print(f"  ⚠ 清理 Image Milvus 旧数据时出错: {e}")

    try:
        parent_chunk_store.delete_by_filename(filename)
    except Exception as e:
        print(f"  ⚠ 清理 PostgreSQL 旧数据时出错: {e}")

    _cleanup_reference_images(filename)

    print(f"  ♻ 已清理旧数据: {filename}")


# ── 请求/响应模型 ──────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    image_path: str | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str


class TTSRequest(BaseModel):
    text: str


class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    encoding: str = "wav"


class URLImportRequest(BaseModel):
    url: str
    use_llm_naming: bool = False
    use_vlm_description: bool = True


# ── 聊天路由 ──────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _: bool = Depends(verify_admin)):
    """同步聊天接口。"""
    if not req.message.strip() and not req.image_path:
        raise HTTPException(status_code=400, detail="消息不能为空")
    try:
        response = chat_sync(req.message, req.image_path, session_id=req.session_id)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聊天失败: {str(e)}")


@router.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest, _: bool = Depends(verify_admin)):
    """SSE 流式聊天接口，遵循 SuperMew 的异步生成器模式。"""
    if not req.message.strip() and not req.image_path:
        raise HTTPException(status_code=400, detail="消息不能为空")

    return StreamingResponse(
        chat_stream(req.message, req.image_path, session_id=req.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/tts", response_model=TTSResponse)
async def tts(req: TTSRequest, _: bool = Depends(verify_admin)):
    """语音合成接口，供 Web 前端调试播放 Agent 回复。"""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="文本不能为空")

    audio_base64 = await tts_synthesize(text)
    if not audio_base64:
        raise HTTPException(status_code=500, detail="TTS 合成失败，请检查配置或后端日志")

    return TTSResponse(audio_base64=audio_base64, sample_rate=TTS_SAMPLE_RATE)


# ── 文档管理路由（仅管理员） ──────────────────────

@router.post("/documents/upload")
async def upload_document(
    files: list[UploadFile] = File(...),
    use_llm_naming: bool = Form(False),
    use_vlm_description: bool = Form(True),
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
        if not (file_lower.endswith(".pdf") or file_lower.endswith((".docx", ".doc", ".html", ".htm"))):
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
            ingest_document(
                str(file_path),
                use_llm_naming=use_llm_naming,
                use_vlm_description=use_vlm_description,
            )
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


@router.post("/documents/import-url")
async def import_url_document(
    req: URLImportRequest,
    _: bool = Depends(verify_admin),
):
    """下载公开 URL 并录入文档，支持 PDF/Word/HTML。"""
    try:
        downloaded = await run_in_threadpool(download_url_document, req.url, UPLOAD_DIR)
    except URLIngestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        raise HTTPException(status_code=400, detail=f"URL 下载失败: HTTP {status_code}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"URL 下载失败: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL 下载失败: {str(e)}")

    try:
        # URL 文件名由来源和 URL hash 生成。同一 URL 重复导入时先清理旧索引。
        _cleanup_old_document_data(downloaded.filename)
        await run_in_threadpool(
            ingest_document,
            downloaded.file_path,
            use_llm_naming=req.use_llm_naming,
            use_vlm_description=req.use_vlm_description,
            source_url=downloaded.final_url,
        )
        return {
            "status": "success",
            "filename": downloaded.filename,
            "source_url": downloaded.source_url,
            "final_url": downloaded.final_url,
            "content_type": downloaded.content_type,
            "size": downloaded.size_bytes,
            "message": f"URL 文档 {downloaded.filename} 已成功处理",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL 文档处理失败: {str(e)}")


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

    suffix = Path(file.filename).suffix.lower()
    content = await file.read()
    digest = hashlib.sha256(content).hexdigest()
    safe_name = f"{digest}{suffix}"
    file_path = IMAGE_UPLOAD_DIR / safe_name
    created = False
    if not file_path.exists():
        file_path.write_bytes(content)
        created = True

    return {
        "status": "ok",
        "image_path": str(file_path),
        "original_name": file.filename,
        "deduplicated": not created,
        "sha256": digest,
    }


@router.post("/auth/verify")
async def verify_token(_: bool = Depends(verify_admin)):
    """验证管理员 token 是否有效。"""
    return {"status": "ok", "role": "admin", "username": "admin"}

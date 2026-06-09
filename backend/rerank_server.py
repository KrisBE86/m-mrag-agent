"""
Local rerank API for MRagAgent.

Run:
    uvicorn backend.rerank_server:app --host 127.0.0.1 --port 8001

The endpoint is compatible with backend.rag_utils._rerank_documents:
    POST /v1/rerank
"""

import os
import threading
from typing import Any

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_MODEL = os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL) or DEFAULT_RERANK_MODEL
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "1024"))

app = FastAPI(title="MRagAgent Local Rerank API")
_tokenizer = None
_model = None
_model_lock = threading.Lock()


class RerankRequest(BaseModel):
    model: str = Field(default=RERANK_MODEL)
    query: str
    documents: list[str]
    top_n: int = 5
    return_documents: bool = False


def _get_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
        _model.to(RERANK_DEVICE)
        _model.eval()
    return _tokenizer, _model


def _check_auth(authorization: str | None) -> None:
    if not RERANK_API_KEY:
        return
    expected = f"Bearer {RERANK_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid rerank API key")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": RERANK_MODEL,
        "device": RERANK_DEVICE,
        "loaded": _model is not None,
    }


@app.post("/v1/rerank")
def rerank(req: RerankRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_auth(authorization)

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    if not req.documents:
        return {"results": []}

    top_n = max(0, min(req.top_n, len(req.documents)))
    if top_n == 0:
        return {"results": []}

    tokenizer, model = _get_model()
    with _model_lock, torch.no_grad():
        inputs = tokenizer(
            [req.query] * len(req.documents),
            [doc or "" for doc in req.documents],
            padding=True,
            truncation=True,
            max_length=RERANK_MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {key: value.to(RERANK_DEVICE) for key, value in inputs.items()}
        scores = model(**inputs).logits.view(-1).detach().cpu().tolist()

    ranked = sorted(
        enumerate(scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:top_n]

    results = []
    for index, score in ranked:
        item: dict[str, Any] = {
            "index": index,
            "relevance_score": float(score),
        }
        if req.return_documents:
            item["document"] = req.documents[index]
        results.append(item)

    return {
        "model": req.model or RERANK_MODEL,
        "results": results,
    }

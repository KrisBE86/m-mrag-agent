"""
Multi-model embedding service: ChineseCLIP (image/text) + BGE-M3 (text dense) + BM25 (text sparse).

- ChineseCLIPEmbeddings: wraps cn-clip for image→image and short text→image cross-modal search.
  Outputs 768-dim L2-normalized vectors for Image Milvus Collection.
- BGEM3Embeddings: wraps sentence-transformers BAAI/bge-m3 for long-text semantic search.
  Outputs 1024-dim L2-normalized vectors for Text Milvus Collection (dense field).
- ChineseBM25: custom BM25 implementation for Chinese keyword search.
  Persists vocabulary + document frequency to JSON. Outputs sparse vectors for Text Milvus Collection.
- Singleton instances at module level, aligned with SuperMew pattern.
"""

import json
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

_DEFAULT_BM25_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_state.json"


# ═══════════════════════════════════════════════════════════════════
# Chinese-CLIP embeddings (image + short text, 768d)
# ═══════════════════════════════════════════════════════════════════

class ChineseCLIPEmbeddings:
    """Chinese-CLIP ViT-L/14 wrapper for image and short-text embeddings.

    Both modalities map into the same L2-normalized 768d space,
    enabling cross-modal image↔text search within the same vector field.

    Uses the official cn-clip package (pip install cn-clip).
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        device: Optional[str] = None,
        download_root: str = "./models/",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model_name = model_name
        self._device = device

        import cn_clip.clip as clip
        from cn_clip.clip import load_from_name

        self._model, self._preprocess = load_from_name(
            model_name, device=device, download_root=download_root,
        )
        self._model.eval()

        self._dim_map = {"ViT-B-16": 512, "ViT-L-14": 768, "ViT-L-14-336": 768, "ViT-H-14": 1024}

    @property
    def dimension(self) -> int:
        return self._dim_map.get(self._model_name, 768)

    def embed_image(self, image) -> List[float]:
        """Embed a single image (file path str, Path, or PIL Image). Returns 768d list."""
        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().tolist()

    def embed_images(self, images: list) -> List[List[float]]:
        """Batch-embed multiple images."""
        return [self.embed_image(img) for img in images]

    def embed_query(self, text: str) -> List[float]:
        """Embed a short Chinese text query (≤77 tokens) into image space."""
        import cn_clip.clip as clip

        text_tokens = clip.tokenize([text]).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(text_tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple short texts."""
        return [self.embed_query(t) for t in texts]


# ═══════════════════════════════════════════════════════════════════
# BGE-M3 dense text embeddings (1024d, semantic search)
# ═══════════════════════════════════════════════════════════════════

class BGEM3Embeddings:
    """BGE-M3 dense text embeddings for semantic search.

    Wraps sentence-transformers. Outputs 1024-dim L2-normalized vectors.
    Used for Text Milvus Collection (dense_embedding field).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self._model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self._device = device or os.getenv("EMBEDDING_DEVICE", "cpu")
        # `local_files_only` 优先使用本地缓存，避免每次启动都访问 HuggingFace Hub。
        # 国内网络环境下 HF Hub 可能不稳定，若模型已下载到本地缓存（默认路径
        # ~/.cache/huggingface/hub/），开启此选项可跳过在线检查，加快启动速度。
        # 首次使用或需要更新模型时，可将环境变量 HF_HUB_OFFLINE=false。
        use_local = os.getenv("HF_HUB_OFFLINE", "true").lower() != "false"
        self._model = SentenceTransformer(
            self._model_name, device=self._device, local_files_only=use_local,
        )

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        embedding = self._model.encode(
            text, normalize_embeddings=True, show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch-embed multiple documents."""
        if not texts:
            return []
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )
        return embeddings.tolist()


# ═══════════════════════════════════════════════════════════════════
# BM25 sparse text embeddings (keyword search)
# ═══════════════════════════════════════════════════════════════════

class ChineseBM25:
    """BM25 sparse vector service with Chinese tokenization.

    Tokenizes: Chinese single-char + English word-level.
    Persists vocabulary + document frequency to JSON.

    Aligned with SuperMew's BM25 implementation.
    """

    def __init__(self, state_path: Optional[Path | str] = None):
        self._state_path = Path(
            state_path or os.getenv("BM25_STATE_PATH", _DEFAULT_BM25_STATE_PATH)
        )
        self._lock = threading.Lock()

        # BM25 parameters
        self.k1 = 1.5
        self.b = 0.75

        self._vocab: dict[str, int] = {}
        self._vocab_counter = 0
        self._doc_freq: Counter[str] = Counter()
        self._total_docs = 0
        self._sum_token_len = 0
        self._avg_doc_len = 1.0

        self._load_state()

    # ── persistence ──────────────────────────────────────────────

    def _recompute_avg_len(self) -> None:
        self._avg_doc_len = (
            self._sum_token_len / self._total_docs if self._total_docs > 0 else 1.0
        )

    def _load_state(self) -> None:
        path = self._state_path
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if raw.get("version") != 1:
            return
        self._vocab = {str(k): int(v) for k, v in raw.get("vocab", {}).items()}
        self._doc_freq = Counter({str(k): int(v) for k, v in raw.get("doc_freq", {}).items()})
        self._total_docs = int(raw.get("total_docs", 0))
        self._sum_token_len = int(raw.get("sum_token_len", 0))
        if self._vocab:
            self._vocab_counter = max(self._vocab.values()) + 1
        else:
            self._vocab_counter = 0
        self._recompute_avg_len()

    def _persist_unlocked(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "total_docs": self._total_docs,
            "sum_token_len": self._sum_token_len,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
        }
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._state_path)

    def _persist(self) -> None:
        with self._lock:
            self._persist_unlocked()

    # ── incremental updates (aligned with SuperMew) ──────────────

    def increment_add_documents(self, texts: list[str]) -> None:
        """Incrementally add documents to BM25 statistics."""
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len += doc_len
                self._total_docs += 1
                for token in set(tokens):
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    self._doc_freq[token] += 1
            self._recompute_avg_len()
            self._persist_unlocked()

    def increment_remove_documents(self, texts: list[str]) -> None:
        """Incrementally remove documents from BM25 statistics."""
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len = max(0, self._sum_token_len - doc_len)
                self._total_docs = max(0, self._total_docs - 1)
                for token in set(tokens):
                    if token not in self._doc_freq:
                        continue
                    self._doc_freq[token] -= 1
                    if self._doc_freq[token] <= 0:
                        del self._doc_freq[token]
            self._recompute_avg_len()
            self._persist_unlocked()

    # ── tokenization ─────────────────────────────────────────────

    def tokenize(self, text: str) -> list[str]:
        """Chinese single-char + English word-level tokenization."""
        text = text.lower()
        tokens: list[str] = []
        chinese_pattern = re.compile(r"[一-鿿]")
        english_pattern = re.compile(r"[a-zA-Z]+")
        i = 0
        while i < len(text):
            char = text[i]
            if chinese_pattern.match(char):
                tokens.append(char)
                i += 1
            elif english_pattern.match(char):
                match = english_pattern.match(text[i:])
                if match:
                    tokens.append(match.group())
                    i += len(match.group())
            else:
                i += 1
        return tokens

    # ── sparse vector generation ────────────────────────────────

    def _sparse_vector_for_text_unlocked(self, text: str) -> tuple[dict, bool]:
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)
        sparse_vector: dict[int, float] = {}
        vocab_changed = False
        n = max(self._total_docs, 0)
        avg = max(self._avg_doc_len, 1.0)

        for token, freq in tf.items():
            if token not in self._vocab:
                self._vocab[token] = self._vocab_counter
                self._vocab_counter += 1
                vocab_changed = True

            idx = self._vocab[token]
            df = self._doc_freq.get(token, 0)
            idf = math.log((n + 1) / 1) if df == 0 else math.log(
                (n - df + 0.5) / (df + 0.5) + 1
            )

            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avg)
            score = idf * numerator / denominator
            if score > 0:
                sparse_vector[idx] = float(score)

        return sparse_vector, vocab_changed

    def get_sparse_embedding(self, text: str) -> dict:
        """Generate a BM25 sparse vector dict for a single text."""
        with self._lock:
            sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
            if vocab_changed:
                self._persist_unlocked()
        return sparse_vector

    def get_sparse_embeddings(self, texts: list[str]) -> list[dict]:
        """Generate BM25 sparse vector dicts for a list of texts."""
        if not texts:
            return []
        with self._lock:
            out: list[dict] = []
            any_new_vocab = False
            for text in texts:
                sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
                out.append(sparse_vector)
                any_new_vocab = any_new_vocab or vocab_changed
            if any_new_vocab:
                self._persist_unlocked()
        return out


# ═══════════════════════════════════════════════════════════════════
# Module-level singletons (aligned with SuperMew pattern)
# ═══════════════════════════════════════════════════════════════════

clip_embeddings = ChineseCLIPEmbeddings(
    model_name=os.getenv("CLIP_MODEL", "ViT-L-14"),
    device=os.getenv("CLIP_DEVICE", "cpu"),
)

bge_embeddings = BGEM3Embeddings()

bm25 = ChineseBM25()

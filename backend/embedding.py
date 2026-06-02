"""
多模型 Embedding 服务：ChineseCLIP（图像/文本）+ BGE-M3（文本稠密）+ BM25（文本稀疏）。

- ChineseCLIPEmbeddings：封装 cn-clip，用于图像→图像和短文本→图像跨模态搜索。
  输出 768 维 L2 归一化向量，存入 Image Milvus Collection。
- BGEM3Embeddings：封装 sentence-transformers BAAI/bge-m3，用于长文本语义搜索。
  输出 1024 维 L2 归一化向量，存入 Text Milvus Collection（dense 字段）。
- ChineseBM25：自定义中文 BM25 实现，用于关键词搜索。
  将词表 + 文档频率持久化到 JSON。输出稀疏向量，存入 Text Milvus Collection。
- 模块级单例实例，对齐 SuperMew 模式。
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
# Chinese-CLIP Embedding（图像 + 短文本，768 维）
# ═══════════════════════════════════════════════════════════════════

class ChineseCLIPEmbeddings:
    """Chinese-CLIP ViT-L/14 封装，用于图像和短文本 Embedding。

    两种模态映射到相同的 L2 归一化 768 维空间，
    支持在同一个向量字段中进行跨模态图像↔文本搜索。

    使用官方 cn-clip 包（pip install cn-clip）。
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
        """对单张图像编码（支持文件路径 str、Path 或 PIL Image）。返回 768 维列表。"""
        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().tolist()

    def embed_images(self, images: list) -> List[List[float]]:
        """批量编码多张图像。"""
        return [self.embed_image(img) for img in images]

    def embed_query(self, text: str) -> List[float]:
        """将中文短文本查询（≤77 tokens）编码到图像空间。"""
        import cn_clip.clip as clip

        text_tokens = clip.tokenize([text]).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(text_tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量编码多条短文本。"""
        return [self.embed_query(t) for t in texts]


# ═══════════════════════════════════════════════════════════════════
# BGE-M3 稠密文本 Embedding（1024 维，语义搜索）
# ═══════════════════════════════════════════════════════════════════

class BGEM3Embeddings:
    """BGE-M3 稠密文本 Embedding，用于语义搜索。

    封装 sentence-transformers。输出 1024 维 L2 归一化向量。
    用于 Text Milvus Collection（dense_embedding 字段）。
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
        """对单条查询文本编码。"""
        embedding = self._model.encode(
            text, normalize_embeddings=True, show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量编码多条文档。"""
        if not texts:
            return []
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )
        return embeddings.tolist()


# ═══════════════════════════════════════════════════════════════════
# BM25 稀疏文本 Embedding（关键词搜索）
# ═══════════════════════════════════════════════════════════════════

class ChineseBM25:
    """BM25 稀疏向量服务，支持中文分词。

    分词方式：中文逐字 + 英文词级。
    将词表 + 文档频率持久化到 JSON。

    对齐 SuperMew 的 BM25 实现。
    """

    def __init__(self, state_path: Optional[Path | str] = None):
        self._state_path = Path(
            state_path or os.getenv("BM25_STATE_PATH", _DEFAULT_BM25_STATE_PATH)
        )
        self._lock = threading.Lock()

        # BM25 参数
        self.k1 = 1.5
        self.b = 0.75

        self._vocab: dict[str, int] = {}
        self._vocab_counter = 0
        self._doc_freq: Counter[str] = Counter()
        self._total_docs = 0
        self._sum_token_len = 0
        self._avg_doc_len = 1.0

        self._load_state()

    # ── 持久化 ───────────────────────────────────────────────────

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

    # ── 增量更新（对齐 SuperMew）────────────────────────────────

    def increment_add_documents(self, texts: list[str]) -> None:
        """增量添加文档到 BM25 统计。"""
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
        """增量从 BM25 统计中移除文档。"""
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

    # ── 分词 ────────────────────────────────────────────────────

    def tokenize(self, text: str) -> list[str]:
        """中文逐字 + 英文词级分词。"""
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

    # ── 稀疏向量生成 ───────────────────────────────────────────

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
        """为单条文本生成 BM25 稀疏向量字典。"""
        with self._lock:
            sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
            if vocab_changed:
                self._persist_unlocked()
        return sparse_vector

    def get_sparse_embeddings(self, texts: list[str]) -> list[dict]:
        """为多条文本批量生成 BM25 稀疏向量字典。"""
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
# 模块级单例（对齐 SuperMew 模式）
# ═══════════════════════════════════════════════════════════════════

clip_embeddings = ChineseCLIPEmbeddings(
    model_name=os.getenv("CLIP_MODEL", "ViT-L-14"),
    device=os.getenv("CLIP_DEVICE", "cpu"),
)

bge_embeddings = BGEM3Embeddings()

bm25 = ChineseBM25()

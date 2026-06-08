"""
双 Milvus Collection 的批量写入器。

- write_image_pois()：用 Chinese-CLIP 编码参考图像 → 插入 image_poi_collection。
- write_text_chunks()：用 BGE-M3 + BM25 编码 L3 文本 → 插入 text_chunk_collection。
  对齐 SuperMew 的 MilvusWriter 模式。
"""

from typing import Callable, Optional

from backend.embedding import bge_embeddings, bm25, clip_embeddings
from backend.milvus_client import milvus_manager


def _truncate_utf8(value, max_bytes: int) -> str:
    """Truncate a string to a Milvus VARCHAR byte limit without splitting UTF-8."""
    text = "" if value is None else str(value)
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore").rstrip()


class MilvusWriter:
    """图像 POI 和文本块 collection 的批量写入器。"""

    def __init__(self):
        self._clip = clip_embeddings
        self._bge = bge_embeddings
        self._bm25 = bm25
        self._milvus = milvus_manager

    # ── 图像 POI Collection（Collection 1）──────────────────────

    def write_image_pois(
        self,
        image_pois: list[dict],
        batch_size: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        用 Chinese-CLIP 编码参考图像并写入 image_poi_collection。

        image_pois 中每个元素应包含：
            chunk_id, filename, image_path, source_url, poi_name, site, cave,
            poi_description, distinguishing_features, tags
        """
        if not image_pois:
            return

        self._milvus.init_image_collection(dense_dim=self._clip.dimension)
        total = len(image_pois)

        for i in range(0, total, batch_size):
            batch = image_pois[i : i + batch_size]
            image_paths = [item["image_path"] for item in batch]
            image_vectors = self._clip.embed_images(image_paths)

            insert_data = []
            for item, vec in zip(batch, image_vectors):
                insert_data.append({
                    "chunk_id": _truncate_utf8(item["chunk_id"], 512),
                    "filename": _truncate_utf8(item.get("filename", ""), 255),
                    "image_vector": vec,
                    "image_path": _truncate_utf8(item["image_path"], 512),
                    "source_url": _truncate_utf8(item.get("source_url", ""), 2048),
                    "poi_name": _truncate_utf8(item.get("poi_name", ""), 256),
                    "site": _truncate_utf8(item.get("site", ""), 128),
                    "cave": _truncate_utf8(item.get("cave", ""), 128),
                    "poi_description": _truncate_utf8(item.get("poi_description", ""), 2000),
                    "distinguishing_features": _truncate_utf8(item.get("distinguishing_features", ""), 1000),
                    "tags": _truncate_utf8(item.get("tags", ""), 512),
                })

            self._milvus.insert_to_image_collection(insert_data)

            if progress_callback:
                processed = min(i + batch_size, total)
                progress_callback(processed, total)

    # ── 文本块 Collection（Collection 2，对齐 SuperMew）─────────

    def write_text_chunks(
        self,
        documents: list[dict],
        batch_size: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        用 BGE-M3（稠密）+ BM25（稀疏）编码 L3 文本块，写入
        text_chunk_collection。增量更新 BM25 统计。

        每个文档字典应包含：
            text, filename, file_type, file_path, page_number, chunk_idx,
            chunk_id, parent_chunk_id, root_chunk_id, chunk_level,
            site, cave, poi_name, source_url
        """
        if not documents:
            return

        self._milvus.init_text_collection(dense_dim=self._bge.dimension)

        # 写入前同步 BM25 统计（对齐 SuperMew）。
        all_texts = [doc["text"] for doc in documents]
        self._bm25.increment_add_documents(all_texts)

        total = len(documents)

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc["text"] for doc in batch]

            dense_vectors = self._bge.embed_documents(texts)
            sparse_vectors = self._bm25.get_sparse_embeddings(texts)

            insert_data = []
            for doc, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors):
                insert_data.append({
                    "dense_embedding": dense_vec,
                    "sparse_embedding": sparse_vec,
                    "text": _truncate_utf8(doc["text"], 4000),
                    "filename": _truncate_utf8(doc.get("filename", ""), 255),
                    "file_type": _truncate_utf8(doc.get("file_type", ""), 50),
                    "file_path": _truncate_utf8(doc.get("file_path", ""), 1024),
                    "source_url": _truncate_utf8(doc.get("source_url", ""), 2048),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": _truncate_utf8(doc.get("chunk_id", ""), 512),
                    "parent_chunk_id": _truncate_utf8(doc.get("parent_chunk_id", ""), 512),
                    "root_chunk_id": _truncate_utf8(doc.get("root_chunk_id", ""), 512),
                    "chunk_level": doc.get("chunk_level", 0),
                    "site": _truncate_utf8(doc.get("site", ""), 128),
                    "cave": _truncate_utf8(doc.get("cave", ""), 128),
                    "poi_name": _truncate_utf8(doc.get("poi_name", ""), 256),
                })

            self._milvus.insert_to_text_collection(insert_data)

            if progress_callback:
                processed = min(i + batch_size, total)
                progress_callback(processed, total)


# 模块级单例。
milvus_writer = MilvusWriter()

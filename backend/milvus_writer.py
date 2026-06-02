"""
双 Milvus Collection 的批量写入器。

- write_image_pois()：用 Chinese-CLIP 编码参考图像 → 插入 image_poi_collection。
- write_text_chunks()：用 BGE-M3 + BM25 编码 L3 文本 → 插入 text_chunk_collection。
  对齐 SuperMew 的 MilvusWriter 模式。
"""

from typing import Callable, Optional

from backend.embedding import bge_embeddings, bm25, clip_embeddings
from backend.milvus_client import milvus_manager


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
            chunk_id, image_path, poi_name, site, cave,
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
                    "chunk_id": item["chunk_id"],
                    "image_vector": vec,
                    "image_path": item["image_path"],
                    "poi_name": item.get("poi_name", ""),
                    "site": item.get("site", ""),
                    "cave": item.get("cave", ""),
                    "poi_description": item.get("poi_description", ""),
                    "distinguishing_features": item.get("distinguishing_features", ""),
                    "tags": item.get("tags", ""),
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
            site, cave, poi_name
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
                    "text": doc["text"],
                    "filename": doc.get("filename", ""),
                    "file_type": doc.get("file_type", ""),
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                    "site": doc.get("site", ""),
                    "cave": doc.get("cave", ""),
                    "poi_name": doc.get("poi_name", ""),
                })

            self._milvus.insert_to_text_collection(insert_data)

            if progress_callback:
                processed = min(i + batch_size, total)
                progress_callback(processed, total)


# 模块级单例。
milvus_writer = MilvusWriter()

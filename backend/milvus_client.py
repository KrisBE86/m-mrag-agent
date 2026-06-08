"""
Milvus 向量数据库客户端 — 双 Collection 架构。

Collection 1 — image_poi_collection：
  Chinese-CLIP 稠密向量（768 维），用于图像到图像和短文本到图像搜索。
  HNSW 索引，IP 度量。不需要稀疏字段。

Collection 2 — text_chunk_collection：
  BGE-M3 稠密（1024 维）+ BM25 稀疏，用于混合文本检索。
  HNSW + SPARSE_INVERTED_INDEX，RRFRanker 融合。
  完全对齐 SuperMew 的文本 collection schema。

功能特性：
- 惰性连接 + gRPC "closed channel" 错误时自动重连。
- init_image_collection() / init_text_collection()：创建 schema + 索引。
- image_search()：Collection 1 上的纯稠密 CLIP 搜索。
- hybrid_retrieve()：Collection 2 上的稠密 + 稀疏 + RRF（与 SuperMew 一致）。
- CRUD：两个 collection 的插入、查询、删除操作。
"""

import os
import threading
from typing import Callable, TypeVar

from dotenv import load_dotenv
from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

load_dotenv()

QUERY_MAX_LIMIT = 16384
T = TypeVar("T")


class MilvusManager:
    """Milvus 连接管理器，管理两个 collection。"""

    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "127.0.0.1")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.image_collection = os.getenv("MILVUS_IMAGE_COLLECTION", "image_poi_collection")
        self.text_collection = os.getenv("MILVUS_TEXT_COLLECTION", "text_chunk_collection")
        self.uri = f"http://{self.host}:{self.port}"
        self.client: MilvusClient | None = None
        self._client_lock = threading.RLock()

    # ── 连接管理 ───────────────────────────────────────────────

    def _get_client(self) -> MilvusClient:
        with self._client_lock:
            if self.client is None:
                self.client = MilvusClient(uri=self.uri)
            return self.client

    @staticmethod
    def _is_closed_channel_error(exc: Exception) -> bool:
        return isinstance(exc, ValueError) and "closed channel" in str(exc).lower()

    @staticmethod
    def _close_client(client) -> None:
        close = getattr(client, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            pass

    def _reset_client(self, failed_client=None) -> None:
        with self._client_lock:
            if self.client is None:
                return
            if failed_client is not None and self.client is not failed_client:
                return
            client = self.client
            self.client = None
        self._close_client(client)

    def _run_with_reconnect(self, operation: Callable[[MilvusClient], T]) -> T:
        client = self._get_client()
        try:
            return operation(client)
        except Exception as exc:
            if not self._is_closed_channel_error(exc):
                raise
            self._reset_client(client)
            return operation(self._get_client())

    # ══════════════════════════════════════════════════════════════
    # Collection 1：图像 POI（Chinese-CLIP，仅稠密向量）
    # ══════════════════════════════════════════════════════════════

    def init_image_collection(self, dense_dim: int = 768) -> None:
        """初始化图像 POI collection，使用 CLIP 768 维稠密向量。"""

        def _init(client: MilvusClient) -> None:
            if client.has_collection(self.image_collection):
                return

            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("filename", DataType.VARCHAR, max_length=255)
            schema.add_field("image_vector", DataType.FLOAT_VECTOR, dim=dense_dim)
            schema.add_field("image_path", DataType.VARCHAR, max_length=512)
            schema.add_field("source_url", DataType.VARCHAR, max_length=2048)
            schema.add_field("poi_name", DataType.VARCHAR, max_length=256)
            schema.add_field("site", DataType.VARCHAR, max_length=128)
            schema.add_field("cave", DataType.VARCHAR, max_length=128)
            schema.add_field("poi_description", DataType.VARCHAR, max_length=2000)
            schema.add_field("distinguishing_features", DataType.VARCHAR, max_length=1000)
            schema.add_field("tags", DataType.VARCHAR, max_length=512)

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="image_vector",
                index_type="HNSW",
                metric_type="IP",
                params={"M": 16, "efConstruction": 256},
            )

            client.create_collection(
                collection_name=self.image_collection,
                schema=schema,
                index_params=index_params,
            )

        self._run_with_reconnect(_init)

    def image_search(
        self,
        image_vector: list[float],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[dict]:
        """
        Collection 1 上的纯稠密图像向量搜索。
        返回 POI 元数据（含 poi_description），供下游文本 RAG 使用。
        """

        output_fields = [
            "chunk_id", "filename", "image_path", "source_url", "poi_name", "site", "cave",
            "poi_description", "distinguishing_features", "tags",
        ]

        results = self._run_with_reconnect(
            lambda client: client.search(
                collection_name=self.image_collection,
                data=[image_vector],
                anns_field="image_vector",
                search_params={"metric_type": "IP", "params": {"ef": 64}},
                limit=top_k,
                output_fields=output_fields,
                filter=filter_expr,
            )
        )

        formatted = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                formatted.append({
                    "chunk_id": entity.get("chunk_id", ""),
                    "filename": entity.get("filename", ""),
                    "image_path": entity.get("image_path", ""),
                    "source_url": entity.get("source_url", ""),
                    "poi_name": entity.get("poi_name", ""),
                    "site": entity.get("site", ""),
                    "cave": entity.get("cave", ""),
                    "poi_description": entity.get("poi_description", ""),
                    "distinguishing_features": entity.get("distinguishing_features", ""),
                    "tags": entity.get("tags", ""),
                    "score": hit.get("distance", 0.0),
                })
        return formatted

    def insert_to_image_collection(self, data: list[dict]) -> dict:
        """将 POI 图像数据插入 Collection 1。"""
        return self._run_with_reconnect(
            lambda client: client.insert(self.image_collection, data)
        )

    # ══════════════════════════════════════════════════════════════
    # Collection 2：文本块（BGE-M3 + BM25，对齐 SuperMew）
    # ══════════════════════════════════════════════════════════════

    def init_text_collection(self, dense_dim: int = 1024) -> None:
        """初始化文本块 collection（稠密 + 稀疏）。与 SuperMew 一致。"""

        def _init(client: MilvusClient) -> None:
            if client.has_collection(self.text_collection):
                return

            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
            schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("text", DataType.VARCHAR, max_length=4000)
            schema.add_field("filename", DataType.VARCHAR, max_length=255)
            schema.add_field("file_type", DataType.VARCHAR, max_length=50)
            schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
            schema.add_field("source_url", DataType.VARCHAR, max_length=2048)
            schema.add_field("page_number", DataType.INT64)
            schema.add_field("chunk_idx", DataType.INT64)
            schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("chunk_level", DataType.INT64)
            schema.add_field("site", DataType.VARCHAR, max_length=128)
            schema.add_field("cave", DataType.VARCHAR, max_length=128)
            schema.add_field("poi_name", DataType.VARCHAR, max_length=256)

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="dense_embedding",
                index_type="HNSW",
                metric_type="IP",
                params={"M": 16, "efConstruction": 256},
            )
            index_params.add_index(
                field_name="sparse_embedding",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )

            client.create_collection(
                collection_name=self.text_collection,
                schema=schema,
                index_params=index_params,
            )

        self._run_with_reconnect(_init)

    def hybrid_retrieve(
        self,
        dense_embedding: list[float],
        sparse_embedding: dict,
        top_k: int = 5,
        rrf_k: int = 60,
        filter_expr: str = "",
    ) -> list[dict]:
        """
        Collection 2 上的混合检索：稠密（HNSW）+ 稀疏（BM25），通过 RRF 融合。
        完全对齐 SuperMew 的 hybrid_retrieve()。
        """

        output_fields = [
            "text", "filename", "file_type", "file_path", "source_url", "page_number",
            "chunk_id", "parent_chunk_id", "root_chunk_id", "chunk_level",
            "chunk_idx", "site", "cave", "poi_name",
        ]

        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k * 2,
            expr=filter_expr,
        )

        sparse_search = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k * 2,
            expr=filter_expr,
        )

        reranker = RRFRanker(k=rrf_k)

        results = self._run_with_reconnect(
            lambda client: client.hybrid_search(
                collection_name=self.text_collection,
                reqs=[dense_search, sparse_search],
                ranker=reranker,
                limit=top_k,
                output_fields=output_fields,
            )
        )

        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "id": hit.get("id"),
                    "text": hit.get("entity", {}).get("text", ""),
                    "filename": hit.get("entity", {}).get("filename", ""),
                    "file_type": hit.get("entity", {}).get("file_type", ""),
                    "file_path": hit.get("entity", {}).get("file_path", ""),
                    "source_url": hit.get("entity", {}).get("source_url", ""),
                    "page_number": hit.get("entity", {}).get("page_number", 0),
                    "chunk_id": hit.get("entity", {}).get("chunk_id", ""),
                    "parent_chunk_id": hit.get("entity", {}).get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("entity", {}).get("root_chunk_id", ""),
                    "chunk_level": hit.get("entity", {}).get("chunk_level", 0),
                    "chunk_idx": hit.get("entity", {}).get("chunk_idx", 0),
                    "site": hit.get("entity", {}).get("site", ""),
                    "cave": hit.get("entity", {}).get("cave", ""),
                    "poi_name": hit.get("entity", {}).get("poi_name", ""),
                    "score": hit.get("distance", 0.0),
                })
        return formatted

    def insert_to_text_collection(self, data: list[dict]) -> dict:
        """将文本块数据插入 Collection 2。"""
        return self._run_with_reconnect(
            lambda client: client.insert(self.text_collection, data)
        )

    # ── 公共 CRUD ───────────────────────────────────────────────

    def query(
        self,
        collection: str,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
        limit: int = 10000,
        offset: int = 0,
    ) -> list[dict]:
        return self._run_with_reconnect(
            lambda client: client.query(
                collection_name=collection,
                filter=filter_expr,
                output_fields=output_fields or ["chunk_id"],
                limit=min(limit, QUERY_MAX_LIMIT),
                offset=offset,
            )
        )

    def delete(self, collection: str, filter_expr: str) -> dict:
        return self._run_with_reconnect(
            lambda client: client.delete(collection_name=collection, filter=filter_expr)
        )

    def has_collection(self, collection: str) -> bool:
        return self._run_with_reconnect(
            lambda client: client.has_collection(collection)
        )

    def drop_collection(self, collection: str) -> None:
        def _drop(client: MilvusClient) -> None:
            if client.has_collection(collection):
                client.drop_collection(collection)

        self._run_with_reconnect(_drop)

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """按 chunk_id 列表查询 Collection 2（用于 auto-merge）。"""
        ids = [item for item in chunk_ids if item]
        if not ids:
            return []
        quoted = ", ".join([f'"{item}"' for item in ids])
        filter_expr = f"chunk_id in [{quoted}]"
        return self.query(
            collection=self.text_collection,
            filter_expr=filter_expr,
            output_fields=[
                "text", "filename", "file_type", "file_path", "source_url", "page_number",
                "chunk_id", "parent_chunk_id", "root_chunk_id",
                "chunk_level", "chunk_idx", "site", "cave", "poi_name",
            ],
            limit=len(ids),
        )


# 模块级单例，对齐 SuperMew 模式。
milvus_manager = MilvusManager()

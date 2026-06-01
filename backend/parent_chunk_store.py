"""
Parent chunk L1/L2 storage service — PostgreSQL + Redis.

- Aligned with SuperMew's ParentChunkStore pattern.
- upsert_documents: batch write/update parent chunks + Redis cache.
- get_documents_by_ids: batch query by chunk_id, Redis-first then PostgreSQL fallback.
- delete_by_filename: remove all parent chunks + cache for a filename.
"""

from datetime import datetime
from typing import List

from backend.cache import cache
from backend.database import SessionLocal
from backend.models import ParentChunk


class ParentChunkStore:
    """PostgreSQL + Redis parent chunk storage for auto-merging retrieval."""

    @staticmethod
    def _to_dict(item: ParentChunk) -> dict:
        return {
            "text": item.text,
            "filename": item.filename,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "page_number": item.page_number,
            "chunk_id": item.chunk_id,
            "parent_chunk_id": item.parent_chunk_id,
            "root_chunk_id": item.root_chunk_id,
            "chunk_level": item.chunk_level,
            "chunk_idx": item.chunk_idx,
        }

    @staticmethod
    def _cache_key(chunk_id: str) -> str:
        return f"parent_chunk:{chunk_id}"

    def upsert_documents(self, docs: List[dict]) -> int:
        """Write/update parent chunks. Returns upserted count."""
        if not docs:
            return 0

        db = SessionLocal()
        upserted = 0
        try:
            for doc in docs:
                chunk_id = (doc.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue

                record = db.query(ParentChunk).filter(
                    ParentChunk.chunk_id == chunk_id
                ).first()

                payload = {
                    "text": doc.get("text", ""),
                    "filename": doc.get("filename", ""),
                    "file_type": doc.get("file_type", ""),
                    "file_path": doc.get("file_path", ""),
                    "page_number": int(doc.get("page_number", 0) or 0),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": int(doc.get("chunk_level", 0) or 0),
                    "chunk_idx": int(doc.get("chunk_idx", 0) or 0),
                    "updated_at": datetime.utcnow(),
                }
                cache_payload = {
                    "chunk_id": chunk_id,
                    **{k: v for k, v in payload.items() if k != "updated_at"},
                }

                if record:
                    for key, value in payload.items():
                        setattr(record, key, value)
                else:
                    db.add(ParentChunk(chunk_id=chunk_id, **payload))

                cache.set_json(self._cache_key(chunk_id), cache_payload)
                upserted += 1

            db.commit()
        finally:
            db.close()

        return upserted

    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        """Get parent chunks by chunk_id list. Redis-first, PostgreSQL fallback."""
        if not chunk_ids:
            return []

        ordered_results = {}
        missing_ids = []
        for chunk_id in chunk_ids:
            key = (chunk_id or "").strip()
            if not key:
                continue
            cached = cache.get_json(self._cache_key(key))
            if cached:
                ordered_results[key] = cached
            else:
                missing_ids.append(key)

        if missing_ids:
            db = SessionLocal()
            try:
                rows = (
                    db.query(ParentChunk)
                    .filter(ParentChunk.chunk_id.in_(missing_ids))
                    .all()
                )
                for row in rows:
                    payload = self._to_dict(row)
                    ordered_results[row.chunk_id] = payload
                    cache.set_json(self._cache_key(row.chunk_id), payload)
            finally:
                db.close()

        return [ordered_results[item] for item in chunk_ids if item in ordered_results]

    def delete_by_filename(self, filename: str) -> int:
        """Delete all parent chunks for a filename. Returns deleted count."""
        if not filename:
            return 0

        db = SessionLocal()
        try:
            rows = (
                db.query(ParentChunk)
                .filter(ParentChunk.filename == filename)
                .all()
            )
            chunk_ids = [row.chunk_id for row in rows]
            deleted = len(chunk_ids)
            if deleted > 0:
                db.query(ParentChunk).filter(
                    ParentChunk.filename == filename
                ).delete(synchronize_session=False)
                db.commit()
                for chunk_id in chunk_ids:
                    cache.delete(self._cache_key(chunk_id))
            return deleted
        finally:
            db.close()


# Module-level singleton, aligned with SuperMew pattern.
parent_chunk_store = ParentChunkStore()

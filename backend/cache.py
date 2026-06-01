"""
Redis cache wrapper.
- JSON object read/write with auto serialization/deserialization.
- Namespaced keys with configurable TTL.
- Global singleton `cache` for module-level access.
- Aligned with SuperMew's cache.py pattern.
"""

import json
import os
from typing import Any, Optional

import redis


class RedisCache:
    """Redis cache with JSON serialization and key prefix namespacing."""

    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        self.key_prefix = os.getenv("REDIS_KEY_PREFIX", "mragagent")
        self.default_ttl = int(os.getenv("REDIS_CACHE_TTL_SECONDS", "300"))
        self._client: Optional[redis.Redis] = None

    def _get_client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _key(self, key: str) -> str:
        return f"{self.key_prefix}:{key}"

    def get_json(self, key: str) -> Optional[Any]:
        """Get a JSON value by key. Returns None on any error."""
        try:
            value = self._get_client().get(self._key(key))
            if not value:
                return None
            return json.loads(value)
        except Exception:
            return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a JSON value with optional TTL override."""
        try:
            payload = json.dumps(value, ensure_ascii=False)
            self._get_client().setex(self._key(key), ttl or self.default_ttl, payload)
        except Exception:
            return

    def delete(self, key: str) -> None:
        """Delete a key."""
        try:
            self._get_client().delete(self._key(key))
        except Exception:
            return

    def delete_pattern(self, pattern: str) -> None:
        """Delete all keys matching a glob pattern under the namespace."""
        try:
            full_pattern = self._key(pattern)
            keys = self._get_client().keys(full_pattern)
            if keys:
                self._get_client().delete(*keys)
        except Exception:
            return


# Module-level singleton for shared access.
cache = RedisCache()

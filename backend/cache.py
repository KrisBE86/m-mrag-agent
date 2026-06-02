"""
Redis 缓存封装。
- JSON 对象读写，自动序列化/反序列化。
- 带命名空间的 key，支持可配置 TTL。
- 全局单例 `cache`，供模块级直接访问。
- 对齐 SuperMew 的 cache.py 模式。
"""

import json
import os
from typing import Any, Optional

import redis


class RedisCache:
    """Redis 缓存，支持 JSON 序列化和 key 前缀命名空间。"""

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
        """按 key 获取 JSON 值。任何异常返回 None。"""
        try:
            value = self._get_client().get(self._key(key))
            if not value:
                return None
            return json.loads(value)
        except Exception:
            return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置 JSON 值，可选覆盖 TTL。"""
        try:
            payload = json.dumps(value, ensure_ascii=False)
            self._get_client().setex(self._key(key), ttl or self.default_ttl, payload)
        except Exception:
            return

    def delete(self, key: str) -> None:
        """删除一个 key。"""
        try:
            self._get_client().delete(self._key(key))
        except Exception:
            return

    def delete_pattern(self, pattern: str) -> None:
        """删除命名空间下所有匹配 glob 模式的 key。"""
        try:
            full_pattern = self._key(pattern)
            keys = self._get_client().keys(full_pattern)
            if keys:
                self._get_client().delete(*keys)
        except Exception:
            return


# 模块级单例，供全局共享访问。
cache = RedisCache()

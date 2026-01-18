"""
AI Cache Module for Ultimate Trading Bot v2.2.

This module provides caching for AI responses to reduce
API costs and improve response times.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache strategy enumeration."""

    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class CachePriority(str, Enum):
    """Cache priority enumeration."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class CacheEntry(BaseModel):
    """Cache entry model."""

    key: str
    value: Any
    created_at: datetime = Field(default_factory=now_utc)
    last_accessed: datetime = Field(default_factory=now_utc)
    expires_at: Optional[datetime] = None
    access_count: int = Field(default=0)
    size_bytes: int = Field(default=0)
    priority: CachePriority = Field(default=CachePriority.NORMAL)
    metadata: dict = Field(default_factory=dict)


class CacheStats(BaseModel):
    """Cache statistics model."""

    total_entries: int = Field(default=0)
    total_size_bytes: int = Field(default=0)
    hits: int = Field(default=0)
    misses: int = Field(default=0)
    evictions: int = Field(default=0)
    hit_rate: float = Field(default=0.0)
    avg_entry_age_seconds: float = Field(default=0.0)
    cost_savings_estimate: float = Field(default=0.0)


class AICacheConfig(BaseModel):
    """Configuration for AI cache."""

    max_entries: int = Field(default=1000, ge=10, le=100000)
    max_size_bytes: int = Field(default=100 * 1024 * 1024, ge=1024)
    default_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    strategy: CacheStrategy = Field(default=CacheStrategy.LRU)
    enable_compression: bool = Field(default=False)
    cost_per_request: float = Field(default=0.01)
    enable_persistence: bool = Field(default=False)
    persistence_path: str = Field(default="cache/ai_cache.json")


class AICache:
    """
    AI Response Cache for cost optimization.

    Provides:
    - Response caching with configurable TTL
    - Multiple eviction strategies (LRU, LFU, TTL)
    - Cache statistics and cost tracking
    - Priority-based caching
    """

    def __init__(
        self,
        config: Optional[AICacheConfig] = None,
    ) -> None:
        """
        Initialize AICache.

        Args:
            config: Cache configuration
        """
        self._config = config or AICacheConfig()

        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []
        self._lock = asyncio.Lock()

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_size = 0

        logger.info(
            f"AICache initialized with strategy={self._config.strategy.value}, "
            f"max_entries={self._config.max_entries}"
        )

    async def get(
        self,
        key: str,
        default: Any = None,
    ) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return default

            if entry.expires_at and now_utc() > entry.expires_at:
                self._remove_entry(key)
                self._misses += 1
                return default

            entry.last_accessed = now_utc()
            entry.access_count += 1

            self._update_access_order(key)

            self._hits += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        priority: CachePriority = CachePriority.NORMAL,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            priority: Cache priority
            metadata: Optional metadata

        Returns:
            True if cached successfully
        """
        async with self._lock:
            ttl = ttl_seconds or self._config.default_ttl_seconds
            expires_at = now_utc() + timedelta(seconds=ttl)

            size_bytes = self._estimate_size(value)

            if size_bytes > self._config.max_size_bytes:
                logger.warning(f"Value too large to cache: {size_bytes} bytes")
                return False

            while (
                len(self._cache) >= self._config.max_entries or
                self._total_size + size_bytes > self._config.max_size_bytes
            ):
                if not self._evict_one():
                    break

            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                size_bytes=size_bytes,
                priority=priority,
                metadata=metadata or {},
            )

            if key in self._cache:
                old_entry = self._cache[key]
                self._total_size -= old_entry.size_bytes

            self._cache[key] = entry
            self._total_size += size_bytes
            self._update_access_order(key)

            return True

    async def delete(self, key: str) -> bool:
        """
        Delete a value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        async with self._lock:
            return self._remove_entry(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        value = await self.get(key)
        return value is not None

    async def get_or_set(
        self,
        key: str,
        factory: Any,
        ttl_seconds: Optional[int] = None,
        priority: CachePriority = CachePriority.NORMAL,
    ) -> Any:
        """
        Get from cache or compute and set.

        Args:
            key: Cache key
            factory: Async function or value to compute
            ttl_seconds: Time to live
            priority: Cache priority

        Returns:
            Cached or computed value
        """
        value = await self.get(key)
        if value is not None:
            return value

        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        elif callable(factory):
            value = factory()
        else:
            value = factory

        await self.set(key, value, ttl_seconds, priority)
        return value

    def generate_key(
        self,
        operation: str,
        **kwargs,
    ) -> str:
        """
        Generate a cache key from operation and parameters.

        Args:
            operation: Operation name
            **kwargs: Parameters to include in key

        Returns:
            Cache key string
        """
        sorted_params = sorted(kwargs.items())
        params_str = json.dumps(sorted_params, sort_keys=True, default=str)
        content = f"{operation}:{params_str}"

        return hashlib.sha256(content.encode()).hexdigest()[:32]

    async def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            self._total_size = 0
            logger.info(f"Cleared {count} cache entries")
            return count

    async def clear_expired(self) -> int:
        """
        Clear expired cache entries.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            now = now_utc()
            expired: list[str] = []

            for key, entry in self._cache.items():
                if entry.expires_at and now > entry.expires_at:
                    expired.append(key)

            for key in expired:
                self._remove_entry(key)

            if expired:
                logger.debug(f"Cleared {len(expired)} expired entries")

            return len(expired)

    async def clear_by_prefix(self, prefix: str) -> int:
        """
        Clear cache entries by key prefix.

        Args:
            prefix: Key prefix to match

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            to_remove = [
                key for key in self._cache.keys()
                if key.startswith(prefix)
            ]

            for key in to_remove:
                self._remove_entry(key)

            return len(to_remove)

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            now = now_utc()
            ages: list[float] = []
            for entry in self._cache.values():
                age = (now - entry.created_at).total_seconds()
                ages.append(age)

            avg_age = sum(ages) / len(ages) if ages else 0.0

            cost_savings = self._hits * self._config.cost_per_request

            return CacheStats(
                total_entries=len(self._cache),
                total_size_bytes=self._total_size,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                hit_rate=hit_rate,
                avg_entry_age_seconds=avg_age,
                cost_savings_estimate=cost_savings,
            )

    async def get_entry_info(self, key: str) -> Optional[dict]:
        """Get information about a cache entry."""
        async with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None

            return {
                "key": entry.key,
                "created_at": entry.created_at.isoformat(),
                "last_accessed": entry.last_accessed.isoformat(),
                "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                "access_count": entry.access_count,
                "size_bytes": entry.size_bytes,
                "priority": entry.priority.value,
                "metadata": entry.metadata,
            }

    async def list_keys(
        self,
        pattern: Optional[str] = None,
        limit: int = 100,
    ) -> list[str]:
        """
        List cache keys.

        Args:
            pattern: Optional pattern to filter keys
            limit: Maximum keys to return

        Returns:
            List of cache keys
        """
        async with self._lock:
            keys = list(self._cache.keys())

            if pattern:
                keys = [k for k in keys if pattern in k]

            return keys[:limit]

    def _remove_entry(self, key: str) -> bool:
        """Remove an entry from cache."""
        entry = self._cache.get(key)
        if not entry:
            return False

        self._total_size -= entry.size_bytes
        del self._cache[key]

        if key in self._access_order:
            self._access_order.remove(key)

        return True

    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU tracking."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict_one(self) -> bool:
        """Evict one entry based on strategy."""
        if not self._cache:
            return False

        key_to_evict: Optional[str] = None

        if self._config.strategy == CacheStrategy.LRU:
            key_to_evict = self._select_lru()
        elif self._config.strategy == CacheStrategy.LFU:
            key_to_evict = self._select_lfu()
        elif self._config.strategy == CacheStrategy.TTL:
            key_to_evict = self._select_ttl()
        elif self._config.strategy == CacheStrategy.ADAPTIVE:
            key_to_evict = self._select_adaptive()

        if key_to_evict:
            self._remove_entry(key_to_evict)
            self._evictions += 1
            return True

        return False

    def _select_lru(self) -> Optional[str]:
        """Select least recently used entry."""
        low_priority = [
            k for k in self._access_order
            if self._cache.get(k) and
            self._cache[k].priority in [CachePriority.LOW, CachePriority.NORMAL]
        ]

        if low_priority:
            return low_priority[0]

        return self._access_order[0] if self._access_order else None

    def _select_lfu(self) -> Optional[str]:
        """Select least frequently used entry."""
        candidates = [
            (k, e) for k, e in self._cache.items()
            if e.priority in [CachePriority.LOW, CachePriority.NORMAL]
        ]

        if not candidates:
            candidates = list(self._cache.items())

        if not candidates:
            return None

        return min(candidates, key=lambda x: x[1].access_count)[0]

    def _select_ttl(self) -> Optional[str]:
        """Select entry closest to expiration."""
        now = now_utc()
        candidates = [
            (k, e) for k, e in self._cache.items()
            if e.expires_at and e.priority in [CachePriority.LOW, CachePriority.NORMAL]
        ]

        if not candidates:
            candidates = [(k, e) for k, e in self._cache.items() if e.expires_at]

        if not candidates:
            return self._select_lru()

        return min(candidates, key=lambda x: x[1].expires_at)[0]

    def _select_adaptive(self) -> Optional[str]:
        """Select entry using adaptive scoring."""
        now = now_utc()
        scored: list[tuple[str, float]] = []

        for key, entry in self._cache.items():
            if entry.priority == CachePriority.CRITICAL:
                continue

            score = 0.0

            age = (now - entry.last_accessed).total_seconds()
            score += age / 3600

            score -= entry.access_count * 0.1

            if entry.expires_at:
                time_to_expire = (entry.expires_at - now).total_seconds()
                if time_to_expire < 300:
                    score += 10

            priority_weights = {
                CachePriority.LOW: 5,
                CachePriority.NORMAL: 0,
                CachePriority.HIGH: -5,
            }
            score += priority_weights.get(entry.priority, 0)

            scored.append((key, score))

        if not scored:
            return None

        return max(scored, key=lambda x: x[1])[0]

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of a value in bytes."""
        try:
            serialized = json.dumps(value, default=str)
            return len(serialized.encode())
        except Exception:
            return len(str(value).encode())

    async def warm_cache(
        self,
        entries: list[tuple[str, Any, Optional[int]]],
    ) -> int:
        """
        Warm cache with multiple entries.

        Args:
            entries: List of (key, value, ttl) tuples

        Returns:
            Number of entries added
        """
        count = 0
        for entry in entries:
            key = entry[0]
            value = entry[1]
            ttl = entry[2] if len(entry) > 2 else None

            if await self.set(key, value, ttl):
                count += 1

        logger.info(f"Warmed cache with {count} entries")
        return count

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AICache(entries={len(self._cache)}, "
            f"size={self._total_size}B, hits={self._hits})"
        )


class SemanticCache:
    """
    Semantic cache using embeddings for similarity-based retrieval.

    Provides fuzzy matching for similar queries to maximize cache hits.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        max_entries: int = 500,
    ) -> None:
        """
        Initialize SemanticCache.

        Args:
            similarity_threshold: Minimum similarity for cache hit
            max_entries: Maximum entries to store
        """
        self._threshold = similarity_threshold
        self._max_entries = max_entries

        self._entries: list[dict] = []
        self._embeddings: list[list[float]] = []

        self._hits = 0
        self._misses = 0

        logger.info("SemanticCache initialized")

    async def get(
        self,
        query: str,
        embedding: list[float],
    ) -> Optional[Any]:
        """
        Get value by semantic similarity.

        Args:
            query: Query text
            embedding: Query embedding

        Returns:
            Cached value if similar query found
        """
        if not self._embeddings:
            self._misses += 1
            return None

        best_similarity = 0.0
        best_idx = -1

        import numpy as np
        query_vec = np.array(embedding)

        for idx, stored_embedding in enumerate(self._embeddings):
            stored_vec = np.array(stored_embedding)

            dot_product = np.dot(query_vec, stored_vec)
            norm_product = np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)

            if norm_product > 0:
                similarity = dot_product / norm_product
            else:
                similarity = 0.0

            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx

        if best_similarity >= self._threshold and best_idx >= 0:
            self._hits += 1
            return self._entries[best_idx]["value"]

        self._misses += 1
        return None

    async def set(
        self,
        query: str,
        embedding: list[float],
        value: Any,
    ) -> None:
        """
        Store a value with its embedding.

        Args:
            query: Query text
            embedding: Query embedding
            value: Value to store
        """
        if len(self._entries) >= self._max_entries:
            self._entries.pop(0)
            self._embeddings.pop(0)

        self._entries.append({
            "query": query,
            "value": value,
            "timestamp": now_utc().isoformat(),
        })
        self._embeddings.append(embedding)

    def clear(self) -> int:
        """Clear the semantic cache."""
        count = len(self._entries)
        self._entries.clear()
        self._embeddings.clear()
        return count

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"SemanticCache(entries={len(self._entries)}, hits={self._hits})"

"""
Data Cache Module for Ultimate Trading Bot v2.2.

This module provides caching functionality for market data,
including in-memory and Redis-based caching with TTL support.
"""

import asyncio
import hashlib
import json
import logging
import pickle
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, TypeVar, Generic

from pydantic import BaseModel, Field

from src.utils.exceptions import CacheError
from src.utils.helpers import generate_uuid, safe_json_dumps, safe_json_loads
from src.utils.date_utils import now_utc
from src.utils.decorators import singleton


logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheBackend(str, Enum):
    """Cache backend enumeration."""

    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


class CachePolicy(str, Enum):
    """Cache eviction policy enumeration."""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"


class CacheConfig(BaseModel):
    """Configuration for cache."""

    backend: CacheBackend = Field(default=CacheBackend.MEMORY)
    policy: CachePolicy = Field(default=CachePolicy.LRU)

    max_size: int = Field(default=10000, ge=100, le=1000000)
    default_ttl_seconds: int = Field(default=300, ge=1, le=86400)

    redis_url: Optional[str] = None
    redis_prefix: str = Field(default="trading_bot:")
    redis_pool_size: int = Field(default=10, ge=1, le=100)

    enable_compression: bool = Field(default=False)
    compression_threshold_bytes: int = Field(default=1024, ge=100)

    enable_statistics: bool = Field(default=True)
    cleanup_interval_seconds: int = Field(default=60, ge=10, le=3600)


class CacheEntry(BaseModel):
    """Cache entry model."""

    key: str
    value: Any
    created_at: datetime = Field(default_factory=now_utc)
    expires_at: Optional[datetime] = None
    access_count: int = Field(default=0)
    last_accessed: datetime = Field(default_factory=now_utc)
    size_bytes: int = Field(default=0)

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return now_utc() >= self.expires_at

    @property
    def ttl_remaining(self) -> Optional[float]:
        """Get remaining TTL in seconds."""
        if self.expires_at is None:
            return None
        remaining = (self.expires_at - now_utc()).total_seconds()
        return max(0, remaining)

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = now_utc()
        self.access_count += 1


class CacheStatistics(BaseModel):
    """Cache statistics model."""

    hits: int = Field(default=0)
    misses: int = Field(default=0)
    sets: int = Field(default=0)
    deletes: int = Field(default=0)
    evictions: int = Field(default=0)
    expirations: int = Field(default=0)
    current_size: int = Field(default=0)
    total_bytes: int = Field(default=0)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 100 - self.hit_rate


class MemoryCache:
    """
    In-memory cache implementation.

    Provides fast, thread-safe caching with configurable eviction policies.
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
    ) -> None:
        """
        Initialize MemoryCache.

        Args:
            config: Cache configuration
        """
        self._config = config or CacheConfig()
        self._cache: dict[str, CacheEntry] = {}
        self._stats = CacheStatistics()
        self._lock = asyncio.Lock()

        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"MemoryCache initialized (max_size={self._config.max_size}, "
            f"policy={self._config.policy.value})"
        )

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    @property
    def statistics(self) -> CacheStatistics:
        """Get cache statistics."""
        self._stats.current_size = self.size
        return self._stats

    async def start(self) -> None:
        """Start the cache cleanup task."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name="cache_cleanup"
        )

    async def stop(self) -> None:
        """Stop the cache cleanup task."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired entries."""
        while self._running:
            try:
                await asyncio.sleep(self._config.cleanup_interval_seconds)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")

    async def _cleanup_expired(self) -> int:
        """Remove expired entries."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]

            for key in expired_keys:
                del self._cache[key]
                self._stats.expirations += 1

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return default

            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.expirations += 1
                return default

            entry.touch()
            self._stats.hits += 1

            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds

        Returns:
            True if set successfully
        """
        async with self._lock:
            if self.size >= self._config.max_size:
                await self._evict()

            ttl = ttl_seconds or self._config.default_ttl_seconds
            expires_at = now_utc() + timedelta(seconds=ttl) if ttl > 0 else None

            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 0

            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                size_bytes=size_bytes,
            )

            self._cache[key] = entry
            self._stats.sets += 1
            self._stats.total_bytes += size_bytes

            return True

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._stats.total_bytes -= entry.size_bytes
                del self._cache[key]
                self._stats.deletes += 1
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._cache[key]
                return False
            return True

    async def clear(self) -> int:
        """Clear all entries from cache."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.total_bytes = 0
            logger.info(f"Cleared {count} cache entries")
            return count

    async def _evict(self) -> None:
        """Evict entries based on policy."""
        if not self._cache:
            return

        if self._config.policy == CachePolicy.LRU:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].last_accessed
            )
        elif self._config.policy == CachePolicy.LFU:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].access_count
            )
        elif self._config.policy == CachePolicy.FIFO:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].created_at
            )
        else:
            oldest_key = next(iter(self._cache))

        entry = self._cache[oldest_key]
        self._stats.total_bytes -= entry.size_bytes
        del self._cache[oldest_key]
        self._stats.evictions += 1

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from cache."""
        results = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value
        return results

    async def set_many(
        self,
        items: dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Set multiple values in cache."""
        for key, value in items.items():
            await self.set(key, value, ttl_seconds)
        return True

    async def delete_many(self, keys: list[str]) -> int:
        """Delete multiple values from cache."""
        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1
        return count

    async def keys(self, pattern: Optional[str] = None) -> list[str]:
        """Get all keys matching pattern."""
        async with self._lock:
            if pattern is None:
                return list(self._cache.keys())

            import fnmatch
            return [
                key for key in self._cache.keys()
                if fnmatch.fnmatch(key, pattern)
            ]

    def get_entry(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry with metadata."""
        return self._cache.get(key)

    def __repr__(self) -> str:
        """String representation."""
        return f"MemoryCache(size={self.size}, max={self._config.max_size})"


@singleton
class DataCache:
    """
    High-level data cache for trading bot.

    Provides specialized caching for market data with
    automatic key generation and type-safe retrieval.
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
    ) -> None:
        """
        Initialize DataCache.

        Args:
            config: Cache configuration
        """
        self._config = config or CacheConfig()
        self._memory_cache = MemoryCache(config)
        self._redis_client = None

        logger.info("DataCache initialized")

    @property
    def backend(self) -> CacheBackend:
        """Get cache backend type."""
        return self._config.backend

    async def start(self) -> None:
        """Start the cache."""
        await self._memory_cache.start()

        if self._config.backend in (CacheBackend.REDIS, CacheBackend.HYBRID):
            await self._connect_redis()

    async def stop(self) -> None:
        """Stop the cache."""
        await self._memory_cache.stop()

        if self._redis_client:
            await self._redis_client.close()

    async def _connect_redis(self) -> None:
        """Connect to Redis."""
        if not self._config.redis_url:
            logger.warning("Redis URL not configured")
            return

        try:
            import redis.asyncio as redis

            self._redis_client = redis.from_url(
                self._config.redis_url,
                max_connections=self._config.redis_pool_size,
            )

            await self._redis_client.ping()
            logger.info("Connected to Redis")

        except ImportError:
            logger.warning("redis package not installed")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")

    def _make_key(self, prefix: str, *args: Any) -> str:
        """Generate cache key."""
        parts = [prefix] + [str(arg) for arg in args]
        return ":".join(parts)

    def _hash_key(self, key: str) -> str:
        """Hash a long key."""
        if len(key) <= 200:
            return key
        return hashlib.md5(key.encode()).hexdigest()

    async def get_quote(self, symbol: str) -> Optional[Any]:
        """Get cached quote."""
        key = self._make_key("quote", symbol)
        return await self._memory_cache.get(key)

    async def set_quote(
        self,
        symbol: str,
        quote: Any,
        ttl_seconds: int = 5,
    ) -> bool:
        """Cache a quote."""
        key = self._make_key("quote", symbol)
        return await self._memory_cache.set(key, quote, ttl_seconds)

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> Optional[list]:
        """Get cached bars."""
        key = self._make_key("bars", symbol, timeframe, limit)
        return await self._memory_cache.get(key)

    async def set_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        bars: list,
        ttl_seconds: int = 60,
    ) -> bool:
        """Cache bars."""
        key = self._make_key("bars", symbol, timeframe, limit)
        return await self._memory_cache.set(key, bars, ttl_seconds)

    async def get_indicator(
        self,
        symbol: str,
        indicator: str,
        params: dict,
    ) -> Optional[Any]:
        """Get cached indicator value."""
        params_str = safe_json_dumps(params)
        key = self._make_key("indicator", symbol, indicator, params_str)
        key = self._hash_key(key)
        return await self._memory_cache.get(key)

    async def set_indicator(
        self,
        symbol: str,
        indicator: str,
        params: dict,
        value: Any,
        ttl_seconds: int = 60,
    ) -> bool:
        """Cache indicator value."""
        params_str = safe_json_dumps(params)
        key = self._make_key("indicator", symbol, indicator, params_str)
        key = self._hash_key(key)
        return await self._memory_cache.set(key, value, ttl_seconds)

    async def get_analysis(
        self,
        symbol: str,
        analysis_type: str,
    ) -> Optional[Any]:
        """Get cached analysis."""
        key = self._make_key("analysis", symbol, analysis_type)
        return await self._memory_cache.get(key)

    async def set_analysis(
        self,
        symbol: str,
        analysis_type: str,
        analysis: Any,
        ttl_seconds: int = 300,
    ) -> bool:
        """Cache analysis."""
        key = self._make_key("analysis", symbol, analysis_type)
        return await self._memory_cache.set(key, analysis, ttl_seconds)

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        return await self._memory_cache.get(key, default)

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Set value in cache."""
        return await self._memory_cache.set(key, value, ttl_seconds)

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return await self._memory_cache.delete(key)

    async def clear(self) -> int:
        """Clear all cache entries."""
        return await self._memory_cache.clear()

    async def clear_symbol(self, symbol: str) -> int:
        """Clear all cache entries for a symbol."""
        keys = await self._memory_cache.keys(f"*:{symbol}:*")
        return await self._memory_cache.delete_many(keys)

    def get_statistics(self) -> dict:
        """Get cache statistics."""
        stats = self._memory_cache.statistics
        return {
            "backend": self._config.backend.value,
            "size": stats.current_size,
            "max_size": self._config.max_size,
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": stats.hit_rate,
            "sets": stats.sets,
            "deletes": stats.deletes,
            "evictions": stats.evictions,
            "total_bytes": stats.total_bytes,
            "redis_connected": self._redis_client is not None,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"DataCache(backend={self._config.backend.value})"

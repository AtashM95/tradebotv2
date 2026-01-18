"""
UI Cache Module for Ultimate Trading Bot v2.2.

This module provides caching functionality including:
- In-memory caching
- Cache invalidation
- Decorator-based caching
- Fragment caching
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    value: Any
    created_at: float
    expires_at: float
    hits: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() > self.expires_at

    @property
    def ttl_remaining(self) -> float:
        """Get remaining TTL in seconds."""
        remaining = self.expires_at - time.time()
        return max(0, remaining)


class MemoryCache:
    """
    Simple in-memory cache implementation.

    Thread-safe with TTL support and tag-based invalidation.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 300,
        cleanup_interval: int = 60,
    ) -> None:
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
            cleanup_interval: Cleanup interval in seconds
        """
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
        }

        logger.info(f"MemoryCache initialized (max_size={max_size}, ttl={default_ttl})")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        self._maybe_cleanup()

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return default

            if entry.is_expired:
                del self._cache[key]
                self._stats["misses"] += 1
                return default

            entry.hits += 1
            self._stats["hits"] += 1
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
            tags: Optional tags for invalidation
        """
        self._maybe_cleanup()

        ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()

        entry = CacheEntry(
            value=value,
            created_at=now,
            expires_at=now + ttl,
            tags=tags or [],
        )

        with self._lock:
            # Check size limit
            if len(self._cache) >= self._max_size:
                self._evict_oldest()

            self._cache[key] = entry
            self._stats["sets"] += 1

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats["deletes"] += 1
                return True
            return False

    def delete_by_tag(self, tag: str) -> int:
        """
        Delete all entries with a specific tag.

        Args:
            tag: Tag to match

        Returns:
            Number of entries deleted
        """
        deleted = 0
        with self._lock:
            keys_to_delete = [
                key for key, entry in self._cache.items()
                if tag in entry.tags
            ]

            for key in keys_to_delete:
                del self._cache[key]
                deleted += 1
                self._stats["deletes"] += 1

        if deleted > 0:
            logger.debug(f"Deleted {deleted} entries with tag '{tag}'")

        return deleted

    def delete_by_prefix(self, prefix: str) -> int:
        """
        Delete all entries with a key prefix.

        Args:
            prefix: Key prefix to match

        Returns:
            Number of entries deleted
        """
        deleted = 0
        with self._lock:
            keys_to_delete = [
                key for key in self._cache.keys()
                if key.startswith(prefix)
            ]

            for key in keys_to_delete:
                del self._cache[key]
                deleted += 1
                self._stats["deletes"] += 1

        return deleted

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._cache[key]
                return False
            return True

    def get_or_set(
        self,
        key: str,
        default_func: Callable[[], Any],
        ttl: int | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        """
        Get value or set from function if not found.

        Args:
            key: Cache key
            default_func: Function to generate value
            ttl: Optional TTL
            tags: Optional tags

        Returns:
            Cached or generated value
        """
        value = self.get(key)
        if value is not None:
            return value

        value = default_func()
        self.set(key, value, ttl, tags)
        return value

    def _evict_oldest(self) -> None:
        """Evict oldest entry to make room."""
        if not self._cache:
            return

        # Find oldest entry
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at,
        )

        del self._cache[oldest_key]
        self._stats["evictions"] += 1

    def _maybe_cleanup(self) -> None:
        """Run cleanup if needed."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        with self._lock:
            self._last_cleanup = now
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0

            return {
                **self._stats,
                "size": len(self._cache),
                "max_size": self._max_size,
                "hit_rate": round(hit_rate, 4),
            }


# Global cache instance
_cache: MemoryCache | None = None


def get_cache() -> MemoryCache:
    """Get or create global cache instance."""
    global _cache
    if _cache is None:
        _cache = MemoryCache()
    return _cache


def cached(
    ttl: int = 300,
    key_prefix: str = "",
    tags: list[str] | None = None,
    key_builder: Callable[..., str] | None = None,
) -> Callable:
    """
    Decorator for caching function results.

    Args:
        ttl: Cache TTL in seconds
        key_prefix: Key prefix
        tags: Cache tags
        key_builder: Custom key builder function

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_cache()

            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key from function name and arguments
                key_parts = [
                    key_prefix or func.__module__,
                    func.__name__,
                    str(args),
                    str(sorted(kwargs.items())),
                ]
                key_hash = hashlib.md5(
                    ":".join(key_parts).encode()
                ).hexdigest()[:16]
                cache_key = f"{key_prefix or func.__name__}:{key_hash}"

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl, tags)

            return result

        return wrapper

    return decorator


def invalidate_cache(*tags: str) -> None:
    """
    Invalidate cache entries by tags.

    Args:
        *tags: Tags to invalidate
    """
    cache = get_cache()
    for tag in tags:
        cache.delete_by_tag(tag)


class FragmentCache:
    """
    Cache for template fragments.

    Used for caching rendered HTML fragments.
    """

    def __init__(self, cache: MemoryCache | None = None) -> None:
        """Initialize fragment cache."""
        self._cache = cache or get_cache()

    def get(self, name: str, vary_on: dict[str, Any] | None = None) -> str | None:
        """
        Get cached fragment.

        Args:
            name: Fragment name
            vary_on: Variables to vary cache on

        Returns:
            Cached HTML or None
        """
        key = self._build_key(name, vary_on)
        return self._cache.get(key)

    def set(
        self,
        name: str,
        content: str,
        ttl: int = 300,
        vary_on: dict[str, Any] | None = None,
    ) -> None:
        """
        Cache a fragment.

        Args:
            name: Fragment name
            content: HTML content
            ttl: Cache TTL
            vary_on: Variables to vary cache on
        """
        key = self._build_key(name, vary_on)
        self._cache.set(key, content, ttl, tags=["fragment", f"fragment:{name}"])

    def invalidate(self, name: str) -> None:
        """Invalidate all versions of a fragment."""
        self._cache.delete_by_tag(f"fragment:{name}")

    def _build_key(self, name: str, vary_on: dict[str, Any] | None) -> str:
        """Build cache key for fragment."""
        if vary_on:
            vary_hash = hashlib.md5(
                str(sorted(vary_on.items())).encode()
            ).hexdigest()[:8]
            return f"fragment:{name}:{vary_hash}"
        return f"fragment:{name}"


def create_cache(**kwargs: Any) -> MemoryCache:
    """
    Create a new cache instance.

    Args:
        **kwargs: Cache configuration

    Returns:
        MemoryCache instance
    """
    return MemoryCache(**kwargs)


def create_fragment_cache(cache: MemoryCache | None = None) -> FragmentCache:
    """
    Create a fragment cache instance.

    Args:
        cache: Optional underlying cache

    Returns:
        FragmentCache instance
    """
    return FragmentCache(cache)


# Module version
__version__ = "2.2.0"

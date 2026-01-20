
from datetime import datetime, timedelta
from typing import Any, Optional


class CacheManager:
    """Simple in-memory cache with TTL support."""

    def __init__(self) -> None:
        self.cache = {}
        self.expiry = {}

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a cache value.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional)
        """
        self.cache[key] = value
        if ttl is not None:
            self.expiry[key] = datetime.utcnow() + timedelta(seconds=ttl)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a cache value.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        # Check if key exists
        if key not in self.cache:
            return default

        # Check if expired
        if key in self.expiry:
            if datetime.utcnow() > self.expiry[key]:
                # Remove expired entry
                del self.cache[key]
                del self.expiry[key]
                return default

        return self.cache.get(key, default)

    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        self.expiry.clear()

    def remove(self, key: str) -> None:
        """Remove a specific key from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.expiry:
            del self.expiry[key]

    def size(self) -> int:
        """Get number of cached items."""
        return len(self.cache)

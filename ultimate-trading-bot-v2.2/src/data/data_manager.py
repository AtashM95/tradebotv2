"""
Data Manager Module for Ultimate Trading Bot v2.2.

This module provides centralized data management, coordinating
multiple data providers and handling data aggregation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.data.base_provider import (
    BaseDataProvider,
    DataProviderType,
    DataProviderStatus,
    TimeFrame,
    Quote,
    Bar,
    Trade,
)
from src.utils.exceptions import DataFetchError, ValidationError
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc
from src.utils.decorators import singleton


logger = logging.getLogger(__name__)


class DataPriority(str, Enum):
    """Data source priority enumeration."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    FALLBACK = "fallback"


class DataManagerConfig(BaseModel):
    """Configuration for data manager."""

    enable_fallback: bool = Field(default=True)
    fallback_timeout_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    cache_quotes_seconds: int = Field(default=5, ge=1, le=60)
    cache_bars_seconds: int = Field(default=60, ge=10, le=3600)
    max_concurrent_requests: int = Field(default=10, ge=1, le=50)
    aggregate_from_multiple: bool = Field(default=False)
    validate_data: bool = Field(default=True)


class DataSourceConfig(BaseModel):
    """Configuration for a data source."""

    source_id: str = Field(default_factory=generate_uuid)
    provider_type: DataProviderType
    priority: DataPriority = Field(default=DataPriority.PRIMARY)
    enabled: bool = Field(default=True)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    supported_timeframes: list[TimeFrame] = Field(default_factory=list)
    symbols_filter: list[str] = Field(default_factory=list)


class DataSubscription(BaseModel):
    """Data subscription for real-time updates."""

    subscription_id: str = Field(default_factory=generate_uuid)
    symbol: str
    data_type: str = Field(default="quote")
    callback_id: str
    created_at: datetime = Field(default_factory=now_utc)


class AggregatedQuote(BaseModel):
    """Aggregated quote from multiple sources."""

    symbol: str
    sources: list[str] = Field(default_factory=list)
    bid_price: float = Field(default=0.0)
    ask_price: float = Field(default=0.0)
    last_price: float = Field(default=0.0)
    volume: int = Field(default=0)
    spread: float = Field(default=0.0)
    confidence: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=now_utc)


@singleton
class DataManager:
    """
    Centralized data management for the trading bot.

    This class provides:
    - Multi-provider data access
    - Automatic fallback handling
    - Data caching and aggregation
    - Real-time subscriptions
    """

    def __init__(
        self,
        config: Optional[DataManagerConfig] = None,
    ) -> None:
        """
        Initialize DataManager.

        Args:
            config: Data manager configuration
        """
        self._config = config or DataManagerConfig()

        self._providers: dict[str, BaseDataProvider] = {}
        self._source_configs: dict[str, DataSourceConfig] = {}

        self._quote_cache: dict[str, tuple[Quote, datetime]] = {}
        self._bar_cache: dict[str, tuple[list[Bar], datetime]] = {}

        self._subscriptions: dict[str, DataSubscription] = {}
        self._callbacks: dict[str, Callable] = {}

        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_requests)
        self._lock = asyncio.Lock()

        self._request_count = 0
        self._error_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info("DataManager initialized")

    @property
    def providers(self) -> dict[str, BaseDataProvider]:
        """Get registered providers."""
        return self._providers.copy()

    @property
    def provider_count(self) -> int:
        """Get number of providers."""
        return len(self._providers)

    def register_provider(
        self,
        provider: BaseDataProvider,
        priority: DataPriority = DataPriority.PRIMARY,
        weight: float = 1.0,
        supported_timeframes: Optional[list[TimeFrame]] = None,
        symbols_filter: Optional[list[str]] = None,
    ) -> str:
        """
        Register a data provider.

        Args:
            provider: Data provider instance
            priority: Provider priority
            weight: Weight for aggregation
            supported_timeframes: Supported timeframes
            symbols_filter: Symbols this provider handles

        Returns:
            Source ID
        """
        source_config = DataSourceConfig(
            provider_type=provider.provider_type,
            priority=priority,
            weight=weight,
            supported_timeframes=supported_timeframes or list(TimeFrame),
            symbols_filter=symbols_filter or [],
        )

        source_id = source_config.source_id
        self._providers[source_id] = provider
        self._source_configs[source_id] = source_config

        logger.info(
            f"Registered provider {provider.provider_type.value} "
            f"(id={source_id}, priority={priority.value})"
        )

        return source_id

    def unregister_provider(self, source_id: str) -> bool:
        """
        Unregister a data provider.

        Args:
            source_id: Source to unregister

        Returns:
            True if unregistered
        """
        if source_id not in self._providers:
            return False

        del self._providers[source_id]
        del self._source_configs[source_id]

        logger.info(f"Unregistered provider: {source_id}")
        return True

    async def connect_all(self) -> dict[str, bool]:
        """
        Connect all registered providers.

        Returns:
            Dictionary of source_id to connection status
        """
        results = {}

        for source_id, provider in self._providers.items():
            try:
                success = await provider.connect()
                results[source_id] = success
                logger.info(
                    f"Provider {provider.provider_type.value} "
                    f"{'connected' if success else 'failed to connect'}"
                )
            except Exception as e:
                results[source_id] = False
                logger.error(f"Error connecting provider {source_id}: {e}")

        return results

    async def disconnect_all(self) -> None:
        """Disconnect all providers."""
        for source_id, provider in self._providers.items():
            try:
                await provider.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting provider {source_id}: {e}")

        logger.info("All providers disconnected")

    def _get_providers_for_symbol(
        self,
        symbol: str,
        timeframe: Optional[TimeFrame] = None,
    ) -> list[tuple[str, BaseDataProvider]]:
        """Get providers that can handle a symbol, sorted by priority."""
        eligible = []

        for source_id, provider in self._providers.items():
            config = self._source_configs[source_id]

            if not config.enabled:
                continue

            if not provider.is_connected:
                continue

            if config.symbols_filter and symbol not in config.symbols_filter:
                continue

            if timeframe and config.supported_timeframes:
                if timeframe not in config.supported_timeframes:
                    continue

            eligible.append((source_id, provider, config))

        eligible.sort(key=lambda x: (
            0 if x[2].priority == DataPriority.PRIMARY else
            1 if x[2].priority == DataPriority.SECONDARY else 2,
            -x[2].weight
        ))

        return [(sid, p) for sid, p, _ in eligible]

    async def get_quote(
        self,
        symbol: str,
        use_cache: bool = True,
    ) -> Optional[Quote]:
        """
        Get quote for a symbol.

        Args:
            symbol: Trading symbol
            use_cache: Whether to use cached data

        Returns:
            Quote data or None
        """
        if use_cache:
            cached = self._get_cached_quote(symbol)
            if cached:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1

        providers = self._get_providers_for_symbol(symbol)

        if not providers:
            logger.warning(f"No providers available for {symbol}")
            return None

        async with self._semaphore:
            self._request_count += 1

            for source_id, provider in providers:
                try:
                    quote = await asyncio.wait_for(
                        provider.get_quote(symbol),
                        timeout=self._config.fallback_timeout_seconds,
                    )

                    if quote:
                        self._set_cached_quote(symbol, quote)
                        return quote

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout getting quote from {provider.provider_type.value}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error getting quote from {provider.provider_type.value}: {e}"
                    )

                if not self._config.enable_fallback:
                    break

            self._error_count += 1
            return None

    async def get_quotes(
        self,
        symbols: list[str],
        use_cache: bool = True,
    ) -> dict[str, Optional[Quote]]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of symbols
            use_cache: Whether to use cached data

        Returns:
            Dictionary of symbol to quote
        """
        tasks = [self.get_quote(symbol, use_cache) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                output[symbol] = None
            else:
                output[symbol] = result

        return output

    async def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
        use_cache: bool = True,
    ) -> list[Bar]:
        """
        Get historical bar data.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum bars to return
            use_cache: Whether to use cached data

        Returns:
            List of bars
        """
        cache_key = f"{symbol}:{timeframe.value}:{limit}"

        if use_cache:
            cached = self._get_cached_bars(cache_key)
            if cached:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1

        providers = self._get_providers_for_symbol(symbol, timeframe)

        if not providers:
            logger.warning(f"No providers available for {symbol} {timeframe.value}")
            return []

        async with self._semaphore:
            self._request_count += 1

            for source_id, provider in providers:
                try:
                    bars = await asyncio.wait_for(
                        provider.get_bars(symbol, timeframe, start, end, limit),
                        timeout=self._config.fallback_timeout_seconds * 2,
                    )

                    if bars:
                        if self._config.validate_data:
                            bars = self._validate_bars(bars)

                        self._set_cached_bars(cache_key, bars)
                        return bars

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout getting bars from {provider.provider_type.value}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error getting bars from {provider.provider_type.value}: {e}"
                    )

                if not self._config.enable_fallback:
                    break

            self._error_count += 1
            return []

    async def get_bars_df(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get historical bar data as DataFrame.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum bars to return

        Returns:
            DataFrame with OHLCV data
        """
        bars = await self.get_bars(symbol, timeframe, start, end, limit)

        if not bars:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        data = [
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        return df

    async def get_multi_bars(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
    ) -> dict[str, list[Bar]]:
        """
        Get bars for multiple symbols.

        Args:
            symbols: List of symbols
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum bars per symbol

        Returns:
            Dictionary of symbol to bars
        """
        tasks = [
            self.get_bars(symbol, timeframe, start, end, limit)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                output[symbol] = []
            else:
                output[symbol] = result

        return output

    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest price or None
        """
        quote = await self.get_quote(symbol)
        if quote:
            return quote.last_price
        return None

    async def get_latest_prices(
        self,
        symbols: list[str]
    ) -> dict[str, Optional[float]]:
        """
        Get latest prices for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary of symbol to price
        """
        quotes = await self.get_quotes(symbols)
        return {
            symbol: quote.last_price if quote else None
            for symbol, quote in quotes.items()
        }

    def _get_cached_quote(self, symbol: str) -> Optional[Quote]:
        """Get cached quote if valid."""
        if symbol not in self._quote_cache:
            return None

        quote, cached_at = self._quote_cache[symbol]
        age = (now_utc() - cached_at).total_seconds()

        if age > self._config.cache_quotes_seconds:
            del self._quote_cache[symbol]
            return None

        return quote

    def _set_cached_quote(self, symbol: str, quote: Quote) -> None:
        """Set cached quote."""
        self._quote_cache[symbol] = (quote, now_utc())

    def _get_cached_bars(self, key: str) -> Optional[list[Bar]]:
        """Get cached bars if valid."""
        if key not in self._bar_cache:
            return None

        bars, cached_at = self._bar_cache[key]
        age = (now_utc() - cached_at).total_seconds()

        if age > self._config.cache_bars_seconds:
            del self._bar_cache[key]
            return None

        return bars

    def _set_cached_bars(self, key: str, bars: list[Bar]) -> None:
        """Set cached bars."""
        self._bar_cache[key] = (bars, now_utc())

    def _validate_bars(self, bars: list[Bar]) -> list[Bar]:
        """Validate and clean bar data."""
        valid_bars = []

        for bar in bars:
            if bar.open <= 0 or bar.high <= 0 or bar.low <= 0 or bar.close <= 0:
                continue

            if bar.high < bar.low:
                continue

            if bar.high < bar.open or bar.high < bar.close:
                continue

            if bar.low > bar.open or bar.low > bar.close:
                continue

            valid_bars.append(bar)

        return valid_bars

    def subscribe(
        self,
        symbol: str,
        callback: Callable[[Quote], None],
        data_type: str = "quote",
    ) -> str:
        """
        Subscribe to real-time data updates.

        Args:
            symbol: Trading symbol
            callback: Callback function for updates
            data_type: Type of data to subscribe to

        Returns:
            Subscription ID
        """
        callback_id = generate_uuid()
        self._callbacks[callback_id] = callback

        subscription = DataSubscription(
            symbol=symbol,
            data_type=data_type,
            callback_id=callback_id,
        )

        self._subscriptions[subscription.subscription_id] = subscription

        logger.debug(f"Created subscription for {symbol}")
        return subscription.subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from data updates.

        Args:
            subscription_id: Subscription to cancel

        Returns:
            True if unsubscribed
        """
        if subscription_id not in self._subscriptions:
            return False

        subscription = self._subscriptions[subscription_id]

        if subscription.callback_id in self._callbacks:
            del self._callbacks[subscription.callback_id]

        del self._subscriptions[subscription_id]

        logger.debug(f"Removed subscription: {subscription_id}")
        return True

    def clear_cache(self) -> int:
        """Clear all caches."""
        quote_count = len(self._quote_cache)
        bar_count = len(self._bar_cache)

        self._quote_cache.clear()
        self._bar_cache.clear()

        total = quote_count + bar_count
        logger.info(f"Cleared {total} cached items")
        return total

    def get_statistics(self) -> dict:
        """Get data manager statistics."""
        provider_stats = {}
        for source_id, provider in self._providers.items():
            provider_stats[source_id] = {
                "type": provider.provider_type.value,
                "status": provider.status.value,
                "connected": provider.is_connected,
            }

        total_requests = self._cache_hits + self._cache_misses

        return {
            "provider_count": len(self._providers),
            "providers": provider_stats,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._request_count * 100
                if self._request_count > 0 else 0
            ),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": (
                self._cache_hits / total_requests * 100
                if total_requests > 0 else 0
            ),
            "quote_cache_size": len(self._quote_cache),
            "bar_cache_size": len(self._bar_cache),
            "subscription_count": len(self._subscriptions),
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all providers."""
        results = {}

        for source_id, provider in self._providers.items():
            try:
                health = await provider.health_check()
                results[source_id] = health
            except Exception as e:
                results[source_id] = {
                    "status": "error",
                    "error": str(e),
                }

        return results

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DataManager(providers={len(self._providers)}, "
            f"requests={self._request_count})"
        )

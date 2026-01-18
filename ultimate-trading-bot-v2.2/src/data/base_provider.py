"""
Base Data Provider Module for Ultimate Trading Bot v2.2.

This module provides the abstract base class for all data providers,
defining the common interface for market data access.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, date
from enum import Enum
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.utils.exceptions import DataFetchError, DataParseError
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc
from src.utils.decorators import async_retry


logger = logging.getLogger(__name__)


class DataProviderType(str, Enum):
    """Data provider type enumeration."""

    ALPACA = "alpaca"
    YAHOO = "yahoo"
    POLYGON = "polygon"
    IEX = "iex"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    TIINGO = "tiingo"
    QUANDL = "quandl"
    CUSTOM = "custom"


class TimeFrame(str, Enum):
    """Timeframe enumeration for OHLCV data."""

    TICK = "tick"
    SECOND_1 = "1s"
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1hour"
    HOUR_4 = "4hour"
    DAY_1 = "1day"
    WEEK_1 = "1week"
    MONTH_1 = "1month"


class DataProviderStatus(str, Enum):
    """Data provider status enumeration."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    INITIALIZING = "initializing"


class DataProviderConfig(BaseModel):
    """Base configuration for data providers."""

    provider_type: DataProviderType
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None

    timeout_seconds: int = Field(default=30, ge=5, le=120)
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=10.0)

    rate_limit_calls: int = Field(default=100, ge=1)
    rate_limit_period_seconds: int = Field(default=60, ge=1)

    enable_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=60, ge=1, le=3600)

    metadata: dict = Field(default_factory=dict)


class DataRequest(BaseModel):
    """Data request model."""

    request_id: str = Field(default_factory=generate_uuid)
    symbol: str
    timeframe: TimeFrame = Field(default=TimeFrame.DAY_1)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=10000)
    include_extended_hours: bool = Field(default=False)
    adjust: bool = Field(default=True)
    created_at: datetime = Field(default_factory=now_utc)


class DataResponse(BaseModel):
    """Data response model."""

    request_id: str
    symbol: str
    provider: DataProviderType
    success: bool = Field(default=False)
    error_message: Optional[str] = None
    data_points: int = Field(default=0)
    latency_ms: float = Field(default=0.0)
    from_cache: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=now_utc)


class Quote(BaseModel):
    """Real-time quote data model."""

    symbol: str
    bid_price: float = Field(default=0.0)
    bid_size: int = Field(default=0)
    ask_price: float = Field(default=0.0)
    ask_size: int = Field(default=0)
    last_price: float = Field(default=0.0)
    last_size: int = Field(default=0)
    volume: int = Field(default=0)
    timestamp: datetime = Field(default_factory=now_utc)
    exchange: Optional[str] = None

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2
        return self.last_price

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if self.bid_price > 0 and self.ask_price > 0:
            return self.ask_price - self.bid_price
        return 0.0

    @property
    def spread_percent(self) -> float:
        """Calculate spread as percentage."""
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 100
        return 0.0


class Bar(BaseModel):
    """OHLCV bar data model."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = Field(default=0)
    vwap: Optional[float] = None
    trade_count: Optional[int] = None

    @property
    def range(self) -> float:
        """Calculate price range."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Calculate candle body size."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """Check if bar is bullish."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if bar is bearish."""
        return self.close < self.open

    @property
    def upper_wick(self) -> float:
        """Calculate upper wick size."""
        body_high = max(self.open, self.close)
        return self.high - body_high

    @property
    def lower_wick(self) -> float:
        """Calculate lower wick size."""
        body_low = min(self.open, self.close)
        return body_low - self.low


class Trade(BaseModel):
    """Trade tick data model."""

    symbol: str
    price: float
    size: int
    timestamp: datetime
    exchange: Optional[str] = None
    conditions: Optional[list[str]] = None
    trade_id: Optional[str] = None


class BaseDataProvider(ABC):
    """
    Abstract base class for data providers.

    This class defines the interface that all data providers must implement.
    """

    def __init__(
        self,
        config: DataProviderConfig,
    ) -> None:
        """
        Initialize BaseDataProvider.

        Args:
            config: Provider configuration
        """
        self._config = config
        self._status = DataProviderStatus.INITIALIZING
        self._last_request_time: Optional[datetime] = None
        self._request_count = 0
        self._error_count = 0

        self._cache: dict[str, tuple[Any, datetime]] = {}

        self._lock = asyncio.Lock()

        logger.info(f"Initializing {config.provider_type.value} data provider")

    @property
    def provider_type(self) -> DataProviderType:
        """Get provider type."""
        return self._config.provider_type

    @property
    def status(self) -> DataProviderStatus:
        """Get provider status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected."""
        return self._status == DataProviderStatus.CONNECTED

    @property
    def config(self) -> DataProviderConfig:
        """Get provider configuration."""
        return self._config

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the data provider.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data provider."""
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Quote data or None
        """
        pass

    @abstractmethod
    async def get_quotes(self, symbols: list[str]) -> dict[str, Optional[Quote]]:
        """
        Get real-time quotes for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary of symbol to quote
        """
        pass

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Bar]:
        """
        Get historical bar data.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum bars to return

        Returns:
            List of bars
        """
        pass

    @abstractmethod
    async def get_trades(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[Trade]:
        """
        Get trade tick data.

        Args:
            symbol: Trading symbol
            start: Start datetime
            end: End datetime
            limit: Maximum trades to return

        Returns:
            List of trades
        """
        pass

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
                "vwap": bar.vwap,
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
        Get historical bars for multiple symbols.

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
                logger.error(f"Error fetching bars for {symbol}: {result}")
                output[symbol] = []
            else:
                output[symbol] = result

        return output

    async def get_latest_bar(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.DAY_1,
    ) -> Optional[Bar]:
        """
        Get the latest bar for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe

        Returns:
            Latest bar or None
        """
        bars = await self.get_bars(symbol, timeframe, limit=1)
        return bars[0] if bars else None

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

        bar = await self.get_latest_bar(symbol)
        if bar:
            return bar.close

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

    def _get_cache_key(
        self,
        method: str,
        symbol: str,
        **kwargs
    ) -> str:
        """Generate cache key."""
        key_parts = [method, symbol]
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}={v}")
        return ":".join(key_parts)

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if valid."""
        if not self._config.enable_caching:
            return None

        if key not in self._cache:
            return None

        data, cached_at = self._cache[key]
        age = (now_utc() - cached_at).total_seconds()

        if age > self._config.cache_ttl_seconds:
            del self._cache[key]
            return None

        return data

    def _set_cached(self, key: str, data: Any) -> None:
        """Set cached data."""
        if self._config.enable_caching:
            self._cache[key] = (data, now_utc())

    def clear_cache(self) -> int:
        """Clear the cache."""
        count = len(self._cache)
        self._cache.clear()
        return count

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health check results
        """
        try:
            quote = await self.get_quote("AAPL")
            is_healthy = quote is not None

            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "provider": self.provider_type.value,
                "connected": self.is_connected,
                "request_count": self._request_count,
                "error_count": self._error_count,
                "cache_size": len(self._cache),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_type.value,
                "error": str(e),
            }

    def get_statistics(self) -> dict[str, Any]:
        """Get provider statistics."""
        return {
            "provider": self.provider_type.value,
            "status": self._status.value,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": (
                (self._error_count / self._request_count * 100)
                if self._request_count > 0 else 0
            ),
            "cache_size": len(self._cache),
            "last_request": (
                self._last_request_time.isoformat()
                if self._last_request_time else None
            ),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"type={self.provider_type.value}, "
            f"status={self._status.value})"
        )

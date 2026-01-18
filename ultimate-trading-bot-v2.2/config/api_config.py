"""
API Configuration Module for Ultimate Trading Bot v2.2.

This module provides configuration for all external APIs including
Alpaca, data providers, and other third-party services.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from functools import lru_cache
import logging

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


logger = logging.getLogger(__name__)


class APIEnvironment(str, Enum):
    """API environment enumeration."""
    LIVE = "live"
    PAPER = "paper"
    SANDBOX = "sandbox"


class DataFeed(str, Enum):
    """Data feed provider enumeration."""
    ALPACA = "alpaca"
    YAHOO = "yahoo"
    POLYGON = "polygon"
    IEX = "iex"
    QUANDL = "quandl"


class AlpacaEndpoints(BaseModel):
    """Alpaca API endpoints configuration."""

    # Trading API endpoints
    trading_base: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Trading API base URL"
    )
    trading_v2: str = Field(
        default="/v2",
        description="Trading API v2 path"
    )

    # Data API endpoints
    data_base: str = Field(
        default="https://data.alpaca.markets",
        description="Data API base URL"
    )
    data_v2: str = Field(
        default="/v2",
        description="Data API v2 path"
    )

    # Streaming endpoints
    stream_trading: str = Field(
        default="wss://paper-api.alpaca.markets/stream",
        description="Trading stream WebSocket URL"
    )
    stream_data: str = Field(
        default="wss://stream.data.alpaca.markets/v2/iex",
        description="Data stream WebSocket URL"
    )
    stream_news: str = Field(
        default="wss://stream.data.alpaca.markets/v1beta1/news",
        description="News stream WebSocket URL"
    )

    # Account endpoints
    account: str = Field(default="/v2/account", description="Account endpoint")
    positions: str = Field(default="/v2/positions", description="Positions endpoint")
    orders: str = Field(default="/v2/orders", description="Orders endpoint")
    assets: str = Field(default="/v2/assets", description="Assets endpoint")
    clock: str = Field(default="/v2/clock", description="Clock endpoint")
    calendar: str = Field(default="/v2/calendar", description="Calendar endpoint")
    activities: str = Field(default="/v2/account/activities", description="Activities endpoint")
    portfolio_history: str = Field(default="/v2/account/portfolio/history", description="Portfolio history")
    watchlists: str = Field(default="/v2/watchlists", description="Watchlists endpoint")

    # Data endpoints
    bars: str = Field(default="/v2/stocks/{symbol}/bars", description="Bars endpoint")
    trades: str = Field(default="/v2/stocks/{symbol}/trades", description="Trades endpoint")
    quotes: str = Field(default="/v2/stocks/{symbol}/quotes", description="Quotes endpoint")
    snapshots: str = Field(default="/v2/stocks/snapshots", description="Snapshots endpoint")
    news: str = Field(default="/v1beta1/news", description="News endpoint")

    def set_environment(self, is_paper: bool = True) -> None:
        """
        Set endpoints based on trading environment.

        Args:
            is_paper: Whether to use paper trading endpoints
        """
        if is_paper:
            self.trading_base = "https://paper-api.alpaca.markets"
            self.stream_trading = "wss://paper-api.alpaca.markets/stream"
        else:
            self.trading_base = "https://api.alpaca.markets"
            self.stream_trading = "wss://api.alpaca.markets/stream"


class AlpacaAPIConfig(BaseModel):
    """Alpaca API configuration."""

    api_key: SecretStr = Field(
        default=SecretStr(os.getenv("ALPACA_API_KEY", "")),
        description="Alpaca API key"
    )
    api_secret: SecretStr = Field(
        default=SecretStr(os.getenv("ALPACA_API_SECRET", "")),
        description="Alpaca API secret"
    )
    environment: APIEnvironment = Field(
        default=APIEnvironment.PAPER,
        description="Trading environment"
    )
    endpoints: AlpacaEndpoints = Field(
        default_factory=AlpacaEndpoints,
        description="API endpoints"
    )

    # Request settings
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retries")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Retry delay")
    backoff_factor: float = Field(default=2.0, ge=1.0, le=5.0, description="Retry backoff factor")

    # Rate limiting
    rate_limit_requests: int = Field(default=200, ge=1, description="Requests per minute")
    rate_limit_data: int = Field(default=200, ge=1, description="Data requests per minute")

    # Data settings
    data_feed: str = Field(default="iex", description="Data feed (iex or sip)")
    use_fractional_shares: bool = Field(default=True, description="Allow fractional shares")

    @model_validator(mode='after')
    def update_endpoints_for_environment(self) -> 'AlpacaAPIConfig':
        """Update endpoints based on environment."""
        is_paper = self.environment != APIEnvironment.LIVE
        self.endpoints.set_environment(is_paper)
        return self

    @property
    def is_paper(self) -> bool:
        """Check if using paper trading."""
        return self.environment == APIEnvironment.PAPER

    @property
    def is_configured(self) -> bool:
        """Check if API credentials are configured."""
        return bool(
            self.api_key.get_secret_value() and
            self.api_secret.get_secret_value()
        )

    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests.

        Returns:
            Dictionary of authentication headers
        """
        return {
            "APCA-API-KEY-ID": self.api_key.get_secret_value(),
            "APCA-API-SECRET-KEY": self.api_secret.get_secret_value(),
        }

    def get_full_url(self, endpoint: str, base_type: str = "trading") -> str:
        """
        Get full URL for an endpoint.

        Args:
            endpoint: API endpoint path
            base_type: Type of base URL (trading, data)

        Returns:
            Full URL string
        """
        if base_type == "data":
            return f"{self.endpoints.data_base}{endpoint}"
        return f"{self.endpoints.trading_base}{endpoint}"


class YahooFinanceConfig(BaseModel):
    """Yahoo Finance configuration."""

    enabled: bool = Field(default=True, description="Enable Yahoo Finance")
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retries")
    rate_limit: int = Field(default=2000, ge=1, description="Requests per hour")
    proxy: Optional[str] = Field(default=None, description="Proxy URL")

    # Cache settings
    cache_ttl_quotes: int = Field(default=1, ge=0, description="Quote cache TTL seconds")
    cache_ttl_bars: int = Field(default=60, ge=0, description="Bar cache TTL seconds")
    cache_ttl_info: int = Field(default=3600, ge=0, description="Info cache TTL seconds")


class PolygonAPIConfig(BaseModel):
    """Polygon.io API configuration."""

    enabled: bool = Field(default=False, description="Enable Polygon API")
    api_key: SecretStr = Field(
        default=SecretStr(os.getenv("POLYGON_API_KEY", "")),
        description="Polygon API key"
    )
    base_url: str = Field(
        default="https://api.polygon.io",
        description="Base URL"
    )
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retries")

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key.get_secret_value())


class IEXCloudConfig(BaseModel):
    """IEX Cloud API configuration."""

    enabled: bool = Field(default=False, description="Enable IEX Cloud API")
    api_key: SecretStr = Field(
        default=SecretStr(os.getenv("IEX_API_KEY", "")),
        description="IEX API key"
    )
    base_url: str = Field(
        default="https://cloud.iexapis.com/stable",
        description="Base URL"
    )
    sandbox_url: str = Field(
        default="https://sandbox.iexapis.com/stable",
        description="Sandbox URL"
    )
    use_sandbox: bool = Field(default=True, description="Use sandbox environment")
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="Request timeout")

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key.get_secret_value())

    @property
    def active_url(self) -> str:
        """Get active base URL."""
        return self.sandbox_url if self.use_sandbox else self.base_url


class NewsAPIConfig(BaseModel):
    """News API configuration."""

    # Alpaca News
    alpaca_news_enabled: bool = Field(default=True, description="Enable Alpaca news")

    # NewsAPI.org
    newsapi_enabled: bool = Field(default=False, description="Enable NewsAPI")
    newsapi_key: SecretStr = Field(
        default=SecretStr(os.getenv("NEWS_API_KEY", "")),
        description="NewsAPI key"
    )
    newsapi_base_url: str = Field(
        default="https://newsapi.org/v2",
        description="NewsAPI base URL"
    )

    # Finnhub
    finnhub_enabled: bool = Field(default=False, description="Enable Finnhub news")
    finnhub_key: SecretStr = Field(
        default=SecretStr(os.getenv("FINNHUB_API_KEY", "")),
        description="Finnhub API key"
    )

    # Settings
    max_articles: int = Field(default=100, ge=1, le=1000, description="Max articles to fetch")
    lookback_hours: int = Field(default=24, ge=1, le=168, description="News lookback hours")
    cache_ttl: int = Field(default=300, ge=60, description="Cache TTL seconds")


class WebSocketConfig(BaseModel):
    """WebSocket configuration for streaming data."""

    # Connection settings
    ping_interval: int = Field(default=30, ge=5, le=120, description="Ping interval seconds")
    ping_timeout: int = Field(default=10, ge=1, le=60, description="Ping timeout seconds")
    close_timeout: int = Field(default=10, ge=1, le=60, description="Close timeout seconds")
    max_queue_size: int = Field(default=10000, ge=100, description="Max message queue size")

    # Reconnection settings
    auto_reconnect: bool = Field(default=True, description="Auto reconnect on disconnect")
    reconnect_delay: float = Field(default=1.0, ge=0.1, le=60.0, description="Initial reconnect delay")
    max_reconnect_delay: float = Field(default=60.0, ge=1.0, le=300.0, description="Max reconnect delay")
    reconnect_attempts: int = Field(default=10, ge=0, le=100, description="Max reconnect attempts")

    # Data processing
    process_timeout: float = Field(default=1.0, ge=0.1, le=10.0, description="Message process timeout")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Batch size for processing")


@dataclass
class RateLimiter:
    """Rate limiter configuration."""

    requests_per_minute: int = 60
    requests_per_second: float = field(init=False)
    burst_size: int = 10

    def __post_init__(self) -> None:
        """Calculate requests per second."""
        self.requests_per_second = self.requests_per_minute / 60.0


class APIConfig(BaseModel):
    """
    Master API configuration.

    Aggregates all API configurations for the trading bot.
    """

    # Broker APIs
    alpaca: AlpacaAPIConfig = Field(
        default_factory=AlpacaAPIConfig,
        description="Alpaca API configuration"
    )

    # Data provider APIs
    yahoo: YahooFinanceConfig = Field(
        default_factory=YahooFinanceConfig,
        description="Yahoo Finance configuration"
    )
    polygon: PolygonAPIConfig = Field(
        default_factory=PolygonAPIConfig,
        description="Polygon API configuration"
    )
    iex: IEXCloudConfig = Field(
        default_factory=IEXCloudConfig,
        description="IEX Cloud configuration"
    )

    # News APIs
    news: NewsAPIConfig = Field(
        default_factory=NewsAPIConfig,
        description="News API configuration"
    )

    # WebSocket settings
    websocket: WebSocketConfig = Field(
        default_factory=WebSocketConfig,
        description="WebSocket configuration"
    )

    # General settings
    default_data_source: DataFeed = Field(
        default=DataFeed.ALPACA,
        description="Default data source"
    )
    fallback_data_source: DataFeed = Field(
        default=DataFeed.YAHOO,
        description="Fallback data source"
    )
    enable_data_fallback: bool = Field(
        default=True,
        description="Enable fallback to alternate data source"
    )

    # User agent for requests
    user_agent: str = Field(
        default="UltimateTradingBot/2.2.0",
        description="User agent string"
    )

    def get_available_data_sources(self) -> List[DataFeed]:
        """
        Get list of available data sources.

        Returns:
            List of configured data sources
        """
        sources = []

        if self.alpaca.is_configured:
            sources.append(DataFeed.ALPACA)

        if self.yahoo.enabled:
            sources.append(DataFeed.YAHOO)

        if self.polygon.is_configured:
            sources.append(DataFeed.POLYGON)

        if self.iex.is_configured:
            sources.append(DataFeed.IEX)

        return sources

    def get_default_headers(self) -> Dict[str, str]:
        """
        Get default headers for API requests.

        Returns:
            Dictionary of default headers
        """
        return {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }


@lru_cache()
def get_api_config() -> APIConfig:
    """
    Get cached API configuration.

    Returns:
        Singleton APIConfig instance
    """
    return APIConfig()


def reload_api_config() -> APIConfig:
    """
    Reload API configuration.

    Returns:
        New APIConfig instance
    """
    get_api_config.cache_clear()
    return get_api_config()


# Module-level API config instance
api_config = get_api_config()


class EndpointBuilder:
    """Helper class for building API endpoint URLs."""

    def __init__(self, config: APIConfig) -> None:
        """
        Initialize endpoint builder.

        Args:
            config: API configuration
        """
        self.config = config

    def alpaca_trading(self, path: str) -> str:
        """
        Build Alpaca trading API URL.

        Args:
            path: API path

        Returns:
            Full URL
        """
        base = self.config.alpaca.endpoints.trading_base
        return f"{base}{path}"

    def alpaca_data(self, path: str) -> str:
        """
        Build Alpaca data API URL.

        Args:
            path: API path

        Returns:
            Full URL
        """
        base = self.config.alpaca.endpoints.data_base
        return f"{base}{path}"

    def alpaca_bars(self, symbol: str) -> str:
        """
        Build Alpaca bars endpoint URL.

        Args:
            symbol: Stock symbol

        Returns:
            Full URL
        """
        path = self.config.alpaca.endpoints.bars.format(symbol=symbol)
        return self.alpaca_data(path)

    def alpaca_quotes(self, symbol: str) -> str:
        """
        Build Alpaca quotes endpoint URL.

        Args:
            symbol: Stock symbol

        Returns:
            Full URL
        """
        path = self.config.alpaca.endpoints.quotes.format(symbol=symbol)
        return self.alpaca_data(path)

    def alpaca_trades(self, symbol: str) -> str:
        """
        Build Alpaca trades endpoint URL.

        Args:
            symbol: Stock symbol

        Returns:
            Full URL
        """
        path = self.config.alpaca.endpoints.trades.format(symbol=symbol)
        return self.alpaca_data(path)

    def polygon(self, path: str) -> str:
        """
        Build Polygon API URL.

        Args:
            path: API path

        Returns:
            Full URL
        """
        return f"{self.config.polygon.base_url}{path}"

    def iex(self, path: str) -> str:
        """
        Build IEX Cloud API URL.

        Args:
            path: API path

        Returns:
            Full URL
        """
        return f"{self.config.iex.active_url}{path}"

    def newsapi(self, path: str) -> str:
        """
        Build NewsAPI URL.

        Args:
            path: API path

        Returns:
            Full URL
        """
        return f"{self.config.news.newsapi_base_url}{path}"

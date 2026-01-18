"""
Data Package for Ultimate Trading Bot v2.2.

This package provides comprehensive market data functionality including:
- Multiple data providers (Alpaca, Yahoo Finance, Polygon)
- Real-time WebSocket streaming
- Historical data management
- Data caching and storage
- Data normalization and validation
- News data management
"""

from src.data.base_provider import (
    DataProviderType,
    TimeFrame,
    DataProviderStatus,
    DataProviderConfig,
    DataRequest,
    DataResponse,
    Quote,
    Bar,
    Trade,
    BaseDataProvider,
)

from src.data.alpaca_provider import (
    AlpacaDataConfig,
    AlpacaDataProvider,
)

from src.data.yahoo_provider import (
    YahooDataConfig,
    YahooDataProvider,
)

from src.data.data_manager import (
    DataPriority,
    DataManagerConfig,
    DataSourceConfig,
    DataSubscription,
    AggregatedQuote,
    DataManager,
)

from src.data.websocket_client import (
    WebSocketState,
    StreamType,
    WebSocketConfig,
    StreamSubscription,
    WebSocketClient,
    AlpacaWebSocketClient,
)

from src.data.data_cache import (
    CacheBackend,
    CachePolicy,
    CacheConfig,
    CacheEntry,
    CacheStatistics,
    MemoryCache,
    DataCache,
)

from src.data.data_storage import (
    StorageBackend,
    StorageConfig,
    DataStorage,
)

from src.data.historical_data import (
    DataQuality,
    HistoricalDataConfig,
    DataGap,
    DataCoverage,
    HistoricalDataManager,
)

from src.data.realtime_data import (
    TickDirection,
    RealtimeConfig,
    QuoteSnapshot,
    TickStatistics,
    SymbolData,
    RealtimeDataHandler,
)

from src.data.data_normalizer import (
    NormalizerConfig,
    DataNormalizer,
)

from src.data.news_data import (
    NewsSource,
    NewsSentiment,
    NewsArticle,
    NewsConfig,
    NewsDataManager,
)


__all__ = [
    # Base Provider
    "DataProviderType",
    "TimeFrame",
    "DataProviderStatus",
    "DataProviderConfig",
    "DataRequest",
    "DataResponse",
    "Quote",
    "Bar",
    "Trade",
    "BaseDataProvider",
    # Alpaca Provider
    "AlpacaDataConfig",
    "AlpacaDataProvider",
    # Yahoo Provider
    "YahooDataConfig",
    "YahooDataProvider",
    # Data Manager
    "DataPriority",
    "DataManagerConfig",
    "DataSourceConfig",
    "DataSubscription",
    "AggregatedQuote",
    "DataManager",
    # WebSocket
    "WebSocketState",
    "StreamType",
    "WebSocketConfig",
    "StreamSubscription",
    "WebSocketClient",
    "AlpacaWebSocketClient",
    # Cache
    "CacheBackend",
    "CachePolicy",
    "CacheConfig",
    "CacheEntry",
    "CacheStatistics",
    "MemoryCache",
    "DataCache",
    # Storage
    "StorageBackend",
    "StorageConfig",
    "DataStorage",
    # Historical Data
    "DataQuality",
    "HistoricalDataConfig",
    "DataGap",
    "DataCoverage",
    "HistoricalDataManager",
    # Realtime Data
    "TickDirection",
    "RealtimeConfig",
    "QuoteSnapshot",
    "TickStatistics",
    "SymbolData",
    "RealtimeDataHandler",
    # Normalizer
    "NormalizerConfig",
    "DataNormalizer",
    # News
    "NewsSource",
    "NewsSentiment",
    "NewsArticle",
    "NewsConfig",
    "NewsDataManager",
]

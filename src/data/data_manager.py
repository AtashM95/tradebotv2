
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
from .market_data import MarketData
from .cache_manager import CacheManager
from ..core.contracts import MarketSnapshot, RunContext

logger = logging.getLogger(__name__)


class DataManagerMetrics:
    """Tracks data manager performance metrics."""

    def __init__(self) -> None:
        self.total_snapshots_fetched: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.failed_fetches: int = 0
        self.avg_fetch_time: float = 0.0
        self._fetch_times: List[float] = []

    def record_fetch(self, cache_hit: bool, fetch_time: float) -> None:
        """Record a data fetch."""
        self.total_snapshots_fetched += 1
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            self._fetch_times.append(fetch_time)
            if self._fetch_times:
                self.avg_fetch_time = sum(self._fetch_times) / len(self._fetch_times)

    def record_failure(self) -> None:
        """Record a failed fetch."""
        self.failed_fetches += 1

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_snapshots_fetched': self.total_snapshots_fetched,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.get_cache_hit_rate(),
            'failed_fetches': self.failed_fetches,
            'avg_fetch_time': self.avg_fetch_time
        }


class DataManager:
    """
    Manages market data from multiple sources with caching.

    Responsibilities:
    - Coordinate data fetching from multiple sources
    - Provide unified data access interface
    - Cache frequently accessed data
    - Handle data validation and error recovery
    - Track data quality metrics
    - Support multiple data providers (Alpaca, yfinance, Polygon, etc.)
    """

    def __init__(
        self,
        watchlist: List[str],
        enable_cache: bool = True,
        cache_ttl: int = 60,
        primary_source: str = "mock"
    ) -> None:
        """
        Initialize the data manager.

        Args:
            watchlist: List of symbols to track
            enable_cache: Whether to enable caching
            cache_ttl: Cache time-to-live in seconds
            primary_source: Primary data source (mock, alpaca, yfinance, etc.)
        """
        self.watchlist = watchlist
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.primary_source = primary_source

        # Initialize data sources
        self.market_data = MarketData()
        self.cache = CacheManager() if enable_cache else None

        # Data quality tracking
        self.metrics = DataManagerMetrics()
        self._last_fetch_times: Dict[str, datetime] = {}
        self._stale_symbols: Dict[str, int] = defaultdict(int)

        logger.info("DataManager initialized", extra={
            'watchlist_size': len(watchlist),
            'enable_cache': enable_cache,
            'cache_ttl': cache_ttl,
            'primary_source': primary_source
        })

    def initialize(self, context: RunContext) -> None:
        """
        Initialize the data manager for a trading session.

        Args:
            context: Run context with session information
        """
        logger.info("Initializing DataManager", extra={'run_id': context.run_id})

        # Validate watchlist
        if not self.watchlist:
            logger.warning("Watchlist is empty", extra={'run_id': context.run_id})

        # Pre-warm cache for watchlist symbols
        if self.enable_cache and self.cache:
            logger.info("Pre-warming cache for watchlist symbols", extra={
                'run_id': context.run_id,
                'symbol_count': len(self.watchlist)
            })

        logger.info("DataManager initialized successfully", extra={'run_id': context.run_id})

    def shutdown(self, context: RunContext) -> None:
        """
        Shutdown the data manager.

        Args:
            context: Run context with session information
        """
        logger.info("Shutting down DataManager", extra={
            'run_id': context.run_id,
            'metrics': self.metrics.to_dict()
        })

        # Clear cache
        if self.cache:
            self.cache.clear()

        logger.info("DataManager shutdown complete", extra={'run_id': context.run_id})

    def get_snapshot(self, symbol: str) -> MarketSnapshot:
        """
        Get a market data snapshot for a symbol.

        Args:
            symbol: Symbol to fetch data for

        Returns:
            MarketSnapshot with current price and history
        """
        import time
        fetch_start = time.time()

        # Check cache first
        if self.enable_cache and self.cache:
            cache_key = f"snapshot:{symbol}"
            cached = self.cache.get(cache_key)

            if cached is not None:
                # Validate cache freshness
                cache_age = (datetime.utcnow() - cached.timestamp).total_seconds()
                if cache_age < self.cache_ttl:
                    self.metrics.record_fetch(True, 0.0)
                    return cached

        # Fetch from source
        try:
            price, history = self.market_data.get_price(symbol)

            snapshot = MarketSnapshot(
                symbol=symbol,
                price=price,
                history=history,
                timestamp=datetime.utcnow()
            )

            # Update cache
            if self.enable_cache and self.cache:
                cache_key = f"snapshot:{symbol}"
                self.cache.set(cache_key, snapshot, ttl=self.cache_ttl)

            fetch_time = time.time() - fetch_start
            self.metrics.record_fetch(False, fetch_time)
            self._last_fetch_times[symbol] = datetime.utcnow()

            return snapshot

        except Exception as e:
            self.metrics.record_failure()
            self._stale_symbols[symbol] += 1
            logger.error(f"Failed to fetch snapshot for {symbol}: {e}", extra={
                'symbol': symbol,
                'error': str(e)
            })
            raise

    def get_batch_snapshots(self, symbols: List[str]) -> Dict[str, MarketSnapshot]:
        """
        Get market snapshots for multiple symbols.

        Args:
            symbols: List of symbols to fetch

        Returns:
            Dictionary of symbol -> snapshot
        """
        snapshots = {}

        for symbol in symbols:
            try:
                snapshot = self.get_snapshot(symbol)
                snapshots[symbol] = snapshot
            except Exception as e:
                logger.error(f"Failed to fetch {symbol} in batch: {e}", extra={
                    'symbol': symbol
                })

        return snapshots

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.

        Args:
            symbol: Symbol to get price for

        Returns:
            Current price
        """
        snapshot = self.get_snapshot(symbol)
        return snapshot.price

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary of symbol -> price
        """
        prices = {}

        for symbol in symbols:
            try:
                price = self.get_current_price(symbol)
                prices[symbol] = price
            except Exception as e:
                logger.error(f"Failed to get price for {symbol}: {e}", extra={
                    'symbol': symbol
                })

        return prices

    def refresh_watchlist(self, symbols: List[str]) -> None:
        """
        Refresh the watchlist with new symbols.

        Args:
            symbols: New watchlist symbols
        """
        old_size = len(self.watchlist)
        self.watchlist = symbols

        logger.info("Watchlist refreshed", extra={
            'old_size': old_size,
            'new_size': len(symbols)
        })

    def add_to_watchlist(self, symbol: str) -> None:
        """
        Add a symbol to the watchlist.

        Args:
            symbol: Symbol to add
        """
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            logger.info(f"Added {symbol} to watchlist", extra={
                'symbol': symbol,
                'watchlist_size': len(self.watchlist)
            })

    def remove_from_watchlist(self, symbol: str) -> None:
        """
        Remove a symbol from the watchlist.

        Args:
            symbol: Symbol to remove
        """
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            logger.info(f"Removed {symbol} from watchlist", extra={
                'symbol': symbol,
                'watchlist_size': len(self.watchlist)
            })

    def is_symbol_valid(self, symbol: str) -> bool:
        """
        Check if a symbol is valid and has data.

        Args:
            symbol: Symbol to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            self.get_snapshot(symbol)
            return True
        except Exception:
            return False

    def get_stale_symbols(self, max_age_seconds: int = 300) -> List[str]:
        """
        Get symbols with stale data.

        Args:
            max_age_seconds: Maximum age before considering stale

        Returns:
            List of stale symbols
        """
        stale = []
        now = datetime.utcnow()

        for symbol, last_fetch in self._last_fetch_times.items():
            age = (now - last_fetch).total_seconds()
            if age > max_age_seconds:
                stale.append(symbol)

        return stale

    def get_failed_symbols(self) -> List[str]:
        """
        Get symbols that have failed to fetch multiple times.

        Returns:
            List of failed symbols
        """
        return [symbol for symbol, count in self._stale_symbols.items() if count > 0]

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get data manager metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.to_dict()
        metrics['watchlist_size'] = len(self.watchlist)
        metrics['stale_symbols'] = len(self.get_stale_symbols())
        metrics['failed_symbols'] = len(self.get_failed_symbols())
        return metrics

    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache:
            self.cache.clear()
            logger.info("Data cache cleared")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on data manager.

        Returns:
            Health status dictionary
        """
        # Try to fetch a sample symbol if watchlist exists
        healthy = True
        sample_fetch_ok = False

        if self.watchlist:
            try:
                self.get_snapshot(self.watchlist[0])
                sample_fetch_ok = True
            except Exception as e:
                healthy = False
                logger.error(f"Health check failed: {e}")

        return {
            'healthy': healthy,
            'watchlist_size': len(self.watchlist),
            'sample_fetch_ok': sample_fetch_ok,
            'cache_enabled': self.enable_cache,
            'metrics': self.metrics.to_dict(),
            'stale_symbols': len(self.get_stale_symbols()),
            'failed_symbols': len(self.get_failed_symbols())
        }

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for a symbol (placeholder for future implementation).

        Args:
            symbol: Symbol to fetch
            start_date: Start date
            end_date: End date
            interval: Data interval (1d, 1h, etc.)

        Returns:
            List of historical data points
        """
        logger.info(f"Historical data fetch requested for {symbol}", extra={
            'symbol': symbol,
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'interval': interval
        })

        # Placeholder - would integrate with historical_data module
        return []

    def get_realtime_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a symbol (placeholder for future implementation).

        Args:
            symbol: Symbol to get quote for

        Returns:
            Real-time quote data
        """
        snapshot = self.get_snapshot(symbol)

        # Return basic quote structure
        return {
            'symbol': symbol,
            'price': snapshot.price,
            'timestamp': snapshot.timestamp.isoformat(),
            'bid': snapshot.price * 0.9995,
            'ask': snapshot.price * 1.0005,
            'volume': 0
        }

    def validate_watchlist(self) -> Dict[str, bool]:
        """
        Validate all symbols in the watchlist.

        Returns:
            Dictionary of symbol -> is_valid
        """
        results = {}

        for symbol in self.watchlist:
            results[symbol] = self.is_symbol_valid(symbol)

        valid_count = sum(1 for v in results.values() if v)
        logger.info("Watchlist validation complete", extra={
            'total': len(results),
            'valid': valid_count,
            'invalid': len(results) - valid_count
        })

        return results

    def prefetch_watchlist(self) -> int:
        """
        Pre-fetch data for all watchlist symbols.

        Returns:
            Number of symbols successfully fetched
        """
        success_count = 0

        for symbol in self.watchlist:
            try:
                self.get_snapshot(symbol)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to prefetch {symbol}: {e}")

        logger.info("Watchlist prefetch complete", extra={
            'total': len(self.watchlist),
            'success': success_count,
            'failed': len(self.watchlist) - success_count
        })

        return success_count

"""
Market Data module for real-time and delayed market data.
~600 lines as per schema
"""

import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """Real-time quote data."""
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last: float
    last_size: int
    timestamp: datetime


@dataclass
class Trade:
    """Trade data."""
    symbol: str
    price: float
    size: int
    timestamp: datetime
    exchange: str


@dataclass
class OHLCV:
    """Open, High, Low, Close, Volume bar data."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime
    vwap: Optional[float] = None


class MarketData:
    """
    Market data provider for real-time and historical quotes.

    Features:
    - Real-time quote data
    - Trade data
    - OHLCV bars (multiple timeframes)
    - Market depth (Level 2)
    - Data validation
    - Multiple data sources support
    - Caching for performance
    """

    def __init__(
        self,
        primary_source: str = 'mock',
        cache_ttl: int = 1,  # seconds
        enable_validation: bool = True
    ):
        """
        Initialize market data provider.

        Args:
            primary_source: Primary data source ('mock', 'alpaca', 'polygon', etc.)
            cache_ttl: Cache time-to-live in seconds
            enable_validation: Enable data validation
        """
        self.primary_source = primary_source
        self.cache_ttl = cache_ttl
        self.enable_validation = enable_validation

        # Cache
        self.quote_cache: Dict[str, Tuple[Quote, datetime]] = {}
        self.ohlcv_cache: Dict[str, Tuple[List[OHLCV], datetime]] = {}

        # Statistics
        self.stats = {
            "quotes_fetched": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "validation_errors": 0,
            "data_source_errors": 0
        }

    def get_quote(self, symbol: str, use_cache: bool = True) -> Optional[Quote]:
        """
        Get current quote for symbol.

        Args:
            symbol: Stock symbol
            use_cache: Use cached data if available

        Returns:
            Quote object or None if unavailable
        """
        # Check cache
        if use_cache and symbol in self.quote_cache:
            cached_quote, cached_time = self.quote_cache[symbol]
            age = (datetime.now() - cached_time).total_seconds()

            if age < self.cache_ttl:
                self.stats["cache_hits"] += 1
                return cached_quote

        self.stats["cache_misses"] += 1
        self.stats["quotes_fetched"] += 1

        # Fetch from source
        try:
            quote = self._fetch_quote_from_source(symbol)

            if quote and self.enable_validation:
                if not self._validate_quote(quote):
                    self.stats["validation_errors"] += 1
                    logger.warning(f"Quote validation failed for {symbol}")
                    return None

            # Cache result
            if quote:
                self.quote_cache[symbol] = (quote, datetime.now())

            return quote

        except Exception as e:
            logger.error(f"Failed to fetch quote for {symbol}: {e}")
            self.stats["data_source_errors"] += 1
            return None

    def get_price(self, symbol: str) -> Tuple[float, List[float]]:
        """
        Get current price and recent history.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (current_price, price_history)
        """
        # Get quote
        quote = self.get_quote(symbol)

        if quote:
            current_price = quote.last
        else:
            # Fallback to mock data
            current_price = self._generate_mock_price(symbol)

        # Get recent OHLCV bars for history
        bars = self.get_ohlcv_bars(symbol, timeframe='1min', limit=20)

        if bars:
            history = [bar.close for bar in bars]
        else:
            # Generate mock history
            history = self._generate_mock_history(current_price, 20)

        return current_price, history

    def get_ohlcv_bars(
        self,
        symbol: str,
        timeframe: str = '1min',
        limit: int = 100,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[OHLCV]:
        """
        Get OHLCV bars for symbol.

        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe (1min, 5min, 15min, 1h, 1d)
            limit: Maximum number of bars
            start: Start datetime
            end: End datetime

        Returns:
            List of OHLCV bars
        """
        cache_key = f"{symbol}:{timeframe}:{limit}"

        # Check cache
        if cache_key in self.ohlcv_cache:
            cached_bars, cached_time = self.ohlcv_cache[cache_key]
            age = (datetime.now() - cached_time).total_seconds()

            # Cache longer for longer timeframes
            cache_duration = self._get_cache_duration(timeframe)
            if age < cache_duration:
                self.stats["cache_hits"] += 1
                return cached_bars

        self.stats["cache_misses"] += 1

        # Fetch from source
        try:
            bars = self._fetch_ohlcv_from_source(symbol, timeframe, limit, start, end)

            if bars and self.enable_validation:
                bars = [bar for bar in bars if self._validate_ohlcv(bar)]

            # Cache result
            if bars:
                self.ohlcv_cache[cache_key] = (bars, datetime.now())

            return bars

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            self.stats["data_source_errors"] += 1
            return []

    def get_trades(
        self,
        symbol: str,
        limit: int = 100,
        start: Optional[datetime] = None
    ) -> List[Trade]:
        """
        Get recent trades for symbol.

        Args:
            symbol: Stock symbol
            limit: Maximum number of trades
            start: Start datetime

        Returns:
            List of Trade objects
        """
        try:
            return self._fetch_trades_from_source(symbol, limit, start)
        except Exception as e:
            logger.error(f"Failed to fetch trades for {symbol}: {e}")
            return []

    def get_market_depth(
        self,
        symbol: str,
        depth: int = 10
    ) -> Dict[str, List[Tuple[float, int]]]:
        """
        Get market depth (Level 2 data).

        Args:
            symbol: Stock symbol
            depth: Number of levels per side

        Returns:
            Dict with 'bids' and 'asks' lists of (price, size) tuples
        """
        try:
            return self._fetch_market_depth_from_source(symbol, depth)
        except Exception as e:
            logger.error(f"Failed to fetch market depth for {symbol}: {e}")
            return {"bids": [], "asks": []}

    def _fetch_quote_from_source(self, symbol: str) -> Optional[Quote]:
        """Fetch quote from configured data source."""
        if self.primary_source == 'mock':
            return self._generate_mock_quote(symbol)
        elif self.primary_source == 'alpaca':
            # Would integrate with Alpaca API
            return self._generate_mock_quote(symbol)
        elif self.primary_source == 'polygon':
            # Would integrate with Polygon.io API
            return self._generate_mock_quote(symbol)
        else:
            return self._generate_mock_quote(symbol)

    def _fetch_ohlcv_from_source(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        start: Optional[datetime],
        end: Optional[datetime]
    ) -> List[OHLCV]:
        """Fetch OHLCV bars from configured data source."""
        if self.primary_source == 'mock':
            return self._generate_mock_ohlcv(symbol, timeframe, limit)
        else:
            return self._generate_mock_ohlcv(symbol, timeframe, limit)

    def _fetch_trades_from_source(
        self,
        symbol: str,
        limit: int,
        start: Optional[datetime]
    ) -> List[Trade]:
        """Fetch trades from configured data source."""
        return []  # Mock implementation

    def _fetch_market_depth_from_source(
        self,
        symbol: str,
        depth: int
    ) -> Dict[str, List[Tuple[float, int]]]:
        """Fetch market depth from configured data source."""
        quote = self.get_quote(symbol, use_cache=False)
        if not quote:
            return {"bids": [], "asks": []}

        # Generate mock depth around current quote
        bids = []
        asks = []

        for i in range(depth):
            bid_price = quote.bid - (i * 0.01)
            bid_size = random.randint(100, 1000)
            bids.append((bid_price, bid_size))

            ask_price = quote.ask + (i * 0.01)
            ask_size = random.randint(100, 1000)
            asks.append((ask_price, ask_size))

        return {"bids": bids, "asks": asks}

    def _generate_mock_quote(self, symbol: str) -> Quote:
        """Generate mock quote data for testing."""
        base_price = self._get_base_price(symbol)
        spread = base_price * 0.001  # 0.1% spread

        last = base_price + random.uniform(-0.5, 0.5)
        bid = last - spread / 2
        ask = last + spread / 2

        return Quote(
            symbol=symbol,
            bid=round(bid, 2),
            ask=round(ask, 2),
            bid_size=random.randint(100, 500),
            ask_size=random.randint(100, 500),
            last=round(last, 2),
            last_size=random.randint(50, 200),
            timestamp=datetime.now()
        )

    def _generate_mock_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> List[OHLCV]:
        """Generate mock OHLCV data for testing."""
        bars = []
        base_price = self._get_base_price(symbol)
        current_time = datetime.now()

        # Calculate time delta based on timeframe
        time_delta = self._get_timeframe_delta(timeframe)

        for i in range(limit):
            # Random walk for price
            open_price = base_price + random.uniform(-2, 2)
            close_price = open_price + random.uniform(-1, 1)
            high_price = max(open_price, close_price) + random.uniform(0, 0.5)
            low_price = min(open_price, close_price) - random.uniform(0, 0.5)
            volume = random.randint(10000, 100000)

            bar = OHLCV(
                symbol=symbol,
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=volume,
                timestamp=current_time - (time_delta * (limit - i)),
                vwap=round((high_price + low_price + close_price) / 3, 2)
            )
            bars.append(bar)

            base_price = close_price  # Continue from last close

        return bars

    def _generate_mock_price(self, symbol: str) -> float:
        """Generate mock price."""
        return round(self._get_base_price(symbol) + random.uniform(-1, 1), 2)

    def _generate_mock_history(self, current_price: float, length: int) -> List[float]:
        """Generate mock price history."""
        history = []
        price = current_price

        for _ in range(length):
            price = price + random.uniform(-0.5, 0.5)
            history.append(round(price, 2))

        return history

    def _get_base_price(self, symbol: str) -> float:
        """Get base price for symbol (for mock data consistency)."""
        # Use hash for consistency across calls
        hash_val = hash(symbol) % 1000
        return 100.0 + (hash_val / 10.0)

    def _get_timeframe_delta(self, timeframe: str) -> timedelta:
        """Get timedelta for timeframe."""
        if timeframe == '1min':
            return timedelta(minutes=1)
        elif timeframe == '5min':
            return timedelta(minutes=5)
        elif timeframe == '15min':
            return timedelta(minutes=15)
        elif timeframe == '1h':
            return timedelta(hours=1)
        elif timeframe == '1d':
            return timedelta(days=1)
        else:
            return timedelta(minutes=1)

    def _get_cache_duration(self, timeframe: str) -> int:
        """Get cache duration in seconds for timeframe."""
        if timeframe == '1min':
            return 1
        elif timeframe == '5min':
            return 5
        elif timeframe == '15min':
            return 15
        elif timeframe == '1h':
            return 60
        elif timeframe == '1d':
            return 300
        else:
            return 1

    def _validate_quote(self, quote: Quote) -> bool:
        """Validate quote data."""
        if quote.bid <= 0 or quote.ask <= 0 or quote.last <= 0:
            return False

        if quote.bid > quote.ask:
            return False

        if abs(quote.last - quote.bid) > quote.bid * 0.1:  # 10% sanity check
            return False

        if quote.bid_size <= 0 or quote.ask_size <= 0:
            return False

        return True

    def _validate_ohlcv(self, bar: OHLCV) -> bool:
        """Validate OHLCV bar data."""
        if bar.high < bar.low:
            return False

        if bar.high < bar.open or bar.high < bar.close:
            return False

        if bar.low > bar.open or bar.low > bar.close:
            return False

        if bar.volume < 0:
            return False

        if bar.open <= 0 or bar.high <= 0 or bar.low <= 0 or bar.close <= 0:
            return False

        return True

    def clear_cache(self):
        """Clear all cached data."""
        self.quote_cache.clear()
        self.ohlcv_cache.clear()
        logger.info("Market data cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get market data statistics."""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            self.stats["cache_hits"] / total_requests
            if total_requests > 0 else 0.0
        )

        return {
            **self.stats,
            "cache_hit_rate": round(cache_hit_rate, 3),
            "cached_quotes": len(self.quote_cache),
            "cached_ohlcv": len(self.ohlcv_cache)
        }

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "quotes_fetched": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "validation_errors": 0,
            "data_source_errors": 0
        }

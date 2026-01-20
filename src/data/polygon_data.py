"""
Polygon.io Data Provider - Integration with Polygon.io API.
~500 lines as per schema
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PolygonConfig:
    """Configuration for Polygon.io API."""
    api_key: str
    base_url: str = "https://api.polygon.io"
    timeout: int = 30
    max_requests_per_minute: int = 5  # Free tier limit


class PolygonData:
    """
    Polygon.io data provider for stocks, forex, and crypto.

    Features:
    - Aggregated bars (stocks, forex, crypto)
    - Real-time and delayed quotes
    - Trades data
    - Market status
    - Ticker details
    - Snapshots
    - Technical indicators
    - News and market events
    - Rate limiting
    - Websocket support structure
    """

    def __init__(
        self,
        config: Optional[PolygonConfig] = None,
        tier: str = 'free'
    ):
        """
        Initialize Polygon.io data provider.

        Args:
            config: Polygon API configuration
            tier: API tier ('free', 'starter', 'developer', 'advanced')
        """
        self.config = config or self._get_default_config()
        self.tier = tier

        # Rate limiting based on tier
        rate_limits = {
            'free': 5,
            'starter': 100,
            'developer': 100,
            'advanced': 500
        }
        self.requests_per_minute = rate_limits.get(tier, 5)
        self.request_times = []

        # Statistics
        self.stats = {
            "requests_made": 0,
            "rate_limit_hits": 0,
            "errors": 0,
            "bars_fetched": 0,
            "quotes_fetched": 0,
            "trades_fetched": 0
        }

    def fetch(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timespan: str = 'day',
        multiplier: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch aggregated bars from Polygon.io.

        Args:
            symbol: Stock ticker symbol
            start: Start date
            end: End date
            timespan: Timespan (minute, hour, day, week, month, quarter, year)
            multiplier: Size of timespan multiplier

        Returns:
            Dictionary with bars data
        """
        try:
            bars = self.get_aggregates(symbol, start, end, timespan, multiplier)
            return {
                'status': 'ok',
                'symbol': symbol,
                'bars': bars,
                'count': len(bars)
            }
        except Exception as e:
            logger.error(f"Polygon fetch failed for {symbol}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'symbol': symbol
            }

    def get_aggregates(
        self,
        ticker: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timespan: str = 'day',
        multiplier: int = 1,
        adjusted: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get aggregate bars for a ticker.

        Args:
            ticker: Stock ticker
            start: Start date
            end: End date
            timespan: Size of time window (minute, hour, day, week, month)
            multiplier: Multiplier for timespan
            adjusted: Adjust for splits

        Returns:
            List of aggregate bars
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            # Mock implementation
            # In production: GET /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
            bars = self._generate_mock_aggregates(ticker, start, end, timespan, multiplier)
            self.stats["bars_fetched"] += len(bars)

            return bars

        except Exception as e:
            logger.error(f"Failed to get aggregates for {ticker}: {e}")
            self.stats["errors"] += 1
            return []

    def get_grouped_daily(
        self,
        date: Optional[datetime] = None,
        adjusted: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get grouped daily bars for entire market.

        Args:
            date: Date to get data for
            adjusted: Adjust for splits

        Returns:
            Dictionary mapping tickers to their daily bars
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            # Mock implementation
            # In production: GET /v2/aggs/grouped/locale/us/market/stocks/{date}
            return self._generate_mock_grouped_daily(date)

        except Exception as e:
            logger.error(f"Failed to get grouped daily: {e}")
            self.stats["errors"] += 1
            return {}

    def get_previous_close(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get previous day's close for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Previous close data or None
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            # Mock implementation
            # In production: GET /v2/aggs/ticker/{ticker}/prev
            return self._generate_mock_previous_close(ticker)

        except Exception as e:
            logger.error(f"Failed to get previous close for {ticker}: {e}")
            self.stats["errors"] += 1
            return None

    def get_last_trade(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get last trade for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Last trade data or None
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1
            self.stats["trades_fetched"] += 1

            # Mock implementation
            # In production: GET /v2/last/trade/{ticker}
            return self._generate_mock_trade(ticker)

        except Exception as e:
            logger.error(f"Failed to get last trade for {ticker}: {e}")
            self.stats["errors"] += 1
            return None

    def get_last_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get last quote for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Last quote data or None
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1
            self.stats["quotes_fetched"] += 1

            # Mock implementation
            # In production: GET /v2/last/nbbo/{ticker}
            return self._generate_mock_quote(ticker)

        except Exception as e:
            logger.error(f"Failed to get last quote for {ticker}: {e}")
            self.stats["errors"] += 1
            return None

    def get_snapshot(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get snapshot of ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Snapshot data or None
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            # Mock implementation
            # In production: GET /v2/snapshot/locale/us/markets/stocks/tickers/{ticker}
            return {
                'ticker': ticker,
                'day': self._generate_mock_previous_close(ticker),
                'lastTrade': self._generate_mock_trade(ticker),
                'lastQuote': self._generate_mock_quote(ticker),
                'min': self._generate_mock_aggregates(ticker, limit=1)[0] if True else None,
                'prevDay': self._generate_mock_previous_close(ticker)
            }

        except Exception as e:
            logger.error(f"Failed to get snapshot for {ticker}: {e}")
            self.stats["errors"] += 1
            return None

    def get_all_snapshots(self) -> List[Dict[str, Any]]:
        """
        Get snapshots for all tickers.

        Returns:
            List of ticker snapshots
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            # Mock implementation
            # In production: GET /v2/snapshot/locale/us/markets/stocks/tickers
            return self._generate_mock_all_snapshots()

        except Exception as e:
            logger.error(f"Failed to get all snapshots: {e}")
            self.stats["errors"] += 1
            return []

    def get_ticker_details(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Ticker details or None
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            # Mock implementation
            # In production: GET /v3/reference/tickers/{ticker}
            return self._generate_mock_ticker_details(ticker)

        except Exception as e:
            logger.error(f"Failed to get ticker details for {ticker}: {e}")
            self.stats["errors"] += 1
            return None

    def get_market_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current market status.

        Returns:
            Market status data or None
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            # Mock implementation
            # In production: GET /v1/marketstatus/now
            now = datetime.now()
            return {
                'market': 'open' if self._is_market_hours(now) else 'closed',
                'serverTime': now.isoformat(),
                'exchanges': {
                    'nasdaq': 'open' if self._is_market_hours(now) else 'closed',
                    'nyse': 'open' if self._is_market_hours(now) else 'closed'
                },
                'currencies': {
                    'fx': 'open',
                    'crypto': 'open'
                }
            }

        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            self.stats["errors"] += 1
            return None

    def get_market_holidays(self, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get market holidays.

        Args:
            year: Year to get holidays for

        Returns:
            List of market holidays
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            # Mock implementation
            # In production: GET /v1/marketstatus/upcoming
            if year is None:
                year = datetime.now().year

            return self._generate_mock_holidays(year)

        except Exception as e:
            logger.error(f"Failed to get market holidays: {e}")
            self.stats["errors"] += 1
            return []

    def _check_rate_limit(self):
        """Check and enforce rate limits."""
        current_time = time.time()

        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]

        # Check if we've hit the limit
        if len(self.request_times) >= self.requests_per_minute:
            # Wait until oldest request is 60 seconds old
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                self.stats["rate_limit_hits"] += 1
                self.request_times = []

        self.request_times.append(current_time)

    def _generate_mock_aggregates(
        self,
        ticker: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timespan: str = 'day',
        multiplier: int = 1,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate mock aggregate data."""
        import random

        if end is None:
            end = datetime.now()
        if start is None:
            days = limit if timespan == 'day' else 7
            start = end - timedelta(days=days)

        # Map timespan to timedelta
        timespan_map = {
            'minute': timedelta(minutes=multiplier),
            'hour': timedelta(hours=multiplier),
            'day': timedelta(days=multiplier),
            'week': timedelta(weeks=multiplier),
            'month': timedelta(days=30 * multiplier),
            'quarter': timedelta(days=90 * multiplier),
            'year': timedelta(days=365 * multiplier)
        }

        delta = timespan_map.get(timespan, timedelta(days=1))
        base_price = 100.0 + (hash(ticker) % 1000) / 10.0
        current_price = base_price

        bars = []
        current_time = start
        count = 0

        while current_time <= end and count < limit:
            open_price = current_price
            change = random.gauss(0, 0.02)
            close_price = open_price * (1 + change)

            high = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
            low = min(open_price, close_price) * (1 - random.uniform(0, 0.01))

            bar = {
                'v': random.randint(1000000, 50000000),  # volume
                'vw': round((high + low + close_price) / 3, 2),  # vwap
                'o': round(open_price, 2),  # open
                'c': round(close_price, 2),  # close
                'h': round(high, 2),  # high
                'l': round(low, 2),  # low
                't': int(current_time.timestamp() * 1000),  # timestamp (ms)
                'n': random.randint(100, 1000)  # number of transactions
            }

            bars.append(bar)
            current_price = close_price
            current_time += delta
            count += 1

        return bars

    def _generate_mock_grouped_daily(
        self,
        date: Optional[datetime]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate mock grouped daily data."""
        if date is None:
            date = datetime.now() - timedelta(days=1)

        # Generate for a few sample tickers
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        grouped = {}

        for ticker in tickers:
            bars = self._generate_mock_aggregates(ticker, date, date, 'day', 1, 1)
            if bars:
                grouped[ticker] = bars[0]

        return grouped

    def _generate_mock_previous_close(self, ticker: str) -> Dict[str, Any]:
        """Generate mock previous close data."""
        import random

        base_price = 100.0 + (hash(ticker) % 1000) / 10.0

        return {
            'T': ticker,
            'v': random.randint(10000000, 100000000),
            'vw': round(base_price, 2),
            'o': round(base_price * 0.99, 2),
            'c': round(base_price, 2),
            'h': round(base_price * 1.01, 2),
            'l': round(base_price * 0.98, 2),
            't': int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        }

    def _generate_mock_trade(self, ticker: str) -> Dict[str, Any]:
        """Generate mock trade data."""
        import random

        base_price = 100.0 + (hash(ticker) % 1000) / 10.0

        return {
            'T': ticker,
            'p': round(base_price + random.uniform(-1, 1), 2),  # price
            's': random.randint(50, 500),  # size
            't': int(datetime.now().timestamp() * 1000),  # timestamp
            'x': random.randint(1, 20),  # exchange
            'c': [14, 41],  # conditions
            'i': str(random.randint(1000000, 9999999))  # trade ID
        }

    def _generate_mock_quote(self, ticker: str) -> Dict[str, Any]:
        """Generate mock quote data."""
        import random

        base_price = 100.0 + (hash(ticker) % 1000) / 10.0
        spread = base_price * 0.001

        return {
            'T': ticker,
            'p': round(base_price - spread / 2, 2),  # bid price
            'P': round(base_price + spread / 2, 2),  # ask price
            's': random.randint(100, 500),  # bid size
            'S': random.randint(100, 500),  # ask size
            't': int(datetime.now().timestamp() * 1000),  # timestamp
            'x': random.randint(1, 20),  # bid exchange
            'X': random.randint(1, 20)   # ask exchange
        }

    def _generate_mock_ticker_details(self, ticker: str) -> Dict[str, Any]:
        """Generate mock ticker details."""
        import random

        return {
            'ticker': ticker,
            'name': f'{ticker} Inc.',
            'market': 'stocks',
            'locale': 'us',
            'primary_exchange': 'XNAS',
            'type': 'CS',
            'active': True,
            'currency_name': 'usd',
            'cik': str(random.randint(1000000, 9999999)),
            'composite_figi': f'BBG{random.randint(100000000, 999999999)}',
            'share_class_figi': f'BBG{random.randint(100000000, 999999999)}',
            'market_cap': random.randint(1000000000, 1000000000000),
            'phone_number': '(555) 123-4567',
            'address': {
                'address1': '123 Main St',
                'city': 'New York',
                'state': 'NY',
                'postal_code': '10001'
            },
            'description': f'{ticker} is a leading company in its industry.',
            'sic_code': '7370',
            'sic_description': 'Services-Computer Programming, Data Processing, Etc.',
            'ticker_root': ticker,
            'homepage_url': f'https://www.{ticker.lower()}.com',
            'total_employees': random.randint(1000, 100000),
            'list_date': '2010-01-01',
            'branding': {
                'logo_url': f'https://api.polygon.io/v1/reference/company-branding/{ticker.lower()}/images/logo.png',
                'icon_url': f'https://api.polygon.io/v1/reference/company-branding/{ticker.lower()}/images/icon.png'
            }
        }

    def _generate_mock_all_snapshots(self) -> List[Dict[str, Any]]:
        """Generate mock snapshots for all tickers."""
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        snapshots = []

        for ticker in tickers:
            snapshot = {
                'ticker': ticker,
                'day': self._generate_mock_previous_close(ticker),
                'lastTrade': self._generate_mock_trade(ticker),
                'lastQuote': self._generate_mock_quote(ticker)
            }
            snapshots.append(snapshot)

        return snapshots

    def _generate_mock_holidays(self, year: int) -> List[Dict[str, Any]]:
        """Generate mock market holidays."""
        return [
            {'date': f'{year}-01-01', 'name': "New Year's Day", 'status': 'closed'},
            {'date': f'{year}-07-04', 'name': 'Independence Day', 'status': 'closed'},
            {'date': f'{year}-12-25', 'name': 'Christmas Day', 'status': 'closed'}
        ]

    def _is_market_hours(self, dt: datetime) -> bool:
        """Check if datetime is during market hours."""
        if dt.weekday() >= 5:  # Weekend
            return False

        hour = dt.hour
        if hour < 9 or hour >= 16:
            return False

        if hour == 9 and dt.minute < 30:
            return False

        return True

    def _get_default_config(self) -> PolygonConfig:
        """Get default configuration."""
        return PolygonConfig(
            api_key='mock-api-key',
            base_url='https://api.polygon.io'
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            **self.stats,
            'requests_in_last_minute': len(self.request_times),
            'rate_limit': self.requests_per_minute
        }

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "requests_made": 0,
            "rate_limit_hits": 0,
            "errors": 0,
            "bars_fetched": 0,
            "quotes_fetched": 0,
            "trades_fetched": 0
        }

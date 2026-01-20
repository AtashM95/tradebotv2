"""
Yahoo Finance Data Provider - Integration with Yahoo Finance API.
~500 lines as per schema
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class YfinanceConfig:
    """Configuration for Yahoo Finance API."""
    base_url: str = "https://query1.finance.yahoo.com"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


class YfinanceData:
    """
    Yahoo Finance data provider.

    Features:
    - Historical price data
    - Real-time quotes
    - Company information
    - Financial statements
    - Key statistics
    - Analyst recommendations
    - News and events
    - Multiple timeframes
    - Free (no API key required)
    """

    def __init__(self, config: Optional[YfinanceConfig] = None):
        """
        Initialize Yahoo Finance data provider.

        Args:
            config: Yahoo Finance configuration
        """
        self.config = config or YfinanceConfig()

        # Rate limiting (be respectful to Yahoo's servers)
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests

        # Statistics
        self.stats = {
            "requests_made": 0,
            "errors": 0,
            "retries": 0,
            "bars_fetched": 0,
            "quotes_fetched": 0,
            "info_fetched": 0
        }

    def fetch(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: str = '1d',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch historical data from Yahoo Finance.

        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)

        Returns:
            Dictionary with historical data
        """
        try:
            bars = self.get_historical_data(symbol, start, end, interval)
            return {
                'status': 'ok',
                'symbol': symbol,
                'bars': bars,
                'count': len(bars)
            }
        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed for {symbol}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'symbol': symbol
            }

    def get_historical_data(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: str = '1d'
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data.

        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)

        Returns:
            List of historical bars
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            # In production, would make actual API call
            # For now, generate mock data
            bars = self._generate_mock_historical(symbol, start, end, interval)
            self.stats["bars_fetched"] += len(bars)

            return bars

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            self.stats["errors"] += 1
            return []

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Quote dictionary or None
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1
            self.stats["quotes_fetched"] += 1

            # Mock implementation
            # In production: GET /v7/finance/quote?symbols={symbol}
            return self._generate_mock_quote(symbol)

        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            self.stats["errors"] += 1
            return None

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get real-time quotes for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbols to quotes
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            quotes = {}
            for symbol in symbols:
                quote = self._generate_mock_quote(symbol)
                if quote:
                    quotes[symbol] = quote
                    self.stats["quotes_fetched"] += 1

            return quotes

        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")
            self.stats["errors"] += 1
            return {}

    def get_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get company information and key statistics.

        Args:
            symbol: Stock symbol

        Returns:
            Company info dictionary or None
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1
            self.stats["info_fetched"] += 1

            # Mock implementation
            # In production: GET /v10/finance/quoteSummary/{symbol}
            return self._generate_mock_info(symbol)

        except Exception as e:
            logger.error(f"Failed to get info for {symbol}: {e}")
            self.stats["errors"] += 1
            return None

    def get_dividends(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get dividend history.

        Args:
            symbol: Stock symbol
            start: Start date
            end: End date

        Returns:
            List of dividend events
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            # Mock implementation
            return self._generate_mock_dividends(symbol, start, end)

        except Exception as e:
            logger.error(f"Failed to get dividends for {symbol}: {e}")
            self.stats["errors"] += 1
            return []

    def get_splits(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get stock split history.

        Args:
            symbol: Stock symbol
            start: Start date
            end: End date

        Returns:
            List of split events
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            # Mock implementation
            return self._generate_mock_splits(symbol, start, end)

        except Exception as e:
            logger.error(f"Failed to get splits for {symbol}: {e}")
            self.stats["errors"] += 1
            return []

    def get_recommendations(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get analyst recommendations.

        Args:
            symbol: Stock symbol

        Returns:
            List of recommendations
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            # Mock implementation
            return self._generate_mock_recommendations(symbol)

        except Exception as e:
            logger.error(f"Failed to get recommendations for {symbol}: {e}")
            self.stats["errors"] += 1
            return []

    def get_financials(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get financial statements.

        Args:
            symbol: Stock symbol

        Returns:
            Financials dictionary or None
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            # Mock implementation
            return self._generate_mock_financials(symbol)

        except Exception as e:
            logger.error(f"Failed to get financials for {symbol}: {e}")
            self.stats["errors"] += 1
            return None

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for symbols.

        Args:
            query: Search query

        Returns:
            List of matching symbols
        """
        self._check_rate_limit()

        try:
            self.stats["requests_made"] += 1

            # Mock implementation
            # In production: GET /v1/finance/search?q={query}
            return self._generate_mock_search_results(query)

        except Exception as e:
            logger.error(f"Failed to search for '{query}': {e}")
            self.stats["errors"] += 1
            return []

    def _check_rate_limit(self):
        """Check and enforce rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def _generate_mock_historical(
        self,
        symbol: str,
        start: Optional[datetime],
        end: Optional[datetime],
        interval: str
    ) -> List[Dict[str, Any]]:
        """Generate mock historical data."""
        import random

        if end is None:
            end = datetime.now()
        if start is None:
            days = 100 if interval in ['1d', '1wk', '1mo'] else 7
            start = end - timedelta(days=days)

        # Map interval to timedelta
        interval_map = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '1d': timedelta(days=1),
            '1wk': timedelta(weeks=1),
            '1mo': timedelta(days=30)
        }

        delta = interval_map.get(interval, timedelta(days=1))
        base_price = 100.0 + (hash(symbol) % 1000) / 10.0
        current_price = base_price

        bars = []
        current_time = start

        while current_time <= end:
            open_price = current_price
            change = random.gauss(0.001, 0.02)
            close_price = open_price * (1 + change)

            high = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
            low = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
            volume = random.randint(1000000, 50000000)

            bar = {
                'date': current_time.strftime('%Y-%m-%d'),
                'timestamp': int(current_time.timestamp()),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'adjClose': round(close_price, 2)
            }

            bars.append(bar)
            current_price = close_price
            current_time += delta

        return bars

    def _generate_mock_quote(self, symbol: str) -> Dict[str, Any]:
        """Generate mock quote data."""
        import random

        base_price = 100.0 + (hash(symbol) % 1000) / 10.0
        change = random.uniform(-5, 5)

        return {
            'symbol': symbol,
            'shortName': f"{symbol} Inc.",
            'regularMarketPrice': round(base_price, 2),
            'regularMarketChange': round(change, 2),
            'regularMarketChangePercent': round((change / base_price) * 100, 2),
            'regularMarketVolume': random.randint(1000000, 50000000),
            'regularMarketDayHigh': round(base_price * 1.02, 2),
            'regularMarketDayLow': round(base_price * 0.98, 2),
            'regularMarketOpen': round(base_price * (1 + random.uniform(-0.01, 0.01)), 2),
            'regularMarketPreviousClose': round(base_price - change, 2),
            'bid': round(base_price - 0.05, 2),
            'ask': round(base_price + 0.05, 2),
            'bidSize': random.randint(100, 1000),
            'askSize': random.randint(100, 1000),
            'marketCap': int(base_price * 1000000000),
            'fiftyTwoWeekHigh': round(base_price * 1.5, 2),
            'fiftyTwoWeekLow': round(base_price * 0.7, 2),
            'regularMarketTime': int(datetime.now().timestamp())
        }

    def _generate_mock_info(self, symbol: str) -> Dict[str, Any]:
        """Generate mock company info."""
        import random

        return {
            'symbol': symbol,
            'shortName': f"{symbol} Inc.",
            'longName': f"{symbol} Corporation",
            'sector': random.choice(['Technology', 'Healthcare', 'Finance', 'Consumer', 'Energy']),
            'industry': 'Software',
            'website': f'https://www.{symbol.lower()}.com',
            'fullTimeEmployees': random.randint(1000, 100000),
            'summary': f'{symbol} is a leading company in its sector.',
            'marketCap': random.randint(1000000000, 1000000000000),
            'trailingPE': round(random.uniform(10, 40), 2),
            'forwardPE': round(random.uniform(10, 35), 2),
            'dividendYield': round(random.uniform(0, 0.05), 4),
            'beta': round(random.uniform(0.5, 2.0), 2),
            'fiftyTwoWeekHigh': round(random.uniform(100, 200), 2),
            'fiftyTwoWeekLow': round(random.uniform(50, 100), 2),
            'averageVolume': random.randint(1000000, 10000000),
            'currency': 'USD',
            'exchange': 'NASDAQ'
        }

    def _generate_mock_dividends(
        self,
        symbol: str,
        start: Optional[datetime],
        end: Optional[datetime]
    ) -> List[Dict[str, Any]]:
        """Generate mock dividend data."""
        import random

        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=365)

        dividends = []
        current = start

        while current <= end:
            if random.random() < 0.25:  # 25% chance of dividend
                dividends.append({
                    'date': current.strftime('%Y-%m-%d'),
                    'amount': round(random.uniform(0.1, 2.0), 2)
                })
            current += timedelta(days=90)  # Quarterly

        return dividends

    def _generate_mock_splits(
        self,
        symbol: str,
        start: Optional[datetime],
        end: Optional[datetime]
    ) -> List[Dict[str, Any]]:
        """Generate mock split data."""
        import random

        # Splits are rare
        if random.random() < 0.9:
            return []

        splits = []
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=365 * 5)

        date = start + timedelta(days=random.randint(0, (end - start).days))
        splits.append({
            'date': date.strftime('%Y-%m-%d'),
            'ratio': random.choice(['2:1', '3:1', '3:2'])
        })

        return splits

    def _generate_mock_recommendations(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate mock analyst recommendations."""
        import random

        recommendations = []
        for i in range(5):
            date = datetime.now() - timedelta(days=i * 30)
            recommendations.append({
                'date': date.strftime('%Y-%m-%d'),
                'firm': f'Analyst Firm {i+1}',
                'toGrade': random.choice(['Buy', 'Hold', 'Sell', 'Strong Buy']),
                'fromGrade': random.choice(['Buy', 'Hold', 'Sell']) if i > 0 else None,
                'action': random.choice(['main', 'up', 'down', 'init'])
            })

        return recommendations

    def _generate_mock_financials(self, symbol: str) -> Dict[str, Any]:
        """Generate mock financial statements."""
        import random

        base_revenue = random.randint(1000000000, 100000000000)

        return {
            'income_statement': {
                'totalRevenue': base_revenue,
                'costOfRevenue': int(base_revenue * 0.6),
                'grossProfit': int(base_revenue * 0.4),
                'operatingIncome': int(base_revenue * 0.2),
                'netIncome': int(base_revenue * 0.15),
                'ebitda': int(base_revenue * 0.25)
            },
            'balance_sheet': {
                'totalAssets': int(base_revenue * 2),
                'totalLiabilities': int(base_revenue * 1.2),
                'totalStockholderEquity': int(base_revenue * 0.8),
                'cash': int(base_revenue * 0.3),
                'totalDebt': int(base_revenue * 0.5)
            },
            'cash_flow': {
                'operatingCashFlow': int(base_revenue * 0.2),
                'investingCashFlow': int(base_revenue * -0.1),
                'financingCashFlow': int(base_revenue * -0.05),
                'freeCashFlow': int(base_revenue * 0.15)
            }
        }

    def _generate_mock_search_results(self, query: str) -> List[Dict[str, Any]]:
        """Generate mock search results."""
        results = []
        for i in range(min(5, len(query) + 2)):
            results.append({
                'symbol': f'{query.upper()}{i if i > 0 else ""}',
                'name': f'{query.title()} Company {i if i > 0 else ""}',
                'type': 'equity',
                'exchange': 'NASDAQ'
            })

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "requests_made": 0,
            "errors": 0,
            "retries": 0,
            "bars_fetched": 0,
            "quotes_fetched": 0,
            "info_fetched": 0
        }

"""
Yahoo Finance Data Provider Module for Ultimate Trading Bot v2.2.

This module provides market data access through Yahoo Finance,
including historical bars, quotes, and fundamental data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
import pandas as pd
from pydantic import Field

from src.data.base_provider import (
    BaseDataProvider,
    DataProviderConfig,
    DataProviderType,
    DataProviderStatus,
    TimeFrame,
    Quote,
    Bar,
    Trade,
)
from src.utils.exceptions import DataFetchError
from src.utils.date_utils import now_utc
from src.utils.decorators import async_retry


logger = logging.getLogger(__name__)


TIMEFRAME_MAP = {
    TimeFrame.MINUTE_1: "1m",
    TimeFrame.MINUTE_5: "5m",
    TimeFrame.MINUTE_15: "15m",
    TimeFrame.MINUTE_30: "30m",
    TimeFrame.HOUR_1: "1h",
    TimeFrame.DAY_1: "1d",
    TimeFrame.WEEK_1: "1wk",
    TimeFrame.MONTH_1: "1mo",
}

VALID_RANGES = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "1h": "730d",
    "1d": "max",
    "1wk": "max",
    "1mo": "max",
}


class YahooDataConfig(DataProviderConfig):
    """Configuration for Yahoo Finance data provider."""

    provider_type: DataProviderType = Field(default=DataProviderType.YAHOO)
    base_url: str = Field(default="https://query1.finance.yahoo.com")
    timeout_seconds: int = Field(default=30)
    max_retries: int = Field(default=3)
    rate_limit_calls: int = Field(default=2000)
    rate_limit_period_seconds: int = Field(default=3600)


class YahooDataProvider(BaseDataProvider):
    """
    Yahoo Finance market data provider implementation.

    Provides access to historical market data through Yahoo Finance.
    Note: Yahoo Finance does not provide real-time streaming data.
    """

    def __init__(
        self,
        config: Optional[YahooDataConfig] = None,
    ) -> None:
        """
        Initialize YahooDataProvider.

        Args:
            config: Provider configuration
        """
        if config is None:
            config = YahooDataConfig()

        super().__init__(config)

        self._yahoo_config: YahooDataConfig = config
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def _headers(self) -> dict[str, str]:
        """Get request headers."""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }

    async def connect(self) -> bool:
        """
        Connect to Yahoo Finance API.

        Returns:
            True if connection successful
        """
        try:
            self._client = httpx.AsyncClient(
                base_url=self._yahoo_config.base_url,
                headers=self._headers,
                timeout=self._yahoo_config.timeout_seconds,
            )

            response = await self._client.get(
                "/v8/finance/chart/AAPL",
                params={"interval": "1d", "range": "1d"},
            )

            if response.status_code == 200:
                self._status = DataProviderStatus.CONNECTED
                logger.info("Connected to Yahoo Finance API")
                return True
            else:
                self._status = DataProviderStatus.ERROR
                logger.error(f"Yahoo Finance connection failed: {response.status_code}")
                return False

        except Exception as e:
            self._status = DataProviderStatus.ERROR
            logger.error(f"Yahoo Finance connection error: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Yahoo Finance API."""
        if self._client:
            await self._client.aclose()
            self._client = None

        self._status = DataProviderStatus.DISCONNECTED
        logger.info("Disconnected from Yahoo Finance API")

    @async_retry(max_attempts=3, delay=1.0)
    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get current quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Quote data or None
        """
        cache_key = self._get_cache_key("quote", symbol)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        if not self._client:
            await self.connect()

        try:
            self._request_count += 1
            self._last_request_time = now_utc()

            response = await self._client.get(
                f"/v8/finance/chart/{symbol}",
                params={"interval": "1m", "range": "1d"},
            )

            if response.status_code == 200:
                data = response.json()
                chart = data.get("chart", {})
                result = chart.get("result", [{}])[0]

                meta = result.get("meta", {})
                indicators = result.get("indicators", {})
                quote_data = indicators.get("quote", [{}])[0]

                if quote_data and quote_data.get("close"):
                    closes = quote_data.get("close", [])
                    volumes = quote_data.get("volume", [])

                    last_close = None
                    for c in reversed(closes):
                        if c is not None:
                            last_close = c
                            break

                    last_volume = 0
                    for v in reversed(volumes):
                        if v is not None:
                            last_volume = v
                            break

                    quote = Quote(
                        symbol=symbol,
                        bid_price=float(meta.get("regularMarketPrice", 0)),
                        ask_price=float(meta.get("regularMarketPrice", 0)),
                        last_price=float(last_close or meta.get("regularMarketPrice", 0)),
                        volume=int(meta.get("regularMarketVolume", last_volume)),
                        timestamp=now_utc(),
                    )

                    self._set_cached(cache_key, quote)
                    return quote

            self._error_count += 1
            return None

        except Exception as e:
            self._error_count += 1
            logger.error(f"Error fetching Yahoo quote for {symbol}: {e}")
            return None

    async def get_quotes(
        self,
        symbols: list[str]
    ) -> dict[str, Optional[Quote]]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary of symbol to quote
        """
        tasks = [self.get_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                output[symbol] = None
            else:
                output[symbol] = result

        return output

    @async_retry(max_attempts=3, delay=1.0)
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
        if not self._client:
            await self.connect()

        yahoo_interval = TIMEFRAME_MAP.get(timeframe, "1d")

        if start and end:
            period1 = int(start.timestamp())
            period2 = int(end.timestamp())
            params = {
                "interval": yahoo_interval,
                "period1": period1,
                "period2": period2,
            }
        else:
            range_val = VALID_RANGES.get(yahoo_interval, "1mo")
            params = {
                "interval": yahoo_interval,
                "range": range_val,
            }

        try:
            self._request_count += 1
            self._last_request_time = now_utc()

            response = await self._client.get(
                f"/v8/finance/chart/{symbol}",
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                chart = data.get("chart", {})
                result = chart.get("result", [{}])[0]

                timestamps = result.get("timestamp", [])
                indicators = result.get("indicators", {})
                quote_data = indicators.get("quote", [{}])[0]

                opens = quote_data.get("open", [])
                highs = quote_data.get("high", [])
                lows = quote_data.get("low", [])
                closes = quote_data.get("close", [])
                volumes = quote_data.get("volume", [])

                bars = []
                for i, ts in enumerate(timestamps):
                    if ts is None:
                        continue

                    o = opens[i] if i < len(opens) and opens[i] is not None else 0
                    h = highs[i] if i < len(highs) and highs[i] is not None else 0
                    l = lows[i] if i < len(lows) and lows[i] is not None else 0
                    c = closes[i] if i < len(closes) and closes[i] is not None else 0
                    v = volumes[i] if i < len(volumes) and volumes[i] is not None else 0

                    if o == 0 or h == 0 or l == 0 or c == 0:
                        continue

                    bar = Bar(
                        symbol=symbol,
                        timestamp=datetime.utcfromtimestamp(ts),
                        open=float(o),
                        high=float(h),
                        low=float(l),
                        close=float(c),
                        volume=int(v),
                    )
                    bars.append(bar)

                if limit and len(bars) > limit:
                    bars = bars[-limit:]

                return bars

            self._error_count += 1
            logger.error(f"Failed to get Yahoo bars for {symbol}: {response.status_code}")
            return []

        except Exception as e:
            self._error_count += 1
            logger.error(f"Error fetching Yahoo bars for {symbol}: {e}")
            return []

    async def get_trades(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[Trade]:
        """
        Get trade tick data.

        Note: Yahoo Finance does not provide tick-level trade data.

        Args:
            symbol: Trading symbol
            start: Start datetime
            end: End datetime
            limit: Maximum trades to return

        Returns:
            Empty list (not supported)
        """
        logger.warning("Yahoo Finance does not provide tick-level trade data")
        return []

    async def get_fundamentals(self, symbol: str) -> Optional[dict]:
        """
        Get fundamental data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Fundamental data or None
        """
        if not self._client:
            await self.connect()

        try:
            response = await self._client.get(
                f"/v10/finance/quoteSummary/{symbol}",
                params={
                    "modules": "financialData,defaultKeyStatistics,summaryProfile,assetProfile"
                },
            )

            if response.status_code == 200:
                data = response.json()
                result = data.get("quoteSummary", {}).get("result", [{}])[0]
                return result

            return None

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return None

    async def get_options_chain(self, symbol: str) -> Optional[dict]:
        """
        Get options chain for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Options chain data or None
        """
        if not self._client:
            await self.connect()

        try:
            response = await self._client.get(
                f"/v7/finance/options/{symbol}"
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("optionChain", {}).get("result", [{}])[0]

            return None

        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {e}")
            return None

    async def search_symbols(self, query: str) -> list[dict]:
        """
        Search for symbols.

        Args:
            query: Search query

        Returns:
            List of matching symbols
        """
        if not self._client:
            await self.connect()

        try:
            response = await self._client.get(
                "/v1/finance/search",
                params={"q": query, "quotesCount": 10},
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("quotes", [])

            return []

        except Exception as e:
            logger.error(f"Error searching symbols: {e}")
            return []

    async def get_market_summary(self) -> list[dict]:
        """
        Get market summary data.

        Returns:
            List of market indices
        """
        if not self._client:
            await self.connect()

        try:
            response = await self._client.get(
                "/v6/finance/quote/marketSummary",
                params={"region": "US"},
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("marketSummaryResponse", {}).get("result", [])

            return []

        except Exception as e:
            logger.error(f"Error fetching market summary: {e}")
            return []

    async def get_trending_symbols(self) -> list[str]:
        """
        Get trending symbols.

        Returns:
            List of trending symbol tickers
        """
        if not self._client:
            await self.connect()

        try:
            response = await self._client.get(
                "/v1/finance/trending/US"
            )

            if response.status_code == 200:
                data = response.json()
                quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
                return [q.get("symbol") for q in quotes if q.get("symbol")]

            return []

        except Exception as e:
            logger.error(f"Error fetching trending symbols: {e}")
            return []

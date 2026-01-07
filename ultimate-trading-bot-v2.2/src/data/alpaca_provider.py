"""
Alpaca Data Provider Module for Ultimate Trading Bot v2.2.

This module provides market data access through the Alpaca API,
including real-time quotes, historical bars, and trade data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
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
from src.utils.exceptions import (
    APIError,
    APIConnectionError,
    APIAuthenticationError,
    APIRateLimitError,
    DataFetchError,
)
from src.utils.date_utils import now_utc, to_utc, parse_datetime
from src.utils.decorators import async_retry


logger = logging.getLogger(__name__)


class AlpacaDataConfig(DataProviderConfig):
    """Configuration for Alpaca data provider."""

    provider_type: DataProviderType = Field(default=DataProviderType.ALPACA)
    api_key: str = Field(default="")
    api_secret: str = Field(default="")

    data_url: str = Field(default="https://data.alpaca.markets")
    paper_url: str = Field(default="https://paper-api.alpaca.markets")
    live_url: str = Field(default="https://api.alpaca.markets")

    use_paper: bool = Field(default=True)
    feed: str = Field(default="iex")

    timeout_seconds: int = Field(default=30)
    max_retries: int = Field(default=3)


TIMEFRAME_MAP = {
    TimeFrame.MINUTE_1: "1Min",
    TimeFrame.MINUTE_5: "5Min",
    TimeFrame.MINUTE_15: "15Min",
    TimeFrame.MINUTE_30: "30Min",
    TimeFrame.HOUR_1: "1Hour",
    TimeFrame.HOUR_4: "4Hour",
    TimeFrame.DAY_1: "1Day",
    TimeFrame.WEEK_1: "1Week",
    TimeFrame.MONTH_1: "1Month",
}


class AlpacaDataProvider(BaseDataProvider):
    """
    Alpaca market data provider implementation.

    Provides access to real-time and historical market data
    through the Alpaca Data API.
    """

    def __init__(
        self,
        config: Optional[AlpacaDataConfig] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        use_paper: bool = True,
    ) -> None:
        """
        Initialize AlpacaDataProvider.

        Args:
            config: Provider configuration
            api_key: Alpaca API key (overrides config)
            api_secret: Alpaca API secret (overrides config)
            use_paper: Use paper trading environment
        """
        if config is None:
            config = AlpacaDataConfig(
                api_key=api_key or "",
                api_secret=api_secret or "",
                use_paper=use_paper,
            )

        super().__init__(config)

        self._alpaca_config: AlpacaDataConfig = config

        self._client: Optional[httpx.AsyncClient] = None
        self._data_client: Optional[httpx.AsyncClient] = None

    @property
    def _headers(self) -> dict[str, str]:
        """Get API headers."""
        return {
            "APCA-API-KEY-ID": self._alpaca_config.api_key,
            "APCA-API-SECRET-KEY": self._alpaca_config.api_secret,
        }

    @property
    def _base_url(self) -> str:
        """Get base API URL."""
        if self._alpaca_config.use_paper:
            return self._alpaca_config.paper_url
        return self._alpaca_config.live_url

    async def connect(self) -> bool:
        """
        Connect to Alpaca API.

        Returns:
            True if connection successful
        """
        try:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._headers,
                timeout=self._alpaca_config.timeout_seconds,
            )

            self._data_client = httpx.AsyncClient(
                base_url=self._alpaca_config.data_url,
                headers=self._headers,
                timeout=self._alpaca_config.timeout_seconds,
            )

            response = await self._client.get("/v2/account")

            if response.status_code == 200:
                self._status = DataProviderStatus.CONNECTED
                logger.info("Connected to Alpaca API")
                return True
            elif response.status_code == 401:
                raise APIAuthenticationError("Invalid Alpaca credentials")
            elif response.status_code == 403:
                raise APIAuthenticationError("Alpaca API access forbidden")
            else:
                raise APIConnectionError(
                    f"Alpaca connection failed: {response.status_code}"
                )

        except httpx.TimeoutException:
            self._status = DataProviderStatus.ERROR
            raise APIConnectionError("Alpaca API connection timeout")

        except httpx.RequestError as e:
            self._status = DataProviderStatus.ERROR
            raise APIConnectionError(f"Alpaca API connection error: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        if self._client:
            await self._client.aclose()
            self._client = None

        if self._data_client:
            await self._data_client.aclose()
            self._data_client = None

        self._status = DataProviderStatus.DISCONNECTED
        logger.info("Disconnected from Alpaca API")

    @async_retry(max_attempts=3, delay=1.0)
    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Quote data or None
        """
        cache_key = self._get_cache_key("quote", symbol)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        if not self._data_client:
            raise APIConnectionError("Not connected to Alpaca API")

        try:
            self._request_count += 1
            self._last_request_time = now_utc()

            response = await self._data_client.get(
                f"/v2/stocks/{symbol}/quotes/latest",
                params={"feed": self._alpaca_config.feed},
            )

            if response.status_code == 200:
                data = response.json()
                quote_data = data.get("quote", {})

                quote = Quote(
                    symbol=symbol,
                    bid_price=float(quote_data.get("bp", 0)),
                    bid_size=int(quote_data.get("bs", 0)),
                    ask_price=float(quote_data.get("ap", 0)),
                    ask_size=int(quote_data.get("as", 0)),
                    timestamp=parse_datetime(quote_data.get("t")),
                )

                self._set_cached(cache_key, quote)
                return quote

            elif response.status_code == 429:
                self._status = DataProviderStatus.RATE_LIMITED
                raise APIRateLimitError("Alpaca API rate limit exceeded")

            else:
                self._error_count += 1
                logger.error(
                    f"Failed to get quote for {symbol}: {response.status_code}"
                )
                return None

        except httpx.TimeoutException:
            self._error_count += 1
            raise DataFetchError(f"Timeout fetching quote for {symbol}")

        except httpx.RequestError as e:
            self._error_count += 1
            raise DataFetchError(f"Error fetching quote for {symbol}: {e}")

    async def get_quotes(
        self,
        symbols: list[str]
    ) -> dict[str, Optional[Quote]]:
        """
        Get real-time quotes for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary of symbol to quote
        """
        if not self._data_client:
            raise APIConnectionError("Not connected to Alpaca API")

        tasks = [self.get_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching quote for {symbol}: {result}")
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
        if not self._data_client:
            raise APIConnectionError("Not connected to Alpaca API")

        alpaca_timeframe = TIMEFRAME_MAP.get(timeframe, "1Day")

        params = {
            "timeframe": alpaca_timeframe,
            "limit": min(limit, 10000),
            "feed": self._alpaca_config.feed,
            "adjustment": "all",
        }

        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        try:
            self._request_count += 1
            self._last_request_time = now_utc()

            response = await self._data_client.get(
                f"/v2/stocks/{symbol}/bars",
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                bars_data = data.get("bars", [])

                bars = []
                for bar_data in bars_data:
                    bar = Bar(
                        symbol=symbol,
                        timestamp=parse_datetime(bar_data.get("t")),
                        open=float(bar_data.get("o", 0)),
                        high=float(bar_data.get("h", 0)),
                        low=float(bar_data.get("l", 0)),
                        close=float(bar_data.get("c", 0)),
                        volume=int(bar_data.get("v", 0)),
                        vwap=float(bar_data.get("vw", 0)) if bar_data.get("vw") else None,
                        trade_count=int(bar_data.get("n", 0)) if bar_data.get("n") else None,
                    )
                    bars.append(bar)

                return bars

            elif response.status_code == 429:
                self._status = DataProviderStatus.RATE_LIMITED
                raise APIRateLimitError("Alpaca API rate limit exceeded")

            else:
                self._error_count += 1
                logger.error(
                    f"Failed to get bars for {symbol}: {response.status_code}"
                )
                return []

        except httpx.TimeoutException:
            self._error_count += 1
            raise DataFetchError(f"Timeout fetching bars for {symbol}")

        except httpx.RequestError as e:
            self._error_count += 1
            raise DataFetchError(f"Error fetching bars for {symbol}: {e}")

    @async_retry(max_attempts=3, delay=1.0)
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
        if not self._data_client:
            raise APIConnectionError("Not connected to Alpaca API")

        params = {
            "limit": min(limit, 10000),
            "feed": self._alpaca_config.feed,
        }

        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        try:
            self._request_count += 1
            self._last_request_time = now_utc()

            response = await self._data_client.get(
                f"/v2/stocks/{symbol}/trades",
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                trades_data = data.get("trades", [])

                trades = []
                for trade_data in trades_data:
                    trade = Trade(
                        symbol=symbol,
                        price=float(trade_data.get("p", 0)),
                        size=int(trade_data.get("s", 0)),
                        timestamp=parse_datetime(trade_data.get("t")),
                        exchange=trade_data.get("x"),
                        conditions=trade_data.get("c"),
                        trade_id=trade_data.get("i"),
                    )
                    trades.append(trade)

                return trades

            elif response.status_code == 429:
                self._status = DataProviderStatus.RATE_LIMITED
                raise APIRateLimitError("Alpaca API rate limit exceeded")

            else:
                self._error_count += 1
                logger.error(
                    f"Failed to get trades for {symbol}: {response.status_code}"
                )
                return []

        except httpx.TimeoutException:
            self._error_count += 1
            raise DataFetchError(f"Timeout fetching trades for {symbol}")

        except httpx.RequestError as e:
            self._error_count += 1
            raise DataFetchError(f"Error fetching trades for {symbol}: {e}")

    async def get_snapshot(self, symbol: str) -> Optional[dict]:
        """
        Get market snapshot for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Snapshot data or None
        """
        if not self._data_client:
            raise APIConnectionError("Not connected to Alpaca API")

        try:
            response = await self._data_client.get(
                f"/v2/stocks/{symbol}/snapshot",
                params={"feed": self._alpaca_config.feed},
            )

            if response.status_code == 200:
                data = response.json()
                snapshot = data.get("snapshot", {})

                return {
                    "symbol": symbol,
                    "latest_trade": snapshot.get("latestTrade"),
                    "latest_quote": snapshot.get("latestQuote"),
                    "minute_bar": snapshot.get("minuteBar"),
                    "daily_bar": snapshot.get("dailyBar"),
                    "prev_daily_bar": snapshot.get("prevDailyBar"),
                }

            return None

        except Exception as e:
            logger.error(f"Error fetching snapshot for {symbol}: {e}")
            return None

    async def get_snapshots(
        self,
        symbols: list[str]
    ) -> dict[str, Optional[dict]]:
        """
        Get market snapshots for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary of symbol to snapshot
        """
        if not self._data_client:
            raise APIConnectionError("Not connected to Alpaca API")

        try:
            symbols_param = ",".join(symbols)
            response = await self._data_client.get(
                "/v2/stocks/snapshots",
                params={
                    "symbols": symbols_param,
                    "feed": self._alpaca_config.feed,
                },
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("snapshots", {})

            return {symbol: None for symbol in symbols}

        except Exception as e:
            logger.error(f"Error fetching snapshots: {e}")
            return {symbol: None for symbol in symbols}

    async def get_account(self) -> Optional[dict]:
        """
        Get Alpaca account information.

        Returns:
            Account data or None
        """
        if not self._client:
            raise APIConnectionError("Not connected to Alpaca API")

        try:
            response = await self._client.get("/v2/account")

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"Error fetching account: {e}")
            return None

    async def get_positions(self) -> list[dict]:
        """
        Get all open positions.

        Returns:
            List of positions
        """
        if not self._client:
            raise APIConnectionError("Not connected to Alpaca API")

        try:
            response = await self._client.get("/v2/positions")

            if response.status_code == 200:
                return response.json()

            return []

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    async def get_position(self, symbol: str) -> Optional[dict]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position data or None
        """
        if not self._client:
            raise APIConnectionError("Not connected to Alpaca API")

        try:
            response = await self._client.get(f"/v2/positions/{symbol}")

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"Error fetching position for {symbol}: {e}")
            return None

    async def get_orders(
        self,
        status: str = "all",
        limit: int = 100,
    ) -> list[dict]:
        """
        Get orders.

        Args:
            status: Order status filter
            limit: Maximum orders to return

        Returns:
            List of orders
        """
        if not self._client:
            raise APIConnectionError("Not connected to Alpaca API")

        try:
            response = await self._client.get(
                "/v2/orders",
                params={"status": status, "limit": limit},
            )

            if response.status_code == 200:
                return response.json()

            return []

        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []

    async def get_asset(self, symbol: str) -> Optional[dict]:
        """
        Get asset information.

        Args:
            symbol: Trading symbol

        Returns:
            Asset data or None
        """
        if not self._client:
            raise APIConnectionError("Not connected to Alpaca API")

        try:
            response = await self._client.get(f"/v2/assets/{symbol}")

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"Error fetching asset {symbol}: {e}")
            return None

    async def get_calendar(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Get market calendar.

        Args:
            start: Start date
            end: End date

        Returns:
            List of calendar entries
        """
        if not self._client:
            raise APIConnectionError("Not connected to Alpaca API")

        params = {}
        if start:
            params["start"] = start.strftime("%Y-%m-%d")
        if end:
            params["end"] = end.strftime("%Y-%m-%d")

        try:
            response = await self._client.get("/v2/calendar", params=params)

            if response.status_code == 200:
                return response.json()

            return []

        except Exception as e:
            logger.error(f"Error fetching calendar: {e}")
            return []

    async def get_clock(self) -> Optional[dict]:
        """
        Get market clock.

        Returns:
            Clock data or None
        """
        if not self._client:
            raise APIConnectionError("Not connected to Alpaca API")

        try:
            response = await self._client.get("/v2/clock")

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"Error fetching clock: {e}")
            return None

"""
Historical Data Module for Ultimate Trading Bot v2.2.

This module provides functionality for managing historical market data,
including data fetching, storage, and retrieval with gap detection.
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from enum import Enum
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.data.base_provider import BaseDataProvider, TimeFrame, Bar
from src.data.data_storage import DataStorage
from src.utils.exceptions import DataFetchError
from src.utils.helpers import generate_uuid
from src.utils.date_utils import (
    now_utc,
    is_trading_day,
    get_trading_days_between,
    get_market_open,
    get_market_close,
)


logger = logging.getLogger(__name__)


class DataQuality(str, Enum):
    """Data quality enumeration."""

    GOOD = "good"
    GAPS = "gaps"
    INCOMPLETE = "incomplete"
    MISSING = "missing"


class HistoricalDataConfig(BaseModel):
    """Configuration for historical data manager."""

    default_lookback_days: int = Field(default=365, ge=1, le=3650)
    max_bars_per_request: int = Field(default=10000, ge=100, le=50000)
    auto_fill_gaps: bool = Field(default=True)
    gap_threshold_minutes: int = Field(default=5, ge=1, le=60)
    cache_historical_data: bool = Field(default=True)
    concurrent_fetch_limit: int = Field(default=5, ge=1, le=20)


class DataGap(BaseModel):
    """Data gap model."""

    symbol: str
    timeframe: TimeFrame
    start: datetime
    end: datetime
    expected_bars: int = Field(default=0)
    filled: bool = Field(default=False)


class DataCoverage(BaseModel):
    """Data coverage information."""

    symbol: str
    timeframe: TimeFrame
    first_bar: Optional[datetime] = None
    last_bar: Optional[datetime] = None
    total_bars: int = Field(default=0)
    gaps: list[DataGap] = Field(default_factory=list)
    quality: DataQuality = Field(default=DataQuality.MISSING)

    @property
    def coverage_days(self) -> int:
        """Calculate coverage in days."""
        if not self.first_bar or not self.last_bar:
            return 0
        return (self.last_bar - self.first_bar).days

    @property
    def gap_count(self) -> int:
        """Get number of gaps."""
        return len([g for g in self.gaps if not g.filled])


class HistoricalDataManager:
    """
    Manages historical market data.

    Provides functionality for:
    - Fetching historical data from providers
    - Storing and retrieving from local storage
    - Gap detection and filling
    - Data quality assessment
    """

    def __init__(
        self,
        config: Optional[HistoricalDataConfig] = None,
        provider: Optional[BaseDataProvider] = None,
        storage: Optional[DataStorage] = None,
    ) -> None:
        """
        Initialize HistoricalDataManager.

        Args:
            config: Configuration
            provider: Data provider
            storage: Data storage
        """
        self._config = config or HistoricalDataConfig()
        self._provider = provider
        self._storage = storage

        self._coverage_cache: dict[str, DataCoverage] = {}
        self._lock = asyncio.Lock()

        self._fetch_count = 0
        self._storage_count = 0

        logger.info("HistoricalDataManager initialized")

    def set_provider(self, provider: BaseDataProvider) -> None:
        """Set the data provider."""
        self._provider = provider

    def set_storage(self, storage: DataStorage) -> None:
        """Set the data storage."""
        self._storage = storage

    async def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
        use_storage: bool = True,
        fill_gaps: bool = True,
    ) -> list[Bar]:
        """
        Get historical bar data.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum bars
            use_storage: Check storage first
            fill_gaps: Fill any gaps

        Returns:
            List of bars
        """
        if not end:
            end = now_utc()

        if not start:
            start = end - timedelta(days=self._config.default_lookback_days)

        bars = []

        if use_storage and self._storage:
            bars = await self._load_from_storage(symbol, timeframe, start, end, limit)

            if bars and fill_gaps:
                gaps = self._detect_gaps(bars, timeframe)
                if gaps:
                    await self._fill_gaps(symbol, timeframe, gaps)
                    bars = await self._load_from_storage(symbol, timeframe, start, end, limit)

            if bars:
                return bars

        if self._provider:
            bars = await self._fetch_from_provider(symbol, timeframe, start, end, limit)

            if bars and self._storage:
                await self._storage.store_bars(symbol, timeframe.value, bars)

        return bars

    async def get_bars_df(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Get historical bar data as DataFrame.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum bars

        Returns:
            DataFrame with OHLCV data
        """
        bars = await self.get_bars(symbol, timeframe, start, end, limit)

        if not bars:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
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
        limit: int = 1000,
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
        semaphore = asyncio.Semaphore(self._config.concurrent_fetch_limit)

        async def fetch_with_limit(symbol: str) -> tuple[str, list[Bar]]:
            async with semaphore:
                bars = await self.get_bars(symbol, timeframe, start, end, limit)
                return symbol, bars

        tasks = [fetch_with_limit(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error fetching bars: {result}")
            else:
                symbol, bars = result
                output[symbol] = bars

        return output

    async def _load_from_storage(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[Bar]:
        """Load bars from storage."""
        if not self._storage:
            return []

        try:
            return await self._storage.load_bars(
                symbol,
                timeframe.value,
                start,
                end,
                limit,
            )
        except Exception as e:
            logger.error(f"Error loading from storage: {e}")
            return []

    async def _fetch_from_provider(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[Bar]:
        """Fetch bars from provider."""
        if not self._provider:
            return []

        try:
            self._fetch_count += 1
            return await self._provider.get_bars(symbol, timeframe, start, end, limit)
        except Exception as e:
            logger.error(f"Error fetching from provider: {e}")
            return []

    def _detect_gaps(
        self,
        bars: list[Bar],
        timeframe: TimeFrame,
    ) -> list[DataGap]:
        """Detect gaps in bar data."""
        if len(bars) < 2:
            return []

        gaps = []
        expected_delta = self._get_timeframe_delta(timeframe)

        for i in range(1, len(bars)):
            time_diff = bars[i].timestamp - bars[i - 1].timestamp

            if time_diff > expected_delta * 2:
                if not self._is_expected_gap(bars[i - 1].timestamp, bars[i].timestamp):
                    gap = DataGap(
                        symbol=bars[0].symbol,
                        timeframe=timeframe,
                        start=bars[i - 1].timestamp,
                        end=bars[i].timestamp,
                        expected_bars=int(time_diff / expected_delta) - 1,
                    )
                    gaps.append(gap)

        return gaps

    def _get_timeframe_delta(self, timeframe: TimeFrame) -> timedelta:
        """Get expected time delta for timeframe."""
        deltas = {
            TimeFrame.MINUTE_1: timedelta(minutes=1),
            TimeFrame.MINUTE_5: timedelta(minutes=5),
            TimeFrame.MINUTE_15: timedelta(minutes=15),
            TimeFrame.MINUTE_30: timedelta(minutes=30),
            TimeFrame.HOUR_1: timedelta(hours=1),
            TimeFrame.HOUR_4: timedelta(hours=4),
            TimeFrame.DAY_1: timedelta(days=1),
            TimeFrame.WEEK_1: timedelta(weeks=1),
            TimeFrame.MONTH_1: timedelta(days=30),
        }
        return deltas.get(timeframe, timedelta(days=1))

    def _is_expected_gap(
        self,
        start: datetime,
        end: datetime,
    ) -> bool:
        """Check if gap is expected (weekends, holidays)."""
        current = start
        while current < end:
            current += timedelta(days=1)
            if is_trading_day(current.date()):
                return False
        return True

    async def _fill_gaps(
        self,
        symbol: str,
        timeframe: TimeFrame,
        gaps: list[DataGap],
    ) -> int:
        """Fill data gaps from provider."""
        if not self._provider or not gaps:
            return 0

        filled = 0

        for gap in gaps:
            try:
                bars = await self._provider.get_bars(
                    symbol,
                    timeframe,
                    gap.start,
                    gap.end,
                    gap.expected_bars + 10,
                )

                if bars and self._storage:
                    await self._storage.store_bars(symbol, timeframe.value, bars)
                    gap.filled = True
                    filled += 1

            except Exception as e:
                logger.error(f"Error filling gap: {e}")

        return filled

    async def get_coverage(
        self,
        symbol: str,
        timeframe: TimeFrame,
    ) -> DataCoverage:
        """
        Get data coverage information.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe

        Returns:
            Data coverage information
        """
        cache_key = f"{symbol}:{timeframe.value}"

        if cache_key in self._coverage_cache:
            return self._coverage_cache[cache_key]

        coverage = DataCoverage(symbol=symbol, timeframe=timeframe)

        if self._storage:
            bars = await self._storage.load_bars(symbol, timeframe.value, limit=100000)

            if bars:
                coverage.first_bar = bars[0].timestamp
                coverage.last_bar = bars[-1].timestamp
                coverage.total_bars = len(bars)
                coverage.gaps = self._detect_gaps(bars, timeframe)

                if not coverage.gaps:
                    coverage.quality = DataQuality.GOOD
                elif len(coverage.gaps) < 5:
                    coverage.quality = DataQuality.GAPS
                else:
                    coverage.quality = DataQuality.INCOMPLETE
            else:
                coverage.quality = DataQuality.MISSING

        self._coverage_cache[cache_key] = coverage
        return coverage

    async def update_historical_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        days_back: Optional[int] = None,
    ) -> int:
        """
        Update historical data to current.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            days_back: Days to look back for updates

        Returns:
            Number of new bars added
        """
        if not self._provider:
            return 0

        days = days_back or 7

        end = now_utc()
        start = end - timedelta(days=days)

        bars = await self._fetch_from_provider(symbol, timeframe, start, end, 10000)

        if bars and self._storage:
            await self._storage.store_bars(symbol, timeframe.value, bars)
            self._storage_count += len(bars)

            cache_key = f"{symbol}:{timeframe.value}"
            self._coverage_cache.pop(cache_key, None)

            logger.info(f"Updated {len(bars)} bars for {symbol}")
            return len(bars)

        return 0

    async def backfill_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: Optional[datetime] = None,
        chunk_days: int = 30,
    ) -> int:
        """
        Backfill historical data.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            start: Start date for backfill
            end: End date for backfill
            chunk_days: Days per fetch chunk

        Returns:
            Total bars backfilled
        """
        if not self._provider:
            return 0

        end = end or now_utc()
        total_bars = 0

        current_end = end
        while current_end > start:
            current_start = max(start, current_end - timedelta(days=chunk_days))

            try:
                bars = await self._fetch_from_provider(
                    symbol,
                    timeframe,
                    current_start,
                    current_end,
                    self._config.max_bars_per_request,
                )

                if bars and self._storage:
                    await self._storage.store_bars(symbol, timeframe.value, bars)
                    total_bars += len(bars)
                    logger.debug(f"Backfilled {len(bars)} bars for {symbol}")

            except Exception as e:
                logger.error(f"Error backfilling {symbol}: {e}")

            current_end = current_start

            await asyncio.sleep(0.5)

        cache_key = f"{symbol}:{timeframe.value}"
        self._coverage_cache.pop(cache_key, None)

        logger.info(f"Backfilled total {total_bars} bars for {symbol}")
        return total_bars

    async def bulk_download(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> dict[str, int]:
        """
        Bulk download historical data for multiple symbols.

        Args:
            symbols: List of symbols
            timeframe: Bar timeframe
            start: Start date
            end: End date

        Returns:
            Dictionary of symbol to bars downloaded
        """
        results = {}

        semaphore = asyncio.Semaphore(self._config.concurrent_fetch_limit)

        async def download_symbol(symbol: str) -> tuple[str, int]:
            async with semaphore:
                count = await self.backfill_data(symbol, timeframe, start, end)
                return symbol, count

        tasks = [download_symbol(symbol) for symbol in symbols]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in task_results:
            if isinstance(result, Exception):
                logger.error(f"Bulk download error: {result}")
            else:
                symbol, count = result
                results[symbol] = count

        return results

    def get_statistics(self) -> dict:
        """Get manager statistics."""
        return {
            "fetch_count": self._fetch_count,
            "storage_count": self._storage_count,
            "coverage_cache_size": len(self._coverage_cache),
            "has_provider": self._provider is not None,
            "has_storage": self._storage is not None,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HistoricalDataManager(fetched={self._fetch_count}, "
            f"stored={self._storage_count})"
        )

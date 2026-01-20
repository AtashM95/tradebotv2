"""
Historical Data module for backtesting and analysis.
~550 lines as per schema
"""

import logging
import sqlite3
import csv
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HistoricalBar:
    """Historical OHLCV bar with metadata."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str = '1d'
    vwap: Optional[float] = None
    trade_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class HistoricalData:
    """
    Historical data provider for backtesting and analysis.

    Features:
    - Multi-source historical data fetching
    - SQLite database caching
    - Date range queries
    - Timeframe aggregation
    - Data validation and cleaning
    - Gap detection
    - CSV import/export
    - Data completeness checks
    """

    def __init__(
        self,
        db_path: str = 'data/historical.db',
        cache_enabled: bool = True,
        primary_source: str = 'mock'
    ):
        """
        Initialize historical data provider.

        Args:
            db_path: Path to SQLite database for caching
            cache_enabled: Enable database caching
            primary_source: Primary data source ('mock', 'alpaca', 'polygon', 'yfinance')
        """
        self.db_path = db_path
        self.cache_enabled = cache_enabled
        self.primary_source = primary_source

        # Initialize database
        if cache_enabled:
            self._init_database()

        # Statistics
        self.stats = {
            "bars_fetched": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "data_source_errors": 0,
            "validation_failures": 0,
            "gaps_detected": 0
        }

    def fetch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        timeframe: str = '1d',
        force_refresh: bool = False
    ) -> List[HistoricalBar]:
        """
        Fetch historical data for symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date (defaults to now)
            timeframe: Bar timeframe ('1min', '5min', '15min', '1h', '1d')
            force_refresh: Force refresh from source, skip cache

        Returns:
            List of HistoricalBar objects
        """
        if end_date is None:
            end_date = datetime.now()

        # Validate date range
        if start_date >= end_date:
            logger.warning(f"Invalid date range: {start_date} to {end_date}")
            return []

        # Check cache first
        if self.cache_enabled and not force_refresh:
            cached_bars = self._fetch_from_cache(symbol, start_date, end_date, timeframe)
            if cached_bars:
                self.stats["cache_hits"] += 1
                return cached_bars

        self.stats["cache_misses"] += 1

        # Fetch from source
        try:
            bars = self._fetch_from_source(symbol, start_date, end_date, timeframe)

            # Validate data
            bars = self._validate_and_clean(bars)

            # Check for gaps
            gaps = self._detect_gaps(bars, timeframe)
            if gaps:
                self.stats["gaps_detected"] += len(gaps)
                logger.info(f"Detected {len(gaps)} gaps in {symbol} data")

            # Cache results
            if self.cache_enabled and bars:
                self._save_to_cache(bars)

            self.stats["bars_fetched"] += len(bars)
            return bars

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            self.stats["data_source_errors"] += 1
            return []

    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        timeframe: str = '1d'
    ) -> Dict[str, List[HistoricalBar]]:
        """
        Fetch historical data for multiple symbols.

        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            timeframe: Bar timeframe

        Returns:
            Dictionary mapping symbols to bar lists
        """
        results = {}

        for symbol in symbols:
            bars = self.fetch(symbol, start_date, end_date, timeframe)
            if bars:
                results[symbol] = bars

        return results

    def get_latest(
        self,
        symbol: str,
        timeframe: str = '1d',
        limit: int = 100
    ) -> List[HistoricalBar]:
        """
        Get latest N bars for symbol.

        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe
            limit: Number of bars

        Returns:
            List of most recent bars
        """
        end_date = datetime.now()
        start_date = self._calculate_lookback_date(end_date, timeframe, limit)

        bars = self.fetch(symbol, start_date, end_date, timeframe)
        return bars[-limit:] if len(bars) > limit else bars

    def aggregate_timeframe(
        self,
        bars: List[HistoricalBar],
        target_timeframe: str
    ) -> List[HistoricalBar]:
        """
        Aggregate bars to a different timeframe.

        Args:
            bars: Source bars
            target_timeframe: Target timeframe (must be larger)

        Returns:
            Aggregated bars
        """
        if not bars:
            return []

        # Group bars by target timeframe periods
        grouped = self._group_by_timeframe(bars, target_timeframe)

        # Aggregate each group
        aggregated = []
        for period_start, period_bars in grouped.items():
            if not period_bars:
                continue

            # Calculate OHLCV
            agg_bar = HistoricalBar(
                symbol=period_bars[0].symbol,
                timestamp=period_start,
                open=period_bars[0].open,
                high=max(b.high for b in period_bars),
                low=min(b.low for b in period_bars),
                close=period_bars[-1].close,
                volume=sum(b.volume for b in period_bars),
                timeframe=target_timeframe
            )

            aggregated.append(agg_bar)

        return aggregated

    def export_to_csv(
        self,
        bars: List[HistoricalBar],
        output_path: str
    ) -> bool:
        """
        Export bars to CSV file.

        Args:
            bars: Bars to export
            output_path: Output CSV file path

        Returns:
            True if successful
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='') as f:
                if not bars:
                    return True

                fieldnames = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'timeframe']
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                writer.writeheader()
                for bar in bars:
                    writer.writerow({
                        'symbol': bar.symbol,
                        'timestamp': bar.timestamp.isoformat(),
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                        'timeframe': bar.timeframe
                    })

            logger.info(f"Exported {len(bars)} bars to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return False

    def import_from_csv(
        self,
        input_path: str,
        save_to_cache: bool = True
    ) -> List[HistoricalBar]:
        """
        Import bars from CSV file.

        Args:
            input_path: Input CSV file path
            save_to_cache: Save imported data to cache

        Returns:
            List of imported bars
        """
        try:
            bars = []

            with open(input_path, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    bar = HistoricalBar(
                        symbol=row['symbol'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=int(row['volume']),
                        timeframe=row.get('timeframe', '1d')
                    )
                    bars.append(bar)

            if save_to_cache and self.cache_enabled:
                self._save_to_cache(bars)

            logger.info(f"Imported {len(bars)} bars from {input_path}")
            return bars

        except Exception as e:
            logger.error(f"Failed to import from CSV: {e}")
            return []

    def _init_database(self):
        """Initialize SQLite database."""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_bars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    timeframe TEXT NOT NULL,
                    vwap REAL,
                    trade_count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, timeframe)
                )
            """)

            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp
                ON historical_bars(symbol, timeframe, timestamp)
            """)

            conn.commit()
            conn.close()

            logger.info("Historical data database initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def _fetch_from_cache(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Optional[List[HistoricalBar]]:
        """Fetch data from cache."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT symbol, timestamp, open, high, low, close, volume, timeframe, vwap, trade_count
                FROM historical_bars
                WHERE symbol = ? AND timeframe = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (symbol, timeframe, start_date.isoformat(), end_date.isoformat()))

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return None

            bars = []
            for row in rows:
                bar = HistoricalBar(
                    symbol=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    open=row[2],
                    high=row[3],
                    low=row[4],
                    close=row[5],
                    volume=row[6],
                    timeframe=row[7],
                    vwap=row[8],
                    trade_count=row[9]
                )
                bars.append(bar)

            return bars

        except Exception as e:
            logger.error(f"Failed to fetch from cache: {e}")
            return None

    def _save_to_cache(self, bars: List[HistoricalBar]):
        """Save bars to cache."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for bar in bars:
                cursor.execute("""
                    INSERT OR REPLACE INTO historical_bars
                    (symbol, timestamp, open, high, low, close, volume, timeframe, vwap, trade_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    bar.symbol,
                    bar.timestamp.isoformat(),
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                    bar.timeframe,
                    bar.vwap,
                    bar.trade_count
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")

    def _fetch_from_source(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> List[HistoricalBar]:
        """Fetch data from configured source."""
        if self.primary_source == 'mock':
            return self._generate_mock_data(symbol, start_date, end_date, timeframe)
        elif self.primary_source == 'alpaca':
            # Would integrate with Alpaca API
            return self._generate_mock_data(symbol, start_date, end_date, timeframe)
        elif self.primary_source == 'polygon':
            # Would integrate with Polygon.io API
            return self._generate_mock_data(symbol, start_date, end_date, timeframe)
        elif self.primary_source == 'yfinance':
            # Would integrate with Yahoo Finance
            return self._generate_mock_data(symbol, start_date, end_date, timeframe)
        else:
            return self._generate_mock_data(symbol, start_date, end_date, timeframe)

    def _generate_mock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> List[HistoricalBar]:
        """Generate mock historical data for testing."""
        import random

        bars = []
        current_date = start_date
        time_delta = self._get_timeframe_delta(timeframe)

        # Starting price based on symbol hash for consistency
        base_price = 100.0 + (hash(symbol) % 1000) / 10.0
        current_price = base_price

        while current_date <= end_date:
            # Random walk with slight upward bias
            open_price = current_price
            change_pct = random.gauss(0.001, 0.02)  # 0.1% drift, 2% volatility
            close_price = open_price * (1 + change_pct)

            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))

            volume = random.randint(100000, 10000000)

            bar = HistoricalBar(
                symbol=symbol,
                timestamp=current_date,
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=volume,
                timeframe=timeframe,
                vwap=round((high_price + low_price + close_price) / 3, 2)
            )

            bars.append(bar)
            current_price = close_price
            current_date += time_delta

        return bars

    def _validate_and_clean(self, bars: List[HistoricalBar]) -> List[HistoricalBar]:
        """Validate and clean bar data."""
        cleaned = []

        for bar in bars:
            # Validate OHLCV relationships
            if not self._validate_bar(bar):
                self.stats["validation_failures"] += 1
                continue

            cleaned.append(bar)

        return cleaned

    def _validate_bar(self, bar: HistoricalBar) -> bool:
        """Validate individual bar."""
        # Check positive prices
        if bar.open <= 0 or bar.high <= 0 or bar.low <= 0 or bar.close <= 0:
            return False

        # Check OHLC relationships
        if bar.high < bar.low:
            return False

        if bar.high < bar.open or bar.high < bar.close:
            return False

        if bar.low > bar.open or bar.low > bar.close:
            return False

        # Check volume
        if bar.volume < 0:
            return False

        return True

    def _detect_gaps(
        self,
        bars: List[HistoricalBar],
        timeframe: str
    ) -> List[Tuple[datetime, datetime]]:
        """Detect gaps in bar sequence."""
        if len(bars) < 2:
            return []

        gaps = []
        time_delta = self._get_timeframe_delta(timeframe)

        for i in range(1, len(bars)):
            expected_time = bars[i-1].timestamp + time_delta
            actual_time = bars[i].timestamp

            # Allow some tolerance for market hours
            if actual_time > expected_time + time_delta:
                gaps.append((bars[i-1].timestamp, bars[i].timestamp))

        return gaps

    def _group_by_timeframe(
        self,
        bars: List[HistoricalBar],
        timeframe: str
    ) -> Dict[datetime, List[HistoricalBar]]:
        """Group bars by timeframe periods."""
        grouped = {}

        for bar in bars:
            period_start = self._get_period_start(bar.timestamp, timeframe)

            if period_start not in grouped:
                grouped[period_start] = []

            grouped[period_start].append(bar)

        return grouped

    def _get_period_start(self, timestamp: datetime, timeframe: str) -> datetime:
        """Get period start for timestamp."""
        if timeframe == '1d':
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe == '1h':
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif timeframe == '15min':
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == '5min':
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == '1min':
            return timestamp.replace(second=0, microsecond=0)
        else:
            return timestamp

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

    def _calculate_lookback_date(
        self,
        end_date: datetime,
        timeframe: str,
        bar_count: int
    ) -> datetime:
        """Calculate start date for N bars lookback."""
        delta = self._get_timeframe_delta(timeframe)
        # Add buffer for weekends/holidays
        buffer_multiplier = 1.5 if timeframe == '1d' else 1.1
        return end_date - (delta * bar_count * buffer_multiplier)

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data."""
        if not self.cache_enabled:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if symbol:
                cursor.execute("DELETE FROM historical_bars WHERE symbol = ?", (symbol,))
                logger.info(f"Cleared cache for {symbol}")
            else:
                cursor.execute("DELETE FROM historical_bars")
                logger.info("Cleared all cached data")

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "bars_fetched": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "data_source_errors": 0,
            "validation_failures": 0,
            "gaps_detected": 0
        }

"""
Data Storage Module for Ultimate Trading Bot v2.2.

This module provides persistent data storage functionality using
SQLite for local storage and optional PostgreSQL for production.
"""

import asyncio
import logging
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import aiosqlite
import pandas as pd
from pydantic import BaseModel, Field

from src.data.base_provider import Bar, Quote, Trade
from src.utils.exceptions import DatabaseError
from src.utils.helpers import generate_uuid, ensure_directory
from src.utils.date_utils import now_utc, format_datetime, parse_datetime


logger = logging.getLogger(__name__)


class StorageBackend(str, Enum):
    """Storage backend enumeration."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class StorageConfig(BaseModel):
    """Configuration for data storage."""

    backend: StorageBackend = Field(default=StorageBackend.SQLITE)

    sqlite_path: str = Field(default="data/trading_bot.db")
    sqlite_wal_mode: bool = Field(default=True)

    postgresql_url: Optional[str] = None
    postgresql_pool_size: int = Field(default=5, ge=1, le=20)

    auto_create_tables: bool = Field(default=True)
    enable_compression: bool = Field(default=False)

    max_bars_per_symbol: int = Field(default=100000, ge=1000)
    max_trades_per_symbol: int = Field(default=1000000, ge=10000)


class DataStorage:
    """
    Persistent data storage for market data.

    Provides functionality to store and retrieve:
    - Historical bar data
    - Trade tick data
    - Quote snapshots
    - Custom time series data
    """

    def __init__(
        self,
        config: Optional[StorageConfig] = None,
    ) -> None:
        """
        Initialize DataStorage.

        Args:
            config: Storage configuration
        """
        self._config = config or StorageConfig()
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

        logger.info(f"DataStorage initialized (backend={self._config.backend.value})")

    @property
    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._db is not None

    async def connect(self) -> bool:
        """
        Connect to the database.

        Returns:
            True if connection successful
        """
        try:
            if self._config.backend == StorageBackend.SQLITE:
                return await self._connect_sqlite()
            elif self._config.backend == StorageBackend.POSTGRESQL:
                return await self._connect_postgresql()
            return False

        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise DatabaseError(f"Failed to connect: {e}")

    async def _connect_sqlite(self) -> bool:
        """Connect to SQLite database."""
        db_path = Path(self._config.sqlite_path)
        ensure_directory(str(db_path.parent))

        self._db = await aiosqlite.connect(
            str(db_path),
            isolation_level=None,
        )

        if self._config.sqlite_wal_mode:
            await self._db.execute("PRAGMA journal_mode=WAL")

        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.execute("PRAGMA cache_size=-64000")

        if self._config.auto_create_tables:
            await self._create_tables()

        logger.info(f"Connected to SQLite: {db_path}")
        return True

    async def _connect_postgresql(self) -> bool:
        """Connect to PostgreSQL database."""
        logger.warning("PostgreSQL not fully implemented, using SQLite")
        return await self._connect_sqlite()

    async def disconnect(self) -> None:
        """Disconnect from the database."""
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("Disconnected from database")

    async def _create_tables(self) -> None:
        """Create database tables."""
        await self._db.executescript("""
            -- Bars table
            CREATE TABLE IF NOT EXISTS bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER DEFAULT 0,
                vwap REAL,
                trade_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_bars_symbol_timeframe
                ON bars(symbol, timeframe);
            CREATE INDEX IF NOT EXISTS idx_bars_timestamp
                ON bars(timestamp);

            -- Trades table
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                size INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                exchange TEXT,
                conditions TEXT,
                trade_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_trades_symbol
                ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp
                ON trades(timestamp);

            -- Quotes table
            CREATE TABLE IF NOT EXISTS quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                bid_price REAL,
                bid_size INTEGER,
                ask_price REAL,
                ask_size INTEGER,
                last_price REAL,
                volume INTEGER,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_quotes_symbol
                ON quotes(symbol);
            CREATE INDEX IF NOT EXISTS idx_quotes_timestamp
                ON quotes(timestamp);

            -- Signals table
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                strength REAL,
                price REAL,
                strategy TEXT,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_signals_symbol
                ON signals(symbol);
            CREATE INDEX IF NOT EXISTS idx_signals_timestamp
                ON signals(timestamp);

            -- Orders table
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                client_order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL,
                stop_price REAL,
                status TEXT NOT NULL,
                filled_qty INTEGER DEFAULT 0,
                filled_avg_price REAL,
                submitted_at TEXT,
                filled_at TEXT,
                cancelled_at TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_orders_symbol
                ON orders(symbol);
            CREATE INDEX IF NOT EXISTS idx_orders_status
                ON orders(status);

            -- Positions table
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                unrealized_pnl REAL,
                realized_pnl REAL DEFAULT 0,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                status TEXT DEFAULT 'open',
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_positions_symbol
                ON positions(symbol);
            CREATE INDEX IF NOT EXISTS idx_positions_status
                ON positions(status);

            -- Portfolio snapshots table
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id TEXT UNIQUE NOT NULL,
                equity REAL NOT NULL,
                cash REAL NOT NULL,
                buying_power REAL,
                portfolio_value REAL,
                unrealized_pnl REAL,
                realized_pnl REAL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp
                ON portfolio_snapshots(timestamp);

            -- Key-value store table
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value TEXT,
                expires_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        logger.info("Database tables created")

    async def store_bars(
        self,
        symbol: str,
        timeframe: str,
        bars: list[Bar],
    ) -> int:
        """
        Store bar data.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            bars: List of bars to store

        Returns:
            Number of bars stored
        """
        if not self._db or not bars:
            return 0

        async with self._lock:
            try:
                data = [
                    (
                        symbol,
                        timeframe,
                        format_datetime(bar.timestamp),
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                        bar.vwap,
                        bar.trade_count,
                    )
                    for bar in bars
                ]

                await self._db.executemany(
                    """
                    INSERT OR REPLACE INTO bars
                    (symbol, timeframe, timestamp, open, high, low, close, volume, vwap, trade_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    data,
                )

                await self._db.commit()
                logger.debug(f"Stored {len(bars)} bars for {symbol}")
                return len(bars)

            except Exception as e:
                logger.error(f"Error storing bars: {e}")
                raise DatabaseError(f"Failed to store bars: {e}")

    async def load_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[Bar]:
        """
        Load bar data.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum bars to return

        Returns:
            List of bars
        """
        if not self._db:
            return []

        try:
            query = """
                SELECT timestamp, open, high, low, close, volume, vwap, trade_count
                FROM bars
                WHERE symbol = ? AND timeframe = ?
            """
            params: list[Any] = [symbol, timeframe]

            if start:
                query += " AND timestamp >= ?"
                params.append(format_datetime(start))

            if end:
                query += " AND timestamp <= ?"
                params.append(format_datetime(end))

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            async with self._db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            bars = []
            for row in reversed(rows):
                bar = Bar(
                    symbol=symbol,
                    timestamp=parse_datetime(row[0]),
                    open=row[1],
                    high=row[2],
                    low=row[3],
                    close=row[4],
                    volume=row[5] or 0,
                    vwap=row[6],
                    trade_count=row[7],
                )
                bars.append(bar)

            return bars

        except Exception as e:
            logger.error(f"Error loading bars: {e}")
            return []

    async def load_bars_df(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Load bar data as DataFrame.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum bars to return

        Returns:
            DataFrame with OHLCV data
        """
        bars = await self.load_bars(symbol, timeframe, start, end, limit)

        if not bars:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
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
        return df

    async def store_trades(
        self,
        symbol: str,
        trades: list[Trade],
    ) -> int:
        """
        Store trade data.

        Args:
            symbol: Trading symbol
            trades: List of trades to store

        Returns:
            Number of trades stored
        """
        if not self._db or not trades:
            return 0

        async with self._lock:
            try:
                data = [
                    (
                        symbol,
                        trade.price,
                        trade.size,
                        format_datetime(trade.timestamp),
                        trade.exchange,
                        ",".join(trade.conditions) if trade.conditions else None,
                        trade.trade_id,
                    )
                    for trade in trades
                ]

                await self._db.executemany(
                    """
                    INSERT INTO trades
                    (symbol, price, size, timestamp, exchange, conditions, trade_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    data,
                )

                await self._db.commit()
                return len(trades)

            except Exception as e:
                logger.error(f"Error storing trades: {e}")
                raise DatabaseError(f"Failed to store trades: {e}")

    async def store_quote(self, quote: Quote) -> bool:
        """
        Store a quote snapshot.

        Args:
            quote: Quote to store

        Returns:
            True if stored successfully
        """
        if not self._db:
            return False

        async with self._lock:
            try:
                await self._db.execute(
                    """
                    INSERT INTO quotes
                    (symbol, bid_price, bid_size, ask_price, ask_size,
                     last_price, volume, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        quote.symbol,
                        quote.bid_price,
                        quote.bid_size,
                        quote.ask_price,
                        quote.ask_size,
                        quote.last_price,
                        quote.volume,
                        format_datetime(quote.timestamp),
                    ),
                )

                await self._db.commit()
                return True

            except Exception as e:
                logger.error(f"Error storing quote: {e}")
                return False

    async def store_signal(
        self,
        signal_id: str,
        symbol: str,
        signal_type: str,
        direction: str,
        strength: float,
        price: float,
        strategy: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Store a trading signal.

        Args:
            signal_id: Unique signal ID
            symbol: Trading symbol
            signal_type: Type of signal
            direction: Signal direction
            strength: Signal strength
            price: Price at signal
            strategy: Strategy name
            metadata: Additional metadata

        Returns:
            True if stored successfully
        """
        if not self._db:
            return False

        async with self._lock:
            try:
                import json
                metadata_json = json.dumps(metadata) if metadata else None

                await self._db.execute(
                    """
                    INSERT INTO signals
                    (signal_id, symbol, signal_type, direction, strength,
                     price, strategy, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal_id,
                        symbol,
                        signal_type,
                        direction,
                        strength,
                        price,
                        strategy,
                        metadata_json,
                        format_datetime(now_utc()),
                    ),
                )

                await self._db.commit()
                return True

            except Exception as e:
                logger.error(f"Error storing signal: {e}")
                return False

    async def store_order(self, order_data: dict) -> bool:
        """
        Store order data.

        Args:
            order_data: Order data dictionary

        Returns:
            True if stored successfully
        """
        if not self._db:
            return False

        async with self._lock:
            try:
                import json
                metadata_json = json.dumps(order_data.get("metadata")) if order_data.get("metadata") else None

                await self._db.execute(
                    """
                    INSERT OR REPLACE INTO orders
                    (order_id, client_order_id, symbol, side, order_type, quantity,
                     price, stop_price, status, filled_qty, filled_avg_price,
                     submitted_at, filled_at, cancelled_at, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        order_data.get("order_id"),
                        order_data.get("client_order_id"),
                        order_data.get("symbol"),
                        order_data.get("side"),
                        order_data.get("order_type"),
                        order_data.get("quantity"),
                        order_data.get("price"),
                        order_data.get("stop_price"),
                        order_data.get("status"),
                        order_data.get("filled_qty", 0),
                        order_data.get("filled_avg_price"),
                        order_data.get("submitted_at"),
                        order_data.get("filled_at"),
                        order_data.get("cancelled_at"),
                        metadata_json,
                        format_datetime(now_utc()),
                    ),
                )

                await self._db.commit()
                return True

            except Exception as e:
                logger.error(f"Error storing order: {e}")
                return False

    async def store_portfolio_snapshot(
        self,
        equity: float,
        cash: float,
        buying_power: float,
        portfolio_value: float,
        unrealized_pnl: float,
        realized_pnl: float,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Store a portfolio snapshot.

        Args:
            equity: Account equity
            cash: Available cash
            buying_power: Buying power
            portfolio_value: Total portfolio value
            unrealized_pnl: Unrealized P&L
            realized_pnl: Realized P&L
            metadata: Additional metadata

        Returns:
            True if stored successfully
        """
        if not self._db:
            return False

        async with self._lock:
            try:
                import json
                metadata_json = json.dumps(metadata) if metadata else None

                await self._db.execute(
                    """
                    INSERT INTO portfolio_snapshots
                    (snapshot_id, equity, cash, buying_power, portfolio_value,
                     unrealized_pnl, realized_pnl, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        generate_uuid(),
                        equity,
                        cash,
                        buying_power,
                        portfolio_value,
                        unrealized_pnl,
                        realized_pnl,
                        format_datetime(now_utc()),
                        metadata_json,
                    ),
                )

                await self._db.commit()
                return True

            except Exception as e:
                logger.error(f"Error storing portfolio snapshot: {e}")
                return False

    async def load_portfolio_history(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Load portfolio history.

        Args:
            start: Start datetime
            end: End datetime
            limit: Maximum records

        Returns:
            DataFrame with portfolio history
        """
        if not self._db:
            return pd.DataFrame()

        try:
            query = """
                SELECT timestamp, equity, cash, buying_power, portfolio_value,
                       unrealized_pnl, realized_pnl
                FROM portfolio_snapshots
                WHERE 1=1
            """
            params: list[Any] = []

            if start:
                query += " AND timestamp >= ?"
                params.append(format_datetime(start))

            if end:
                query += " AND timestamp <= ?"
                params.append(format_datetime(end))

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            async with self._db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            if not rows:
                return pd.DataFrame()

            data = [
                {
                    "timestamp": parse_datetime(row[0]),
                    "equity": row[1],
                    "cash": row[2],
                    "buying_power": row[3],
                    "portfolio_value": row[4],
                    "unrealized_pnl": row[5],
                    "realized_pnl": row[6],
                }
                for row in reversed(rows)
            ]

            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            return df

        except Exception as e:
            logger.error(f"Error loading portfolio history: {e}")
            return pd.DataFrame()

    async def set_kv(
        self,
        key: str,
        value: Any,
        expires_at: Optional[datetime] = None,
    ) -> bool:
        """Set a key-value pair."""
        if not self._db:
            return False

        try:
            import json
            value_json = json.dumps(value)
            expires_str = format_datetime(expires_at) if expires_at else None

            await self._db.execute(
                """
                INSERT OR REPLACE INTO kv_store (key, value, expires_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, value_json, expires_str, format_datetime(now_utc())),
            )

            await self._db.commit()
            return True

        except Exception as e:
            logger.error(f"Error setting KV: {e}")
            return False

    async def get_kv(self, key: str, default: Any = None) -> Any:
        """Get a key-value pair."""
        if not self._db:
            return default

        try:
            async with self._db.execute(
                "SELECT value, expires_at FROM kv_store WHERE key = ?",
                (key,),
            ) as cursor:
                row = await cursor.fetchone()

            if not row:
                return default

            value_json, expires_str = row

            if expires_str:
                expires_at = parse_datetime(expires_str)
                if now_utc() >= expires_at:
                    await self.delete_kv(key)
                    return default

            import json
            return json.loads(value_json)

        except Exception as e:
            logger.error(f"Error getting KV: {e}")
            return default

    async def delete_kv(self, key: str) -> bool:
        """Delete a key-value pair."""
        if not self._db:
            return False

        try:
            await self._db.execute("DELETE FROM kv_store WHERE key = ?", (key,))
            await self._db.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting KV: {e}")
            return False

    async def cleanup_old_data(
        self,
        days_to_keep: int = 30,
    ) -> dict[str, int]:
        """
        Clean up old data.

        Args:
            days_to_keep: Number of days of data to retain

        Returns:
            Dictionary of table to deleted count
        """
        if not self._db:
            return {}

        from datetime import timedelta
        cutoff = format_datetime(now_utc() - timedelta(days=days_to_keep))

        results = {}

        tables = ["quotes", "trades", "portfolio_snapshots"]

        for table in tables:
            try:
                cursor = await self._db.execute(
                    f"DELETE FROM {table} WHERE timestamp < ?",
                    (cutoff,),
                )
                results[table] = cursor.rowcount
                await self._db.commit()
            except Exception as e:
                logger.error(f"Error cleaning {table}: {e}")
                results[table] = 0

        logger.info(f"Cleaned up old data: {results}")
        return results

    async def get_statistics(self) -> dict:
        """Get storage statistics."""
        if not self._db:
            return {"connected": False}

        stats = {"connected": True}

        tables = ["bars", "trades", "quotes", "signals", "orders", "positions", "portfolio_snapshots"]

        for table in tables:
            try:
                async with self._db.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ) as cursor:
                    row = await cursor.fetchone()
                    stats[f"{table}_count"] = row[0] if row else 0
            except Exception:
                stats[f"{table}_count"] = 0

        return stats

    def __repr__(self) -> str:
        """String representation."""
        return f"DataStorage(backend={self._config.backend.value}, connected={self.is_connected})"

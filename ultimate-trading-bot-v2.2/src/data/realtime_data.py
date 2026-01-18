"""
Real-time Data Module for Ultimate Trading Bot v2.2.

This module provides real-time market data handling,
including quote aggregation, VWAP calculation, and tick analysis.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from src.data.base_provider import Quote, Bar, Trade
from src.utils.exceptions import ValidationError
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc
from src.utils.math_utils import safe_divide


logger = logging.getLogger(__name__)


class TickDirection(str, Enum):
    """Tick direction enumeration."""

    UP = "up"
    DOWN = "down"
    UNCHANGED = "unchanged"


class RealtimeConfig(BaseModel):
    """Configuration for real-time data handler."""

    max_quote_history: int = Field(default=1000, ge=100, le=10000)
    max_trade_history: int = Field(default=5000, ge=100, le=50000)
    bar_aggregation_interval_seconds: int = Field(default=60, ge=1, le=3600)
    vwap_window_trades: int = Field(default=100, ge=10, le=1000)
    stale_quote_threshold_seconds: float = Field(default=60.0, ge=1.0, le=300.0)
    enable_tick_analysis: bool = Field(default=True)


class QuoteSnapshot(BaseModel):
    """Extended quote snapshot with analytics."""

    symbol: str
    bid_price: float = Field(default=0.0)
    bid_size: int = Field(default=0)
    ask_price: float = Field(default=0.0)
    ask_size: int = Field(default=0)
    last_price: float = Field(default=0.0)
    last_size: int = Field(default=0)
    volume: int = Field(default=0)
    timestamp: datetime = Field(default_factory=now_utc)

    mid_price: float = Field(default=0.0)
    spread: float = Field(default=0.0)
    spread_bps: float = Field(default=0.0)

    vwap: float = Field(default=0.0)
    tick_direction: TickDirection = Field(default=TickDirection.UNCHANGED)

    high_of_day: float = Field(default=0.0)
    low_of_day: float = Field(default=0.0)
    open_price: float = Field(default=0.0)

    change: float = Field(default=0.0)
    change_percent: float = Field(default=0.0)

    @property
    def is_stale(self) -> bool:
        """Check if quote is stale."""
        age = (now_utc() - self.timestamp).total_seconds()
        return age > 60.0


class TickStatistics(BaseModel):
    """Tick statistics model."""

    symbol: str
    window_seconds: int = Field(default=60)

    tick_count: int = Field(default=0)
    up_ticks: int = Field(default=0)
    down_ticks: int = Field(default=0)
    unchanged_ticks: int = Field(default=0)

    avg_tick_size: float = Field(default=0.0)
    max_tick_up: float = Field(default=0.0)
    max_tick_down: float = Field(default=0.0)

    buy_volume: int = Field(default=0)
    sell_volume: int = Field(default=0)
    total_volume: int = Field(default=0)

    @property
    def tick_ratio(self) -> float:
        """Calculate up/down tick ratio."""
        if self.down_ticks == 0:
            return float("inf") if self.up_ticks > 0 else 1.0
        return self.up_ticks / self.down_ticks

    @property
    def volume_imbalance(self) -> float:
        """Calculate volume imbalance (-1 to 1)."""
        total = self.buy_volume + self.sell_volume
        if total == 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / total


class SymbolData:
    """Real-time data container for a symbol."""

    def __init__(
        self,
        symbol: str,
        config: RealtimeConfig,
    ) -> None:
        """
        Initialize SymbolData.

        Args:
            symbol: Trading symbol
            config: Real-time configuration
        """
        self.symbol = symbol
        self._config = config

        self._quotes: deque[Quote] = deque(maxlen=config.max_quote_history)
        self._trades: deque[Trade] = deque(maxlen=config.max_trade_history)

        self._current_snapshot: Optional[QuoteSnapshot] = None
        self._previous_price: Optional[float] = None

        self._high_of_day: float = 0.0
        self._low_of_day: float = float("inf")
        self._open_price: float = 0.0
        self._day_volume: int = 0

        self._vwap_sum: float = 0.0
        self._vwap_volume: int = 0

        self._tick_stats = TickStatistics(symbol=symbol)
        self._tick_stats_reset_time = now_utc()

    @property
    def current_quote(self) -> Optional[QuoteSnapshot]:
        """Get current quote snapshot."""
        return self._current_snapshot

    @property
    def last_price(self) -> float:
        """Get last price."""
        if self._current_snapshot:
            return self._current_snapshot.last_price
        return 0.0

    @property
    def bid_price(self) -> float:
        """Get bid price."""
        if self._current_snapshot:
            return self._current_snapshot.bid_price
        return 0.0

    @property
    def ask_price(self) -> float:
        """Get ask price."""
        if self._current_snapshot:
            return self._current_snapshot.ask_price
        return 0.0

    @property
    def mid_price(self) -> float:
        """Get mid price."""
        if self._current_snapshot:
            return self._current_snapshot.mid_price
        return 0.0

    @property
    def spread(self) -> float:
        """Get spread."""
        if self._current_snapshot:
            return self._current_snapshot.spread
        return 0.0

    @property
    def vwap(self) -> float:
        """Get VWAP."""
        if self._vwap_volume > 0:
            return self._vwap_sum / self._vwap_volume
        return self.last_price

    @property
    def tick_statistics(self) -> TickStatistics:
        """Get tick statistics."""
        return self._tick_stats

    def process_quote(self, quote: Quote) -> QuoteSnapshot:
        """
        Process a new quote.

        Args:
            quote: Quote to process

        Returns:
            Updated snapshot
        """
        self._quotes.append(quote)

        mid = (quote.bid_price + quote.ask_price) / 2 if quote.bid_price > 0 and quote.ask_price > 0 else quote.last_price
        spread = quote.ask_price - quote.bid_price if quote.bid_price > 0 else 0.0
        spread_bps = (spread / mid * 10000) if mid > 0 else 0.0

        tick_direction = TickDirection.UNCHANGED
        if self._previous_price is not None and quote.last_price > 0:
            if quote.last_price > self._previous_price:
                tick_direction = TickDirection.UP
            elif quote.last_price < self._previous_price:
                tick_direction = TickDirection.DOWN

        if quote.last_price > 0:
            if self._open_price == 0:
                self._open_price = quote.last_price

            if quote.last_price > self._high_of_day:
                self._high_of_day = quote.last_price

            if quote.last_price < self._low_of_day:
                self._low_of_day = quote.last_price

        change = 0.0
        change_percent = 0.0
        if self._open_price > 0 and quote.last_price > 0:
            change = quote.last_price - self._open_price
            change_percent = (change / self._open_price) * 100

        snapshot = QuoteSnapshot(
            symbol=self.symbol,
            bid_price=quote.bid_price,
            bid_size=quote.bid_size,
            ask_price=quote.ask_price,
            ask_size=quote.ask_size,
            last_price=quote.last_price,
            volume=quote.volume,
            timestamp=quote.timestamp,
            mid_price=mid,
            spread=spread,
            spread_bps=spread_bps,
            vwap=self.vwap,
            tick_direction=tick_direction,
            high_of_day=self._high_of_day,
            low_of_day=self._low_of_day if self._low_of_day != float("inf") else 0.0,
            open_price=self._open_price,
            change=change,
            change_percent=change_percent,
        )

        self._current_snapshot = snapshot
        self._previous_price = quote.last_price

        return snapshot

    def process_trade(self, trade: Trade) -> None:
        """
        Process a new trade.

        Args:
            trade: Trade to process
        """
        self._trades.append(trade)

        self._vwap_sum += trade.price * trade.size
        self._vwap_volume += trade.size
        self._day_volume += trade.size

        if trade.price > self._high_of_day:
            self._high_of_day = trade.price

        if trade.price < self._low_of_day:
            self._low_of_day = trade.price

        if self._open_price == 0:
            self._open_price = trade.price

        if self._config.enable_tick_analysis:
            self._update_tick_stats(trade)

    def _update_tick_stats(self, trade: Trade) -> None:
        """Update tick statistics."""
        if (now_utc() - self._tick_stats_reset_time).total_seconds() > self._tick_stats.window_seconds:
            self._reset_tick_stats()

        self._tick_stats.tick_count += 1
        self._tick_stats.total_volume += trade.size

        if self._previous_price is not None:
            tick_size = trade.price - self._previous_price

            if tick_size > 0:
                self._tick_stats.up_ticks += 1
                self._tick_stats.buy_volume += trade.size
                if tick_size > self._tick_stats.max_tick_up:
                    self._tick_stats.max_tick_up = tick_size

            elif tick_size < 0:
                self._tick_stats.down_ticks += 1
                self._tick_stats.sell_volume += trade.size
                if abs(tick_size) > self._tick_stats.max_tick_down:
                    self._tick_stats.max_tick_down = abs(tick_size)

            else:
                self._tick_stats.unchanged_ticks += 1

        if self._tick_stats.tick_count > 0:
            total_tick = self._tick_stats.max_tick_up + self._tick_stats.max_tick_down
            self._tick_stats.avg_tick_size = total_tick / 2

    def _reset_tick_stats(self) -> None:
        """Reset tick statistics."""
        self._tick_stats = TickStatistics(symbol=self.symbol)
        self._tick_stats_reset_time = now_utc()

    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self._high_of_day = 0.0
        self._low_of_day = float("inf")
        self._open_price = 0.0
        self._day_volume = 0
        self._vwap_sum = 0.0
        self._vwap_volume = 0
        self._reset_tick_stats()

    def get_recent_quotes(self, count: int = 100) -> list[Quote]:
        """Get recent quotes."""
        return list(self._quotes)[-count:]

    def get_recent_trades(self, count: int = 100) -> list[Trade]:
        """Get recent trades."""
        return list(self._trades)[-count:]

    def get_quote_history_df(self):
        """Get quote history as DataFrame."""
        import pandas as pd

        if not self._quotes:
            return pd.DataFrame()

        data = [
            {
                "timestamp": q.timestamp,
                "bid_price": q.bid_price,
                "ask_price": q.ask_price,
                "last_price": q.last_price,
                "volume": q.volume,
            }
            for q in self._quotes
        ]

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df


class RealtimeDataHandler:
    """
    Handles real-time market data.

    Provides functionality for:
    - Processing incoming quotes and trades
    - Maintaining per-symbol data
    - Quote aggregation and analytics
    - Callback management
    """

    def __init__(
        self,
        config: Optional[RealtimeConfig] = None,
    ) -> None:
        """
        Initialize RealtimeDataHandler.

        Args:
            config: Real-time configuration
        """
        self._config = config or RealtimeConfig()
        self._symbols: dict[str, SymbolData] = {}

        self._quote_callbacks: list[Callable[[QuoteSnapshot], None]] = []
        self._trade_callbacks: list[Callable[[Trade], None]] = []

        self._running = False
        self._lock = asyncio.Lock()

        self._quotes_processed = 0
        self._trades_processed = 0

        logger.info("RealtimeDataHandler initialized")

    @property
    def symbols(self) -> list[str]:
        """Get tracked symbols."""
        return list(self._symbols.keys())

    def on_quote(self, callback: Callable[[QuoteSnapshot], None]) -> None:
        """Register quote callback."""
        self._quote_callbacks.append(callback)

    def on_trade(self, callback: Callable[[Trade], None]) -> None:
        """Register trade callback."""
        self._trade_callbacks.append(callback)

    def add_symbol(self, symbol: str) -> None:
        """
        Add a symbol to track.

        Args:
            symbol: Trading symbol
        """
        if symbol not in self._symbols:
            self._symbols[symbol] = SymbolData(symbol, self._config)
            logger.debug(f"Added symbol: {symbol}")

    def remove_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from tracking.

        Args:
            symbol: Trading symbol
        """
        if symbol in self._symbols:
            del self._symbols[symbol]
            logger.debug(f"Removed symbol: {symbol}")

    async def process_quote(self, quote: Quote) -> Optional[QuoteSnapshot]:
        """
        Process an incoming quote.

        Args:
            quote: Quote to process

        Returns:
            Updated snapshot
        """
        symbol = quote.symbol

        if symbol not in self._symbols:
            self.add_symbol(symbol)

        symbol_data = self._symbols[symbol]
        snapshot = symbol_data.process_quote(quote)

        self._quotes_processed += 1

        for callback in self._quote_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(snapshot)
                else:
                    callback(snapshot)
            except Exception as e:
                logger.error(f"Error in quote callback: {e}")

        return snapshot

    async def process_trade(self, trade: Trade) -> None:
        """
        Process an incoming trade.

        Args:
            trade: Trade to process
        """
        symbol = trade.symbol

        if symbol not in self._symbols:
            self.add_symbol(symbol)

        symbol_data = self._symbols[symbol]
        symbol_data.process_trade(trade)

        self._trades_processed += 1

        for callback in self._trade_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(trade)
                else:
                    callback(trade)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")

    def get_quote(self, symbol: str) -> Optional[QuoteSnapshot]:
        """Get current quote for symbol."""
        if symbol in self._symbols:
            return self._symbols[symbol].current_quote
        return None

    def get_quotes(self, symbols: Optional[list[str]] = None) -> dict[str, Optional[QuoteSnapshot]]:
        """Get current quotes for symbols."""
        symbols = symbols or list(self._symbols.keys())
        return {symbol: self.get_quote(symbol) for symbol in symbols}

    def get_last_price(self, symbol: str) -> float:
        """Get last price for symbol."""
        if symbol in self._symbols:
            return self._symbols[symbol].last_price
        return 0.0

    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        """Get bid/ask for symbol."""
        if symbol in self._symbols:
            data = self._symbols[symbol]
            return data.bid_price, data.ask_price
        return 0.0, 0.0

    def get_spread(self, symbol: str) -> float:
        """Get spread for symbol."""
        if symbol in self._symbols:
            return self._symbols[symbol].spread
        return 0.0

    def get_vwap(self, symbol: str) -> float:
        """Get VWAP for symbol."""
        if symbol in self._symbols:
            return self._symbols[symbol].vwap
        return 0.0

    def get_tick_stats(self, symbol: str) -> Optional[TickStatistics]:
        """Get tick statistics for symbol."""
        if symbol in self._symbols:
            return self._symbols[symbol].tick_statistics
        return None

    def reset_daily_stats(self) -> None:
        """Reset daily statistics for all symbols."""
        for symbol_data in self._symbols.values():
            symbol_data.reset_daily_stats()
        logger.info("Reset daily stats for all symbols")

    def get_statistics(self) -> dict:
        """Get handler statistics."""
        return {
            "tracked_symbols": len(self._symbols),
            "quotes_processed": self._quotes_processed,
            "trades_processed": self._trades_processed,
            "quote_callbacks": len(self._quote_callbacks),
            "trade_callbacks": len(self._trade_callbacks),
        }

    def get_symbol_summary(self, symbol: str) -> Optional[dict]:
        """Get summary for a symbol."""
        if symbol not in self._symbols:
            return None

        data = self._symbols[symbol]
        snapshot = data.current_quote

        if not snapshot:
            return {"symbol": symbol, "status": "no_data"}

        return {
            "symbol": symbol,
            "last_price": snapshot.last_price,
            "bid": snapshot.bid_price,
            "ask": snapshot.ask_price,
            "spread": snapshot.spread,
            "spread_bps": snapshot.spread_bps,
            "vwap": snapshot.vwap,
            "high": snapshot.high_of_day,
            "low": snapshot.low_of_day,
            "change": snapshot.change,
            "change_pct": snapshot.change_percent,
            "tick_direction": snapshot.tick_direction.value,
            "is_stale": snapshot.is_stale,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RealtimeDataHandler(symbols={len(self._symbols)}, "
            f"quotes={self._quotes_processed}, trades={self._trades_processed})"
        )

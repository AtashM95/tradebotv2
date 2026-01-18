"""
Base Strategy Module for Ultimate Trading Bot v2.2.

This module provides the abstract base class for all trading strategies
with common functionality and interfaces.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class StrategyState(str, Enum):
    """Strategy state enumeration."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class PositionSide(str, Enum):
    """Position side enumeration."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SignalAction(str, Enum):
    """Signal action enumeration."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


class StrategySignal(BaseModel):
    """Trading signal from strategy."""

    signal_id: str = Field(default_factory=generate_uuid)
    strategy_id: str
    strategy_name: str
    symbol: str
    action: SignalAction
    side: PositionSide
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: Optional[float] = None
    reason: str = Field(default="")
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now_utc)
    expires_at: Optional[datetime] = None


class StrategyMetrics(BaseModel):
    """Strategy performance metrics."""

    strategy_id: str
    total_signals: int = Field(default=0)
    winning_signals: int = Field(default=0)
    losing_signals: int = Field(default=0)
    win_rate: float = Field(default=0.0)
    avg_profit: float = Field(default=0.0)
    avg_loss: float = Field(default=0.0)
    profit_factor: float = Field(default=0.0)
    total_pnl: float = Field(default=0.0)
    max_drawdown: float = Field(default=0.0)
    sharpe_ratio: float = Field(default=0.0)
    last_signal_time: Optional[datetime] = None


class StrategyConfig(BaseModel):
    """Base strategy configuration."""

    strategy_id: str = Field(default_factory=generate_uuid)
    name: str = Field(default="BaseStrategy")
    enabled: bool = Field(default=True)
    symbols: list[str] = Field(default_factory=list)
    timeframe: str = Field(default="1d")
    max_positions: int = Field(default=5, ge=1, le=100)
    position_size_pct: float = Field(default=0.1, ge=0.01, le=1.0)
    stop_loss_pct: Optional[float] = Field(default=0.02, ge=0.001, le=0.5)
    take_profit_pct: Optional[float] = Field(default=0.04, ge=0.001, le=1.0)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    cooldown_minutes: int = Field(default=5, ge=0, le=1440)
    use_trailing_stop: bool = Field(default=False)
    trailing_stop_pct: float = Field(default=0.02, ge=0.001, le=0.2)


class MarketData(BaseModel):
    """Market data for strategy evaluation."""

    symbol: str
    timestamp: datetime = Field(default_factory=now_utc)
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    indicators: dict = Field(default_factory=dict)


class StrategyContext(BaseModel):
    """Context for strategy execution."""

    account_value: float = Field(default=100000.0)
    buying_power: float = Field(default=100000.0)
    current_positions: dict[str, dict] = Field(default_factory=dict)
    pending_orders: list[dict] = Field(default_factory=list)
    market_regime: str = Field(default="neutral")
    vix: Optional[float] = None
    risk_budget_remaining: float = Field(default=1.0)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Provides:
    - Common initialization and configuration
    - Signal generation interface
    - Position management
    - Risk management integration
    - Performance tracking
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
    ) -> None:
        """
        Initialize BaseStrategy.

        Args:
            config: Strategy configuration
        """
        self._config = config or StrategyConfig()
        self._state = StrategyState.INITIALIZED
        self._metrics = StrategyMetrics(strategy_id=self._config.strategy_id)

        self._last_signal_time: dict[str, datetime] = {}
        self._active_signals: dict[str, StrategySignal] = {}
        self._signal_history: list[StrategySignal] = []

        logger.info(f"Strategy {self._config.name} initialized")

    @property
    def strategy_id(self) -> str:
        """Get strategy ID."""
        return self._config.strategy_id

    @property
    def name(self) -> str:
        """Get strategy name."""
        return self._config.name

    @property
    def state(self) -> StrategyState:
        """Get current state."""
        return self._state

    @property
    def config(self) -> StrategyConfig:
        """Get configuration."""
        return self._config

    @property
    def metrics(self) -> StrategyMetrics:
        """Get performance metrics."""
        return self._metrics

    @abstractmethod
    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate strategy and generate signals.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of trading signals
        """
        pass

    @abstractmethod
    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict:
        """
        Calculate strategy-specific indicators.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        pass

    async def start(self) -> None:
        """Start the strategy."""
        if self._state in [StrategyState.RUNNING]:
            logger.warning(f"Strategy {self.name} already running")
            return

        self._state = StrategyState.RUNNING
        logger.info(f"Strategy {self.name} started")

    async def stop(self) -> None:
        """Stop the strategy."""
        self._state = StrategyState.STOPPED
        logger.info(f"Strategy {self.name} stopped")

    async def pause(self) -> None:
        """Pause the strategy."""
        self._state = StrategyState.PAUSED
        logger.info(f"Strategy {self.name} paused")

    async def resume(self) -> None:
        """Resume the strategy."""
        if self._state == StrategyState.PAUSED:
            self._state = StrategyState.RUNNING
            logger.info(f"Strategy {self.name} resumed")

    def is_enabled(self) -> bool:
        """Check if strategy is enabled."""
        return self._config.enabled and self._state == StrategyState.RUNNING

    def can_trade_symbol(self, symbol: str) -> bool:
        """Check if strategy can trade a symbol."""
        if not self._config.symbols:
            return True
        return symbol in self._config.symbols

    def check_cooldown(self, symbol: str) -> bool:
        """
        Check if symbol is in cooldown period.

        Args:
            symbol: Trading symbol

        Returns:
            True if in cooldown (cannot trade)
        """
        if self._config.cooldown_minutes <= 0:
            return False

        last_signal = self._last_signal_time.get(symbol)
        if not last_signal:
            return False

        elapsed = (now_utc() - last_signal).total_seconds() / 60
        return elapsed < self._config.cooldown_minutes

    def create_signal(
        self,
        symbol: str,
        action: SignalAction,
        side: PositionSide,
        entry_price: float,
        strength: float = 0.5,
        confidence: float = 0.5,
        reason: str = "",
        metadata: Optional[dict] = None,
    ) -> Optional[StrategySignal]:
        """
        Create a trading signal.

        Args:
            symbol: Trading symbol
            action: Signal action
            side: Position side
            entry_price: Entry price
            strength: Signal strength
            confidence: Signal confidence
            reason: Signal reason
            metadata: Additional metadata

        Returns:
            StrategySignal or None if filtered
        """
        if confidence < self._config.min_confidence:
            logger.debug(
                f"Signal filtered: confidence {confidence:.2f} < {self._config.min_confidence:.2f}"
            )
            return None

        if self.check_cooldown(symbol):
            logger.debug(f"Signal filtered: {symbol} in cooldown")
            return None

        stop_loss = None
        take_profit = None

        if self._config.stop_loss_pct:
            if side == PositionSide.LONG:
                stop_loss = entry_price * (1 - self._config.stop_loss_pct)
            else:
                stop_loss = entry_price * (1 + self._config.stop_loss_pct)

        if self._config.take_profit_pct:
            if side == PositionSide.LONG:
                take_profit = entry_price * (1 + self._config.take_profit_pct)
            else:
                take_profit = entry_price * (1 - self._config.take_profit_pct)

        signal = StrategySignal(
            strategy_id=self._config.strategy_id,
            strategy_name=self._config.name,
            symbol=symbol,
            action=action,
            side=side,
            strength=strength,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=self._config.position_size_pct,
            reason=reason,
            metadata=metadata or {},
        )

        self._last_signal_time[symbol] = now_utc()
        self._active_signals[symbol] = signal
        self._signal_history.append(signal)
        self._metrics.total_signals += 1
        self._metrics.last_signal_time = now_utc()

        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-1000:]

        logger.info(
            f"Strategy {self.name} generated {action.value} signal for {symbol} "
            f"(confidence: {confidence:.2f})"
        )

        return signal

    def record_trade_result(
        self,
        symbol: str,
        pnl: float,
        is_winner: bool,
    ) -> None:
        """
        Record trade result for metrics.

        Args:
            symbol: Trading symbol
            pnl: Profit/loss amount
            is_winner: Whether trade was profitable
        """
        if is_winner:
            self._metrics.winning_signals += 1
            if self._metrics.avg_profit == 0:
                self._metrics.avg_profit = pnl
            else:
                self._metrics.avg_profit = (
                    self._metrics.avg_profit * 0.9 + pnl * 0.1
                )
        else:
            self._metrics.losing_signals += 1
            if self._metrics.avg_loss == 0:
                self._metrics.avg_loss = abs(pnl)
            else:
                self._metrics.avg_loss = (
                    self._metrics.avg_loss * 0.9 + abs(pnl) * 0.1
                )

        self._metrics.total_pnl += pnl

        total = self._metrics.winning_signals + self._metrics.losing_signals
        if total > 0:
            self._metrics.win_rate = self._metrics.winning_signals / total

        if self._metrics.avg_loss > 0:
            self._metrics.profit_factor = (
                self._metrics.avg_profit / self._metrics.avg_loss
            )

    def should_exit_position(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        side: PositionSide,
    ) -> tuple[bool, str]:
        """
        Check if position should be exited.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            entry_price: Position entry price
            side: Position side

        Returns:
            Tuple of (should_exit, reason)
        """
        if side == PositionSide.LONG:
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        if self._config.stop_loss_pct and pnl_pct <= -self._config.stop_loss_pct:
            return True, f"Stop loss hit at {pnl_pct*100:.2f}%"

        if self._config.take_profit_pct and pnl_pct >= self._config.take_profit_pct:
            return True, f"Take profit hit at {pnl_pct*100:.2f}%"

        return False, ""

    def get_active_signal(self, symbol: str) -> Optional[StrategySignal]:
        """Get active signal for symbol."""
        return self._active_signals.get(symbol)

    def clear_active_signal(self, symbol: str) -> None:
        """Clear active signal for symbol."""
        if symbol in self._active_signals:
            del self._active_signals[symbol]

    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> list[StrategySignal]:
        """Get signal history."""
        signals = self._signal_history

        if symbol:
            signals = [s for s in signals if s.symbol == symbol]

        return signals[-limit:]

    def update_config(self, **kwargs) -> None:
        """Update strategy configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        logger.info(f"Strategy {self.name} config updated: {kwargs}")

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._metrics = StrategyMetrics(strategy_id=self._config.strategy_id)
        logger.info(f"Strategy {self.name} metrics reset")

    def to_dict(self) -> dict:
        """Convert strategy to dictionary."""
        return {
            "strategy_id": self._config.strategy_id,
            "name": self._config.name,
            "state": self._state.value,
            "enabled": self._config.enabled,
            "symbols": self._config.symbols,
            "metrics": self._metrics.model_dump(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name}, state={self._state.value})"

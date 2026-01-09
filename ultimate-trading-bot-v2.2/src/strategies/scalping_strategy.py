"""
Scalping Strategy Module for Ultimate Trading Bot v2.2.

This module implements a high-frequency scalping strategy
for capturing small price movements.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.strategies.base_strategy import (
    BaseStrategy,
    StrategyConfig,
    StrategySignal,
    SignalAction,
    SignalSide,
    MarketData,
    StrategyContext,
)
from src.analysis.technical_indicators import TechnicalIndicators
from src.utils.helpers import generate_uuid


logger = logging.getLogger(__name__)


class ScalpingConfig(StrategyConfig):
    """Configuration for scalping strategy."""

    name: str = Field(default="Scalping Strategy")
    description: str = Field(
        default="High-frequency scalping for small price movements"
    )

    tick_threshold: float = Field(default=0.0005, ge=0.0001, le=0.01)
    min_spread_pct: float = Field(default=0.0002, ge=0.0001, le=0.005)
    max_spread_pct: float = Field(default=0.002, ge=0.0005, le=0.01)

    rsi_period: int = Field(default=7, ge=3, le=14)
    rsi_overbought: float = Field(default=75.0, ge=60.0, le=90.0)
    rsi_oversold: float = Field(default=25.0, ge=10.0, le=40.0)

    ema_fast: int = Field(default=5, ge=2, le=10)
    ema_slow: int = Field(default=13, ge=8, le=20)

    vwap_deviation_pct: float = Field(default=0.002, ge=0.0005, le=0.01)

    volume_spike_threshold: float = Field(default=1.5, ge=1.2, le=3.0)

    target_profit_pct: float = Field(default=0.002, ge=0.0005, le=0.01)
    max_loss_pct: float = Field(default=0.001, ge=0.0003, le=0.005)

    max_trades_per_hour: int = Field(default=20, ge=5, le=100)
    min_time_between_trades: int = Field(default=30, ge=10, le=300)

    require_momentum: bool = Field(default=True)
    use_order_flow: bool = Field(default=True)


class ScalpingStrategy(BaseStrategy):
    """
    High-frequency scalping strategy.

    Features:
    - Sub-minute trade execution
    - Tight profit targets and stop losses
    - Volume and momentum confirmation
    - Spread analysis
    - VWAP deviation trading
    """

    def __init__(
        self,
        config: Optional[ScalpingConfig] = None,
    ) -> None:
        """
        Initialize ScalpingStrategy.

        Args:
            config: Scalping configuration
        """
        config = config or ScalpingConfig()
        super().__init__(config)

        self._indicators = TechnicalIndicators()
        self._scalp_config = config

        self._trades_this_hour: dict[str, int] = {}
        self._last_trade_time: dict[str, datetime] = {}

        logger.info(f"ScalpingStrategy initialized: {self.name}")

    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict[str, Any]:
        """
        Calculate scalping indicators.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        closes = data.closes
        highs = data.highs
        lows = data.lows
        volumes = data.volumes

        if len(closes) < 20:
            return {}

        rsi = self._indicators.rsi(closes, self._scalp_config.rsi_period)
        current_rsi = rsi[-1] if rsi else 50.0

        ema_fast = self._indicators.ema(closes, self._scalp_config.ema_fast)
        ema_slow = self._indicators.ema(closes, self._scalp_config.ema_slow)

        current_ema_fast = ema_fast[-1] if ema_fast else closes[-1]
        current_ema_slow = ema_slow[-1] if ema_slow else closes[-1]

        vwap = self._calculate_vwap(highs, lows, closes, volumes)

        current_price = closes[-1]
        vwap_deviation = (current_price - vwap) / vwap if vwap > 0 else 0

        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        momentum = self._calculate_momentum(closes)

        spread = self._estimate_spread(highs, lows, closes)

        tick_direction = self._get_tick_direction(closes)

        return {
            "rsi": current_rsi,
            "ema_fast": current_ema_fast,
            "ema_slow": current_ema_slow,
            "vwap": vwap,
            "vwap_deviation": vwap_deviation,
            "volume_ratio": volume_ratio,
            "momentum": momentum,
            "spread": spread,
            "tick_direction": tick_direction,
            "current_price": current_price,
        }

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate scalping opportunities.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of scalping signals
        """
        signals: list[StrategySignal] = []

        for symbol in self.config.symbols:
            if symbol not in market_data:
                continue

            data = market_data[symbol]

            if not self._can_trade(symbol, context):
                continue

            indicators = self.calculate_indicators(symbol, data)
            if not indicators:
                continue

            signal = self._generate_scalp_signal(symbol, indicators, context)

            if signal:
                signals.append(signal)
                self._record_trade(symbol)

        return signals

    def _generate_scalp_signal(
        self,
        symbol: str,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Generate scalping signal based on indicators."""
        current_price = indicators["current_price"]
        rsi = indicators["rsi"]
        ema_fast = indicators["ema_fast"]
        ema_slow = indicators["ema_slow"]
        vwap = indicators["vwap"]
        vwap_deviation = indicators["vwap_deviation"]
        volume_ratio = indicators["volume_ratio"]
        momentum = indicators["momentum"]
        spread = indicators["spread"]
        tick_direction = indicators["tick_direction"]

        if spread > self._scalp_config.max_spread_pct:
            return None
        if spread < self._scalp_config.min_spread_pct:
            return None

        buy_score = 0.0
        sell_score = 0.0

        if rsi < self._scalp_config.rsi_oversold:
            buy_score += 0.25
        elif rsi > self._scalp_config.rsi_overbought:
            sell_score += 0.25

        if ema_fast > ema_slow:
            buy_score += 0.2
        elif ema_fast < ema_slow:
            sell_score += 0.2

        if vwap_deviation < -self._scalp_config.vwap_deviation_pct:
            buy_score += 0.2
        elif vwap_deviation > self._scalp_config.vwap_deviation_pct:
            sell_score += 0.2

        if volume_ratio > self._scalp_config.volume_spike_threshold:
            if tick_direction > 0:
                buy_score += 0.15
            elif tick_direction < 0:
                sell_score += 0.15

        if self._scalp_config.require_momentum:
            if momentum > 0:
                buy_score += 0.2
            elif momentum < 0:
                sell_score += 0.2

        position = context.positions.get(symbol)

        if position and position.get("side") == "long":
            entry_price = position.get("entry_price", current_price)
            pnl_pct = (current_price - entry_price) / entry_price

            if pnl_pct >= self._scalp_config.target_profit_pct:
                return self._create_exit_signal(
                    symbol, current_price, "take_profit", indicators
                )
            if pnl_pct <= -self._scalp_config.max_loss_pct:
                return self._create_exit_signal(
                    symbol, current_price, "stop_loss", indicators
                )

            return None

        elif position and position.get("side") == "short":
            entry_price = position.get("entry_price", current_price)
            pnl_pct = (entry_price - current_price) / entry_price

            if pnl_pct >= self._scalp_config.target_profit_pct:
                return self._create_exit_signal(
                    symbol, current_price, "take_profit", indicators
                )
            if pnl_pct <= -self._scalp_config.max_loss_pct:
                return self._create_exit_signal(
                    symbol, current_price, "stop_loss", indicators
                )

            return None

        min_score = 0.6

        if buy_score >= min_score and buy_score > sell_score:
            stop_loss = current_price * (1 - self._scalp_config.max_loss_pct)
            take_profit = current_price * (1 + self._scalp_config.target_profit_pct)

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.BUY,
                side=SignalSide.LONG,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strength=buy_score,
                confidence=min(0.85, buy_score + 0.1),
                reason=f"Scalp buy: RSI={rsi:.1f}, VWAP_dev={vwap_deviation:.4f}",
                metadata={
                    "indicators": indicators,
                    "strategy_type": "scalping",
                    "trade_type": "quick_scalp",
                },
            )

        elif sell_score >= min_score and sell_score > buy_score:
            stop_loss = current_price * (1 + self._scalp_config.max_loss_pct)
            take_profit = current_price * (1 - self._scalp_config.target_profit_pct)

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL,
                side=SignalSide.SHORT,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strength=sell_score,
                confidence=min(0.85, sell_score + 0.1),
                reason=f"Scalp sell: RSI={rsi:.1f}, VWAP_dev={vwap_deviation:.4f}",
                metadata={
                    "indicators": indicators,
                    "strategy_type": "scalping",
                    "trade_type": "quick_scalp",
                },
            )

        return None

    def _create_exit_signal(
        self,
        symbol: str,
        current_price: float,
        reason: str,
        indicators: dict[str, Any],
    ) -> StrategySignal:
        """Create exit signal for scalp trade."""
        return self.create_signal(
            symbol=symbol,
            action=SignalAction.SELL,
            side=SignalSide.FLAT,
            entry_price=current_price,
            strength=1.0,
            confidence=0.95,
            reason=f"Scalp exit: {reason}",
            metadata={
                "indicators": indicators,
                "strategy_type": "scalping",
                "exit_reason": reason,
            },
        )

    def _calculate_vwap(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
    ) -> float:
        """Calculate VWAP for the session."""
        n = min(len(closes), len(volumes))
        if n == 0:
            return 0.0

        typical_prices = [
            (highs[i] + lows[i] + closes[i]) / 3
            for i in range(-n, 0)
        ]

        total_volume = sum(volumes[-n:])
        if total_volume == 0:
            return closes[-1]

        vwap = sum(
            typical_prices[i] * volumes[-n + i]
            for i in range(n)
        ) / total_volume

        return vwap

    def _calculate_momentum(self, closes: list[float]) -> float:
        """Calculate short-term momentum."""
        if len(closes) < 5:
            return 0.0

        recent = closes[-5:]
        price_change = (recent[-1] - recent[0]) / recent[0] if recent[0] != 0 else 0

        up_moves = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        down_moves = len(recent) - 1 - up_moves

        direction_bias = (up_moves - down_moves) / (len(recent) - 1)

        return (price_change * 100 + direction_bias) / 2

    def _estimate_spread(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> float:
        """Estimate bid-ask spread from OHLC data."""
        if len(closes) < 10:
            return 0.001

        ranges = [
            (highs[i] - lows[i]) / closes[i]
            for i in range(-10, 0)
        ]

        avg_range = sum(ranges) / len(ranges)

        estimated_spread = avg_range * 0.3

        return max(0.0001, min(0.01, estimated_spread))

    def _get_tick_direction(self, closes: list[float]) -> int:
        """Get recent tick direction."""
        if len(closes) < 3:
            return 0

        recent = closes[-3:]
        up_ticks = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        down_ticks = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])

        if up_ticks > down_ticks:
            return 1
        elif down_ticks > up_ticks:
            return -1
        return 0

    def _can_trade(self, symbol: str, context: StrategyContext) -> bool:
        """Check if trading is allowed for symbol."""
        current_hour = context.timestamp.strftime("%Y-%m-%d-%H")

        if symbol not in self._trades_this_hour:
            self._trades_this_hour[symbol] = 0

        key = f"{symbol}_{current_hour}"
        trades = self._trades_this_hour.get(symbol, 0)

        if trades >= self._scalp_config.max_trades_per_hour:
            return False

        if symbol in self._last_trade_time:
            last_trade = self._last_trade_time[symbol]
            seconds_since = (context.timestamp - last_trade).total_seconds()
            if seconds_since < self._scalp_config.min_time_between_trades:
                return False

        return True

    def _record_trade(self, symbol: str) -> None:
        """Record trade for rate limiting."""
        from src.utils.date_utils import now_utc

        self._trades_this_hour[symbol] = self._trades_this_hour.get(symbol, 0) + 1
        self._last_trade_time[symbol] = now_utc()

    def reset_hourly_counters(self) -> None:
        """Reset hourly trade counters."""
        self._trades_this_hour.clear()

    def get_scalping_stats(self) -> dict:
        """Get scalping-specific statistics."""
        return {
            "trades_this_hour": dict(self._trades_this_hour),
            "last_trade_times": {
                k: v.isoformat() for k, v in self._last_trade_time.items()
            },
            "max_trades_per_hour": self._scalp_config.max_trades_per_hour,
            "target_profit_pct": self._scalp_config.target_profit_pct,
            "max_loss_pct": self._scalp_config.max_loss_pct,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"ScalpingStrategy(name={self.name})"

"""
Trend Following Strategy Module for Ultimate Trading Bot v2.2.

This module implements trend following trading strategies
that capitalize on sustained price movements.
"""

import logging
from typing import Optional

from pydantic import Field

from src.strategies.base_strategy import (
    BaseStrategy,
    StrategyConfig,
    StrategySignal,
    MarketData,
    StrategyContext,
    SignalAction,
    PositionSide,
)
from src.analysis.technical_indicators import TechnicalIndicators


logger = logging.getLogger(__name__)


class TrendFollowingConfig(StrategyConfig):
    """Trend following strategy configuration."""

    name: str = Field(default="TrendFollowingStrategy")
    fast_ma_period: int = Field(default=10, ge=5, le=50)
    slow_ma_period: int = Field(default=50, ge=20, le=200)
    signal_ma_period: int = Field(default=20, ge=10, le=100)
    adx_period: int = Field(default=14, ge=5, le=30)
    adx_threshold: float = Field(default=25.0, ge=15.0, le=50.0)
    atr_period: int = Field(default=14, ge=5, le=30)
    atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    use_ema: bool = Field(default=True)
    require_adx_confirmation: bool = Field(default=True)
    trail_stop_on_profit: bool = Field(default=True)
    pyramiding_enabled: bool = Field(default=False)
    max_pyramid_entries: int = Field(default=3, ge=1, le=5)


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following trading strategy.

    Identifies and follows trends using:
    - Moving average crossovers
    - ADX for trend strength
    - ATR for volatility-based stops
    - Pyramiding on strong trends
    """

    def __init__(
        self,
        config: Optional[TrendFollowingConfig] = None,
    ) -> None:
        """
        Initialize TrendFollowingStrategy.

        Args:
            config: Trend following strategy configuration
        """
        super().__init__(config or TrendFollowingConfig())
        self._indicators = TechnicalIndicators()
        self._price_history: dict[str, list[float]] = {}
        self._high_history: dict[str, list[float]] = {}
        self._low_history: dict[str, list[float]] = {}
        self._pyramid_counts: dict[str, int] = {}

        logger.info("TrendFollowingStrategy initialized")

    @property
    def tf_config(self) -> TrendFollowingConfig:
        """Get trend following specific config."""
        return self._config  # type: ignore

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate trend following signals.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of trading signals
        """
        signals: list[StrategySignal] = []

        if not self.is_enabled():
            return signals

        for symbol, data in market_data.items():
            if not self.can_trade_symbol(symbol):
                continue

            self._update_history(symbol, data)

            if len(self._price_history.get(symbol, [])) < self.tf_config.slow_ma_period + 10:
                continue

            indicators = self.calculate_indicators(symbol, data)

            if symbol in context.current_positions:
                exit_signal = self._check_exit(symbol, data, indicators, context)
                if exit_signal:
                    signals.append(exit_signal)
                elif self.tf_config.pyramiding_enabled:
                    pyramid_signal = self._check_pyramid(symbol, data, indicators, context)
                    if pyramid_signal:
                        signals.append(pyramid_signal)
            else:
                entry_signal = self._generate_entry_signal(symbol, data, indicators, context)
                if entry_signal:
                    signals.append(entry_signal)

        return signals

    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict:
        """
        Calculate trend following indicators.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        closes = self._price_history.get(symbol, [])
        highs = self._high_history.get(symbol, [])
        lows = self._low_history.get(symbol, [])

        if len(closes) < self.tf_config.slow_ma_period:
            return {}

        if self.tf_config.use_ema:
            fast_ma = self._indicators.ema(closes, self.tf_config.fast_ma_period)
            slow_ma = self._indicators.ema(closes, self.tf_config.slow_ma_period)
            signal_ma = self._indicators.ema(closes, self.tf_config.signal_ma_period)
        else:
            fast_ma = self._indicators.sma(closes, self.tf_config.fast_ma_period)
            slow_ma = self._indicators.sma(closes, self.tf_config.slow_ma_period)
            signal_ma = self._indicators.sma(closes, self.tf_config.signal_ma_period)

        current_fast = fast_ma[-1] if fast_ma else closes[-1]
        current_slow = slow_ma[-1] if slow_ma else closes[-1]
        current_signal = signal_ma[-1] if signal_ma else closes[-1]
        prev_fast = fast_ma[-2] if len(fast_ma) >= 2 else current_fast
        prev_slow = slow_ma[-2] if len(slow_ma) >= 2 else current_slow

        adx_values = self._indicators.adx(
            highs, lows, closes,
            self.tf_config.adx_period,
        )
        current_adx = adx_values[-1] if adx_values else 0

        atr_values = self._indicators.atr(
            highs, lows, closes,
            self.tf_config.atr_period,
        )
        current_atr = atr_values[-1] if atr_values else 0

        golden_cross = prev_fast <= prev_slow and current_fast > current_slow
        death_cross = prev_fast >= prev_slow and current_fast < current_slow

        trend_up = current_fast > current_slow and closes[-1] > current_signal
        trend_down = current_fast < current_slow and closes[-1] < current_signal

        trend_strength = current_adx if current_adx else 0
        is_strong_trend = trend_strength >= self.tf_config.adx_threshold

        return {
            "fast_ma": current_fast,
            "slow_ma": current_slow,
            "signal_ma": current_signal,
            "adx": current_adx,
            "atr": current_atr,
            "price": closes[-1],
            "golden_cross": golden_cross,
            "death_cross": death_cross,
            "trend_up": trend_up,
            "trend_down": trend_down,
            "trend_strength": trend_strength,
            "is_strong_trend": is_strong_trend,
            "price_above_fast": closes[-1] > current_fast,
            "price_above_slow": closes[-1] > current_slow,
        }

    def _generate_entry_signal(
        self,
        symbol: str,
        data: MarketData,
        indicators: dict,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Generate entry signal based on trend following."""
        if not indicators:
            return None

        golden_cross = indicators.get("golden_cross", False)
        death_cross = indicators.get("death_cross", False)
        trend_up = indicators.get("trend_up", False)
        trend_down = indicators.get("trend_down", False)
        is_strong_trend = indicators.get("is_strong_trend", False)
        adx = indicators.get("adx", 0)
        atr = indicators.get("atr", 0)

        if self.tf_config.require_adx_confirmation and not is_strong_trend:
            return None

        if golden_cross or (trend_up and is_strong_trend):
            stop_loss = data.close - (atr * self.tf_config.atr_multiplier)

            strength = min(1.0, adx / 50) if adx else 0.5
            confidence = 0.6 if golden_cross else 0.5
            if is_strong_trend:
                confidence += 0.2

            self._pyramid_counts[symbol] = 1

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.BUY,
                side=PositionSide.LONG,
                entry_price=data.close,
                strength=strength,
                confidence=min(1.0, confidence),
                reason=f"Trend following long: {'Golden cross' if golden_cross else 'Trend continuation'}, ADX={adx:.1f}",
                metadata={**indicators, "atr_stop": stop_loss},
            )

        if death_cross or (trend_down and is_strong_trend):
            stop_loss = data.close + (atr * self.tf_config.atr_multiplier)

            strength = min(1.0, adx / 50) if adx else 0.5
            confidence = 0.6 if death_cross else 0.5
            if is_strong_trend:
                confidence += 0.2

            self._pyramid_counts[symbol] = 1

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL,
                side=PositionSide.SHORT,
                entry_price=data.close,
                strength=strength,
                confidence=min(1.0, confidence),
                reason=f"Trend following short: {'Death cross' if death_cross else 'Trend continuation'}, ADX={adx:.1f}",
                metadata={**indicators, "atr_stop": stop_loss},
            )

        return None

    def _check_pyramid(
        self,
        symbol: str,
        data: MarketData,
        indicators: dict,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Check for pyramid entry opportunity."""
        position = context.current_positions.get(symbol, {})
        if not position:
            return None

        current_pyramids = self._pyramid_counts.get(symbol, 0)
        if current_pyramids >= self.tf_config.max_pyramid_entries:
            return None

        side = position.get("side", "long")
        entry_price = position.get("entry_price", data.close)
        current_pnl_pct = (data.close - entry_price) / entry_price

        if side == "short":
            current_pnl_pct = -current_pnl_pct

        if current_pnl_pct < 0.02:
            return None

        is_strong_trend = indicators.get("is_strong_trend", False)
        trend_up = indicators.get("trend_up", False)
        trend_down = indicators.get("trend_down", False)

        if not is_strong_trend:
            return None

        if side == "long" and trend_up:
            self._pyramid_counts[symbol] = current_pyramids + 1
            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SCALE_IN,
                side=PositionSide.LONG,
                entry_price=data.close,
                strength=0.5,
                confidence=0.7,
                reason=f"Pyramid entry {current_pyramids + 1} on strong uptrend",
                metadata=indicators,
            )

        if side == "short" and trend_down:
            self._pyramid_counts[symbol] = current_pyramids + 1
            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SCALE_IN,
                side=PositionSide.SHORT,
                entry_price=data.close,
                strength=0.5,
                confidence=0.7,
                reason=f"Pyramid entry {current_pyramids + 1} on strong downtrend",
                metadata=indicators,
            )

        return None

    def _check_exit(
        self,
        symbol: str,
        data: MarketData,
        indicators: dict,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Check if position should be exited."""
        if not indicators:
            return None

        position = context.current_positions.get(symbol, {})
        if not position:
            return None

        side = position.get("side", "long")
        entry_price = position.get("entry_price", data.close)

        should_exit, reason = self.should_exit_position(
            symbol, data.close, entry_price,
            PositionSide.LONG if side == "long" else PositionSide.SHORT,
        )

        if should_exit:
            self._pyramid_counts[symbol] = 0
            return self.create_signal(
                symbol=symbol,
                action=SignalAction.CLOSE,
                side=PositionSide.FLAT,
                entry_price=data.close,
                strength=1.0,
                confidence=1.0,
                reason=reason,
                metadata=indicators,
            )

        trend_up = indicators.get("trend_up", False)
        trend_down = indicators.get("trend_down", False)
        death_cross = indicators.get("death_cross", False)
        golden_cross = indicators.get("golden_cross", False)

        if side == "long" and (death_cross or trend_down):
            self._pyramid_counts[symbol] = 0
            return self.create_signal(
                symbol=symbol,
                action=SignalAction.CLOSE,
                side=PositionSide.FLAT,
                entry_price=data.close,
                strength=0.9,
                confidence=0.9,
                reason="Trend reversal: exiting long position",
                metadata=indicators,
            )

        if side == "short" and (golden_cross or trend_up):
            self._pyramid_counts[symbol] = 0
            return self.create_signal(
                symbol=symbol,
                action=SignalAction.CLOSE,
                side=PositionSide.FLAT,
                entry_price=data.close,
                strength=0.9,
                confidence=0.9,
                reason="Trend reversal: exiting short position",
                metadata=indicators,
            )

        return None

    def _update_history(self, symbol: str, data: MarketData) -> None:
        """Update price history."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
            self._high_history[symbol] = []
            self._low_history[symbol] = []

        self._price_history[symbol].append(data.close)
        self._high_history[symbol].append(data.high)
        self._low_history[symbol].append(data.low)

        max_history = 300
        if len(self._price_history[symbol]) > max_history:
            self._price_history[symbol] = self._price_history[symbol][-max_history:]
            self._high_history[symbol] = self._high_history[symbol][-max_history:]
            self._low_history[symbol] = self._low_history[symbol][-max_history:]

    def __repr__(self) -> str:
        """String representation."""
        return f"TrendFollowingStrategy(name={self.name}, state={self._state.value})"

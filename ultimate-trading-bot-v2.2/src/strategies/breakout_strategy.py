"""
Breakout Strategy Module for Ultimate Trading Bot v2.2.

This module implements breakout trading strategies
that capitalize on price breaking through key levels.
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


class BreakoutConfig(StrategyConfig):
    """Breakout strategy configuration."""

    name: str = Field(default="BreakoutStrategy")
    lookback_period: int = Field(default=20, ge=5, le=100)
    breakout_threshold_pct: float = Field(default=1.0, ge=0.1, le=5.0)
    volume_confirmation_mult: float = Field(default=1.5, ge=1.0, le=3.0)
    atr_period: int = Field(default=14, ge=5, le=30)
    atr_stop_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    consolidation_threshold: float = Field(default=0.03, ge=0.01, le=0.1)
    min_consolidation_bars: int = Field(default=5, ge=3, le=20)
    require_volume_confirmation: bool = Field(default=True)
    require_consolidation: bool = Field(default=True)
    use_donchian_channels: bool = Field(default=True)
    filter_false_breakouts: bool = Field(default=True)


class BreakoutStrategy(BaseStrategy):
    """
    Breakout trading strategy.

    Identifies breakout opportunities using:
    - Support/Resistance level breaks
    - Donchian channel breakouts
    - Volume confirmation
    - Consolidation pattern detection
    - False breakout filtering
    """

    def __init__(
        self,
        config: Optional[BreakoutConfig] = None,
    ) -> None:
        """
        Initialize BreakoutStrategy.

        Args:
            config: Breakout strategy configuration
        """
        super().__init__(config or BreakoutConfig())
        self._indicators = TechnicalIndicators()
        self._price_history: dict[str, list[float]] = {}
        self._high_history: dict[str, list[float]] = {}
        self._low_history: dict[str, list[float]] = {}
        self._volume_history: dict[str, list[float]] = {}
        self._open_history: dict[str, list[float]] = {}

        logger.info("BreakoutStrategy initialized")

    @property
    def bo_config(self) -> BreakoutConfig:
        """Get breakout specific config."""
        return self._config  # type: ignore

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate breakout signals.

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

            if len(self._price_history.get(symbol, [])) < self.bo_config.lookback_period + 10:
                continue

            indicators = self.calculate_indicators(symbol, data)

            if symbol in context.current_positions:
                exit_signal = self._check_exit(symbol, data, indicators, context)
                if exit_signal:
                    signals.append(exit_signal)
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
        Calculate breakout indicators.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        closes = self._price_history.get(symbol, [])
        highs = self._high_history.get(symbol, [])
        lows = self._low_history.get(symbol, [])
        volumes = self._volume_history.get(symbol, [])

        if len(closes) < self.bo_config.lookback_period:
            return {}

        lookback = self.bo_config.lookback_period
        recent_high = max(highs[-lookback:-1])
        recent_low = min(lows[-lookback:-1])

        donchian = self._indicators.donchian_channels(
            highs, lows, lookback,
        )
        current_donchian = donchian[-1] if donchian else None

        atr_values = self._indicators.atr(
            highs, lows, closes,
            self.bo_config.atr_period,
        )
        current_atr = atr_values[-1] if atr_values else 0

        avg_volume = sum(volumes[-lookback:]) / lookback if len(volumes) >= lookback else volumes[-1]
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        is_consolidating = self._check_consolidation(highs, lows, closes)

        breakout_up = closes[-1] > recent_high
        breakout_down = closes[-1] < recent_low

        breakout_strength = 0.0
        if breakout_up:
            breakout_strength = (closes[-1] - recent_high) / recent_high * 100
        elif breakout_down:
            breakout_strength = (recent_low - closes[-1]) / recent_low * 100

        is_false_breakout = False
        if self.bo_config.filter_false_breakouts:
            is_false_breakout = self._detect_false_breakout(
                closes, highs, lows, recent_high, recent_low,
            )

        return {
            "recent_high": recent_high,
            "recent_low": recent_low,
            "donchian_upper": current_donchian["upper"] if current_donchian else recent_high,
            "donchian_lower": current_donchian["lower"] if current_donchian else recent_low,
            "donchian_middle": current_donchian["middle"] if current_donchian else (recent_high + recent_low) / 2,
            "atr": current_atr,
            "volume_ratio": volume_ratio,
            "is_consolidating": is_consolidating,
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
            "breakout_strength": breakout_strength,
            "is_false_breakout": is_false_breakout,
            "price": closes[-1],
            "volume_confirmed": volume_ratio >= self.bo_config.volume_confirmation_mult,
        }

    def _generate_entry_signal(
        self,
        symbol: str,
        data: MarketData,
        indicators: dict,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Generate entry signal based on breakout."""
        if not indicators:
            return None

        breakout_up = indicators.get("breakout_up", False)
        breakout_down = indicators.get("breakout_down", False)
        breakout_strength = indicators.get("breakout_strength", 0)
        is_consolidating = indicators.get("is_consolidating", False)
        volume_confirmed = indicators.get("volume_confirmed", False)
        is_false_breakout = indicators.get("is_false_breakout", False)
        atr = indicators.get("atr", 0)

        if not breakout_up and not breakout_down:
            return None

        if breakout_strength < self.bo_config.breakout_threshold_pct:
            return None

        if self.bo_config.require_consolidation and not is_consolidating:
            return None

        if self.bo_config.require_volume_confirmation and not volume_confirmed:
            return None

        if is_false_breakout:
            return None

        confidence = 0.5
        if volume_confirmed:
            confidence += 0.2
        if is_consolidating:
            confidence += 0.15
        confidence += min(0.15, breakout_strength / 10)

        strength = min(1.0, breakout_strength / 5)

        if breakout_up:
            stop_loss = data.close - (atr * self.bo_config.atr_stop_multiplier)
            target = data.close + (atr * self.bo_config.atr_stop_multiplier * 2)

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.BUY,
                side=PositionSide.LONG,
                entry_price=data.close,
                strength=strength,
                confidence=min(1.0, confidence),
                reason=f"Breakout above ${indicators['recent_high']:.2f}, strength={breakout_strength:.1f}%",
                metadata={**indicators, "target": target, "atr_stop": stop_loss},
            )

        if breakout_down:
            stop_loss = data.close + (atr * self.bo_config.atr_stop_multiplier)
            target = data.close - (atr * self.bo_config.atr_stop_multiplier * 2)

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL,
                side=PositionSide.SHORT,
                entry_price=data.close,
                strength=strength,
                confidence=min(1.0, confidence),
                reason=f"Breakout below ${indicators['recent_low']:.2f}, strength={breakout_strength:.1f}%",
                metadata={**indicators, "target": target, "atr_stop": stop_loss},
            )

        return None

    def _check_consolidation(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> bool:
        """Check if price is in consolidation."""
        min_bars = self.bo_config.min_consolidation_bars
        if len(closes) < min_bars:
            return False

        recent_highs = highs[-min_bars:-1]
        recent_lows = lows[-min_bars:-1]

        price_range = max(recent_highs) - min(recent_lows)
        avg_price = sum(closes[-min_bars:-1]) / (min_bars - 1)

        range_pct = price_range / avg_price if avg_price > 0 else 1

        return range_pct <= self.bo_config.consolidation_threshold

    def _detect_false_breakout(
        self,
        closes: list[float],
        highs: list[float],
        lows: list[float],
        recent_high: float,
        recent_low: float,
    ) -> bool:
        """Detect potential false breakout."""
        if len(closes) < 3:
            return False

        if closes[-1] > recent_high:
            if highs[-2] > recent_high and closes[-2] < recent_high:
                return True

            if closes[-1] < highs[-1] * 0.98:
                return True

        if closes[-1] < recent_low:
            if lows[-2] < recent_low and closes[-2] > recent_low:
                return True

            if closes[-1] > lows[-1] * 1.02:
                return True

        return False

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

        donchian_middle = indicators.get("donchian_middle", data.close)

        if side == "long" and data.close < donchian_middle:
            return self.create_signal(
                symbol=symbol,
                action=SignalAction.CLOSE,
                side=PositionSide.FLAT,
                entry_price=data.close,
                strength=0.8,
                confidence=0.85,
                reason="Price fell below Donchian middle",
                metadata=indicators,
            )

        if side == "short" and data.close > donchian_middle:
            return self.create_signal(
                symbol=symbol,
                action=SignalAction.CLOSE,
                side=PositionSide.FLAT,
                entry_price=data.close,
                strength=0.8,
                confidence=0.85,
                reason="Price rose above Donchian middle",
                metadata=indicators,
            )

        return None

    def _update_history(self, symbol: str, data: MarketData) -> None:
        """Update price history."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
            self._high_history[symbol] = []
            self._low_history[symbol] = []
            self._volume_history[symbol] = []
            self._open_history[symbol] = []

        self._price_history[symbol].append(data.close)
        self._high_history[symbol].append(data.high)
        self._low_history[symbol].append(data.low)
        self._volume_history[symbol].append(data.volume)
        self._open_history[symbol].append(data.open)

        max_history = 200
        if len(self._price_history[symbol]) > max_history:
            self._price_history[symbol] = self._price_history[symbol][-max_history:]
            self._high_history[symbol] = self._high_history[symbol][-max_history:]
            self._low_history[symbol] = self._low_history[symbol][-max_history:]
            self._volume_history[symbol] = self._volume_history[symbol][-max_history:]
            self._open_history[symbol] = self._open_history[symbol][-max_history:]

    def __repr__(self) -> str:
        """String representation."""
        return f"BreakoutStrategy(name={self.name}, state={self._state.value})"

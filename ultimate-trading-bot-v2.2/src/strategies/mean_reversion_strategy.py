"""
Mean Reversion Strategy Module for Ultimate Trading Bot v2.2.

This module implements mean reversion trading strategies
that capitalize on price returning to average levels.
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


class MeanReversionConfig(StrategyConfig):
    """Mean reversion strategy configuration."""

    name: str = Field(default="MeanReversionStrategy")
    bb_period: int = Field(default=20, ge=10, le=50)
    bb_std: float = Field(default=2.0, ge=1.0, le=3.0)
    rsi_period: int = Field(default=14, ge=5, le=30)
    rsi_oversold: float = Field(default=30.0, ge=10.0, le=40.0)
    rsi_overbought: float = Field(default=70.0, ge=60.0, le=90.0)
    zscore_threshold: float = Field(default=2.0, ge=1.0, le=4.0)
    mean_period: int = Field(default=20, ge=5, le=100)
    require_rsi_confirmation: bool = Field(default=True)
    require_bb_touch: bool = Field(default=True)
    exit_at_mean: bool = Field(default=True)
    max_holding_days: int = Field(default=5, ge=1, le=30)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy.

    Identifies oversold/overbought conditions using:
    - Bollinger Bands for price extremes
    - RSI for momentum confirmation
    - Z-Score for statistical deviation
    - Mean crossing for exit signals
    """

    def __init__(
        self,
        config: Optional[MeanReversionConfig] = None,
    ) -> None:
        """
        Initialize MeanReversionStrategy.

        Args:
            config: Mean reversion strategy configuration
        """
        super().__init__(config or MeanReversionConfig())
        self._indicators = TechnicalIndicators()
        self._price_history: dict[str, list[float]] = {}
        self._high_history: dict[str, list[float]] = {}
        self._low_history: dict[str, list[float]] = {}

        logger.info("MeanReversionStrategy initialized")

    @property
    def mr_config(self) -> MeanReversionConfig:
        """Get mean reversion specific config."""
        return self._config  # type: ignore

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate mean reversion signals.

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

            if len(self._price_history.get(symbol, [])) < self.mr_config.mean_period + 10:
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
        Calculate mean reversion indicators.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        closes = self._price_history.get(symbol, [])

        if len(closes) < self.mr_config.mean_period:
            return {}

        bb_results = self._indicators.bollinger_bands(
            closes,
            self.mr_config.bb_period,
            self.mr_config.bb_std,
        )
        current_bb = bb_results[-1] if bb_results else None

        rsi_values = self._indicators.rsi(closes, self.mr_config.rsi_period)
        current_rsi = rsi_values[-1] if rsi_values else 50

        mean = sum(closes[-self.mr_config.mean_period:]) / self.mr_config.mean_period
        variance = sum(
            (p - mean) ** 2 for p in closes[-self.mr_config.mean_period:]
        ) / self.mr_config.mean_period
        std = variance ** 0.5
        zscore = (closes[-1] - mean) / std if std > 0 else 0

        price_vs_mean_pct = (closes[-1] - mean) / mean * 100

        at_lower_band = False
        at_upper_band = False
        if current_bb:
            at_lower_band = closes[-1] <= current_bb.lower
            at_upper_band = closes[-1] >= current_bb.upper

        return {
            "bb_upper": current_bb.upper if current_bb else 0,
            "bb_middle": current_bb.middle if current_bb else mean,
            "bb_lower": current_bb.lower if current_bb else 0,
            "bb_percent_b": current_bb.percent_b if current_bb else 0.5,
            "bb_bandwidth": current_bb.bandwidth if current_bb else 0,
            "rsi": current_rsi,
            "mean": mean,
            "std": std,
            "zscore": zscore,
            "price": closes[-1],
            "price_vs_mean_pct": price_vs_mean_pct,
            "at_lower_band": at_lower_band,
            "at_upper_band": at_upper_band,
            "is_oversold": current_rsi < self.mr_config.rsi_oversold,
            "is_overbought": current_rsi > self.mr_config.rsi_overbought,
        }

    def _generate_entry_signal(
        self,
        symbol: str,
        data: MarketData,
        indicators: dict,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Generate entry signal based on mean reversion."""
        if not indicators:
            return None

        zscore = indicators.get("zscore", 0)
        rsi = indicators.get("rsi", 50)
        at_lower_band = indicators.get("at_lower_band", False)
        at_upper_band = indicators.get("at_upper_band", False)
        is_oversold = indicators.get("is_oversold", False)
        is_overbought = indicators.get("is_overbought", False)

        long_conditions = []
        short_conditions = []

        if zscore < -self.mr_config.zscore_threshold:
            long_conditions.append("zscore_extreme")

        if zscore > self.mr_config.zscore_threshold:
            short_conditions.append("zscore_extreme")

        if self.mr_config.require_bb_touch:
            if at_lower_band:
                long_conditions.append("bb_touch")
            if at_upper_band:
                short_conditions.append("bb_touch")

        if self.mr_config.require_rsi_confirmation:
            if is_oversold:
                long_conditions.append("rsi_oversold")
            if is_overbought:
                short_conditions.append("rsi_overbought")

        required_conditions = 1
        if self.mr_config.require_bb_touch:
            required_conditions += 1
        if self.mr_config.require_rsi_confirmation:
            required_conditions += 1

        if len(long_conditions) >= required_conditions:
            strength = min(1.0, len(long_conditions) / 3)
            confidence = 0.5 + abs(zscore) / 10 + (0.2 if is_oversold else 0)

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.BUY,
                side=PositionSide.LONG,
                entry_price=data.close,
                strength=strength,
                confidence=min(1.0, confidence),
                reason=f"Mean reversion long: Z={zscore:.2f}, RSI={rsi:.1f}",
                metadata=indicators,
            )

        if len(short_conditions) >= required_conditions:
            strength = min(1.0, len(short_conditions) / 3)
            confidence = 0.5 + abs(zscore) / 10 + (0.2 if is_overbought else 0)

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL,
                side=PositionSide.SHORT,
                entry_price=data.close,
                strength=strength,
                confidence=min(1.0, confidence),
                reason=f"Mean reversion short: Z={zscore:.2f}, RSI={rsi:.1f}",
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

        zscore = indicators.get("zscore", 0)
        price = indicators.get("price", data.close)
        mean = indicators.get("mean", price)

        should_exit, reason = self.should_exit_position(
            symbol, price, entry_price,
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

        if self.mr_config.exit_at_mean:
            if side == "long" and price >= mean:
                return self.create_signal(
                    symbol=symbol,
                    action=SignalAction.CLOSE,
                    side=PositionSide.FLAT,
                    entry_price=data.close,
                    strength=0.8,
                    confidence=0.9,
                    reason=f"Price reached mean (target): {mean:.2f}",
                    metadata=indicators,
                )

            if side == "short" and price <= mean:
                return self.create_signal(
                    symbol=symbol,
                    action=SignalAction.CLOSE,
                    side=PositionSide.FLAT,
                    entry_price=data.close,
                    strength=0.8,
                    confidence=0.9,
                    reason=f"Price reached mean (target): {mean:.2f}",
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

        max_history = 200
        if len(self._price_history[symbol]) > max_history:
            self._price_history[symbol] = self._price_history[symbol][-max_history:]
            self._high_history[symbol] = self._high_history[symbol][-max_history:]
            self._low_history[symbol] = self._low_history[symbol][-max_history:]

    def __repr__(self) -> str:
        """String representation."""
        return f"MeanReversionStrategy(name={self.name}, state={self._state.value})"

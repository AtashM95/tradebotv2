"""
Momentum Strategy Module for Ultimate Trading Bot v2.2.

This module implements momentum-based trading strategies
that capitalize on price continuation patterns.
"""

import logging
from typing import Optional

from pydantic import BaseModel, Field

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


class MomentumConfig(StrategyConfig):
    """Momentum strategy configuration."""

    name: str = Field(default="MomentumStrategy")
    rsi_period: int = Field(default=14, ge=5, le=30)
    rsi_overbought: float = Field(default=70.0, ge=60.0, le=90.0)
    rsi_oversold: float = Field(default=30.0, ge=10.0, le=40.0)
    macd_fast: int = Field(default=12, ge=5, le=20)
    macd_slow: int = Field(default=26, ge=15, le=50)
    macd_signal: int = Field(default=9, ge=5, le=15)
    roc_period: int = Field(default=10, ge=5, le=30)
    roc_threshold: float = Field(default=2.0, ge=0.5, le=10.0)
    adx_period: int = Field(default=14, ge=5, le=30)
    adx_threshold: float = Field(default=25.0, ge=15.0, le=50.0)
    require_trend_confirmation: bool = Field(default=True)
    use_volume_confirmation: bool = Field(default=True)
    volume_multiplier: float = Field(default=1.5, ge=1.0, le=3.0)


class MomentumStrategy(BaseStrategy):
    """
    Momentum trading strategy.

    Identifies and trades momentum using:
    - RSI for momentum strength
    - MACD for momentum direction
    - Rate of Change (ROC) for momentum magnitude
    - ADX for trend strength confirmation
    - Volume confirmation for signal quality
    """

    def __init__(
        self,
        config: Optional[MomentumConfig] = None,
    ) -> None:
        """
        Initialize MomentumStrategy.

        Args:
            config: Momentum strategy configuration
        """
        super().__init__(config or MomentumConfig())
        self._indicators = TechnicalIndicators()
        self._price_history: dict[str, list[float]] = {}
        self._volume_history: dict[str, list[float]] = {}
        self._high_history: dict[str, list[float]] = {}
        self._low_history: dict[str, list[float]] = {}

        logger.info("MomentumStrategy initialized")

    @property
    def momentum_config(self) -> MomentumConfig:
        """Get momentum-specific config."""
        return self._config  # type: ignore

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate momentum signals.

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

            if len(self._price_history.get(symbol, [])) < 50:
                continue

            indicators = self.calculate_indicators(symbol, data)

            signal = self._generate_signal(symbol, data, indicators, context)
            if signal:
                signals.append(signal)

        return signals

    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict:
        """
        Calculate momentum indicators.

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

        if len(closes) < 30:
            return {}

        rsi_values = self._indicators.rsi(closes, self.momentum_config.rsi_period)
        current_rsi = rsi_values[-1] if rsi_values else 50

        macd_results = self._indicators.macd(
            closes,
            self.momentum_config.macd_fast,
            self.momentum_config.macd_slow,
            self.momentum_config.macd_signal,
        )
        current_macd = macd_results[-1] if macd_results else None

        if len(closes) > self.momentum_config.roc_period:
            roc = (
                (closes[-1] - closes[-self.momentum_config.roc_period - 1]) /
                closes[-self.momentum_config.roc_period - 1] * 100
            )
        else:
            roc = 0.0

        adx_values = self._indicators.adx(
            highs, lows, closes,
            self.momentum_config.adx_period,
        )
        current_adx = adx_values[-1] if adx_values else 0

        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

        sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]
        sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else closes[-1]

        return {
            "rsi": current_rsi,
            "macd_line": current_macd.macd_line if current_macd else 0,
            "macd_signal": current_macd.signal_line if current_macd else 0,
            "macd_histogram": current_macd.histogram if current_macd else 0,
            "roc": roc,
            "adx": current_adx,
            "volume_ratio": volume_ratio,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "price": closes[-1],
            "trend_bullish": sma_20 > sma_50,
            "trend_bearish": sma_20 < sma_50,
        }

    def _generate_signal(
        self,
        symbol: str,
        data: MarketData,
        indicators: dict,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Generate trading signal from indicators."""
        if not indicators:
            return None

        rsi = indicators.get("rsi", 50)
        macd_hist = indicators.get("macd_histogram", 0)
        roc = indicators.get("roc", 0)
        adx = indicators.get("adx", 0)
        volume_ratio = indicators.get("volume_ratio", 1.0)
        trend_bullish = indicators.get("trend_bullish", False)
        trend_bearish = indicators.get("trend_bearish", False)

        if symbol in context.current_positions:
            return None

        bullish_score = 0.0
        bearish_score = 0.0

        if rsi > 50 and rsi < self.momentum_config.rsi_overbought:
            bullish_score += 0.25
        elif rsi < 50 and rsi > self.momentum_config.rsi_oversold:
            bearish_score += 0.25

        if macd_hist > 0:
            bullish_score += 0.25
        elif macd_hist < 0:
            bearish_score += 0.25

        if roc > self.momentum_config.roc_threshold:
            bullish_score += 0.25
        elif roc < -self.momentum_config.roc_threshold:
            bearish_score += 0.25

        if adx > self.momentum_config.adx_threshold:
            if trend_bullish:
                bullish_score += 0.25
            elif trend_bearish:
                bearish_score += 0.25

        if self.momentum_config.use_volume_confirmation:
            if volume_ratio >= self.momentum_config.volume_multiplier:
                bullish_score *= 1.2
                bearish_score *= 1.2

        if self.momentum_config.require_trend_confirmation:
            if bullish_score > bearish_score and not trend_bullish:
                bullish_score *= 0.5
            if bearish_score > bullish_score and not trend_bearish:
                bearish_score *= 0.5

        min_score = 0.5

        if bullish_score >= min_score and bullish_score > bearish_score:
            return self.create_signal(
                symbol=symbol,
                action=SignalAction.BUY,
                side=PositionSide.LONG,
                entry_price=data.close,
                strength=min(1.0, bullish_score),
                confidence=min(1.0, bullish_score * 0.8 + 0.2),
                reason=f"Momentum bullish: RSI={rsi:.1f}, ROC={roc:.1f}%, ADX={adx:.1f}",
                metadata=indicators,
            )

        elif bearish_score >= min_score and bearish_score > bullish_score:
            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL,
                side=PositionSide.SHORT,
                entry_price=data.close,
                strength=min(1.0, bearish_score),
                confidence=min(1.0, bearish_score * 0.8 + 0.2),
                reason=f"Momentum bearish: RSI={rsi:.1f}, ROC={roc:.1f}%, ADX={adx:.1f}",
                metadata=indicators,
            )

        return None

    def _update_history(self, symbol: str, data: MarketData) -> None:
        """Update price and volume history."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
            self._volume_history[symbol] = []
            self._high_history[symbol] = []
            self._low_history[symbol] = []

        self._price_history[symbol].append(data.close)
        self._volume_history[symbol].append(data.volume)
        self._high_history[symbol].append(data.high)
        self._low_history[symbol].append(data.low)

        max_history = 200
        if len(self._price_history[symbol]) > max_history:
            self._price_history[symbol] = self._price_history[symbol][-max_history:]
            self._volume_history[symbol] = self._volume_history[symbol][-max_history:]
            self._high_history[symbol] = self._high_history[symbol][-max_history:]
            self._low_history[symbol] = self._low_history[symbol][-max_history:]

    def __repr__(self) -> str:
        """String representation."""
        return f"MomentumStrategy(name={self.name}, state={self._state.value})"

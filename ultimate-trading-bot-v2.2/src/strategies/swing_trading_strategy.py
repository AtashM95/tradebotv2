"""
Swing Trading Strategy Module for Ultimate Trading Bot v2.2.

This module implements a swing trading strategy that captures
multi-day price movements based on technical patterns.
"""

import logging
from datetime import datetime, timedelta
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
from src.analysis.pattern_recognition import PatternRecognition
from src.utils.helpers import generate_uuid


logger = logging.getLogger(__name__)


class SwingTradingConfig(StrategyConfig):
    """Configuration for swing trading strategy."""

    name: str = Field(default="Swing Trading Strategy")
    description: str = Field(
        default="Multi-day swing trading based on technical patterns"
    )

    min_holding_days: int = Field(default=2, ge=1, le=10)
    max_holding_days: int = Field(default=15, ge=5, le=60)
    target_holding_days: int = Field(default=5, ge=2, le=30)

    ema_fast: int = Field(default=9, ge=5, le=15)
    ema_slow: int = Field(default=21, ge=15, le=30)
    ema_trend: int = Field(default=50, ge=30, le=100)

    rsi_period: int = Field(default=14, ge=7, le=21)
    rsi_overbought: float = Field(default=70.0, ge=60.0, le=85.0)
    rsi_oversold: float = Field(default=30.0, ge=15.0, le=40.0)
    rsi_divergence_enabled: bool = Field(default=True)

    macd_fast: int = Field(default=12, ge=8, le=16)
    macd_slow: int = Field(default=26, ge=20, le=32)
    macd_signal: int = Field(default=9, ge=5, le=12)

    atr_period: int = Field(default=14, ge=7, le=21)
    atr_stop_multiplier: float = Field(default=2.0, ge=1.0, le=4.0)
    atr_target_multiplier: float = Field(default=3.0, ge=1.5, le=6.0)

    support_resistance_lookback: int = Field(default=50, ge=20, le=100)

    use_patterns: bool = Field(default=True)
    use_fibonacci: bool = Field(default=True)

    min_risk_reward: float = Field(default=2.0, ge=1.5, le=5.0)


class SwingTrade(BaseModel):
    """Model for tracking swing trades."""

    symbol: str
    entry_date: datetime
    entry_price: float
    side: str
    stop_loss: float
    take_profit: float
    target_days: int
    pattern_type: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


class SwingTradingStrategy(BaseStrategy):
    """
    Swing trading strategy for multi-day movements.

    Features:
    - Pattern-based entry signals
    - Fibonacci retracements
    - Support/resistance levels
    - RSI divergence detection
    - Time-based exit management
    - Risk/reward optimization
    """

    def __init__(
        self,
        config: Optional[SwingTradingConfig] = None,
    ) -> None:
        """
        Initialize SwingTradingStrategy.

        Args:
            config: Swing trading configuration
        """
        config = config or SwingTradingConfig()
        super().__init__(config)

        self._indicators = TechnicalIndicators()
        self._patterns = PatternRecognition()
        self._swing_config = config

        self._active_swings: dict[str, SwingTrade] = {}
        self._swing_history: list[SwingTrade] = []

        logger.info(f"SwingTradingStrategy initialized: {self.name}")

    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict[str, Any]:
        """
        Calculate swing trading indicators.

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

        if len(closes) < 60:
            return {}

        ema_fast = self._indicators.ema(closes, self._swing_config.ema_fast)
        ema_slow = self._indicators.ema(closes, self._swing_config.ema_slow)
        ema_trend = self._indicators.ema(closes, self._swing_config.ema_trend)

        rsi = self._indicators.rsi(closes, self._swing_config.rsi_period)

        macd = self._indicators.macd(
            closes,
            self._swing_config.macd_fast,
            self._swing_config.macd_slow,
            self._swing_config.macd_signal,
        )

        atr = self._indicators.atr(
            highs, lows, closes, self._swing_config.atr_period
        )

        support, resistance = self._find_support_resistance(
            highs, lows, closes,
            self._swing_config.support_resistance_lookback,
        )

        fib_levels = self._calculate_fibonacci_levels(highs, lows)

        trend = self._determine_trend(
            closes[-1], ema_fast[-1], ema_slow[-1], ema_trend[-1]
        )

        divergence = self._detect_rsi_divergence(closes, rsi)

        patterns_detected = []
        if self._swing_config.use_patterns:
            patterns_detected = self._detect_swing_patterns(
                data.opens, highs, lows, closes
            )

        return {
            "ema_fast": ema_fast[-1],
            "ema_slow": ema_slow[-1],
            "ema_trend": ema_trend[-1],
            "rsi": rsi[-1] if rsi else 50.0,
            "macd_line": macd[-1].macd_line if macd else 0,
            "macd_signal": macd[-1].signal_line if macd else 0,
            "macd_histogram": macd[-1].histogram if macd else 0,
            "atr": atr[-1] if atr else 0,
            "support": support,
            "resistance": resistance,
            "fibonacci": fib_levels,
            "trend": trend,
            "divergence": divergence,
            "patterns": patterns_detected,
            "current_price": closes[-1],
            "high_20": max(highs[-20:]),
            "low_20": min(lows[-20:]),
        }

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate swing trading opportunities.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of swing trading signals
        """
        signals: list[StrategySignal] = []

        for symbol in self.config.symbols:
            if symbol not in market_data:
                continue

            data = market_data[symbol]

            exit_signal = self._check_swing_exit(symbol, data, context)
            if exit_signal:
                signals.append(exit_signal)
                continue

            if symbol in self._active_swings:
                continue

            indicators = self.calculate_indicators(symbol, data)
            if not indicators:
                continue

            signal = self._generate_swing_signal(symbol, indicators, context)

            if signal:
                signals.append(signal)
                self._record_swing_entry(symbol, signal, indicators, context)

        return signals

    def _generate_swing_signal(
        self,
        symbol: str,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Generate swing trading signal."""
        current_price = indicators["current_price"]
        trend = indicators["trend"]
        rsi = indicators["rsi"]
        macd_histogram = indicators["macd_histogram"]
        divergence = indicators["divergence"]
        support = indicators["support"]
        resistance = indicators["resistance"]
        atr = indicators["atr"]
        patterns = indicators["patterns"]
        fibonacci = indicators["fibonacci"]

        buy_score = 0.0
        sell_score = 0.0
        reasons: list[str] = []

        if trend == "bullish":
            buy_score += 0.2
            reasons.append("bullish_trend")
        elif trend == "bearish":
            sell_score += 0.2
            reasons.append("bearish_trend")

        if rsi < self._swing_config.rsi_oversold:
            buy_score += 0.15
            reasons.append(f"rsi_oversold_{rsi:.1f}")
        elif rsi > self._swing_config.rsi_overbought:
            sell_score += 0.15
            reasons.append(f"rsi_overbought_{rsi:.1f}")

        if macd_histogram > 0 and macd_histogram > indicators.get("prev_macd_hist", 0):
            buy_score += 0.1
        elif macd_histogram < 0 and macd_histogram < indicators.get("prev_macd_hist", 0):
            sell_score += 0.1

        if divergence == "bullish":
            buy_score += 0.2
            reasons.append("bullish_divergence")
        elif divergence == "bearish":
            sell_score += 0.2
            reasons.append("bearish_divergence")

        if support and current_price <= support * 1.02:
            buy_score += 0.15
            reasons.append(f"near_support_{support:.2f}")
        if resistance and current_price >= resistance * 0.98:
            sell_score += 0.15
            reasons.append(f"near_resistance_{resistance:.2f}")

        if self._swing_config.use_fibonacci and fibonacci:
            for level_name, level_price in fibonacci.items():
                if abs(current_price - level_price) / current_price < 0.01:
                    if level_name in ["38.2%", "50%", "61.8%"]:
                        if trend == "bullish":
                            buy_score += 0.1
                            reasons.append(f"fib_{level_name}_support")
                        else:
                            sell_score += 0.1
                            reasons.append(f"fib_{level_name}_resistance")

        if patterns:
            for pattern in patterns:
                if pattern.get("type") in ["double_bottom", "inverse_head_shoulders", "bullish_engulfing"]:
                    buy_score += 0.15
                    reasons.append(pattern["type"])
                elif pattern.get("type") in ["double_top", "head_shoulders", "bearish_engulfing"]:
                    sell_score += 0.15
                    reasons.append(pattern["type"])

        min_score = 0.5

        if buy_score >= min_score and buy_score > sell_score:
            stop_loss = current_price - (atr * self._swing_config.atr_stop_multiplier)
            take_profit = current_price + (atr * self._swing_config.atr_target_multiplier)

            if support:
                stop_loss = min(stop_loss, support * 0.995)

            risk = current_price - stop_loss
            reward = take_profit - current_price
            risk_reward = reward / risk if risk > 0 else 0

            if risk_reward < self._swing_config.min_risk_reward:
                return None

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.BUY,
                side=SignalSide.LONG,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strength=buy_score,
                confidence=min(0.85, buy_score + 0.15),
                reason=f"Swing buy: {', '.join(reasons[:3])}",
                metadata={
                    "indicators": indicators,
                    "strategy_type": "swing_trading",
                    "expected_hold_days": self._swing_config.target_holding_days,
                    "risk_reward": risk_reward,
                    "patterns": [p["type"] for p in patterns] if patterns else [],
                },
            )

        elif sell_score >= min_score and sell_score > buy_score:
            stop_loss = current_price + (atr * self._swing_config.atr_stop_multiplier)
            take_profit = current_price - (atr * self._swing_config.atr_target_multiplier)

            if resistance:
                stop_loss = max(stop_loss, resistance * 1.005)

            risk = stop_loss - current_price
            reward = current_price - take_profit
            risk_reward = reward / risk if risk > 0 else 0

            if risk_reward < self._swing_config.min_risk_reward:
                return None

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL,
                side=SignalSide.SHORT,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strength=sell_score,
                confidence=min(0.85, sell_score + 0.15),
                reason=f"Swing sell: {', '.join(reasons[:3])}",
                metadata={
                    "indicators": indicators,
                    "strategy_type": "swing_trading",
                    "expected_hold_days": self._swing_config.target_holding_days,
                    "risk_reward": risk_reward,
                    "patterns": [p["type"] for p in patterns] if patterns else [],
                },
            )

        return None

    def _check_swing_exit(
        self,
        symbol: str,
        data: MarketData,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Check if swing trade should exit."""
        if symbol not in self._active_swings:
            return None

        swing = self._active_swings[symbol]
        current_price = data.closes[-1]
        days_held = (context.timestamp - swing.entry_date).days

        exit_reason = None

        if swing.side == "long":
            if current_price <= swing.stop_loss:
                exit_reason = "stop_loss"
            elif current_price >= swing.take_profit:
                exit_reason = "take_profit"
        else:
            if current_price >= swing.stop_loss:
                exit_reason = "stop_loss"
            elif current_price <= swing.take_profit:
                exit_reason = "take_profit"

        if days_held >= self._swing_config.max_holding_days:
            exit_reason = "max_holding_days"

        if days_held >= self._swing_config.target_holding_days:
            pnl_pct = (
                (current_price - swing.entry_price) / swing.entry_price
                if swing.side == "long"
                else (swing.entry_price - current_price) / swing.entry_price
            )

            if pnl_pct > 0.01:
                exit_reason = "target_days_profit"

        if exit_reason:
            del self._active_swings[symbol]
            self._swing_history.append(swing)

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL if swing.side == "long" else SignalAction.BUY,
                side=SignalSide.FLAT,
                entry_price=current_price,
                strength=1.0,
                confidence=0.95,
                reason=f"Swing exit: {exit_reason}",
                metadata={
                    "exit_reason": exit_reason,
                    "days_held": days_held,
                    "entry_price": swing.entry_price,
                    "strategy_type": "swing_trading",
                },
            )

        return None

    def _record_swing_entry(
        self,
        symbol: str,
        signal: StrategySignal,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> None:
        """Record new swing trade entry."""
        swing = SwingTrade(
            symbol=symbol,
            entry_date=context.timestamp,
            entry_price=signal.entry_price or indicators["current_price"],
            side="long" if signal.side == SignalSide.LONG else "short",
            stop_loss=signal.stop_loss or 0,
            take_profit=signal.take_profit or 0,
            target_days=self._swing_config.target_holding_days,
            pattern_type=indicators.get("patterns", [{}])[0].get("type") if indicators.get("patterns") else None,
        )

        self._active_swings[symbol] = swing

    def _find_support_resistance(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        lookback: int,
    ) -> tuple[Optional[float], Optional[float]]:
        """Find support and resistance levels."""
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        current_price = closes[-1]

        swing_highs: list[float] = []
        swing_lows: list[float] = []

        for i in range(2, len(recent_highs) - 2):
            if (recent_highs[i] > recent_highs[i - 1] and
                recent_highs[i] > recent_highs[i - 2] and
                recent_highs[i] > recent_highs[i + 1] and
                recent_highs[i] > recent_highs[i + 2]):
                swing_highs.append(recent_highs[i])

            if (recent_lows[i] < recent_lows[i - 1] and
                recent_lows[i] < recent_lows[i - 2] and
                recent_lows[i] < recent_lows[i + 1] and
                recent_lows[i] < recent_lows[i + 2]):
                swing_lows.append(recent_lows[i])

        support = None
        resistance = None

        below_price = [l for l in swing_lows if l < current_price]
        if below_price:
            support = max(below_price)

        above_price = [h for h in swing_highs if h > current_price]
        if above_price:
            resistance = min(above_price)

        return support, resistance

    def _calculate_fibonacci_levels(
        self,
        highs: list[float],
        lows: list[float],
    ) -> dict[str, float]:
        """Calculate Fibonacci retracement levels."""
        lookback = min(50, len(highs))
        recent_high = max(highs[-lookback:])
        recent_low = min(lows[-lookback:])

        diff = recent_high - recent_low

        return {
            "0%": recent_high,
            "23.6%": recent_high - diff * 0.236,
            "38.2%": recent_high - diff * 0.382,
            "50%": recent_high - diff * 0.5,
            "61.8%": recent_high - diff * 0.618,
            "78.6%": recent_high - diff * 0.786,
            "100%": recent_low,
        }

    def _determine_trend(
        self,
        price: float,
        ema_fast: float,
        ema_slow: float,
        ema_trend: float,
    ) -> str:
        """Determine overall trend direction."""
        if price > ema_fast > ema_slow > ema_trend:
            return "bullish"
        elif price < ema_fast < ema_slow < ema_trend:
            return "bearish"
        elif price > ema_trend:
            return "weak_bullish"
        elif price < ema_trend:
            return "weak_bearish"
        else:
            return "neutral"

    def _detect_rsi_divergence(
        self,
        closes: list[float],
        rsi: list[float],
    ) -> Optional[str]:
        """Detect RSI divergence."""
        if len(closes) < 20 or len(rsi) < 20:
            return None

        price_lows: list[tuple[int, float]] = []
        price_highs: list[tuple[int, float]] = []

        for i in range(2, min(20, len(closes) - 2)):
            idx = -i
            if closes[idx] < closes[idx - 1] and closes[idx] < closes[idx + 1]:
                price_lows.append((idx, closes[idx]))
            if closes[idx] > closes[idx - 1] and closes[idx] > closes[idx + 1]:
                price_highs.append((idx, closes[idx]))

        if len(price_lows) >= 2:
            if (price_lows[0][1] < price_lows[1][1] and
                rsi[price_lows[0][0]] > rsi[price_lows[1][0]]):
                return "bullish"

        if len(price_highs) >= 2:
            if (price_highs[0][1] > price_highs[1][1] and
                rsi[price_highs[0][0]] < rsi[price_highs[1][0]]):
                return "bearish"

        return None

    def _detect_swing_patterns(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> list[dict]:
        """Detect chart patterns relevant to swing trading."""
        patterns: list[dict] = []

        if len(closes) >= 20:
            lows_20 = lows[-20:]
            min_idx_1 = lows_20.index(min(lows_20))

            if min_idx_1 > 3:
                lows_before = lows_20[:min_idx_1 - 2]
                if lows_before:
                    min_idx_2 = lows_before.index(min(lows_before))

                    if abs(lows_20[min_idx_1] - lows_before[min_idx_2]) / lows_20[min_idx_1] < 0.02:
                        patterns.append({
                            "type": "double_bottom",
                            "confidence": 0.7,
                        })

            highs_20 = highs[-20:]
            max_idx_1 = highs_20.index(max(highs_20))

            if max_idx_1 > 3:
                highs_before = highs_20[:max_idx_1 - 2]
                if highs_before:
                    max_idx_2 = highs_before.index(max(highs_before))

                    if abs(highs_20[max_idx_1] - highs_before[max_idx_2]) / highs_20[max_idx_1] < 0.02:
                        patterns.append({
                            "type": "double_top",
                            "confidence": 0.7,
                        })

        if len(closes) >= 3:
            if (closes[-2] < opens[-2] and
                closes[-1] > opens[-1] and
                closes[-1] > opens[-2] and
                opens[-1] < closes[-2]):
                patterns.append({
                    "type": "bullish_engulfing",
                    "confidence": 0.65,
                })

            if (closes[-2] > opens[-2] and
                closes[-1] < opens[-1] and
                closes[-1] < opens[-2] and
                opens[-1] > closes[-2]):
                patterns.append({
                    "type": "bearish_engulfing",
                    "confidence": 0.65,
                })

        return patterns

    def get_active_swings(self) -> dict[str, SwingTrade]:
        """Get active swing trades."""
        return self._active_swings.copy()

    def get_swing_history(self, limit: int = 50) -> list[SwingTrade]:
        """Get swing trade history."""
        return self._swing_history[-limit:]

    def __repr__(self) -> str:
        """String representation."""
        return f"SwingTradingStrategy(name={self.name}, active_swings={len(self._active_swings)})"

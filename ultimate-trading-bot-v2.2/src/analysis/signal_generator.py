"""
Signal Generator Module for Ultimate Trading Bot v2.2.

This module generates trading signals based on technical analysis,
combining multiple indicators and patterns.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.analysis.technical_indicators import TechnicalIndicators, MACDResult
from src.analysis.trend_analysis import TrendAnalyzer, TrendDirection, TrendStrength
from src.analysis.pattern_recognition import PatternRecognition, PatternType
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Signal type enumeration."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class SignalSource(str, Enum):
    """Signal source enumeration."""

    TREND = "trend"
    MOMENTUM = "momentum"
    PATTERN = "pattern"
    VOLUME = "volume"
    COMBINED = "combined"
    MA_CROSSOVER = "ma_crossover"
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER = "bollinger"


class SignalTimeframe(str, Enum):
    """Signal timeframe enumeration."""

    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"
    LONG_TERM = "long_term"


class TradingSignal(BaseModel):
    """Trading signal model."""

    signal_id: str = Field(default_factory=generate_uuid)
    symbol: str
    signal_type: SignalType
    source: SignalSource
    timeframe: SignalTimeframe
    strength: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    supporting_signals: list[str] = Field(default_factory=list)
    conflicting_signals: list[str] = Field(default_factory=list)
    notes: str = Field(default="")
    created_at: datetime = Field(default_factory=now_utc)
    expires_at: Optional[datetime] = None


class SignalSummary(BaseModel):
    """Summary of all signals for a symbol."""

    symbol: str
    overall_signal: SignalType
    overall_score: float = Field(ge=-100, le=100)
    buy_signals: int = Field(default=0)
    sell_signals: int = Field(default=0)
    neutral_signals: int = Field(default=0)
    trend_alignment: str = Field(default="")
    key_levels: dict = Field(default_factory=dict)
    recommendation: str = Field(default="")
    created_at: datetime = Field(default_factory=now_utc)


class SignalGeneratorConfig(BaseModel):
    """Configuration for signal generator."""

    rsi_oversold: float = Field(default=30.0, ge=10.0, le=40.0)
    rsi_overbought: float = Field(default=70.0, ge=60.0, le=90.0)
    rsi_period: int = Field(default=14, ge=5, le=30)
    macd_fast: int = Field(default=12, ge=5, le=20)
    macd_slow: int = Field(default=26, ge=15, le=50)
    macd_signal: int = Field(default=9, ge=5, le=15)
    bb_period: int = Field(default=20, ge=10, le=50)
    bb_std: float = Field(default=2.0, ge=1.0, le=3.0)
    short_ma: int = Field(default=10, ge=5, le=30)
    long_ma: int = Field(default=50, ge=20, le=200)
    require_trend_alignment: bool = Field(default=True)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SignalGenerator:
    """
    Trading signal generator.

    Combines multiple technical analysis methods:
    - Moving average crossovers
    - RSI signals
    - MACD signals
    - Bollinger Band signals
    - Pattern-based signals
    - Trend-aligned signals
    """

    def __init__(
        self,
        config: Optional[SignalGeneratorConfig] = None,
    ) -> None:
        """
        Initialize SignalGenerator.

        Args:
            config: Signal generator configuration
        """
        self._config = config or SignalGeneratorConfig()
        self._indicators = TechnicalIndicators()
        self._trend_analyzer = TrendAnalyzer()
        self._pattern_recognition = PatternRecognition()

        logger.info("SignalGenerator initialized")

    def generate_all_signals(
        self,
        symbol: str,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        timeframe: SignalTimeframe = SignalTimeframe.SWING,
    ) -> list[TradingSignal]:
        """
        Generate all available signals for a symbol.

        Args:
            symbol: Trading symbol
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices
            volumes: Volume data
            timeframe: Signal timeframe

        Returns:
            List of TradingSignal objects
        """
        signals: list[TradingSignal] = []

        ma_signal = self.generate_ma_crossover_signal(
            symbol, closes, timeframe,
        )
        if ma_signal:
            signals.append(ma_signal)

        rsi_signal = self.generate_rsi_signal(
            symbol, closes, timeframe,
        )
        if rsi_signal:
            signals.append(rsi_signal)

        macd_signal = self.generate_macd_signal(
            symbol, closes, timeframe,
        )
        if macd_signal:
            signals.append(macd_signal)

        bb_signal = self.generate_bollinger_signal(
            symbol, closes, timeframe,
        )
        if bb_signal:
            signals.append(bb_signal)

        pattern_signals = self.generate_pattern_signals(
            symbol, opens, highs, lows, closes, timeframe,
        )
        signals.extend(pattern_signals)

        trend_signal = self.generate_trend_signal(
            symbol, highs, lows, closes, volumes, timeframe,
        )
        if trend_signal:
            signals.append(trend_signal)

        return signals

    def generate_ma_crossover_signal(
        self,
        symbol: str,
        closes: list[float],
        timeframe: SignalTimeframe,
    ) -> Optional[TradingSignal]:
        """
        Generate moving average crossover signal.

        Args:
            symbol: Trading symbol
            closes: Close prices
            timeframe: Signal timeframe

        Returns:
            TradingSignal or None
        """
        short_ma = self._indicators.ema(closes, self._config.short_ma)
        long_ma = self._indicators.ema(closes, self._config.long_ma)

        if len(short_ma) < 2 or len(long_ma) < 2:
            return None

        current_short = short_ma[-1]
        current_long = long_ma[-1]
        prev_short = short_ma[-2]
        prev_long = long_ma[-2]

        if any(x != x for x in [current_short, current_long, prev_short, prev_long]):
            return None

        golden_cross = prev_short <= prev_long and current_short > current_long
        death_cross = prev_short >= prev_long and current_short < current_long

        if not golden_cross and not death_cross:
            return None

        if golden_cross:
            signal_type = SignalType.BUY
            notes = f"Golden cross: {self._config.short_ma} EMA crossed above {self._config.long_ma} EMA"
        else:
            signal_type = SignalType.SELL
            notes = f"Death cross: {self._config.short_ma} EMA crossed below {self._config.long_ma} EMA"

        current_price = closes[-1]
        spread = abs(current_short - current_long) / current_price

        strength = min(1.0, spread * 20)
        confidence = 0.6 + strength * 0.3

        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            source=SignalSource.MA_CROSSOVER,
            timeframe=timeframe,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            notes=notes,
        )

    def generate_rsi_signal(
        self,
        symbol: str,
        closes: list[float],
        timeframe: SignalTimeframe,
    ) -> Optional[TradingSignal]:
        """
        Generate RSI-based signal.

        Args:
            symbol: Trading symbol
            closes: Close prices
            timeframe: Signal timeframe

        Returns:
            TradingSignal or None
        """
        rsi_values = self._indicators.rsi(closes, self._config.rsi_period)

        if not rsi_values or len(rsi_values) < 2:
            return None

        current_rsi = rsi_values[-1]
        prev_rsi = rsi_values[-2]

        if current_rsi != current_rsi:
            return None

        signal_type = None
        notes = ""

        if current_rsi < self._config.rsi_oversold:
            if prev_rsi < current_rsi:
                signal_type = SignalType.BUY
                notes = f"RSI oversold ({current_rsi:.1f}) with bullish divergence"
            else:
                signal_type = SignalType.HOLD
                notes = f"RSI oversold ({current_rsi:.1f}) but still falling"

        elif current_rsi > self._config.rsi_overbought:
            if prev_rsi > current_rsi:
                signal_type = SignalType.SELL
                notes = f"RSI overbought ({current_rsi:.1f}) with bearish divergence"
            else:
                signal_type = SignalType.HOLD
                notes = f"RSI overbought ({current_rsi:.1f}) but still rising"

        if not signal_type:
            return None

        if signal_type == SignalType.BUY:
            strength = (self._config.rsi_oversold - current_rsi) / self._config.rsi_oversold
        elif signal_type == SignalType.SELL:
            strength = (current_rsi - self._config.rsi_overbought) / (100 - self._config.rsi_overbought)
        else:
            strength = 0.3

        strength = max(0.0, min(1.0, strength))
        confidence = 0.5 + strength * 0.4

        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            source=SignalSource.RSI,
            timeframe=timeframe,
            strength=strength,
            confidence=confidence,
            entry_price=closes[-1],
            notes=notes,
        )

    def generate_macd_signal(
        self,
        symbol: str,
        closes: list[float],
        timeframe: SignalTimeframe,
    ) -> Optional[TradingSignal]:
        """
        Generate MACD-based signal.

        Args:
            symbol: Trading symbol
            closes: Close prices
            timeframe: Signal timeframe

        Returns:
            TradingSignal or None
        """
        macd_results = self._indicators.macd(
            closes,
            self._config.macd_fast,
            self._config.macd_slow,
            self._config.macd_signal,
        )

        if len(macd_results) < 2:
            return None

        current = macd_results[-1]
        prev = macd_results[-2]

        bullish_cross = prev.histogram <= 0 and current.histogram > 0
        bearish_cross = prev.histogram >= 0 and current.histogram < 0

        if not bullish_cross and not bearish_cross:
            return None

        if bullish_cross:
            signal_type = SignalType.BUY
            notes = f"MACD bullish crossover (histogram: {current.histogram:.4f})"
        else:
            signal_type = SignalType.SELL
            notes = f"MACD bearish crossover (histogram: {current.histogram:.4f})"

        current_price = closes[-1]
        normalized_hist = abs(current.histogram) / current_price if current_price > 0 else 0

        strength = min(1.0, normalized_hist * 100)
        confidence = 0.55 + strength * 0.35

        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            source=SignalSource.MACD,
            timeframe=timeframe,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            notes=notes,
        )

    def generate_bollinger_signal(
        self,
        symbol: str,
        closes: list[float],
        timeframe: SignalTimeframe,
    ) -> Optional[TradingSignal]:
        """
        Generate Bollinger Bands signal.

        Args:
            symbol: Trading symbol
            closes: Close prices
            timeframe: Signal timeframe

        Returns:
            TradingSignal or None
        """
        bb_results = self._indicators.bollinger_bands(
            closes,
            self._config.bb_period,
            self._config.bb_std,
        )

        if not bb_results:
            return None

        current = bb_results[-1]
        current_price = closes[-1]

        if current.upper == 0 or current.lower == 0:
            return None

        signal_type = None
        notes = ""

        if current_price < current.lower:
            signal_type = SignalType.BUY
            distance = (current.lower - current_price) / current_price
            notes = f"Price below lower Bollinger Band ({distance*100:.1f}% below)"

        elif current_price > current.upper:
            signal_type = SignalType.SELL
            distance = (current_price - current.upper) / current_price
            notes = f"Price above upper Bollinger Band ({distance*100:.1f}% above)"

        elif current.percent_b < 0.1:
            signal_type = SignalType.BUY
            notes = f"%B very low ({current.percent_b:.2f}), potential bounce"

        elif current.percent_b > 0.9:
            signal_type = SignalType.SELL
            notes = f"%B very high ({current.percent_b:.2f}), potential pullback"

        if not signal_type:
            return None

        if signal_type == SignalType.BUY:
            strength = 1.0 - current.percent_b
        else:
            strength = current.percent_b

        strength = max(0.0, min(1.0, strength))
        confidence = 0.5 + strength * 0.35

        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            source=SignalSource.BOLLINGER,
            timeframe=timeframe,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            target_price=current.middle,
            notes=notes,
        )

    def generate_pattern_signals(
        self,
        symbol: str,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
        timeframe: SignalTimeframe,
    ) -> list[TradingSignal]:
        """
        Generate pattern-based signals.

        Args:
            symbol: Trading symbol
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices
            timeframe: Signal timeframe

        Returns:
            List of TradingSignal objects
        """
        signals: list[TradingSignal] = []

        patterns = self._pattern_recognition.detect_all_candle_patterns(
            opens, highs, lows, closes,
        )

        recent_patterns = [p for p in patterns if p.index >= len(closes) - 3]

        for pattern in recent_patterns:
            if pattern.pattern_type == PatternType.BULLISH:
                signal_type = SignalType.BUY
            elif pattern.pattern_type == PatternType.BEARISH:
                signal_type = SignalType.SELL
            else:
                continue

            reliability_scores = {
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4,
            }
            strength = reliability_scores.get(pattern.reliability.value, 0.5)

            signals.append(TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                source=SignalSource.PATTERN,
                timeframe=timeframe,
                strength=strength,
                confidence=strength,
                entry_price=closes[-1],
                notes=f"Pattern: {pattern.name} - {pattern.description}",
            ))

        return signals

    def generate_trend_signal(
        self,
        symbol: str,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        timeframe: SignalTimeframe,
    ) -> Optional[TradingSignal]:
        """
        Generate trend-based signal.

        Args:
            symbol: Trading symbol
            highs: High prices
            lows: Low prices
            closes: Close prices
            volumes: Volume data
            timeframe: Signal timeframe

        Returns:
            TradingSignal or None
        """
        trend_result = self._trend_analyzer.analyze_trend(
            highs, lows, closes, volumes,
        )

        bullish_trends = {
            TrendDirection.STRONG_BULLISH,
            TrendDirection.BULLISH,
        }
        bearish_trends = {
            TrendDirection.STRONG_BEARISH,
            TrendDirection.BEARISH,
        }

        if trend_result.direction in bullish_trends:
            if trend_result.strength in [TrendStrength.STRONG, TrendStrength.VERY_STRONG]:
                signal_type = SignalType.STRONG_BUY
            else:
                signal_type = SignalType.BUY

        elif trend_result.direction in bearish_trends:
            if trend_result.strength in [TrendStrength.STRONG, TrendStrength.VERY_STRONG]:
                signal_type = SignalType.STRONG_SELL
            else:
                signal_type = SignalType.SELL

        else:
            return None

        strength_scores = {
            TrendStrength.VERY_STRONG: 1.0,
            TrendStrength.STRONG: 0.8,
            TrendStrength.MODERATE: 0.6,
            TrendStrength.WEAK: 0.4,
            TrendStrength.NONE: 0.2,
        }
        strength = strength_scores.get(trend_result.strength, 0.5)

        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            source=SignalSource.TREND,
            timeframe=timeframe,
            strength=strength,
            confidence=trend_result.confidence,
            entry_price=closes[-1],
            target_price=trend_result.resistance_level if signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else trend_result.support_level,
            stop_loss=trend_result.support_level if signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else trend_result.resistance_level,
            notes=trend_result.description,
        )

    def generate_combined_signal(
        self,
        symbol: str,
        signals: list[TradingSignal],
    ) -> TradingSignal:
        """
        Generate combined signal from multiple signals.

        Args:
            symbol: Trading symbol
            signals: List of individual signals

        Returns:
            Combined TradingSignal
        """
        if not signals:
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                source=SignalSource.COMBINED,
                timeframe=SignalTimeframe.SWING,
                strength=0.0,
                confidence=0.0,
                notes="No signals available",
            )

        score = 0.0
        total_weight = 0.0

        signal_weights = {
            SignalSource.TREND: 2.0,
            SignalSource.MA_CROSSOVER: 1.5,
            SignalSource.MACD: 1.3,
            SignalSource.RSI: 1.2,
            SignalSource.BOLLINGER: 1.0,
            SignalSource.PATTERN: 1.0,
        }

        type_scores = {
            SignalType.STRONG_BUY: 2.0,
            SignalType.BUY: 1.0,
            SignalType.HOLD: 0.0,
            SignalType.SELL: -1.0,
            SignalType.STRONG_SELL: -2.0,
        }

        supporting: list[str] = []
        conflicting: list[str] = []

        for signal in signals:
            weight = signal_weights.get(signal.source, 1.0)
            type_score = type_scores.get(signal.signal_type, 0.0)

            weighted_score = type_score * signal.strength * signal.confidence * weight
            score += weighted_score
            total_weight += weight

            signal_desc = f"{signal.source.value}: {signal.signal_type.value}"
            if type_score > 0:
                supporting.append(signal_desc)
            elif type_score < 0:
                conflicting.append(signal_desc)

        normalized_score = score / total_weight if total_weight > 0 else 0

        if normalized_score > 1.0:
            signal_type = SignalType.STRONG_BUY
        elif normalized_score > 0.3:
            signal_type = SignalType.BUY
        elif normalized_score < -1.0:
            signal_type = SignalType.STRONG_SELL
        elif normalized_score < -0.3:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        strength = min(1.0, abs(normalized_score) / 2)

        buy_count = sum(1 for s in signals if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY])
        sell_count = sum(1 for s in signals if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL])
        total_count = len(signals)

        if total_count > 0:
            agreement = max(buy_count, sell_count) / total_count
            confidence = agreement * strength
        else:
            confidence = 0.0

        avg_entry = sum(s.entry_price for s in signals if s.entry_price) / len([s for s in signals if s.entry_price]) if signals else None

        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            source=SignalSource.COMBINED,
            timeframe=signals[0].timeframe if signals else SignalTimeframe.SWING,
            strength=strength,
            confidence=confidence,
            entry_price=avg_entry,
            supporting_signals=supporting,
            conflicting_signals=conflicting,
            notes=f"Combined from {len(signals)} signals, score: {normalized_score:.2f}",
        )

    def get_signal_summary(
        self,
        symbol: str,
        signals: list[TradingSignal],
    ) -> SignalSummary:
        """
        Get summary of all signals for a symbol.

        Args:
            symbol: Trading symbol
            signals: List of signals

        Returns:
            SignalSummary object
        """
        buy_signals = sum(1 for s in signals if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY])
        sell_signals = sum(1 for s in signals if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL])
        neutral_signals = sum(1 for s in signals if s.signal_type == SignalType.HOLD)

        combined = self.generate_combined_signal(symbol, signals)

        score_map = {
            SignalType.STRONG_BUY: 80,
            SignalType.BUY: 40,
            SignalType.HOLD: 0,
            SignalType.SELL: -40,
            SignalType.STRONG_SELL: -80,
        }
        overall_score = score_map.get(combined.signal_type, 0) * combined.strength

        trend_signals = [s for s in signals if s.source == SignalSource.TREND]
        if trend_signals:
            trend_alignment = trend_signals[0].signal_type.value
        else:
            trend_alignment = "unknown"

        key_levels = {}
        for signal in signals:
            if signal.target_price:
                key_levels["target"] = signal.target_price
            if signal.stop_loss:
                key_levels["stop_loss"] = signal.stop_loss

        if combined.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            recommendation = f"Consider LONG position with {combined.confidence*100:.0f}% confidence"
        elif combined.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            recommendation = f"Consider SHORT/EXIT position with {combined.confidence*100:.0f}% confidence"
        else:
            recommendation = "Wait for clearer signal alignment"

        return SignalSummary(
            symbol=symbol,
            overall_signal=combined.signal_type,
            overall_score=overall_score,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            neutral_signals=neutral_signals,
            trend_alignment=trend_alignment,
            key_levels=key_levels,
            recommendation=recommendation,
        )

    def __repr__(self) -> str:
        """String representation."""
        return "SignalGenerator()"

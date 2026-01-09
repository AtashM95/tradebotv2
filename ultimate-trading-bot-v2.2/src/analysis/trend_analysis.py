"""
Trend Analysis Module for Ultimate Trading Bot v2.2.

This module provides comprehensive trend detection and analysis
for trading decisions.
"""

import logging
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from src.analysis.technical_indicators import TechnicalIndicators


logger = logging.getLogger(__name__)


class TrendDirection(str, Enum):
    """Trend direction enumeration."""

    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    WEAK_BULLISH = "weak_bullish"
    NEUTRAL = "neutral"
    WEAK_BEARISH = "weak_bearish"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class TrendStrength(str, Enum):
    """Trend strength enumeration."""

    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


class TrendPhase(str, Enum):
    """Market trend phase enumeration."""

    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    RANGING = "ranging"


class TrendResult(BaseModel):
    """Trend analysis result."""

    direction: TrendDirection
    strength: TrendStrength
    phase: TrendPhase
    confidence: float = Field(ge=0.0, le=1.0)
    adx_value: float = Field(default=0.0)
    slope: float = Field(default=0.0)
    ma_alignment: str = Field(default="")
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    description: str = Field(default="")


class TrendChange(BaseModel):
    """Trend change detection result."""

    detected: bool = Field(default=False)
    previous_direction: Optional[TrendDirection] = None
    new_direction: Optional[TrendDirection] = None
    change_index: int = Field(default=-1)
    confirmation_level: str = Field(default="unconfirmed")
    description: str = Field(default="")


class TrendLine(BaseModel):
    """Trend line model."""

    start_index: int
    end_index: int
    start_price: float
    end_price: float
    slope: float
    line_type: str = Field(default="support")
    touches: int = Field(default=2)
    strength: float = Field(ge=0.0, le=1.0, default=0.5)


class TrendAnalysisConfig(BaseModel):
    """Configuration for trend analysis."""

    short_period: int = Field(default=10, ge=2, le=50)
    medium_period: int = Field(default=20, ge=5, le=100)
    long_period: int = Field(default=50, ge=10, le=200)
    adx_period: int = Field(default=14, ge=5, le=30)
    adx_threshold_strong: float = Field(default=25.0, ge=15.0, le=50.0)
    adx_threshold_weak: float = Field(default=20.0, ge=10.0, le=30.0)
    slope_threshold: float = Field(default=0.001)


class TrendAnalyzer:
    """
    Comprehensive trend analyzer.

    Provides:
    - Multi-timeframe trend detection
    - Trend strength measurement
    - Trend change detection
    - Trend line identification
    - Market phase detection
    """

    def __init__(
        self,
        config: Optional[TrendAnalysisConfig] = None,
    ) -> None:
        """
        Initialize TrendAnalyzer.

        Args:
            config: Trend analysis configuration
        """
        self._config = config or TrendAnalysisConfig()
        self._indicators = TechnicalIndicators()

        logger.info("TrendAnalyzer initialized")

    def analyze_trend(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
        volume: Optional[list[float]] = None,
    ) -> TrendResult:
        """
        Perform comprehensive trend analysis.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Optional volume data

        Returns:
            TrendResult with analysis
        """
        direction = self._determine_direction(close)
        strength = self._determine_strength(high, low, close)
        phase = self._determine_phase(close, volume)

        adx_values = self._indicators.adx(
            high, low, close,
            self._config.adx_period,
        )
        adx_value = adx_values[-1] if adx_values and not np.isnan(adx_values[-1]) else 0.0

        slope = self._calculate_trend_slope(close, self._config.medium_period)

        ma_alignment = self._check_ma_alignment(close)

        confidence = self._calculate_confidence(
            direction, strength, adx_value, ma_alignment,
        )

        support, resistance = self._find_sr_levels(high, low, close)

        description = self._generate_description(
            direction, strength, phase, adx_value, ma_alignment,
        )

        return TrendResult(
            direction=direction,
            strength=strength,
            phase=phase,
            confidence=confidence,
            adx_value=adx_value,
            slope=slope,
            ma_alignment=ma_alignment,
            support_level=support,
            resistance_level=resistance,
            description=description,
        )

    def detect_trend_change(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
        lookback: int = 20,
    ) -> TrendChange:
        """
        Detect trend changes.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            lookback: Lookback period for comparison

        Returns:
            TrendChange detection result
        """
        if len(close) < lookback * 2:
            return TrendChange(detected=False)

        past_close = close[:-lookback]
        past_direction = self._determine_direction(past_close)

        current_direction = self._determine_direction(close)

        bullish_dirs = {
            TrendDirection.STRONG_BULLISH,
            TrendDirection.BULLISH,
            TrendDirection.WEAK_BULLISH,
        }
        bearish_dirs = {
            TrendDirection.STRONG_BEARISH,
            TrendDirection.BEARISH,
            TrendDirection.WEAK_BEARISH,
        }

        past_bullish = past_direction in bullish_dirs
        past_bearish = past_direction in bearish_dirs
        current_bullish = current_direction in bullish_dirs
        current_bearish = current_direction in bearish_dirs

        change_detected = (
            (past_bullish and current_bearish) or
            (past_bearish and current_bullish)
        )

        if not change_detected:
            return TrendChange(detected=False)

        change_index = len(close) - lookback

        confirmation = self._assess_change_confirmation(
            high, low, close, change_index,
        )

        return TrendChange(
            detected=True,
            previous_direction=past_direction,
            new_direction=current_direction,
            change_index=change_index,
            confirmation_level=confirmation,
            description=f"Trend change from {past_direction.value} to {current_direction.value}",
        )

    def find_trend_lines(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
        min_touches: int = 2,
    ) -> list[TrendLine]:
        """
        Find significant trend lines.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            min_touches: Minimum touches for valid line

        Returns:
            List of TrendLine objects
        """
        trend_lines: list[TrendLine] = []

        support_lines = self._find_support_lines(low, min_touches)
        for line in support_lines:
            line.line_type = "support"
            trend_lines.append(line)

        resistance_lines = self._find_resistance_lines(high, min_touches)
        for line in resistance_lines:
            line.line_type = "resistance"
            trend_lines.append(line)

        return trend_lines

    def get_multi_timeframe_trend(
        self,
        close: list[float],
    ) -> dict[str, TrendDirection]:
        """
        Get trend direction across multiple timeframes.

        Args:
            close: Close prices

        Returns:
            Dictionary of timeframe -> trend direction
        """
        results: dict[str, TrendDirection] = {}

        if len(close) >= self._config.short_period:
            short_close = close[-self._config.short_period:]
            results["short"] = self._determine_direction(short_close)

        if len(close) >= self._config.medium_period:
            medium_close = close[-self._config.medium_period:]
            results["medium"] = self._determine_direction(medium_close)

        if len(close) >= self._config.long_period:
            long_close = close[-self._config.long_period:]
            results["long"] = self._determine_direction(long_close)

        return results

    def calculate_trend_score(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
    ) -> float:
        """
        Calculate overall trend score (-100 to +100).

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Trend score
        """
        score = 0.0

        direction = self._determine_direction(close)
        direction_scores = {
            TrendDirection.STRONG_BULLISH: 40,
            TrendDirection.BULLISH: 25,
            TrendDirection.WEAK_BULLISH: 10,
            TrendDirection.NEUTRAL: 0,
            TrendDirection.WEAK_BEARISH: -10,
            TrendDirection.BEARISH: -25,
            TrendDirection.STRONG_BEARISH: -40,
        }
        score += direction_scores.get(direction, 0)

        adx_values = self._indicators.adx(
            high, low, close,
            self._config.adx_period,
        )
        adx = adx_values[-1] if adx_values and not np.isnan(adx_values[-1]) else 0

        if adx > 30:
            multiplier = 1.5
        elif adx > 25:
            multiplier = 1.25
        elif adx > 20:
            multiplier = 1.0
        else:
            multiplier = 0.75

        score *= multiplier

        ma_alignment = self._check_ma_alignment(close)
        if ma_alignment == "bullish":
            score += 20
        elif ma_alignment == "bearish":
            score -= 20

        return max(-100, min(100, score))

    def _determine_direction(self, close: list[float]) -> TrendDirection:
        """Determine trend direction from close prices."""
        if len(close) < 10:
            return TrendDirection.NEUTRAL

        sma_short = self._indicators.sma(close, min(10, len(close)))
        sma_medium = self._indicators.sma(close, min(20, len(close)))

        current_price = close[-1]
        short_ma = sma_short[-1] if sma_short and not np.isnan(sma_short[-1]) else current_price
        medium_ma = sma_medium[-1] if sma_medium and not np.isnan(sma_medium[-1]) else current_price

        slope = self._calculate_trend_slope(close, min(20, len(close)))

        price_vs_short = (current_price - short_ma) / short_ma if short_ma != 0 else 0
        price_vs_medium = (current_price - medium_ma) / medium_ma if medium_ma != 0 else 0

        if price_vs_short > 0.02 and price_vs_medium > 0.03 and slope > 0.002:
            return TrendDirection.STRONG_BULLISH
        elif price_vs_short > 0.01 and price_vs_medium > 0.015:
            return TrendDirection.BULLISH
        elif price_vs_short > 0 and slope > 0:
            return TrendDirection.WEAK_BULLISH
        elif price_vs_short < -0.02 and price_vs_medium < -0.03 and slope < -0.002:
            return TrendDirection.STRONG_BEARISH
        elif price_vs_short < -0.01 and price_vs_medium < -0.015:
            return TrendDirection.BEARISH
        elif price_vs_short < 0 and slope < 0:
            return TrendDirection.WEAK_BEARISH
        else:
            return TrendDirection.NEUTRAL

    def _determine_strength(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
    ) -> TrendStrength:
        """Determine trend strength using ADX."""
        adx_values = self._indicators.adx(
            high, low, close,
            self._config.adx_period,
        )

        adx = adx_values[-1] if adx_values and not np.isnan(adx_values[-1]) else 0

        if adx > 50:
            return TrendStrength.VERY_STRONG
        elif adx > 35:
            return TrendStrength.STRONG
        elif adx > 25:
            return TrendStrength.MODERATE
        elif adx > 15:
            return TrendStrength.WEAK
        else:
            return TrendStrength.NONE

    def _determine_phase(
        self,
        close: list[float],
        volume: Optional[list[float]],
    ) -> TrendPhase:
        """Determine market phase (Wyckoff-inspired)."""
        if len(close) < 20:
            return TrendPhase.RANGING

        recent_close = close[-20:]
        price_range = max(recent_close) - min(recent_close)
        avg_price = sum(recent_close) / len(recent_close)
        volatility = price_range / avg_price if avg_price != 0 else 0

        slope = self._calculate_trend_slope(close, 20)

        volume_trend = "neutral"
        if volume and len(volume) >= 20:
            recent_volume = volume[-20:]
            first_half_vol = sum(recent_volume[:10]) / 10
            second_half_vol = sum(recent_volume[10:]) / 10
            if second_half_vol > first_half_vol * 1.2:
                volume_trend = "increasing"
            elif second_half_vol < first_half_vol * 0.8:
                volume_trend = "decreasing"

        if volatility < 0.03 and abs(slope) < 0.001:
            if volume_trend == "decreasing":
                return TrendPhase.ACCUMULATION
            else:
                return TrendPhase.RANGING

        if slope > 0.001:
            if volume_trend == "increasing":
                return TrendPhase.MARKUP
            elif volatility < 0.04:
                return TrendPhase.DISTRIBUTION
            else:
                return TrendPhase.MARKUP

        if slope < -0.001:
            if volume_trend == "increasing":
                return TrendPhase.MARKDOWN
            elif volatility < 0.04:
                return TrendPhase.ACCUMULATION
            else:
                return TrendPhase.MARKDOWN

        return TrendPhase.RANGING

    def _calculate_trend_slope(
        self,
        close: list[float],
        period: int,
    ) -> float:
        """Calculate normalized slope of price trend."""
        if len(close) < period:
            period = len(close)

        if period < 2:
            return 0.0

        recent = close[-period:]
        n = len(recent)

        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        normalized_slope = slope / y_mean if y_mean != 0 else 0

        return normalized_slope

    def _check_ma_alignment(self, close: list[float]) -> str:
        """Check moving average alignment."""
        if len(close) < self._config.long_period:
            return "insufficient_data"

        short_ma = self._indicators.sma(close, self._config.short_period)
        medium_ma = self._indicators.sma(close, self._config.medium_period)
        long_ma = self._indicators.sma(close, self._config.long_period)

        short_val = short_ma[-1] if not np.isnan(short_ma[-1]) else 0
        medium_val = medium_ma[-1] if not np.isnan(medium_ma[-1]) else 0
        long_val = long_ma[-1] if not np.isnan(long_ma[-1]) else 0

        if short_val > medium_val > long_val:
            return "bullish"
        elif short_val < medium_val < long_val:
            return "bearish"
        elif short_val > long_val:
            return "mixed_bullish"
        elif short_val < long_val:
            return "mixed_bearish"
        else:
            return "neutral"

    def _calculate_confidence(
        self,
        direction: TrendDirection,
        strength: TrendStrength,
        adx: float,
        ma_alignment: str,
    ) -> float:
        """Calculate confidence in trend analysis."""
        confidence = 0.5

        if strength == TrendStrength.VERY_STRONG:
            confidence += 0.25
        elif strength == TrendStrength.STRONG:
            confidence += 0.15
        elif strength == TrendStrength.MODERATE:
            confidence += 0.05

        if direction in [TrendDirection.STRONG_BULLISH, TrendDirection.STRONG_BEARISH]:
            confidence += 0.1
        elif direction in [TrendDirection.BULLISH, TrendDirection.BEARISH]:
            confidence += 0.05

        if ma_alignment == "bullish" and "bullish" in direction.value.lower():
            confidence += 0.1
        elif ma_alignment == "bearish" and "bearish" in direction.value.lower():
            confidence += 0.1

        return min(1.0, confidence)

    def _find_sr_levels(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """Find nearest support and resistance levels."""
        if len(close) < 10:
            return None, None

        current_price = close[-1]

        recent_lows = low[-50:] if len(low) >= 50 else low
        recent_highs = high[-50:] if len(high) >= 50 else high

        support_candidates = [
            l for l in recent_lows
            if l < current_price
        ]
        resistance_candidates = [
            h for h in recent_highs
            if h > current_price
        ]

        support = max(support_candidates) if support_candidates else None
        resistance = min(resistance_candidates) if resistance_candidates else None

        return support, resistance

    def _assess_change_confirmation(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
        change_index: int,
    ) -> str:
        """Assess confirmation level of trend change."""
        if change_index < 0 or change_index >= len(close):
            return "unconfirmed"

        post_change = close[change_index:]
        if len(post_change) < 3:
            return "unconfirmed"

        direction = self._determine_direction(post_change)
        strength = self._determine_strength(
            high[change_index:],
            low[change_index:],
            post_change,
        )

        if strength in [TrendStrength.STRONG, TrendStrength.VERY_STRONG]:
            return "confirmed"
        elif strength == TrendStrength.MODERATE:
            return "partial"
        else:
            return "unconfirmed"

    def _find_support_lines(
        self,
        low: list[float],
        min_touches: int,
    ) -> list[TrendLine]:
        """Find support trend lines."""
        lines: list[TrendLine] = []
        n = len(low)

        if n < 10:
            return lines

        troughs: list[int] = []
        for i in range(2, n - 2):
            if low[i] <= low[i - 1] and low[i] <= low[i - 2]:
                if low[i] <= low[i + 1] and low[i] <= low[i + 2]:
                    troughs.append(i)

        for i in range(len(troughs)):
            for j in range(i + 1, len(troughs)):
                start_idx = troughs[i]
                end_idx = troughs[j]

                slope = (low[end_idx] - low[start_idx]) / (end_idx - start_idx)

                touches = 0
                for k in range(start_idx, end_idx + 1):
                    line_value = low[start_idx] + slope * (k - start_idx)
                    if abs(low[k] - line_value) < line_value * 0.01:
                        touches += 1

                if touches >= min_touches:
                    lines.append(TrendLine(
                        start_index=start_idx,
                        end_index=end_idx,
                        start_price=low[start_idx],
                        end_price=low[end_idx],
                        slope=slope,
                        touches=touches,
                        strength=min(1.0, touches / 5),
                    ))

        return lines

    def _find_resistance_lines(
        self,
        high: list[float],
        min_touches: int,
    ) -> list[TrendLine]:
        """Find resistance trend lines."""
        lines: list[TrendLine] = []
        n = len(high)

        if n < 10:
            return lines

        peaks: list[int] = []
        for i in range(2, n - 2):
            if high[i] >= high[i - 1] and high[i] >= high[i - 2]:
                if high[i] >= high[i + 1] and high[i] >= high[i + 2]:
                    peaks.append(i)

        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                start_idx = peaks[i]
                end_idx = peaks[j]

                slope = (high[end_idx] - high[start_idx]) / (end_idx - start_idx)

                touches = 0
                for k in range(start_idx, end_idx + 1):
                    line_value = high[start_idx] + slope * (k - start_idx)
                    if abs(high[k] - line_value) < line_value * 0.01:
                        touches += 1

                if touches >= min_touches:
                    lines.append(TrendLine(
                        start_index=start_idx,
                        end_index=end_idx,
                        start_price=high[start_idx],
                        end_price=high[end_idx],
                        slope=slope,
                        touches=touches,
                        strength=min(1.0, touches / 5),
                    ))

        return lines

    def _generate_description(
        self,
        direction: TrendDirection,
        strength: TrendStrength,
        phase: TrendPhase,
        adx: float,
        ma_alignment: str,
    ) -> str:
        """Generate human-readable trend description."""
        parts = []

        direction_desc = {
            TrendDirection.STRONG_BULLISH: "Strong uptrend",
            TrendDirection.BULLISH: "Uptrend",
            TrendDirection.WEAK_BULLISH: "Weak uptrend",
            TrendDirection.NEUTRAL: "Sideways/No trend",
            TrendDirection.WEAK_BEARISH: "Weak downtrend",
            TrendDirection.BEARISH: "Downtrend",
            TrendDirection.STRONG_BEARISH: "Strong downtrend",
        }
        parts.append(direction_desc.get(direction, "Unknown"))

        parts.append(f"with {strength.value} strength (ADX: {adx:.1f})")

        parts.append(f"in {phase.value} phase")

        if ma_alignment in ["bullish", "bearish"]:
            parts.append(f"MAs {ma_alignment}ly aligned")

        return ", ".join(parts)

    def __repr__(self) -> str:
        """String representation."""
        return "TrendAnalyzer()"

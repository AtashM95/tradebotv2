"""
Price Analysis Module for Ultimate Trading Bot v2.2.

This module provides price action analysis, support/resistance
detection, and price structure analysis.
"""

import logging
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class PriceStructure(str, Enum):
    """Price structure enumeration."""

    HIGHER_HIGHS_HIGHER_LOWS = "higher_highs_higher_lows"
    LOWER_HIGHS_LOWER_LOWS = "lower_highs_lower_lows"
    CONSOLIDATION = "consolidation"
    REVERSAL_TOP = "reversal_top"
    REVERSAL_BOTTOM = "reversal_bottom"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"


class PriceLevel(BaseModel):
    """Price level model."""

    price: float
    level_type: str = Field(default="support")
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    touches: int = Field(default=1)
    last_touch_index: int = Field(default=-1)
    is_broken: bool = Field(default=False)


class PriceRange(BaseModel):
    """Price range model."""

    high: float
    low: float
    range_size: float
    range_pct: float
    is_consolidating: bool = Field(default=False)


class PriceAnalysisResult(BaseModel):
    """Price analysis result model."""

    current_price: float
    structure: PriceStructure
    trend_bias: str = Field(default="neutral")
    support_levels: list[PriceLevel] = Field(default_factory=list)
    resistance_levels: list[PriceLevel] = Field(default_factory=list)
    price_range: Optional[PriceRange] = None
    atr: float = Field(default=0.0)
    volatility_pct: float = Field(default=0.0)
    price_momentum: float = Field(default=0.0)
    pivot_points: dict = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class PriceAnalysisConfig(BaseModel):
    """Configuration for price analysis."""

    swing_lookback: int = Field(default=5, ge=2, le=20)
    sr_tolerance_pct: float = Field(default=0.5, ge=0.1, le=2.0)
    min_touches_for_level: int = Field(default=2, ge=1, le=5)
    atr_period: int = Field(default=14, ge=5, le=30)
    consolidation_threshold: float = Field(default=0.03, ge=0.01, le=0.1)


class PriceAnalyzer:
    """
    Price action analyzer.

    Provides:
    - Price structure analysis
    - Support/resistance detection
    - Price momentum calculation
    - Volatility analysis
    - Pivot point calculation
    - Range analysis
    """

    def __init__(
        self,
        config: Optional[PriceAnalysisConfig] = None,
    ) -> None:
        """
        Initialize PriceAnalyzer.

        Args:
            config: Price analysis configuration
        """
        self._config = config or PriceAnalysisConfig()

        logger.info("PriceAnalyzer initialized")

    def analyze(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        opens: Optional[list[float]] = None,
    ) -> PriceAnalysisResult:
        """
        Perform comprehensive price analysis.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            opens: Optional open prices

        Returns:
            PriceAnalysisResult with analysis
        """
        if not closes:
            return PriceAnalysisResult(
                current_price=0.0,
                structure=PriceStructure.CONSOLIDATION,
            )

        current_price = closes[-1]
        notes: list[str] = []

        structure = self._analyze_structure(highs, lows, closes)
        notes.append(f"Price structure: {structure.value}")

        trend_bias = self._determine_trend_bias(highs, lows, closes)
        notes.append(f"Trend bias: {trend_bias}")

        support_levels = self._find_support_levels(lows, closes)
        resistance_levels = self._find_resistance_levels(highs, closes)

        price_range = self._analyze_range(highs, lows, closes)
        if price_range.is_consolidating:
            notes.append("Price is consolidating")

        atr = self._calculate_atr(highs, lows, closes)
        volatility_pct = (atr / current_price * 100) if current_price > 0 else 0
        notes.append(f"ATR: ${atr:.2f} ({volatility_pct:.1f}%)")

        momentum = self._calculate_momentum(closes)
        notes.append(f"Momentum: {momentum:+.2f}%")

        pivot_points = self._calculate_pivot_points(highs, lows, closes)

        return PriceAnalysisResult(
            current_price=current_price,
            structure=structure,
            trend_bias=trend_bias,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            price_range=price_range,
            atr=atr,
            volatility_pct=volatility_pct,
            price_momentum=momentum,
            pivot_points=pivot_points,
            notes=notes,
        )

    def _analyze_structure(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> PriceStructure:
        """Analyze price structure."""
        if len(highs) < 20:
            return PriceStructure.CONSOLIDATION

        swing_highs = self._find_swing_highs(highs)
        swing_lows = self._find_swing_lows(lows)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return PriceStructure.CONSOLIDATION

        recent_swing_highs = swing_highs[-3:]
        recent_swing_lows = swing_lows[-3:]

        higher_highs = all(
            recent_swing_highs[i] > recent_swing_highs[i - 1]
            for i in range(1, len(recent_swing_highs))
        )
        higher_lows = all(
            recent_swing_lows[i] > recent_swing_lows[i - 1]
            for i in range(1, len(recent_swing_lows))
        )

        lower_highs = all(
            recent_swing_highs[i] < recent_swing_highs[i - 1]
            for i in range(1, len(recent_swing_highs))
        )
        lower_lows = all(
            recent_swing_lows[i] < recent_swing_lows[i - 1]
            for i in range(1, len(recent_swing_lows))
        )

        if higher_highs and higher_lows:
            return PriceStructure.HIGHER_HIGHS_HIGHER_LOWS
        elif lower_highs and lower_lows:
            return PriceStructure.LOWER_HIGHS_LOWER_LOWS
        elif lower_highs and higher_lows:
            if closes[-1] > swing_highs[-1]:
                return PriceStructure.BREAKOUT
            elif closes[-1] < swing_lows[-1]:
                return PriceStructure.BREAKDOWN
            return PriceStructure.CONSOLIDATION
        elif higher_highs and lower_lows:
            return PriceStructure.CONSOLIDATION

        return PriceStructure.CONSOLIDATION

    def _determine_trend_bias(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> str:
        """Determine overall trend bias."""
        if len(closes) < 20:
            return "neutral"

        sma_10 = sum(closes[-10:]) / 10
        sma_20 = sum(closes[-20:]) / 20

        current_price = closes[-1]

        if current_price > sma_10 > sma_20:
            return "bullish"
        elif current_price < sma_10 < sma_20:
            return "bearish"
        elif current_price > sma_20:
            return "slightly_bullish"
        elif current_price < sma_20:
            return "slightly_bearish"
        else:
            return "neutral"

    def _find_swing_highs(self, highs: list[float]) -> list[float]:
        """Find swing high values."""
        lookback = self._config.swing_lookback
        swings: list[float] = []

        for i in range(lookback, len(highs) - lookback):
            is_swing = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing = False
                    break

            if is_swing:
                swings.append(highs[i])

        return swings

    def _find_swing_lows(self, lows: list[float]) -> list[float]:
        """Find swing low values."""
        lookback = self._config.swing_lookback
        swings: list[float] = []

        for i in range(lookback, len(lows) - lookback):
            is_swing = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing = False
                    break

            if is_swing:
                swings.append(lows[i])

        return swings

    def _find_support_levels(
        self,
        lows: list[float],
        closes: list[float],
    ) -> list[PriceLevel]:
        """Find support levels."""
        if len(lows) < 10:
            return []

        current_price = closes[-1]
        tolerance = current_price * self._config.sr_tolerance_pct / 100

        swing_lows = self._find_swing_lows(lows)
        if not swing_lows:
            return []

        levels: dict[float, PriceLevel] = {}

        for low in swing_lows:
            found_cluster = False
            for level_price in levels:
                if abs(low - level_price) < tolerance:
                    levels[level_price].touches += 1
                    levels[level_price].strength = min(
                        1.0,
                        levels[level_price].touches / 5,
                    )
                    found_cluster = True
                    break

            if not found_cluster:
                levels[low] = PriceLevel(
                    price=low,
                    level_type="support",
                    strength=0.3,
                    touches=1,
                )

        filtered_levels = [
            level for level in levels.values()
            if level.touches >= self._config.min_touches_for_level
            and level.price < current_price
        ]

        filtered_levels.sort(key=lambda x: x.price, reverse=True)

        return filtered_levels[:5]

    def _find_resistance_levels(
        self,
        highs: list[float],
        closes: list[float],
    ) -> list[PriceLevel]:
        """Find resistance levels."""
        if len(highs) < 10:
            return []

        current_price = closes[-1]
        tolerance = current_price * self._config.sr_tolerance_pct / 100

        swing_highs = self._find_swing_highs(highs)
        if not swing_highs:
            return []

        levels: dict[float, PriceLevel] = {}

        for high in swing_highs:
            found_cluster = False
            for level_price in levels:
                if abs(high - level_price) < tolerance:
                    levels[level_price].touches += 1
                    levels[level_price].strength = min(
                        1.0,
                        levels[level_price].touches / 5,
                    )
                    found_cluster = True
                    break

            if not found_cluster:
                levels[high] = PriceLevel(
                    price=high,
                    level_type="resistance",
                    strength=0.3,
                    touches=1,
                )

        filtered_levels = [
            level for level in levels.values()
            if level.touches >= self._config.min_touches_for_level
            and level.price > current_price
        ]

        filtered_levels.sort(key=lambda x: x.price)

        return filtered_levels[:5]

    def _analyze_range(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> PriceRange:
        """Analyze price range."""
        lookback = min(20, len(closes))

        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]

        range_high = max(recent_highs)
        range_low = min(recent_lows)
        range_size = range_high - range_low

        avg_price = (range_high + range_low) / 2
        range_pct = (range_size / avg_price) if avg_price > 0 else 0

        is_consolidating = range_pct < self._config.consolidation_threshold

        return PriceRange(
            high=range_high,
            low=range_low,
            range_size=range_size,
            range_pct=range_pct,
            is_consolidating=is_consolidating,
        )

    def _calculate_atr(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> float:
        """Calculate Average True Range."""
        period = min(self._config.atr_period, len(closes) - 1)

        if period < 1:
            return 0.0

        tr_values: list[float] = []

        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr_values.append(max(hl, hc, lc))

        if len(tr_values) < period:
            return sum(tr_values) / len(tr_values) if tr_values else 0.0

        atr = sum(tr_values[:period]) / period

        for i in range(period, len(tr_values)):
            atr = (atr * (period - 1) + tr_values[i]) / period

        return atr

    def _calculate_momentum(
        self,
        closes: list[float],
        period: int = 10,
    ) -> float:
        """Calculate price momentum as percentage change."""
        if len(closes) <= period:
            return 0.0

        current = closes[-1]
        past = closes[-period - 1]

        if past == 0:
            return 0.0

        return (current - past) / past * 100

    def _calculate_pivot_points(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> dict:
        """Calculate pivot points."""
        if not highs or not lows or not closes:
            return {}

        high = highs[-1]
        low = lows[-1]
        close = closes[-1]

        pivot = (high + low + close) / 3

        r1 = 2 * pivot - low
        s1 = 2 * pivot - high

        r2 = pivot + (high - low)
        s2 = pivot - (high - low)

        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)

        return {
            "pivot": round(pivot, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2),
            "r3": round(r3, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
            "s3": round(s3, 2),
        }

    def calculate_fibonacci_levels(
        self,
        high: float,
        low: float,
        direction: str = "up",
    ) -> dict:
        """
        Calculate Fibonacci retracement levels.

        Args:
            high: High price
            low: Low price
            direction: Trend direction ("up" or "down")

        Returns:
            Dictionary of Fibonacci levels
        """
        diff = high - low

        if direction == "up":
            return {
                "0.0%": low,
                "23.6%": low + diff * 0.236,
                "38.2%": low + diff * 0.382,
                "50.0%": low + diff * 0.500,
                "61.8%": low + diff * 0.618,
                "78.6%": low + diff * 0.786,
                "100.0%": high,
                "161.8%": low + diff * 1.618,
                "261.8%": low + diff * 2.618,
            }
        else:
            return {
                "0.0%": high,
                "23.6%": high - diff * 0.236,
                "38.2%": high - diff * 0.382,
                "50.0%": high - diff * 0.500,
                "61.8%": high - diff * 0.618,
                "78.6%": high - diff * 0.786,
                "100.0%": low,
                "161.8%": high - diff * 1.618,
                "261.8%": high - diff * 2.618,
            }

    def analyze_price_action(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> dict:
        """
        Analyze recent price action.

        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices

        Returns:
            Price action analysis dictionary
        """
        if len(closes) < 3:
            return {"analysis": "insufficient_data"}

        current_body = closes[-1] - opens[-1]
        current_range = highs[-1] - lows[-1]
        upper_shadow = highs[-1] - max(opens[-1], closes[-1])
        lower_shadow = min(opens[-1], closes[-1]) - lows[-1]

        if current_range > 0:
            body_ratio = abs(current_body) / current_range
            upper_shadow_ratio = upper_shadow / current_range
            lower_shadow_ratio = lower_shadow / current_range
        else:
            body_ratio = 0
            upper_shadow_ratio = 0
            lower_shadow_ratio = 0

        candle_type = "neutral"
        if current_body > 0:
            candle_type = "bullish"
        elif current_body < 0:
            candle_type = "bearish"

        pattern = "normal"
        if body_ratio < 0.1:
            pattern = "doji"
        elif lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1:
            pattern = "hammer" if candle_type == "bullish" else "hanging_man"
        elif upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1:
            pattern = "shooting_star" if candle_type == "bearish" else "inverted_hammer"
        elif body_ratio > 0.8:
            pattern = "marubozu"

        prev_closes = closes[-5:-1]
        momentum = "neutral"
        if all(prev_closes[i] < prev_closes[i + 1] for i in range(len(prev_closes) - 1)):
            momentum = "bullish"
        elif all(prev_closes[i] > prev_closes[i + 1] for i in range(len(prev_closes) - 1)):
            momentum = "bearish"

        return {
            "candle_type": candle_type,
            "pattern": pattern,
            "body_ratio": body_ratio,
            "upper_shadow_ratio": upper_shadow_ratio,
            "lower_shadow_ratio": lower_shadow_ratio,
            "momentum": momentum,
            "current_range": current_range,
        }

    def find_breakout_levels(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        lookback: int = 20,
    ) -> dict:
        """
        Find potential breakout levels.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            lookback: Lookback period

        Returns:
            Breakout levels dictionary
        """
        if len(highs) < lookback:
            return {}

        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]

        resistance = max(recent_highs[:-1])
        support = min(recent_lows[:-1])

        current_price = closes[-1]

        distance_to_resistance = (resistance - current_price) / current_price * 100
        distance_to_support = (current_price - support) / current_price * 100

        return {
            "resistance": resistance,
            "support": support,
            "distance_to_resistance_pct": distance_to_resistance,
            "distance_to_support_pct": distance_to_support,
            "breakout_bias": "bullish" if distance_to_resistance < distance_to_support else "bearish",
        }

    def __repr__(self) -> str:
        """String representation."""
        return "PriceAnalyzer()"

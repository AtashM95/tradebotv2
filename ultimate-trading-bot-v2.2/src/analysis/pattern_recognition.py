"""
Pattern Recognition Module for Ultimate Trading Bot v2.2.

This module provides candlestick pattern detection and
chart pattern recognition for technical analysis.
"""

import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class PatternType(str, Enum):
    """Pattern type enumeration."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    CONTINUATION = "continuation"
    REVERSAL = "reversal"


class PatternReliability(str, Enum):
    """Pattern reliability level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CandlePattern(BaseModel):
    """Candlestick pattern result."""

    name: str
    pattern_type: PatternType
    reliability: PatternReliability
    index: int
    description: str = Field(default="")
    confirmation_needed: bool = Field(default=True)


class ChartPattern(BaseModel):
    """Chart pattern result."""

    name: str
    pattern_type: PatternType
    start_index: int
    end_index: int
    breakout_level: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    reliability: PatternReliability = Field(default=PatternReliability.MEDIUM)
    description: str = Field(default="")


class PatternRecognition:
    """
    Pattern recognition for technical analysis.

    Provides detection for:
    - Single candlestick patterns
    - Multi-candle patterns
    - Chart patterns (Head & Shoulders, etc.)
    - Support/Resistance patterns
    """

    def __init__(self) -> None:
        """Initialize PatternRecognition."""
        logger.info("PatternRecognition initialized")

    def detect_all_candle_patterns(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> list[CandlePattern]:
        """
        Detect all candlestick patterns.

        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices

        Returns:
            List of detected patterns
        """
        patterns: list[CandlePattern] = []

        patterns.extend(self.detect_doji(opens, highs, lows, closes))
        patterns.extend(self.detect_hammer(opens, highs, lows, closes))
        patterns.extend(self.detect_engulfing(opens, highs, lows, closes))
        patterns.extend(self.detect_morning_evening_star(opens, highs, lows, closes))
        patterns.extend(self.detect_three_white_soldiers(opens, highs, lows, closes))
        patterns.extend(self.detect_three_black_crows(opens, highs, lows, closes))
        patterns.extend(self.detect_harami(opens, highs, lows, closes))
        patterns.extend(self.detect_marubozu(opens, highs, lows, closes))

        return patterns

    def detect_doji(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
        threshold: float = 0.1,
    ) -> list[CandlePattern]:
        """
        Detect Doji patterns.

        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices
            threshold: Body/range ratio threshold

        Returns:
            List of Doji patterns
        """
        patterns: list[CandlePattern] = []

        for i in range(len(closes)):
            body = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]

            if total_range == 0:
                continue

            body_ratio = body / total_range

            if body_ratio < threshold:
                upper_shadow = highs[i] - max(opens[i], closes[i])
                lower_shadow = min(opens[i], closes[i]) - lows[i]

                if abs(upper_shadow - lower_shadow) < total_range * 0.1:
                    name = "doji"
                    description = "Indecision pattern, equal buying/selling pressure"
                elif upper_shadow > lower_shadow * 2:
                    name = "gravestone_doji"
                    description = "Bearish reversal signal at resistance"
                elif lower_shadow > upper_shadow * 2:
                    name = "dragonfly_doji"
                    description = "Bullish reversal signal at support"
                else:
                    name = "long_legged_doji"
                    description = "High indecision, potential reversal"

                patterns.append(CandlePattern(
                    name=name,
                    pattern_type=PatternType.NEUTRAL,
                    reliability=PatternReliability.MEDIUM,
                    index=i,
                    description=description,
                ))

        return patterns

    def detect_hammer(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> list[CandlePattern]:
        """
        Detect Hammer and Hanging Man patterns.

        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices

        Returns:
            List of Hammer/Hanging Man patterns
        """
        patterns: list[CandlePattern] = []

        for i in range(len(closes)):
            body = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]

            if total_range == 0:
                continue

            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]

            if lower_shadow > body * 2 and upper_shadow < body * 0.5:
                if i >= 5:
                    recent_trend = closes[i] - closes[i - 5]

                    if recent_trend < 0:
                        patterns.append(CandlePattern(
                            name="hammer",
                            pattern_type=PatternType.BULLISH,
                            reliability=PatternReliability.MEDIUM,
                            index=i,
                            description="Bullish reversal after downtrend",
                        ))
                    else:
                        patterns.append(CandlePattern(
                            name="hanging_man",
                            pattern_type=PatternType.BEARISH,
                            reliability=PatternReliability.MEDIUM,
                            index=i,
                            description="Bearish reversal after uptrend",
                        ))

            if upper_shadow > body * 2 and lower_shadow < body * 0.5:
                if i >= 5:
                    recent_trend = closes[i] - closes[i - 5]

                    if recent_trend > 0:
                        patterns.append(CandlePattern(
                            name="shooting_star",
                            pattern_type=PatternType.BEARISH,
                            reliability=PatternReliability.MEDIUM,
                            index=i,
                            description="Bearish reversal after uptrend",
                        ))
                    else:
                        patterns.append(CandlePattern(
                            name="inverted_hammer",
                            pattern_type=PatternType.BULLISH,
                            reliability=PatternReliability.LOW,
                            index=i,
                            description="Potential bullish reversal",
                        ))

        return patterns

    def detect_engulfing(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> list[CandlePattern]:
        """
        Detect Engulfing patterns.

        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices

        Returns:
            List of Engulfing patterns
        """
        patterns: list[CandlePattern] = []

        for i in range(1, len(closes)):
            prev_body_start = min(opens[i - 1], closes[i - 1])
            prev_body_end = max(opens[i - 1], closes[i - 1])
            curr_body_start = min(opens[i], closes[i])
            curr_body_end = max(opens[i], closes[i])

            prev_is_bearish = closes[i - 1] < opens[i - 1]
            curr_is_bullish = closes[i] > opens[i]

            if (
                prev_is_bearish and
                curr_is_bullish and
                curr_body_start < prev_body_start and
                curr_body_end > prev_body_end
            ):
                patterns.append(CandlePattern(
                    name="bullish_engulfing",
                    pattern_type=PatternType.BULLISH,
                    reliability=PatternReliability.HIGH,
                    index=i,
                    description="Strong bullish reversal signal",
                ))

            prev_is_bullish = closes[i - 1] > opens[i - 1]
            curr_is_bearish = closes[i] < opens[i]

            if (
                prev_is_bullish and
                curr_is_bearish and
                curr_body_start < prev_body_start and
                curr_body_end > prev_body_end
            ):
                patterns.append(CandlePattern(
                    name="bearish_engulfing",
                    pattern_type=PatternType.BEARISH,
                    reliability=PatternReliability.HIGH,
                    index=i,
                    description="Strong bearish reversal signal",
                ))

        return patterns

    def detect_morning_evening_star(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> list[CandlePattern]:
        """
        Detect Morning Star and Evening Star patterns.

        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices

        Returns:
            List of Star patterns
        """
        patterns: list[CandlePattern] = []

        for i in range(2, len(closes)):
            first_body = abs(closes[i - 2] - opens[i - 2])
            second_body = abs(closes[i - 1] - opens[i - 1])
            third_body = abs(closes[i] - opens[i])

            first_is_bearish = closes[i - 2] < opens[i - 2]
            third_is_bullish = closes[i] > opens[i]

            avg_body = (first_body + third_body) / 2
            if second_body < avg_body * 0.3:
                if first_is_bearish and third_is_bullish:
                    if closes[i] > (opens[i - 2] + closes[i - 2]) / 2:
                        patterns.append(CandlePattern(
                            name="morning_star",
                            pattern_type=PatternType.BULLISH,
                            reliability=PatternReliability.HIGH,
                            index=i,
                            description="Strong bullish reversal pattern",
                        ))

            first_is_bullish = closes[i - 2] > opens[i - 2]
            third_is_bearish = closes[i] < opens[i]

            if second_body < avg_body * 0.3:
                if first_is_bullish and third_is_bearish:
                    if closes[i] < (opens[i - 2] + closes[i - 2]) / 2:
                        patterns.append(CandlePattern(
                            name="evening_star",
                            pattern_type=PatternType.BEARISH,
                            reliability=PatternReliability.HIGH,
                            index=i,
                            description="Strong bearish reversal pattern",
                        ))

        return patterns

    def detect_three_white_soldiers(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> list[CandlePattern]:
        """
        Detect Three White Soldiers pattern.

        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices

        Returns:
            List of Three White Soldiers patterns
        """
        patterns: list[CandlePattern] = []

        for i in range(2, len(closes)):
            is_bullish = all(
                closes[i - j] > opens[i - j]
                for j in range(3)
            )

            if not is_bullish:
                continue

            is_ascending = (
                closes[i] > closes[i - 1] > closes[i - 2] and
                opens[i] > opens[i - 1] > opens[i - 2]
            )

            if not is_ascending:
                continue

            opens_within_prev_body = all(
                opens[i - j] > opens[i - j - 1] and
                opens[i - j] < closes[i - j - 1]
                for j in range(2)
            )

            if is_ascending and opens_within_prev_body:
                patterns.append(CandlePattern(
                    name="three_white_soldiers",
                    pattern_type=PatternType.BULLISH,
                    reliability=PatternReliability.HIGH,
                    index=i,
                    description="Strong bullish continuation pattern",
                ))

        return patterns

    def detect_three_black_crows(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> list[CandlePattern]:
        """
        Detect Three Black Crows pattern.

        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices

        Returns:
            List of Three Black Crows patterns
        """
        patterns: list[CandlePattern] = []

        for i in range(2, len(closes)):
            is_bearish = all(
                closes[i - j] < opens[i - j]
                for j in range(3)
            )

            if not is_bearish:
                continue

            is_descending = (
                closes[i] < closes[i - 1] < closes[i - 2] and
                opens[i] < opens[i - 1] < opens[i - 2]
            )

            if not is_descending:
                continue

            opens_within_prev_body = all(
                opens[i - j] < opens[i - j - 1] and
                opens[i - j] > closes[i - j - 1]
                for j in range(2)
            )

            if is_descending and opens_within_prev_body:
                patterns.append(CandlePattern(
                    name="three_black_crows",
                    pattern_type=PatternType.BEARISH,
                    reliability=PatternReliability.HIGH,
                    index=i,
                    description="Strong bearish continuation pattern",
                ))

        return patterns

    def detect_harami(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> list[CandlePattern]:
        """
        Detect Harami patterns.

        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices

        Returns:
            List of Harami patterns
        """
        patterns: list[CandlePattern] = []

        for i in range(1, len(closes)):
            prev_body_size = abs(closes[i - 1] - opens[i - 1])
            curr_body_size = abs(closes[i] - opens[i])

            if curr_body_size >= prev_body_size * 0.5:
                continue

            prev_body_high = max(opens[i - 1], closes[i - 1])
            prev_body_low = min(opens[i - 1], closes[i - 1])
            curr_body_high = max(opens[i], closes[i])
            curr_body_low = min(opens[i], closes[i])

            is_contained = (
                curr_body_high < prev_body_high and
                curr_body_low > prev_body_low
            )

            if not is_contained:
                continue

            prev_is_bearish = closes[i - 1] < opens[i - 1]
            curr_is_bullish = closes[i] > opens[i]

            if prev_is_bearish and curr_is_bullish:
                patterns.append(CandlePattern(
                    name="bullish_harami",
                    pattern_type=PatternType.BULLISH,
                    reliability=PatternReliability.MEDIUM,
                    index=i,
                    description="Potential bullish reversal",
                ))

            prev_is_bullish = closes[i - 1] > opens[i - 1]
            curr_is_bearish = closes[i] < opens[i]

            if prev_is_bullish and curr_is_bearish:
                patterns.append(CandlePattern(
                    name="bearish_harami",
                    pattern_type=PatternType.BEARISH,
                    reliability=PatternReliability.MEDIUM,
                    index=i,
                    description="Potential bearish reversal",
                ))

        return patterns

    def detect_marubozu(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
        shadow_threshold: float = 0.02,
    ) -> list[CandlePattern]:
        """
        Detect Marubozu patterns.

        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices
            shadow_threshold: Max shadow ratio

        Returns:
            List of Marubozu patterns
        """
        patterns: list[CandlePattern] = []

        for i in range(len(closes)):
            body = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]

            if total_range == 0 or body == 0:
                continue

            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]

            upper_ratio = upper_shadow / total_range
            lower_ratio = lower_shadow / total_range

            if upper_ratio < shadow_threshold and lower_ratio < shadow_threshold:
                if closes[i] > opens[i]:
                    patterns.append(CandlePattern(
                        name="bullish_marubozu",
                        pattern_type=PatternType.BULLISH,
                        reliability=PatternReliability.MEDIUM,
                        index=i,
                        description="Strong bullish momentum",
                    ))
                else:
                    patterns.append(CandlePattern(
                        name="bearish_marubozu",
                        pattern_type=PatternType.BEARISH,
                        reliability=PatternReliability.MEDIUM,
                        index=i,
                        description="Strong bearish momentum",
                    ))

        return patterns

    def detect_head_and_shoulders(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        window: int = 20,
    ) -> list[ChartPattern]:
        """
        Detect Head and Shoulders pattern.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            window: Window for peak detection

        Returns:
            List of Head and Shoulders patterns
        """
        patterns: list[ChartPattern] = []
        peaks = self._find_peaks(highs, window)

        if len(peaks) < 3:
            return patterns

        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]

            if (
                highs[head] > highs[left_shoulder] and
                highs[head] > highs[right_shoulder] and
                abs(highs[left_shoulder] - highs[right_shoulder]) < highs[head] * 0.03
            ):
                neckline = min(
                    lows[left_shoulder:head],
                    default=lows[left_shoulder],
                )

                target = highs[head] - (highs[head] - neckline)

                patterns.append(ChartPattern(
                    name="head_and_shoulders",
                    pattern_type=PatternType.BEARISH,
                    start_index=left_shoulder,
                    end_index=right_shoulder,
                    breakout_level=neckline,
                    target_price=target,
                    stop_price=highs[head],
                    reliability=PatternReliability.HIGH,
                    description="Bearish reversal pattern",
                ))

        return patterns

    def detect_inverse_head_and_shoulders(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        window: int = 20,
    ) -> list[ChartPattern]:
        """
        Detect Inverse Head and Shoulders pattern.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            window: Window for trough detection

        Returns:
            List of Inverse Head and Shoulders patterns
        """
        patterns: list[ChartPattern] = []
        troughs = self._find_troughs(lows, window)

        if len(troughs) < 3:
            return patterns

        for i in range(len(troughs) - 2):
            left_shoulder = troughs[i]
            head = troughs[i + 1]
            right_shoulder = troughs[i + 2]

            if (
                lows[head] < lows[left_shoulder] and
                lows[head] < lows[right_shoulder] and
                abs(lows[left_shoulder] - lows[right_shoulder]) < abs(lows[head]) * 0.03
            ):
                neckline = max(
                    highs[left_shoulder:head],
                    default=highs[left_shoulder],
                )

                target = neckline + (neckline - lows[head])

                patterns.append(ChartPattern(
                    name="inverse_head_and_shoulders",
                    pattern_type=PatternType.BULLISH,
                    start_index=left_shoulder,
                    end_index=right_shoulder,
                    breakout_level=neckline,
                    target_price=target,
                    stop_price=lows[head],
                    reliability=PatternReliability.HIGH,
                    description="Bullish reversal pattern",
                ))

        return patterns

    def detect_double_top(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        window: int = 15,
        tolerance: float = 0.02,
    ) -> list[ChartPattern]:
        """
        Detect Double Top pattern.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            window: Window for peak detection
            tolerance: Price tolerance for matching peaks

        Returns:
            List of Double Top patterns
        """
        patterns: list[ChartPattern] = []
        peaks = self._find_peaks(highs, window)

        if len(peaks) < 2:
            return patterns

        for i in range(len(peaks) - 1):
            first_peak = peaks[i]
            second_peak = peaks[i + 1]

            if second_peak - first_peak < window:
                continue

            price_diff = abs(highs[first_peak] - highs[second_peak])
            avg_price = (highs[first_peak] + highs[second_peak]) / 2

            if price_diff / avg_price < tolerance:
                neckline = min(lows[first_peak:second_peak])
                height = avg_price - neckline
                target = neckline - height

                patterns.append(ChartPattern(
                    name="double_top",
                    pattern_type=PatternType.BEARISH,
                    start_index=first_peak,
                    end_index=second_peak,
                    breakout_level=neckline,
                    target_price=target,
                    stop_price=avg_price,
                    reliability=PatternReliability.HIGH,
                    description="Bearish reversal at resistance",
                ))

        return patterns

    def detect_double_bottom(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        window: int = 15,
        tolerance: float = 0.02,
    ) -> list[ChartPattern]:
        """
        Detect Double Bottom pattern.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            window: Window for trough detection
            tolerance: Price tolerance for matching troughs

        Returns:
            List of Double Bottom patterns
        """
        patterns: list[ChartPattern] = []
        troughs = self._find_troughs(lows, window)

        if len(troughs) < 2:
            return patterns

        for i in range(len(troughs) - 1):
            first_trough = troughs[i]
            second_trough = troughs[i + 1]

            if second_trough - first_trough < window:
                continue

            price_diff = abs(lows[first_trough] - lows[second_trough])
            avg_price = (lows[first_trough] + lows[second_trough]) / 2

            if price_diff / avg_price < tolerance:
                neckline = max(highs[first_trough:second_trough])
                height = neckline - avg_price
                target = neckline + height

                patterns.append(ChartPattern(
                    name="double_bottom",
                    pattern_type=PatternType.BULLISH,
                    start_index=first_trough,
                    end_index=second_trough,
                    breakout_level=neckline,
                    target_price=target,
                    stop_price=avg_price,
                    reliability=PatternReliability.HIGH,
                    description="Bullish reversal at support",
                ))

        return patterns

    def detect_triangle(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        min_length: int = 10,
    ) -> list[ChartPattern]:
        """
        Detect Triangle patterns (ascending, descending, symmetrical).

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            min_length: Minimum pattern length

        Returns:
            List of Triangle patterns
        """
        patterns: list[ChartPattern] = []
        n = len(closes)

        if n < min_length:
            return patterns

        for start in range(0, n - min_length, 5):
            end = min(start + min_length * 3, n)

            window_highs = highs[start:end]
            window_lows = lows[start:end]

            high_slope = self._calculate_slope(window_highs)
            low_slope = self._calculate_slope(window_lows)

            if low_slope > 0 and abs(high_slope) < 0.001:
                patterns.append(ChartPattern(
                    name="ascending_triangle",
                    pattern_type=PatternType.BULLISH,
                    start_index=start,
                    end_index=end - 1,
                    breakout_level=max(window_highs),
                    target_price=max(window_highs) + (max(window_highs) - min(window_lows)),
                    reliability=PatternReliability.MEDIUM,
                    description="Bullish continuation pattern",
                ))

            elif high_slope < 0 and abs(low_slope) < 0.001:
                patterns.append(ChartPattern(
                    name="descending_triangle",
                    pattern_type=PatternType.BEARISH,
                    start_index=start,
                    end_index=end - 1,
                    breakout_level=min(window_lows),
                    target_price=min(window_lows) - (max(window_highs) - min(window_lows)),
                    reliability=PatternReliability.MEDIUM,
                    description="Bearish continuation pattern",
                ))

            elif high_slope < 0 and low_slope > 0:
                patterns.append(ChartPattern(
                    name="symmetrical_triangle",
                    pattern_type=PatternType.NEUTRAL,
                    start_index=start,
                    end_index=end - 1,
                    reliability=PatternReliability.MEDIUM,
                    description="Continuation pattern, direction depends on breakout",
                ))

        return patterns

    def detect_wedge(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        min_length: int = 15,
    ) -> list[ChartPattern]:
        """
        Detect Wedge patterns (rising/falling).

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            min_length: Minimum pattern length

        Returns:
            List of Wedge patterns
        """
        patterns: list[ChartPattern] = []
        n = len(closes)

        if n < min_length:
            return patterns

        for start in range(0, n - min_length, 5):
            end = min(start + min_length * 2, n)

            window_highs = highs[start:end]
            window_lows = lows[start:end]

            high_slope = self._calculate_slope(window_highs)
            low_slope = self._calculate_slope(window_lows)

            if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
                patterns.append(ChartPattern(
                    name="rising_wedge",
                    pattern_type=PatternType.BEARISH,
                    start_index=start,
                    end_index=end - 1,
                    breakout_level=min(window_lows),
                    reliability=PatternReliability.MEDIUM,
                    description="Bearish reversal pattern",
                ))

            elif high_slope < 0 and low_slope < 0 and high_slope > low_slope:
                patterns.append(ChartPattern(
                    name="falling_wedge",
                    pattern_type=PatternType.BULLISH,
                    start_index=start,
                    end_index=end - 1,
                    breakout_level=max(window_highs),
                    reliability=PatternReliability.MEDIUM,
                    description="Bullish reversal pattern",
                ))

        return patterns

    def find_support_resistance(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        window: int = 10,
        num_levels: int = 5,
    ) -> dict:
        """
        Find support and resistance levels.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            window: Window for level detection
            num_levels: Number of levels to return

        Returns:
            Dictionary with support and resistance levels
        """
        peaks = self._find_peaks(highs, window)
        troughs = self._find_troughs(lows, window)

        resistance_levels = sorted(
            [highs[i] for i in peaks],
            reverse=True,
        )[:num_levels]

        support_levels = sorted(
            [lows[i] for i in troughs],
        )[:num_levels]

        return {
            "resistance": resistance_levels,
            "support": support_levels,
            "current_price": closes[-1] if closes else 0,
            "nearest_resistance": min(
                [r for r in resistance_levels if r > closes[-1]],
                default=None,
            ) if closes else None,
            "nearest_support": max(
                [s for s in support_levels if s < closes[-1]],
                default=None,
            ) if closes else None,
        }

    def _find_peaks(
        self,
        data: list[float],
        window: int,
    ) -> list[int]:
        """Find local peaks in data."""
        peaks: list[int] = []

        for i in range(window, len(data) - window):
            is_peak = True
            for j in range(1, window + 1):
                if data[i] <= data[i - j] or data[i] <= data[i + j]:
                    is_peak = False
                    break

            if is_peak:
                peaks.append(i)

        return peaks

    def _find_troughs(
        self,
        data: list[float],
        window: int,
    ) -> list[int]:
        """Find local troughs in data."""
        troughs: list[int] = []

        for i in range(window, len(data) - window):
            is_trough = True
            for j in range(1, window + 1):
                if data[i] >= data[i - j] or data[i] >= data[i + j]:
                    is_trough = False
                    break

            if is_trough:
                troughs.append(i)

        return troughs

    def _calculate_slope(self, data: list[float]) -> float:
        """Calculate linear regression slope."""
        n = len(data)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2
        y_mean = sum(data) / n

        numerator = sum((i - x_mean) * (data[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def __repr__(self) -> str:
        """String representation."""
        return "PatternRecognition()"

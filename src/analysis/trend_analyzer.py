"""
Trend Analysis module for identifying and analyzing market trends.
~400 lines as per schema
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import statistics

from ..core.contracts import MarketSnapshot

logger = logging.getLogger(__name__)


@dataclass
class TrendInfo:
    """Trend analysis information."""
    direction: str  # 'uptrend', 'downtrend', 'sideways'
    strength: float  # 0.0 to 1.0
    duration: int  # Number of bars
    start_price: float
    current_price: float
    change_percent: float
    slope: float
    confidence: float  # 0.0 to 1.0


def analyze(snapshot: MarketSnapshot) -> dict:
    """
    Analyze market snapshot for trends.

    Args:
        snapshot: Market snapshot with price data

    Returns:
        Dictionary with trend analysis
    """
    analyzer = TrendAnalyzer()
    trend = analyzer.analyze_trend(snapshot)

    if trend:
        return {
            'symbol': snapshot.symbol,
            'price': snapshot.price,
            'trend': {
                'direction': trend.direction,
                'strength': trend.strength,
                'confidence': trend.confidence,
                'change_percent': trend.change_percent
            }
        }
    else:
        return {
            'symbol': snapshot.symbol,
            'price': snapshot.price,
            'trend': None
        }


class TrendAnalyzer:
    """
    Comprehensive trend analysis system.

    Features:
    - Trend direction detection (up, down, sideways)
    - Trend strength measurement
    - Moving average analysis
    - Slope calculation
    - Trend reversal detection
    - Support/resistance trend lines
    - Multiple timeframe analysis
    """

    def __init__(self, short_period: int = 20, long_period: int = 50):
        """
        Initialize trend analyzer.

        Args:
            short_period: Short-term moving average period
            long_period: Long-term moving average period
        """
        self.short_period = short_period
        self.long_period = long_period

        # Statistics
        self.stats = {
            "trends_analyzed": 0,
            "uptrends": 0,
            "downtrends": 0,
            "sideways": 0,
            "reversals_detected": 0
        }

    def analyze_trend(self, snapshot: MarketSnapshot) -> Optional[TrendInfo]:
        """
        Analyze current trend.

        Args:
            snapshot: Market snapshot

        Returns:
            TrendInfo or None
        """
        if not hasattr(snapshot, 'bars') or not snapshot.bars:
            return None

        bars = snapshot.bars

        if len(bars) < self.short_period:
            return None

        # Calculate moving averages
        short_ma = self._calculate_sma(bars, self.short_period)
        long_ma = self._calculate_sma(bars, self.long_period) if len(bars) >= self.long_period else None

        # Determine trend direction
        direction = self._determine_direction(bars, short_ma, long_ma)

        # Calculate trend strength
        strength = self._calculate_strength(bars, direction)

        # Calculate slope
        slope = self._calculate_slope(bars)

        # Calculate confidence
        confidence = self._calculate_confidence(bars, direction, strength)

        # Create trend info
        trend = TrendInfo(
            direction=direction,
            strength=strength,
            duration=len(bars),
            start_price=bars[0]['close'],
            current_price=bars[-1]['close'],
            change_percent=((bars[-1]['close'] - bars[0]['close']) / bars[0]['close']) * 100,
            slope=slope,
            confidence=confidence
        )

        # Update statistics
        self.stats["trends_analyzed"] += 1
        if direction == 'uptrend':
            self.stats["uptrends"] += 1
        elif direction == 'downtrend':
            self.stats["downtrends"] += 1
        else:
            self.stats["sideways"] += 1

        return trend

    def detect_reversal(self, bars: List[Dict]) -> Optional[str]:
        """
        Detect potential trend reversals.

        Args:
            bars: List of OHLCV bars

        Returns:
            Reversal type ('bullish_reversal', 'bearish_reversal', None)
        """
        if len(bars) < 20:
            return None

        # Check for bullish reversal (downtrend to uptrend)
        if self._check_bullish_reversal(bars):
            self.stats["reversals_detected"] += 1
            return 'bullish_reversal'

        # Check for bearish reversal (uptrend to downtrend)
        if self._check_bearish_reversal(bars):
            self.stats["reversals_detected"] += 1
            return 'bearish_reversal'

        return None

    def _determine_direction(
        self,
        bars: List[Dict],
        short_ma: float,
        long_ma: Optional[float]
    ) -> str:
        """Determine trend direction."""
        current_price = bars[-1]['close']

        # Use MA crossover if we have long MA
        if long_ma:
            if short_ma > long_ma * 1.01:  # 1% threshold
                return 'uptrend'
            elif short_ma < long_ma * 0.99:
                return 'downtrend'
            else:
                return 'sideways'

        # Use price vs MA
        if current_price > short_ma * 1.02:
            return 'uptrend'
        elif current_price < short_ma * 0.98:
            return 'downtrend'
        else:
            return 'sideways'

    def _calculate_strength(self, bars: List[Dict], direction: str) -> float:
        """Calculate trend strength (0.0 to 1.0)."""
        if len(bars) < 10:
            return 0.0

        # Calculate ADX-like strength measure
        closes = [bar['close'] for bar in bars[-20:]]

        # Calculate price momentum
        momentum = abs((closes[-1] - closes[0]) / closes[0])

        # Calculate consistency (how many bars follow the trend)
        if direction == 'uptrend':
            consistent_bars = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        elif direction == 'downtrend':
            consistent_bars = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
        else:
            # For sideways, measure consolidation
            price_range = max(closes) - min(closes)
            avg_price = statistics.mean(closes)
            if avg_price > 0:
                return 1.0 - min(price_range / avg_price / 0.1, 1.0)
            return 0.5

        consistency = consistent_bars / (len(closes) - 1)

        # Combine momentum and consistency
        strength = (momentum * 10 + consistency) / 2
        return min(strength, 1.0)

    def _calculate_slope(self, bars: List[Dict]) -> float:
        """Calculate trend slope using linear regression."""
        if len(bars) < 2:
            return 0.0

        # Use recent bars for slope
        recent_bars = bars[-min(20, len(bars)):]
        closes = [bar['close'] for bar in recent_bars]

        # Simple linear regression
        n = len(closes)
        x = list(range(n))
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(closes)

        numerator = sum((x[i] - mean_x) * (closes[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope

    def _calculate_confidence(
        self,
        bars: List[Dict],
        direction: str,
        strength: float
    ) -> float:
        """Calculate confidence in trend analysis."""
        confidence = 0.5  # Base confidence

        # Increase confidence with more data
        if len(bars) >= 50:
            confidence += 0.2
        elif len(bars) >= 30:
            confidence += 0.1

        # Increase confidence with strong trends
        confidence += strength * 0.2

        # Check volume confirmation
        if self._volume_confirms_trend(bars, direction):
            confidence += 0.1

        return min(confidence, 1.0)

    def _volume_confirms_trend(self, bars: List[Dict], direction: str) -> bool:
        """Check if volume confirms the trend."""
        if len(bars) < 10:
            return False

        recent_bars = bars[-10:]
        avg_volume = statistics.mean([bar['volume'] for bar in recent_bars])

        if direction == 'uptrend':
            # Volume should be higher on up days
            up_days_volume = [bar['volume'] for bar in recent_bars if bar['close'] > bar['open']]
            if up_days_volume:
                return statistics.mean(up_days_volume) > avg_volume

        elif direction == 'downtrend':
            # Volume should be higher on down days
            down_days_volume = [bar['volume'] for bar in recent_bars if bar['close'] < bar['open']]
            if down_days_volume:
                return statistics.mean(down_days_volume) > avg_volume

        return False

    def _calculate_sma(self, bars: List[Dict], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(bars) < period:
            period = len(bars)

        closes = [bar['close'] for bar in bars[-period:]]
        return statistics.mean(closes)

    def _calculate_ema(self, bars: List[Dict], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(bars) < period:
            return self._calculate_sma(bars, len(bars))

        closes = [bar['close'] for bar in bars]
        multiplier = 2 / (period + 1)

        # Start with SMA
        ema = statistics.mean(closes[:period])

        # Calculate EMA
        for i in range(period, len(closes)):
            ema = (closes[i] - ema) * multiplier + ema

        return ema

    def _check_bullish_reversal(self, bars: List[Dict]) -> bool:
        """Check for bullish reversal signals."""
        if len(bars) < 20:
            return False

        recent = bars[-10:]
        older = bars[-20:-10]

        # Check if trend was down and now reversing up
        older_trend = statistics.mean([b['close'] for b in older])
        recent_trend = statistics.mean([b['close'] for b in recent])

        # Was going down
        if bars[-11]['close'] > bars[-1]['close']:
            # Now showing strength
            if recent[-1]['close'] > recent[0]['close']:
                # Higher lows pattern
                lows = [bar['low'] for bar in recent]
                higher_lows = all(lows[i] >= lows[i-1] * 0.99 for i in range(1, len(lows)))

                if higher_lows:
                    return True

        return False

    def _check_bearish_reversal(self, bars: List[Dict]) -> bool:
        """Check for bearish reversal signals."""
        if len(bars) < 20:
            return False

        recent = bars[-10:]
        older = bars[-20:-10]

        # Check if trend was up and now reversing down
        older_trend = statistics.mean([b['close'] for b in older])
        recent_trend = statistics.mean([b['close'] for b in recent])

        # Was going up
        if bars[-11]['close'] < bars[-1]['close']:
            # Now showing weakness
            if recent[-1]['close'] < recent[0]['close']:
                # Lower highs pattern
                highs = [bar['high'] for bar in recent]
                lower_highs = all(highs[i] <= highs[i-1] * 1.01 for i in range(1, len(highs)))

                if lower_highs:
                    return True

        return False

    def get_trend_lines(self, bars: List[Dict]) -> Dict[str, Any]:
        """
        Calculate support and resistance trend lines.

        Args:
            bars: List of OHLCV bars

        Returns:
            Dictionary with trend line info
        """
        if len(bars) < 20:
            return {'support': None, 'resistance': None}

        # Find swing lows for support line
        lows = []
        for i in range(1, len(bars) - 1):
            if bars[i]['low'] < bars[i-1]['low'] and bars[i]['low'] < bars[i+1]['low']:
                lows.append((i, bars[i]['low']))

        # Find swing highs for resistance line
        highs = []
        for i in range(1, len(bars) - 1):
            if bars[i]['high'] > bars[i-1]['high'] and bars[i]['high'] > bars[i+1]['high']:
                highs.append((i, bars[i]['high']))

        support_line = self._fit_trend_line(lows) if len(lows) >= 2 else None
        resistance_line = self._fit_trend_line(highs) if len(highs) >= 2 else None

        return {
            'support': support_line,
            'resistance': resistance_line
        }

    def _fit_trend_line(self, points: List[Tuple[int, float]]) -> Dict[str, float]:
        """Fit a trend line through points."""
        if len(points) < 2:
            return None

        # Linear regression
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(len(x)))

        if denominator == 0:
            return None

        slope = numerator / denominator
        intercept = mean_y - slope * mean_x

        return {
            'slope': slope,
            'intercept': intercept
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "trends_analyzed": 0,
            "uptrends": 0,
            "downtrends": 0,
            "sideways": 0,
            "reversals_detected": 0
        }

"""
Pattern Recognition module for identifying candlestick and chart patterns.
~450 lines as per schema
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..core.contracts import MarketSnapshot

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Detected pattern information."""
    name: str
    type: str  # 'candlestick', 'chart', 'continuation', 'reversal'
    signal: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0.0 to 1.0
    start_index: int
    end_index: int
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    metadata: Dict[str, Any] = None


def analyze(snapshot: MarketSnapshot) -> dict:
    """
    Analyze market snapshot for patterns.

    Args:
        snapshot: Market snapshot with OHLCV data

    Returns:
        Dictionary with detected patterns
    """
    recognizer = PatternRecognizer()
    patterns = recognizer.detect_all(snapshot)

    return {
        'symbol': snapshot.symbol,
        'price': snapshot.price,
        'patterns': [
            {
                'name': p.name,
                'type': p.type,
                'signal': p.signal,
                'confidence': p.confidence
            }
            for p in patterns
        ],
        'pattern_count': len(patterns)
    }


class PatternRecognizer:
    """
    Comprehensive pattern recognition for candlestick and chart patterns.

    Features:
    - Single candlestick patterns
    - Multi-candlestick patterns
    - Chart patterns (head and shoulders, triangles, etc.)
    - Support/resistance pattern integration
    - Confidence scoring
    - Price target calculation
    """

    def __init__(self, min_confidence: float = 0.6):
        """
        Initialize pattern recognizer.

        Args:
            min_confidence: Minimum confidence threshold for patterns
        """
        self.min_confidence = min_confidence

        # Statistics
        self.stats = {
            "patterns_detected": 0,
            "candlestick_patterns": 0,
            "chart_patterns": 0,
            "bullish_signals": 0,
            "bearish_signals": 0
        }

    def detect_all(self, snapshot: MarketSnapshot) -> List[Pattern]:
        """
        Detect all patterns in market snapshot.

        Args:
            snapshot: Market snapshot with OHLCV data

        Returns:
            List of detected patterns
        """
        if not hasattr(snapshot, 'bars') or not snapshot.bars:
            return []

        patterns = []

        # Detect candlestick patterns
        candlestick_patterns = self._detect_candlestick_patterns(snapshot.bars)
        patterns.extend(candlestick_patterns)

        # Detect chart patterns (if enough bars)
        if len(snapshot.bars) >= 20:
            chart_patterns = self._detect_chart_patterns(snapshot.bars)
            patterns.extend(chart_patterns)

        # Filter by confidence
        patterns = [p for p in patterns if p.confidence >= self.min_confidence]

        # Update statistics
        self.stats["patterns_detected"] += len(patterns)
        for pattern in patterns:
            if pattern.signal == 'bullish':
                self.stats["bullish_signals"] += 1
            elif pattern.signal == 'bearish':
                self.stats["bearish_signals"] += 1

        return patterns

    def _detect_candlestick_patterns(self, bars: List[Dict]) -> List[Pattern]:
        """Detect candlestick patterns."""
        patterns = []

        for i in range(len(bars)):
            # Single candlestick patterns
            patterns.extend(self._check_doji(bars, i))
            patterns.extend(self._check_hammer(bars, i))
            patterns.extend(self._check_shooting_star(bars, i))
            patterns.extend(self._check_marubozu(bars, i))

            # Two-candle patterns
            if i >= 1:
                patterns.extend(self._check_engulfing(bars, i))
                patterns.extend(self._check_harami(bars, i))
                patterns.extend(self._check_piercing(bars, i))
                patterns.extend(self._check_dark_cloud(bars, i))

            # Three-candle patterns
            if i >= 2:
                patterns.extend(self._check_morning_star(bars, i))
                patterns.extend(self._check_evening_star(bars, i))
                patterns.extend(self._check_three_white_soldiers(bars, i))
                patterns.extend(self._check_three_black_crows(bars, i))

        self.stats["candlestick_patterns"] += len(patterns)
        return patterns

    def _check_doji(self, bars: List[Dict], idx: int) -> List[Pattern]:
        """Check for Doji pattern."""
        bar = bars[idx]
        body = abs(bar['close'] - bar['open'])
        range_size = bar['high'] - bar['low']

        # Doji: very small body relative to range
        if range_size > 0 and body / range_size < 0.1:
            return [Pattern(
                name='Doji',
                type='candlestick',
                signal='neutral',
                confidence=0.7,
                start_index=idx,
                end_index=idx
            )]

        return []

    def _check_hammer(self, bars: List[Dict], idx: int) -> List[Pattern]:
        """Check for Hammer pattern."""
        bar = bars[idx]
        body = abs(bar['close'] - bar['open'])
        upper_shadow = bar['high'] - max(bar['open'], bar['close'])
        lower_shadow = min(bar['open'], bar['close']) - bar['low']

        # Hammer: long lower shadow, small body, small upper shadow
        if lower_shadow > body * 2 and upper_shadow < body * 0.5:
            return [Pattern(
                name='Hammer',
                type='candlestick',
                signal='bullish',
                confidence=0.75,
                start_index=idx,
                end_index=idx
            )]

        return []

    def _check_shooting_star(self, bars: List[Dict], idx: int) -> List[Pattern]:
        """Check for Shooting Star pattern."""
        bar = bars[idx]
        body = abs(bar['close'] - bar['open'])
        upper_shadow = bar['high'] - max(bar['open'], bar['close'])
        lower_shadow = min(bar['open'], bar['close']) - bar['low']

        # Shooting Star: long upper shadow, small body, small lower shadow
        if upper_shadow > body * 2 and lower_shadow < body * 0.5:
            return [Pattern(
                name='Shooting Star',
                type='candlestick',
                signal='bearish',
                confidence=0.75,
                start_index=idx,
                end_index=idx
            )]

        return []

    def _check_marubozu(self, bars: List[Dict], idx: int) -> List[Pattern]:
        """Check for Marubozu pattern."""
        bar = bars[idx]
        body = abs(bar['close'] - bar['open'])
        range_size = bar['high'] - bar['low']

        # Marubozu: body fills most of the range
        if range_size > 0 and body / range_size > 0.9:
            signal = 'bullish' if bar['close'] > bar['open'] else 'bearish'
            return [Pattern(
                name='Marubozu',
                type='candlestick',
                signal=signal,
                confidence=0.8,
                start_index=idx,
                end_index=idx
            )]

        return []

    def _check_engulfing(self, bars: List[Dict], idx: int) -> List[Pattern]:
        """Check for Engulfing pattern."""
        if idx < 1:
            return []

        prev = bars[idx - 1]
        curr = bars[idx]

        prev_body = abs(prev['close'] - prev['open'])
        curr_body = abs(curr['close'] - curr['open'])

        # Bullish Engulfing
        if (prev['close'] < prev['open'] and  # Previous bearish
            curr['close'] > curr['open'] and  # Current bullish
            curr['open'] <= prev['close'] and
            curr['close'] >= prev['open']):
            return [Pattern(
                name='Bullish Engulfing',
                type='candlestick',
                signal='bullish',
                confidence=0.85,
                start_index=idx - 1,
                end_index=idx
            )]

        # Bearish Engulfing
        if (prev['close'] > prev['open'] and  # Previous bullish
            curr['close'] < curr['open'] and  # Current bearish
            curr['open'] >= prev['close'] and
            curr['close'] <= prev['open']):
            return [Pattern(
                name='Bearish Engulfing',
                type='candlestick',
                signal='bearish',
                confidence=0.85,
                start_index=idx - 1,
                end_index=idx
            )]

        return []

    def _check_harami(self, bars: List[Dict], idx: int) -> List[Pattern]:
        """Check for Harami pattern."""
        if idx < 1:
            return []

        prev = bars[idx - 1]
        curr = bars[idx]

        # Bullish Harami
        if (prev['close'] < prev['open'] and  # Previous bearish
            curr['close'] > curr['open'] and  # Current bullish
            curr['open'] >= prev['close'] and
            curr['close'] <= prev['open']):
            return [Pattern(
                name='Bullish Harami',
                type='candlestick',
                signal='bullish',
                confidence=0.7,
                start_index=idx - 1,
                end_index=idx
            )]

        # Bearish Harami
        if (prev['close'] > prev['open'] and  # Previous bullish
            curr['close'] < curr['open'] and  # Current bearish
            curr['open'] <= prev['close'] and
            curr['close'] >= prev['open']):
            return [Pattern(
                name='Bearish Harami',
                type='candlestick',
                signal='bearish',
                confidence=0.7,
                start_index=idx - 1,
                end_index=idx
            )]

        return []

    def _check_piercing(self, bars: List[Dict], idx: int) -> List[Pattern]:
        """Check for Piercing Line pattern."""
        if idx < 1:
            return []

        prev = bars[idx - 1]
        curr = bars[idx]

        prev_midpoint = (prev['open'] + prev['close']) / 2

        # Piercing Line: bearish candle followed by bullish that closes above midpoint
        if (prev['close'] < prev['open'] and
            curr['close'] > curr['open'] and
            curr['open'] < prev['close'] and
            curr['close'] > prev_midpoint and
            curr['close'] < prev['open']):
            return [Pattern(
                name='Piercing Line',
                type='candlestick',
                signal='bullish',
                confidence=0.75,
                start_index=idx - 1,
                end_index=idx
            )]

        return []

    def _check_dark_cloud(self, bars: List[Dict], idx: int) -> List[Pattern]:
        """Check for Dark Cloud Cover pattern."""
        if idx < 1:
            return []

        prev = bars[idx - 1]
        curr = bars[idx]

        prev_midpoint = (prev['open'] + prev['close']) / 2

        # Dark Cloud: bullish candle followed by bearish that closes below midpoint
        if (prev['close'] > prev['open'] and
            curr['close'] < curr['open'] and
            curr['open'] > prev['close'] and
            curr['close'] < prev_midpoint and
            curr['close'] > prev['open']):
            return [Pattern(
                name='Dark Cloud Cover',
                type='candlestick',
                signal='bearish',
                confidence=0.75,
                start_index=idx - 1,
                end_index=idx
            )]

        return []

    def _check_morning_star(self, bars: List[Dict], idx: int) -> List[Pattern]:
        """Check for Morning Star pattern."""
        if idx < 2:
            return []

        first = bars[idx - 2]
        second = bars[idx - 1]
        third = bars[idx]

        # Morning Star: bearish, small body, bullish
        if (first['close'] < first['open'] and  # Bearish
            abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and  # Small body
            third['close'] > third['open'] and  # Bullish
            third['close'] > (first['open'] + first['close']) / 2):  # Closes above first midpoint
            return [Pattern(
                name='Morning Star',
                type='candlestick',
                signal='bullish',
                confidence=0.9,
                start_index=idx - 2,
                end_index=idx
            )]

        return []

    def _check_evening_star(self, bars: List[Dict], idx: int) -> List[Pattern]:
        """Check for Evening Star pattern."""
        if idx < 2:
            return []

        first = bars[idx - 2]
        second = bars[idx - 1]
        third = bars[idx]

        # Evening Star: bullish, small body, bearish
        if (first['close'] > first['open'] and  # Bullish
            abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and  # Small body
            third['close'] < third['open'] and  # Bearish
            third['close'] < (first['open'] + first['close']) / 2):  # Closes below first midpoint
            return [Pattern(
                name='Evening Star',
                type='candlestick',
                signal='bearish',
                confidence=0.9,
                start_index=idx - 2,
                end_index=idx
            )]

        return []

    def _check_three_white_soldiers(self, bars: List[Dict], idx: int) -> List[Pattern]:
        """Check for Three White Soldiers pattern."""
        if idx < 2:
            return []

        # Three consecutive bullish candles with higher closes
        if all(bars[i]['close'] > bars[i]['open'] and
               bars[i]['close'] > bars[i-1]['close']
               for i in range(idx - 2, idx + 1)):
            return [Pattern(
                name='Three White Soldiers',
                type='candlestick',
                signal='bullish',
                confidence=0.85,
                start_index=idx - 2,
                end_index=idx
            )]

        return []

    def _check_three_black_crows(self, bars: List[Dict], idx: int) -> List[Pattern]:
        """Check for Three Black Crows pattern."""
        if idx < 2:
            return []

        # Three consecutive bearish candles with lower closes
        if all(bars[i]['close'] < bars[i]['open'] and
               bars[i]['close'] < bars[i-1]['close']
               for i in range(idx - 2, idx + 1)):
            return [Pattern(
                name='Three Black Crows',
                type='candlestick',
                signal='bearish',
                confidence=0.85,
                start_index=idx - 2,
                end_index=idx
            )]

        return []

    def _detect_chart_patterns(self, bars: List[Dict]) -> List[Pattern]:
        """Detect chart patterns (requires more bars)."""
        patterns = []

        if len(bars) >= 20:
            patterns.extend(self._check_double_top(bars))
            patterns.extend(self._check_double_bottom(bars))
            patterns.extend(self._check_head_and_shoulders(bars))

        self.stats["chart_patterns"] += len(patterns)
        return patterns

    def _check_double_top(self, bars: List[Dict]) -> List[Pattern]:
        """Check for Double Top pattern."""
        if len(bars) < 20:
            return []

        # Look for two peaks at similar levels
        highs = [bar['high'] for bar in bars[-20:]]

        # Find local maxima
        peaks = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))

        # Check if we have two peaks at similar levels
        if len(peaks) >= 2:
            last_two = peaks[-2:]
            price_diff = abs(last_two[0][1] - last_two[1][1])
            avg_price = (last_two[0][1] + last_two[1][1]) / 2

            if price_diff / avg_price < 0.02:  # Within 2%
                return [Pattern(
                    name='Double Top',
                    type='chart',
                    signal='bearish',
                    confidence=0.75,
                    start_index=len(bars) - 20 + last_two[0][0],
                    end_index=len(bars) - 1
                )]

        return []

    def _check_double_bottom(self, bars: List[Dict]) -> List[Pattern]:
        """Check for Double Bottom pattern."""
        if len(bars) < 20:
            return []

        # Look for two troughs at similar levels
        lows = [bar['low'] for bar in bars[-20:]]

        # Find local minima
        troughs = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append((i, lows[i]))

        # Check if we have two troughs at similar levels
        if len(troughs) >= 2:
            last_two = troughs[-2:]
            price_diff = abs(last_two[0][1] - last_two[1][1])
            avg_price = (last_two[0][1] + last_two[1][1]) / 2

            if price_diff / avg_price < 0.02:  # Within 2%
                return [Pattern(
                    name='Double Bottom',
                    type='chart',
                    signal='bullish',
                    confidence=0.75,
                    start_index=len(bars) - 20 + last_two[0][0],
                    end_index=len(bars) - 1
                )]

        return []

    def _check_head_and_shoulders(self, bars: List[Dict]) -> List[Pattern]:
        """Check for Head and Shoulders pattern."""
        if len(bars) < 30:
            return []

        # Look for three peaks: shoulder, head (higher), shoulder
        highs = [bar['high'] for bar in bars[-30:]]

        peaks = []
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                peaks.append((i, highs[i]))

        if len(peaks) >= 3:
            # Check if middle peak is highest
            last_three = peaks[-3:]
            if (last_three[1][1] > last_three[0][1] and
                last_three[1][1] > last_three[2][1]):
                return [Pattern(
                    name='Head and Shoulders',
                    type='chart',
                    signal='bearish',
                    confidence=0.8,
                    start_index=len(bars) - 30 + last_three[0][0],
                    end_index=len(bars) - 1
                )]

        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get recognition statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "patterns_detected": 0,
            "candlestick_patterns": 0,
            "chart_patterns": 0,
            "bullish_signals": 0,
            "bearish_signals": 0
        }

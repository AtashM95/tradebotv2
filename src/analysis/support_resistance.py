"""
Support and Resistance Level Analysis module.
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
class Level:
    """Support or resistance level."""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float  # 0.0 to 1.0
    touches: int  # Number of times price touched this level
    first_touch: datetime
    last_touch: datetime
    broken: bool = False


def analyze(snapshot: MarketSnapshot) -> dict:
    """
    Analyze market snapshot for support and resistance levels.

    Args:
        snapshot: Market snapshot with OHLCV data

    Returns:
        Dictionary with support/resistance analysis
    """
    analyzer = SupportResistanceAnalyzer()
    levels = analyzer.find_levels(snapshot)

    return {
        'symbol': snapshot.symbol,
        'price': snapshot.price,
        'support_levels': [
            {'price': l.price, 'strength': l.strength, 'touches': l.touches}
            for l in levels if l.level_type == 'support'
        ],
        'resistance_levels': [
            {'price': l.price, 'strength': l.strength, 'touches': l.touches}
            for l in levels if l.level_type == 'resistance'
        ]
    }


class SupportResistanceAnalyzer:
    """
    Advanced support and resistance level detection.

    Features:
    - Automatic level detection from price history
    - Level strength calculation
    - Touch counting
    - Breakout detection
    - Dynamic level adjustment
    - Multiple timeframe analysis
    - Clustering algorithm for consolidation
    """

    def __init__(
        self,
        min_touches: int = 2,
        tolerance: float = 0.01,  # 1% price tolerance
        min_strength: float = 0.5
    ):
        """
        Initialize support/resistance analyzer.

        Args:
            min_touches: Minimum touches to form a valid level
            tolerance: Price tolerance for level matching (as percentage)
            min_strength: Minimum strength threshold
        """
        self.min_touches = min_touches
        self.tolerance = tolerance
        self.min_strength = min_strength

        # Statistics
        self.stats = {
            "levels_detected": 0,
            "support_levels": 0,
            "resistance_levels": 0,
            "breakouts_detected": 0
        }

    def find_levels(self, snapshot: MarketSnapshot) -> List[Level]:
        """
        Find support and resistance levels.

        Args:
            snapshot: Market snapshot

        Returns:
            List of Level objects
        """
        if not hasattr(snapshot, 'bars') or not snapshot.bars:
            return []

        bars = snapshot.bars

        if len(bars) < 20:
            return []

        # Find swing points
        swing_highs = self._find_swing_highs(bars)
        swing_lows = self._find_swing_lows(bars)

        # Cluster swing points into levels
        resistance_levels = self._cluster_into_levels(swing_highs, 'resistance', bars)
        support_levels = self._cluster_into_levels(swing_lows, 'support', bars)

        # Combine and filter by strength
        all_levels = resistance_levels + support_levels
        filtered_levels = [l for l in all_levels if l.strength >= self.min_strength]

        # Sort by strength
        filtered_levels.sort(key=lambda x: x.strength, reverse=True)

        # Update statistics
        self.stats["levels_detected"] += len(filtered_levels)
        self.stats["support_levels"] += len([l for l in filtered_levels if l.level_type == 'support'])
        self.stats["resistance_levels"] += len([l for l in filtered_levels if l.level_type == 'resistance'])

        return filtered_levels

    def check_breakout(
        self,
        current_price: float,
        levels: List[Level],
        direction: str = 'both'
    ) -> Optional[Level]:
        """
        Check if price has broken through a level.

        Args:
            current_price: Current price
            levels: List of levels to check
            direction: 'up', 'down', or 'both'

        Returns:
            Broken level or None
        """
        for level in levels:
            if direction in ['up', 'both']:
                if level.level_type == 'resistance' and current_price > level.price * 1.01:
                    level.broken = True
                    self.stats["breakouts_detected"] += 1
                    return level

            if direction in ['down', 'both']:
                if level.level_type == 'support' and current_price < level.price * 0.99:
                    level.broken = True
                    self.stats["breakouts_detected"] += 1
                    return level

        return None

    def get_nearest_levels(
        self,
        current_price: float,
        levels: List[Level],
        count: int = 3
    ) -> Dict[str, List[Level]]:
        """
        Get nearest support and resistance levels to current price.

        Args:
            current_price: Current price
            levels: All levels
            count: Number of levels to return for each type

        Returns:
            Dictionary with nearest support and resistance levels
        """
        support_levels = [l for l in levels if l.level_type == 'support' and l.price < current_price]
        resistance_levels = [l for l in levels if l.level_type == 'resistance' and l.price > current_price]

        # Sort support descending (nearest first)
        support_levels.sort(key=lambda x: x.price, reverse=True)

        # Sort resistance ascending (nearest first)
        resistance_levels.sort(key=lambda x: x.price)

        return {
            'support': support_levels[:count],
            'resistance': resistance_levels[:count]
        }

    def _find_swing_highs(self, bars: List[Dict], window: int = 5) -> List[Tuple[int, float, datetime]]:
        """Find swing high points."""
        swing_highs = []

        for i in range(window, len(bars) - window):
            current_high = bars[i]['high']

            # Check if it's highest in the window
            is_swing_high = True
            for j in range(i - window, i + window + 1):
                if j != i and bars[j]['high'] > current_high:
                    is_swing_high = False
                    break

            if is_swing_high:
                timestamp = bars[i].get('timestamp', datetime.now())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                swing_highs.append((i, current_high, timestamp))

        return swing_highs

    def _find_swing_lows(self, bars: List[Dict], window: int = 5) -> List[Tuple[int, float, datetime]]:
        """Find swing low points."""
        swing_lows = []

        for i in range(window, len(bars) - window):
            current_low = bars[i]['low']

            # Check if it's lowest in the window
            is_swing_low = True
            for j in range(i - window, i + window + 1):
                if j != i and bars[j]['low'] < current_low:
                    is_swing_low = False
                    break

            if is_swing_low:
                timestamp = bars[i].get('timestamp', datetime.now())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                swing_lows.append((i, current_low, timestamp))

        return swing_lows

    def _cluster_into_levels(
        self,
        swing_points: List[Tuple[int, float, datetime]],
        level_type: str,
        bars: List[Dict]
    ) -> List[Level]:
        """Cluster swing points into levels."""
        if not swing_points:
            return []

        # Group swing points by price proximity
        clusters = []

        for idx, price, timestamp in swing_points:
            # Find cluster to add to
            found_cluster = False

            for cluster in clusters:
                cluster_avg_price = statistics.mean([p for _, p, _ in cluster])

                # Check if within tolerance
                if abs(price - cluster_avg_price) / cluster_avg_price <= self.tolerance:
                    cluster.append((idx, price, timestamp))
                    found_cluster = True
                    break

            if not found_cluster:
                clusters.append([(idx, price, timestamp)])

        # Convert clusters to Level objects
        levels = []

        for cluster in clusters:
            if len(cluster) >= self.min_touches:
                # Calculate average price
                avg_price = statistics.mean([p for _, p, _ in cluster])

                # Calculate strength based on touches and recency
                strength = self._calculate_level_strength(cluster, len(bars))

                # Get first and last touch
                sorted_cluster = sorted(cluster, key=lambda x: x[2])
                first_touch = sorted_cluster[0][2]
                last_touch = sorted_cluster[-1][2]

                level = Level(
                    price=avg_price,
                    level_type=level_type,
                    strength=strength,
                    touches=len(cluster),
                    first_touch=first_touch,
                    last_touch=last_touch
                )

                levels.append(level)

        return levels

    def _calculate_level_strength(self, cluster: List[Tuple], total_bars: int) -> float:
        """Calculate strength of a level."""
        # Base strength from number of touches
        touch_strength = min(len(cluster) / 5.0, 1.0)  # Max at 5 touches

        # Recency factor (more recent touches = stronger)
        if cluster:
            most_recent_idx = max(idx for idx, _, _ in cluster)
            recency = most_recent_idx / total_bars
            recency_strength = recency * 0.5  # Up to 0.5 from recency

            # Price consistency (tighter cluster = stronger)
            prices = [p for _, p, _ in cluster]
            if len(prices) > 1:
                price_std = statistics.stdev(prices)
                avg_price = statistics.mean(prices)
                consistency = 1.0 - min(price_std / avg_price / 0.02, 1.0)  # 2% threshold
            else:
                consistency = 1.0

            consistency_strength = consistency * 0.3  # Up to 0.3 from consistency

            total_strength = (touch_strength * 0.5 + recency_strength + consistency_strength)
            return min(total_strength, 1.0)

        return 0.0

    def calculate_risk_reward(
        self,
        entry_price: float,
        direction: str,
        levels: List[Level]
    ) -> Dict[str, Any]:
        """
        Calculate risk/reward ratio for a trade.

        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            levels: All support/resistance levels

        Returns:
            Risk/reward analysis
        """
        if direction == 'long':
            # Stop loss at nearest support
            support_below = [l for l in levels if l.level_type == 'support' and l.price < entry_price]
            stop_loss = max([l.price for l in support_below]) if support_below else entry_price * 0.95

            # Target at nearest resistance
            resistance_above = [l for l in levels if l.level_type == 'resistance' and l.price > entry_price]
            target = min([l.price for l in resistance_above]) if resistance_above else entry_price * 1.05

        else:  # short
            # Stop loss at nearest resistance
            resistance_above = [l for l in levels if l.level_type == 'resistance' and l.price > entry_price]
            stop_loss = min([l.price for l in resistance_above]) if resistance_above else entry_price * 1.05

            # Target at nearest support
            support_below = [l for l in levels if l.level_type == 'support' and l.price < entry_price]
            target = max([l.price for l in support_below]) if support_below else entry_price * 0.95

        # Calculate risk and reward
        risk = abs(entry_price - stop_loss)
        reward = abs(target - entry_price)

        risk_reward_ratio = reward / risk if risk > 0 else 0

        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target': target,
            'risk': risk,
            'reward': reward,
            'risk_reward_ratio': risk_reward_ratio,
            'direction': direction
        }

    def get_pivot_points(self, bars: List[Dict]) -> Dict[str, float]:
        """
        Calculate pivot points (floor trader pivots).

        Args:
            bars: OHLCV bars

        Returns:
            Dictionary with pivot levels
        """
        if not bars:
            return {}

        # Use previous day's data
        prev_bar = bars[-2] if len(bars) >= 2 else bars[-1]

        high = prev_bar['high']
        low = prev_bar['low']
        close = prev_bar['close']

        # Calculate pivot point
        pivot = (high + low + close) / 3

        # Calculate resistance levels
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)

        # Calculate support levels
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)

        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            's1': s1,
            's2': s2,
            's3': s3
        }

    def get_fibonacci_levels(
        self,
        swing_high: float,
        swing_low: float
    ) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.

        Args:
            swing_high: Recent swing high
            swing_low: Recent swing low

        Returns:
            Dictionary with Fibonacci levels
        """
        diff = swing_high - swing_low

        return {
            '0.0': swing_low,
            '23.6': swing_low + diff * 0.236,
            '38.2': swing_low + diff * 0.382,
            '50.0': swing_low + diff * 0.5,
            '61.8': swing_low + diff * 0.618,
            '78.6': swing_low + diff * 0.786,
            '100.0': swing_high
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "levels_detected": 0,
            "support_levels": 0,
            "resistance_levels": 0,
            "breakouts_detected": 0
        }

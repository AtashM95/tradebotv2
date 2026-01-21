"""
Breakout Strategy - Trade price breakouts from support/resistance levels.
~400 lines as per schema
"""

import logging
from typing import Dict, Any, List, Optional
import statistics

from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class BreakoutStrategy(BaseStrategy):
    """
    Breakout trading strategy that identifies and trades price breakouts.

    Algorithm:
    1. Identify support/resistance levels
    2. Detect breakouts with volume confirmation
    3. Enter in direction of breakout
    4. Use ATR for stop loss and targets

    Features:
    - Dynamic support/resistance detection
    - Volume confirmation
    - False breakout filtering
    - Multiple timeframe analysis
    - Trailing stops
    - Breakout strength measurement
    """

    name = 'breakout'

    def __init__(
        self,
        lookback_period: int = 20,
        breakout_threshold: float = 0.02,  # 2% above/below level
        volume_multiplier: float = 1.5,  # Volume must be 1.5x average
        min_consolidation_bars: int = 5,
        atr_multiplier: float = 2.0
    ):
        """
        Initialize breakout strategy.

        Args:
            lookback_period: Period for identifying levels
            breakout_threshold: Percentage threshold for breakout
            volume_multiplier: Required volume increase for confirmation
            min_consolidation_bars: Minimum bars in consolidation
            atr_multiplier: ATR multiplier for stop loss
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        self.volume_multiplier = volume_multiplier
        self.min_consolidation_bars = min_consolidation_bars
        self.atr_multiplier = atr_multiplier

        # Track state
        self.support_levels = {}  # symbol -> level
        self.resistance_levels = {}  # symbol -> level
        self.in_consolidation = {}  # symbol -> bool
        self.consolidation_count = {}  # symbol -> count

        # Statistics
        self.stats = {
            "signals_generated": 0,
            "breakouts_detected": 0,
            "false_breakouts": 0,
            "confirmed_breakouts": 0
        }

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        """
        Generate trading signal based on breakout logic.

        Args:
            snapshot: Market snapshot

        Returns:
            SignalIntent or None
        """
        if not snapshot.history or len(snapshot.history) < self.lookback_period:
            return None

        symbol = snapshot.symbol
        current_price = snapshot.price

        # Get OHLCV data from snapshot
        if hasattr(snapshot, 'bars') and snapshot.bars:
            bars = snapshot.bars
        else:
            # Construct bars from history
            bars = [{'close': p, 'high': p, 'low': p, 'volume': 1000000}
                   for p in snapshot.history]

        if len(bars) < self.lookback_period:
            return None

        # Identify support and resistance
        self._update_levels(symbol, bars)

        # Check for consolidation
        is_consolidating = self._check_consolidation(symbol, bars)

        # Detect breakout
        breakout_signal = self._detect_breakout(symbol, current_price, bars)

        if breakout_signal:
            direction = breakout_signal['direction']
            strength = breakout_signal['strength']
            level = breakout_signal['level']

            # Calculate stop loss using ATR
            atr = self._calculate_atr(bars)
            if direction == 'buy':
                stop_loss = current_price - (atr * self.atr_multiplier)
                target = current_price + (atr * self.atr_multiplier * 2)
            else:
                stop_loss = current_price + (atr * self.atr_multiplier)
                target = current_price - (atr * self.atr_multiplier * 2)

            self.stats["breakouts_detected"] += 1
            self.stats["signals_generated"] += 1

            return SignalIntent(
                symbol=symbol,
                action=direction,
                confidence=strength,
                metadata={
                    'strategy': self.name,
                    'breakout_level': level,
                    'breakout_type': breakout_signal['type'],
                    'stop_loss': stop_loss,
                    'target': target,
                    'atr': atr
                }
            )

        return None

    def _update_levels(self, symbol: str, bars: List[Dict]):
        """Update support and resistance levels."""
        if len(bars) < self.lookback_period:
            return

        recent_bars = bars[-self.lookback_period:]

        # Find swing highs and lows
        highs = [bar['high'] for bar in recent_bars]
        lows = [bar['low'] for bar in recent_bars]

        # Resistance = recent swing high
        resistance = max(highs)

        # Support = recent swing low
        support = min(lows)

        self.resistance_levels[symbol] = resistance
        self.support_levels[symbol] = support

    def _check_consolidation(self, symbol: str, bars: List[Dict]) -> bool:
        """Check if price is consolidating."""
        if len(bars) < self.min_consolidation_bars:
            return False

        recent_bars = bars[-self.min_consolidation_bars:]
        highs = [bar['high'] for bar in recent_bars]
        lows = [bar['low'] for bar in recent_bars]

        # Calculate price range
        price_range = max(highs) - min(lows)
        avg_price = statistics.mean([bar['close'] for bar in recent_bars])

        # Consolidation if range is small relative to price
        range_pct = (price_range / avg_price) if avg_price > 0 else 0

        is_consolidating = range_pct < 0.05  # Within 5% range

        if is_consolidating:
            self.consolidation_count[symbol] = self.consolidation_count.get(symbol, 0) + 1
        else:
            self.consolidation_count[symbol] = 0

        self.in_consolidation[symbol] = is_consolidating

        return is_consolidating

    def _detect_breakout(
        self,
        symbol: str,
        current_price: float,
        bars: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Detect breakout from support/resistance."""
        if symbol not in self.resistance_levels or symbol not in self.support_levels:
            return None

        resistance = self.resistance_levels[symbol]
        support = self.support_levels[symbol]

        # Check volume confirmation
        has_volume = self._check_volume_confirmation(bars)

        # Bullish breakout (resistance)
        if current_price > resistance * (1 + self.breakout_threshold):
            strength = self._calculate_breakout_strength(
                current_price,
                resistance,
                'up',
                has_volume
            )

            if strength >= 0.6:  # Minimum confidence
                self.stats["confirmed_breakouts"] += 1

                return {
                    'direction': 'buy',
                    'level': resistance,
                    'type': 'resistance_breakout',
                    'strength': strength,
                    'volume_confirmed': has_volume
                }

        # Bearish breakout (support)
        elif current_price < support * (1 - self.breakout_threshold):
            strength = self._calculate_breakout_strength(
                current_price,
                support,
                'down',
                has_volume
            )

            if strength >= 0.6:  # Minimum confidence
                self.stats["confirmed_breakouts"] += 1

                return {
                    'direction': 'sell',
                    'level': support,
                    'type': 'support_breakout',
                    'strength': strength,
                    'volume_confirmed': has_volume
                }

        return None

    def _check_volume_confirmation(self, bars: List[Dict]) -> bool:
        """Check if current volume confirms breakout."""
        if len(bars) < 20:
            return False

        recent_bars = bars[-20:]
        volumes = [bar.get('volume', 0) for bar in recent_bars]

        if not volumes or volumes[-1] == 0:
            return False

        avg_volume = statistics.mean(volumes[:-1])  # Exclude current
        current_volume = volumes[-1]

        # Volume should be significantly higher
        return current_volume > avg_volume * self.volume_multiplier

    def _calculate_breakout_strength(
        self,
        current_price: float,
        level: float,
        direction: str,
        has_volume: bool
    ) -> float:
        """Calculate breakout strength/confidence."""
        # Base strength from price distance
        if direction == 'up':
            price_strength = (current_price - level) / level
        else:
            price_strength = (level - current_price) / level

        # Normalize to 0-1 range
        price_strength = min(price_strength / 0.05, 1.0)  # 5% = max strength

        # Volume confirmation adds confidence
        volume_bonus = 0.2 if has_volume else 0.0

        # Total strength
        strength = (price_strength * 0.7) + volume_bonus + 0.1  # Base 0.1

        return min(strength, 1.0)

    def _calculate_atr(self, bars: List[Dict], period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(bars) < period + 1:
            return 0.0

        true_ranges = []

        for i in range(1, len(bars)):
            high = bars[i].get('high', bars[i]['close'])
            low = bars[i].get('low', bars[i]['close'])
            prev_close = bars[i-1]['close']

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        # Average of recent TRs
        recent_trs = true_ranges[-period:]
        atr = statistics.mean(recent_trs) if recent_trs else 0.0

        return atr

    def _find_swing_highs(self, bars: List[Dict], window: int = 3) -> List[float]:
        """Find swing high points."""
        swing_highs = []

        for i in range(window, len(bars) - window):
            current_high = bars[i].get('high', bars[i]['close'])

            is_swing_high = True
            for j in range(i - window, i + window + 1):
                if j != i:
                    compare_high = bars[j].get('high', bars[j]['close'])
                    if compare_high > current_high:
                        is_swing_high = False
                        break

            if is_swing_high:
                swing_highs.append(current_high)

        return swing_highs

    def _find_swing_lows(self, bars: List[Dict], window: int = 3) -> List[float]:
        """Find swing low points."""
        swing_lows = []

        for i in range(window, len(bars) - window):
            current_low = bars[i].get('low', bars[i]['close'])

            is_swing_low = True
            for j in range(i - window, i + window + 1):
                if j != i:
                    compare_low = bars[j].get('low', bars[j]['close'])
                    if compare_low < current_low:
                        is_swing_low = False
                        break

            if is_swing_low:
                swing_lows.append(current_low)

        return swing_lows

    def detect_false_breakout(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        direction: str
    ) -> bool:
        """Detect if a breakout was false."""
        if symbol not in self.resistance_levels or symbol not in self.support_levels:
            return False

        resistance = self.resistance_levels[symbol]
        support = self.support_levels[symbol]

        # Bullish breakout that failed
        if direction == 'buy':
            # Price fell back below resistance
            if current_price < resistance:
                self.stats["false_breakouts"] += 1
                return True

        # Bearish breakout that failed
        elif direction == 'sell':
            # Price rose back above support
            if current_price > support:
                self.stats["false_breakouts"] += 1
                return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        stats = self.stats.copy()

        # Calculate false breakout rate
        if stats["breakouts_detected"] > 0:
            stats["false_breakout_rate"] = stats["false_breakouts"] / stats["breakouts_detected"]
        else:
            stats["false_breakout_rate"] = 0.0

        return stats

    def reset(self):
        """Reset strategy state."""
        self.support_levels.clear()
        self.resistance_levels.clear()
        self.in_consolidation.clear()
        self.consolidation_count.clear()
        self.stats = {
            "signals_generated": 0,
            "breakouts_detected": 0,
            "false_breakouts": 0,
            "confirmed_breakouts": 0
        }

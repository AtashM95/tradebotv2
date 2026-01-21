"""
Channel Breakout Strategy - Trade breakouts from price channels.
~400 lines as per schema
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import statistics

from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class ChannelBreakoutStrategy(BaseStrategy):
    """
    Channel breakout strategy using price channels.

    Algorithm:
    1. Identify price channels (Donchian, Keltner, etc.)
    2. Detect breakouts from channel bounds
    3. Confirm with volume and momentum
    4. Enter on breakout
    5. Exit on opposite channel break or reversal

    Features:
    - Multiple channel types
    - Dynamic channel width
    - Breakout strength measurement
    - False breakout filtering
    - Channel regression
    - Volatility-based channels
    - Parallel channel detection
    """

    name = 'channel_breakout'

    def __init__(
        self,
        channel_period: int = 20,
        channel_type: str = 'donchian',  # donchian, keltner, linear
        breakout_threshold: float = 0.01,  # 1% beyond channel
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        volume_confirmation: bool = True,
        min_channel_width_pct: float = 0.02  # 2% minimum width
    ):
        """
        Initialize channel breakout strategy.

        Args:
            channel_period: Period for channel calculation
            channel_type: Type of channel (donchian, keltner, linear)
            breakout_threshold: Threshold for breakout confirmation
            atr_period: Period for ATR calculation
            atr_multiplier: Multiplier for ATR-based channels
            volume_confirmation: Require volume confirmation
            min_channel_width_pct: Minimum channel width percentage
        """
        super().__init__()
        self.channel_period = channel_period
        self.channel_type = channel_type
        self.breakout_threshold = breakout_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.volume_confirmation = volume_confirmation
        self.min_channel_width_pct = min_channel_width_pct

        # Track channels
        self.channels = {}  # symbol -> channel_data
        self.channel_history = {}  # symbol -> List[historical_channels]

        # Track breakouts
        self.active_breakouts = {}  # symbol -> breakout_data

        # Statistics
        self.stats = {
            "signals_generated": 0,
            "breakouts_detected": 0,
            "false_breakouts": 0,
            "avg_channel_width": 0.0,
            "successful_breakouts": 0
        }

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        """
        Generate trading signal based on channel breakout logic.

        Args:
            snapshot: Market snapshot

        Returns:
            SignalIntent or None
        """
        if not hasattr(snapshot, 'bars') or not snapshot.bars:
            # Try to use history
            if not snapshot.history or len(snapshot.history) < self.channel_period:
                return None
            bars = self._create_bars_from_history(snapshot.history)
        else:
            bars = snapshot.bars

        if len(bars) < self.channel_period:
            return None

        symbol = snapshot.symbol
        current_price = snapshot.price

        # Calculate channel
        channel = self._calculate_channel(symbol, bars)

        if not channel:
            return None

        # Store channel
        self.channels[symbol] = channel

        # Detect breakout
        breakout = self._detect_breakout(symbol, current_price, channel, bars)

        if not breakout:
            return None

        # Confirm with volume if required
        if self.volume_confirmation:
            if not self._confirm_volume(bars):
                return None

        # Generate signal
        signal = self._generate_breakout_signal(
            symbol,
            current_price,
            breakout,
            channel,
            bars
        )

        if signal:
            self.stats["signals_generated"] += 1
            self.stats["breakouts_detected"] += 1

        return signal

    def _create_bars_from_history(self, history: List[float]) -> List[Dict]:
        """Create OHLC bars from price history."""
        bars = []
        for price in history:
            bar = {
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 0
            }
            bars.append(bar)
        return bars

    def _calculate_channel(
        self,
        symbol: str,
        bars: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Calculate price channel based on type."""
        if self.channel_type == 'donchian':
            return self._calculate_donchian_channel(bars)
        elif self.channel_type == 'keltner':
            return self._calculate_keltner_channel(bars)
        elif self.channel_type == 'linear':
            return self._calculate_linear_regression_channel(bars)
        else:
            return self._calculate_donchian_channel(bars)  # Default

    def _calculate_donchian_channel(
        self,
        bars: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Calculate Donchian Channel (highest high, lowest low)."""
        recent_bars = bars[-self.channel_period:]

        highs = [bar['high'] for bar in recent_bars]
        lows = [bar['low'] for bar in recent_bars]

        upper = max(highs)
        lower = min(lows)
        middle = (upper + lower) / 2

        width = upper - lower
        width_pct = width / middle if middle > 0 else 0

        if width_pct < self.min_channel_width_pct:
            return None

        return {
            'type': 'donchian',
            'upper': upper,
            'lower': lower,
            'middle': middle,
            'width': width,
            'width_pct': width_pct
        }

    def _calculate_keltner_channel(
        self,
        bars: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Calculate Keltner Channel (EMA +/- ATR)."""
        recent_bars = bars[-max(self.channel_period, self.atr_period):]

        # Calculate EMA
        closes = [bar['close'] for bar in recent_bars]
        ema = self._calculate_ema(closes, self.channel_period)

        # Calculate ATR
        atr = self._calculate_atr(recent_bars[-self.atr_period:])

        upper = ema + (atr * self.atr_multiplier)
        lower = ema - (atr * self.atr_multiplier)
        middle = ema

        width = upper - lower
        width_pct = width / middle if middle > 0 else 0

        if width_pct < self.min_channel_width_pct:
            return None

        return {
            'type': 'keltner',
            'upper': upper,
            'lower': lower,
            'middle': middle,
            'width': width,
            'width_pct': width_pct,
            'atr': atr
        }

    def _calculate_linear_regression_channel(
        self,
        bars: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Calculate Linear Regression Channel."""
        recent_bars = bars[-self.channel_period:]
        closes = [bar['close'] for bar in recent_bars]

        # Calculate linear regression
        n = len(closes)
        x_values = list(range(n))

        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(closes)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, closes))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return None

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Calculate regression line
        regression_values = [slope * x + intercept for x in x_values]

        # Calculate standard deviation from regression
        residuals = [close - reg for close, reg in zip(closes, regression_values)]
        std_dev = statistics.stdev(residuals) if len(residuals) > 1 else 0

        # Current regression value
        current_regression = slope * (n - 1) + intercept

        upper = current_regression + (std_dev * 2)
        lower = current_regression - (std_dev * 2)
        middle = current_regression

        width = upper - lower
        width_pct = width / middle if middle > 0 else 0

        if width_pct < self.min_channel_width_pct:
            return None

        return {
            'type': 'linear_regression',
            'upper': upper,
            'lower': lower,
            'middle': middle,
            'width': width,
            'width_pct': width_pct,
            'slope': slope,
            'r_squared': self._calculate_r_squared(closes, regression_values)
        }

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if not prices:
            return 0.0

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _calculate_atr(self, bars: List[Dict]) -> float:
        """Calculate Average True Range."""
        if len(bars) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(bars)):
            high = bars[i]['high']
            low = bars[i]['low']
            prev_close = bars[i-1]['close']

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        return statistics.mean(true_ranges) if true_ranges else 0.0

    def _calculate_r_squared(
        self,
        actual: List[float],
        predicted: List[float]
    ) -> float:
        """Calculate R-squared for regression quality."""
        if len(actual) != len(predicted) or len(actual) == 0:
            return 0.0

        mean_actual = statistics.mean(actual)

        ss_total = sum((y - mean_actual) ** 2 for y in actual)
        ss_residual = sum((y - yhat) ** 2 for y, yhat in zip(actual, predicted))

        if ss_total == 0:
            return 0.0

        r_squared = 1 - (ss_residual / ss_total)

        return max(0.0, min(r_squared, 1.0))

    def _detect_breakout(
        self,
        symbol: str,
        current_price: float,
        channel: Dict[str, Any],
        bars: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Detect channel breakout."""
        upper = channel['upper']
        lower = channel['lower']
        middle = channel['middle']

        # Upper breakout
        breakout_upper = upper * (1 + self.breakout_threshold)
        if current_price > breakout_upper:
            return {
                'direction': 'buy',
                'type': 'upper',
                'breakout_price': current_price,
                'channel_level': upper,
                'distance_pct': (current_price - upper) / upper
            }

        # Lower breakout
        breakout_lower = lower * (1 - self.breakout_threshold)
        if current_price < breakout_lower:
            return {
                'direction': 'sell',
                'type': 'lower',
                'breakout_price': current_price,
                'channel_level': lower,
                'distance_pct': (lower - current_price) / lower
            }

        return None

    def _confirm_volume(self, bars: List[Dict]) -> bool:
        """Confirm breakout with volume."""
        if len(bars) < 20:
            return True  # Not enough data

        volumes = [bar.get('volume', 0) for bar in bars[-20:]]

        if not volumes or all(v == 0 for v in volumes):
            return True  # No volume data

        current_volume = volumes[-1]
        avg_volume = statistics.mean(volumes[:-1])

        if avg_volume == 0:
            return True

        # Volume should be above average
        return current_volume > avg_volume * 1.2

    def _generate_breakout_signal(
        self,
        symbol: str,
        current_price: float,
        breakout: Dict[str, Any],
        channel: Dict[str, Any],
        bars: List[Dict]
    ) -> Optional[SignalIntent]:
        """Generate breakout signal."""
        direction = breakout['direction']

        # Calculate stop loss and target
        if direction == 'buy':
            stop_loss = channel['middle']
            target = current_price + (channel['width'] * 1.5)
        else:
            stop_loss = channel['middle']
            target = current_price - (channel['width'] * 1.5)

        # Calculate confidence
        confidence = self._calculate_confidence(
            breakout,
            channel,
            bars
        )

        # Track breakout
        self.active_breakouts[symbol] = {
            'direction': direction,
            'entry': current_price,
            'channel': channel.copy(),
            'breakout': breakout.copy()
        }

        return SignalIntent(
            symbol=symbol,
            action=direction,
            confidence=confidence,
            metadata={
                'strategy': self.name,
                'channel_type': channel['type'],
                'breakout_type': breakout['type'],
                'channel_upper': channel['upper'],
                'channel_lower': channel['lower'],
                'channel_width_pct': channel['width_pct'],
                'stop_loss': stop_loss,
                'target': target,
                'distance_from_channel': breakout['distance_pct']
            }
        )

    def _calculate_confidence(
        self,
        breakout: Dict[str, Any],
        channel: Dict[str, Any],
        bars: List[Dict]
    ) -> float:
        """Calculate signal confidence."""
        confidence = 0.65  # Base confidence

        # Stronger breakout = higher confidence
        distance_pct = breakout['distance_pct']
        if distance_pct > 0.03:  # > 3% beyond channel
            confidence += 0.15
        elif distance_pct > 0.02:  # > 2%
            confidence += 0.10
        elif distance_pct > 0.01:  # > 1%
            confidence += 0.05

        # Wider channels more reliable
        if channel['width_pct'] > 0.05:  # > 5%
            confidence += 0.10
        elif channel['width_pct'] > 0.03:  # > 3%
            confidence += 0.05

        # Linear regression quality
        if channel['type'] == 'linear_regression':
            r_squared = channel.get('r_squared', 0)
            if r_squared > 0.8:
                confidence += 0.10

        return min(confidence, 0.95)

    def check_false_breakout(
        self,
        symbol: str,
        current_price: float
    ) -> bool:
        """Check if breakout was false."""
        if symbol not in self.active_breakouts:
            return False

        breakout = self.active_breakouts[symbol]
        channel = breakout['channel']

        # Check if price returned to channel
        if breakout['direction'] == 'buy':
            if current_price < channel['upper']:
                self.stats["false_breakouts"] += 1
                del self.active_breakouts[symbol]
                return True
        else:
            if current_price > channel['lower']:
                self.stats["false_breakouts"] += 1
                del self.active_breakouts[symbol]
                return True

        return False

    def get_current_channel(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current channel for symbol."""
        return self.channels.get(symbol)

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        stats = self.stats.copy()

        # Calculate false breakout rate
        total = stats["breakouts_detected"]
        if total > 0:
            stats["false_breakout_rate"] = stats["false_breakouts"] / total
            stats["success_rate"] = stats["successful_breakouts"] / total
        else:
            stats["false_breakout_rate"] = 0.0
            stats["success_rate"] = 0.0

        return stats

    def reset(self):
        """Reset strategy state."""
        self.channels.clear()
        self.channel_history.clear()
        self.active_breakouts.clear()
        self.stats = {
            "signals_generated": 0,
            "breakouts_detected": 0,
            "false_breakouts": 0,
            "avg_channel_width": 0.0,
            "successful_breakouts": 0
        }

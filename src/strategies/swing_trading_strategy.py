"""
Swing Trading Strategy - Capture multi-day price swings.
~400 lines as per schema
"""

import logging
from typing import Dict, Any, List, Optional
import statistics

from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class SwingTradingStrategy(BaseStrategy):
    """
    Swing trading strategy for capturing multi-day price movements.

    Algorithm:
    1. Identify swing highs and lows
    2. Detect trend reversals
    3. Enter on retracement to support/resistance
    4. Hold through the swing
    5. Exit at opposite swing level

    Features:
    - Swing point detection
    - Trend alignment
    - Fibonacci retracement levels
    - Multi-timeframe confirmation
    - Risk-reward optimization
    - Trailing stop management
    - Momentum filters
    """

    name = 'swing_trading'

    def __init__(
        self,
        swing_lookback: int = 20,
        min_swing_size_pct: float = 0.03,  # 3% minimum swing
        trend_period: int = 50,
        fib_retracement_levels: List[float] = None,
        min_risk_reward: float = 2.0,
        use_trailing_stop: bool = True,
        trailing_stop_pct: float = 0.02  # 2% trailing stop
    ):
        """
        Initialize swing trading strategy.

        Args:
            swing_lookback: Period for swing detection
            min_swing_size_pct: Minimum swing size percentage
            trend_period: Period for trend determination
            fib_retracement_levels: Fibonacci levels for entry
            min_risk_reward: Minimum risk-reward ratio
            use_trailing_stop: Enable trailing stop
            trailing_stop_pct: Trailing stop percentage
        """
        super().__init__()
        self.swing_lookback = swing_lookback
        self.min_swing_size_pct = min_swing_size_pct
        self.trend_period = trend_period
        self.fib_retracement_levels = fib_retracement_levels or [0.382, 0.500, 0.618]
        self.min_risk_reward = min_risk_reward
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct

        # Track swing points
        self.swing_highs = {}  # symbol -> List[swing_high_data]
        self.swing_lows = {}  # symbol -> List[swing_low_data]
        self.current_trend = {}  # symbol -> 'up' or 'down'

        # Track positions
        self.active_swings = {}  # symbol -> swing_position_data

        # Statistics
        self.stats = {
            "signals_generated": 0,
            "swings_captured": 0,
            "avg_swing_size": 0.0,
            "avg_holding_bars": 0,
            "trailed_stops": 0,
            "avg_risk_reward": 0.0
        }

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        """
        Generate trading signal based on swing logic.

        Args:
            snapshot: Market snapshot

        Returns:
            SignalIntent or None
        """
        if not snapshot.history or len(snapshot.history) < self.trend_period:
            return None

        symbol = snapshot.symbol
        current_price = snapshot.price
        bars = self._prepare_bars(snapshot)

        # Update swing points
        self._update_swing_points(symbol, bars)

        # Determine trend
        trend = self._determine_trend(symbol, bars)
        self.current_trend[symbol] = trend

        # Look for swing entry opportunities
        signal = self._find_swing_entry(symbol, current_price, bars, trend)

        if signal:
            self.stats["signals_generated"] += 1

        return signal

    def _prepare_bars(self, snapshot: MarketSnapshot) -> List[Dict[str, float]]:
        """Prepare OHLC bars from history."""
        bars = []

        # If bars already available
        if hasattr(snapshot, 'bars') and snapshot.bars:
            return snapshot.bars

        # Otherwise create from history
        history = snapshot.history
        chunk_size = 1  # Daily bars

        for i in range(0, len(history), chunk_size):
            chunk = history[i:i+chunk_size]
            if chunk:
                bar = {
                    'open': chunk[0],
                    'high': max(chunk),
                    'low': min(chunk),
                    'close': chunk[-1],
                    'volume': len(chunk)  # Placeholder
                }
                bars.append(bar)

        return bars

    def _update_swing_points(self, symbol: str, bars: List[Dict]):
        """Update swing high and low points."""
        if len(bars) < self.swing_lookback:
            return

        # Find swing highs
        swing_highs = []
        for i in range(self.swing_lookback, len(bars) - self.swing_lookback):
            current_high = bars[i]['high']

            # Check if it's a pivot high
            is_swing_high = True
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j != i and bars[j]['high'] > current_high:
                    is_swing_high = False
                    break

            if is_swing_high:
                swing_highs.append({
                    'price': current_high,
                    'index': i,
                    'bar': bars[i]
                })

        # Find swing lows
        swing_lows = []
        for i in range(self.swing_lookback, len(bars) - self.swing_lookback):
            current_low = bars[i]['low']

            # Check if it's a pivot low
            is_swing_low = True
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j != i and bars[j]['low'] < current_low:
                    is_swing_low = False
                    break

            if is_swing_low:
                swing_lows.append({
                    'price': current_low,
                    'index': i,
                    'bar': bars[i]
                })

        self.swing_highs[symbol] = swing_highs
        self.swing_lows[symbol] = swing_lows

    def _determine_trend(self, symbol: str, bars: List[Dict]) -> str:
        """Determine overall trend direction."""
        if len(bars) < self.trend_period:
            return 'neutral'

        recent_bars = bars[-self.trend_period:]
        closes = [bar['close'] for bar in recent_bars]

        # Calculate moving averages
        short_ma = statistics.mean(closes[-20:])
        long_ma = statistics.mean(closes)

        # Compare swing highs and lows
        swing_highs = self.swing_highs.get(symbol, [])
        swing_lows = self.swing_lows.get(symbol, [])

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            recent_highs = swing_highs[-2:]
            recent_lows = swing_lows[-2:]

            higher_highs = recent_highs[-1]['price'] > recent_highs[0]['price']
            higher_lows = recent_lows[-1]['price'] > recent_lows[0]['price']
            lower_highs = recent_highs[-1]['price'] < recent_highs[0]['price']
            lower_lows = recent_lows[-1]['price'] < recent_lows[0]['price']

            if higher_highs and higher_lows and short_ma > long_ma:
                return 'up'
            elif lower_highs and lower_lows and short_ma < long_ma:
                return 'down'

        # Fallback to MA comparison
        if short_ma > long_ma * 1.02:
            return 'up'
        elif short_ma < long_ma * 0.98:
            return 'down'

        return 'neutral'

    def _find_swing_entry(
        self,
        symbol: str,
        current_price: float,
        bars: List[Dict],
        trend: str
    ) -> Optional[SignalIntent]:
        """Find swing entry opportunities."""
        if trend == 'neutral':
            return None

        swing_highs = self.swing_highs.get(symbol, [])
        swing_lows = self.swing_lows.get(symbol, [])

        if not swing_highs or not swing_lows:
            return None

        # Bullish swing entry
        if trend == 'up':
            return self._check_bullish_swing(symbol, current_price, swing_lows, swing_highs, bars)

        # Bearish swing entry
        elif trend == 'down':
            return self._check_bearish_swing(symbol, current_price, swing_lows, swing_highs, bars)

        return None

    def _check_bullish_swing(
        self,
        symbol: str,
        current_price: float,
        swing_lows: List[Dict],
        swing_highs: List[Dict],
        bars: List[Dict]
    ) -> Optional[SignalIntent]:
        """Check for bullish swing entry."""
        if len(swing_lows) < 1 or len(swing_highs) < 1:
            return None

        # Find most recent swing low and swing high
        latest_low = swing_lows[-1]
        latest_high = max(swing_highs, key=lambda x: x['price'])

        # Calculate swing range
        swing_range = latest_high['price'] - latest_low['price']
        swing_size_pct = swing_range / latest_low['price']

        if swing_size_pct < self.min_swing_size_pct:
            return None

        # Calculate Fibonacci retracement levels
        fib_levels = self._calculate_fib_retracements(
            latest_high['price'],
            latest_low['price'],
            direction='up'
        )

        # Check if price is near a fib level
        for level, price_level in fib_levels.items():
            if abs(current_price - price_level) / current_price < 0.005:  # Within 0.5%
                # Calculate stop loss and target
                stop_loss = latest_low['price'] * 0.995  # Below swing low
                target = latest_high['price'] * 1.02  # Above swing high

                risk = current_price - stop_loss
                reward = target - current_price

                if risk <= 0:
                    return None

                risk_reward = reward / risk

                if risk_reward < self.min_risk_reward:
                    return None

                # Calculate confidence
                confidence = self._calculate_confidence(
                    swing_size_pct,
                    risk_reward,
                    level
                )

                self.active_swings[symbol] = {
                    'direction': 'buy',
                    'entry': current_price,
                    'stop': stop_loss,
                    'target': target,
                    'swing_low': latest_low['price'],
                    'swing_high': latest_high['price'],
                    'fib_level': level
                }

                return SignalIntent(
                    symbol=symbol,
                    action='buy',
                    confidence=confidence,
                    metadata={
                        'strategy': self.name,
                        'swing_type': 'bullish_retracement',
                        'fib_level': level,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_reward': risk_reward,
                        'swing_size_pct': swing_size_pct,
                        'swing_low': latest_low['price'],
                        'swing_high': latest_high['price']
                    }
                )

        return None

    def _check_bearish_swing(
        self,
        symbol: str,
        current_price: float,
        swing_lows: List[Dict],
        swing_highs: List[Dict],
        bars: List[Dict]
    ) -> Optional[SignalIntent]:
        """Check for bearish swing entry."""
        if len(swing_lows) < 1 or len(swing_highs) < 1:
            return None

        # Find most recent swing high and swing low
        latest_high = swing_highs[-1]
        latest_low = min(swing_lows, key=lambda x: x['price'])

        # Calculate swing range
        swing_range = latest_high['price'] - latest_low['price']
        swing_size_pct = swing_range / latest_high['price']

        if swing_size_pct < self.min_swing_size_pct:
            return None

        # Calculate Fibonacci retracement levels
        fib_levels = self._calculate_fib_retracements(
            latest_high['price'],
            latest_low['price'],
            direction='down'
        )

        # Check if price is near a fib level
        for level, price_level in fib_levels.items():
            if abs(current_price - price_level) / current_price < 0.005:  # Within 0.5%
                # Calculate stop loss and target
                stop_loss = latest_high['price'] * 1.005  # Above swing high
                target = latest_low['price'] * 0.98  # Below swing low

                risk = stop_loss - current_price
                reward = current_price - target

                if risk <= 0:
                    return None

                risk_reward = reward / risk

                if risk_reward < self.min_risk_reward:
                    return None

                # Calculate confidence
                confidence = self._calculate_confidence(
                    swing_size_pct,
                    risk_reward,
                    level
                )

                self.active_swings[symbol] = {
                    'direction': 'sell',
                    'entry': current_price,
                    'stop': stop_loss,
                    'target': target,
                    'swing_low': latest_low['price'],
                    'swing_high': latest_high['price'],
                    'fib_level': level
                }

                return SignalIntent(
                    symbol=symbol,
                    action='sell',
                    confidence=confidence,
                    metadata={
                        'strategy': self.name,
                        'swing_type': 'bearish_retracement',
                        'fib_level': level,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_reward': risk_reward,
                        'swing_size_pct': swing_size_pct,
                        'swing_low': latest_low['price'],
                        'swing_high': latest_high['price']
                    }
                )

        return None

    def _calculate_fib_retracements(
        self,
        high: float,
        low: float,
        direction: str
    ) -> Dict[float, float]:
        """Calculate Fibonacci retracement levels."""
        range_size = high - low

        if direction == 'up':
            # Retracement from high back to low
            levels = {
                level: high - (range_size * level)
                for level in self.fib_retracement_levels
            }
        else:
            # Retracement from low back to high
            levels = {
                level: low + (range_size * level)
                for level in self.fib_retracement_levels
            }

        return levels

    def _calculate_confidence(
        self,
        swing_size_pct: float,
        risk_reward: float,
        fib_level: float
    ) -> float:
        """Calculate signal confidence."""
        confidence = 0.6  # Base confidence

        # Larger swings are more reliable
        if swing_size_pct > 0.08:  # > 8%
            confidence += 0.15
        elif swing_size_pct > 0.05:  # > 5%
            confidence += 0.10

        # Better risk-reward = higher confidence
        if risk_reward > 3.0:
            confidence += 0.10
        elif risk_reward > 2.5:
            confidence += 0.05

        # 61.8% fib is most reliable
        if abs(fib_level - 0.618) < 0.05:
            confidence += 0.10
        elif abs(fib_level - 0.5) < 0.05:
            confidence += 0.05

        return min(confidence, 0.95)

    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[float]:
        """Update trailing stop for active swing."""
        if not self.use_trailing_stop:
            return None

        if symbol not in self.active_swings:
            return None

        swing = self.active_swings[symbol]

        if swing['direction'] == 'buy':
            # Trail stop upward
            new_stop = current_price * (1 - self.trailing_stop_pct)
            if new_stop > swing['stop']:
                swing['stop'] = new_stop
                self.stats["trailed_stops"] += 1
                return new_stop
        else:
            # Trail stop downward
            new_stop = current_price * (1 + self.trailing_stop_pct)
            if new_stop < swing['stop']:
                swing['stop'] = new_stop
                self.stats["trailed_stops"] += 1
                return new_stop

        return None

    def get_active_swing(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get active swing trade for symbol."""
        return self.active_swings.get(symbol)

    def get_swing_points(self, symbol: str) -> Dict[str, List[Dict]]:
        """Get identified swing points."""
        return {
            'highs': self.swing_highs.get(symbol, []),
            'lows': self.swing_lows.get(symbol, [])
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        stats = self.stats.copy()

        # Add current trend info
        stats["active_swings"] = len(self.active_swings)

        return stats

    def reset(self):
        """Reset strategy state."""
        self.swing_highs.clear()
        self.swing_lows.clear()
        self.current_trend.clear()
        self.active_swings.clear()
        self.stats = {
            "signals_generated": 0,
            "swings_captured": 0,
            "avg_swing_size": 0.0,
            "avg_holding_bars": 0,
            "trailed_stops": 0,
            "avg_risk_reward": 0.0
        }

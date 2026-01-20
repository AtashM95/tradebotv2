"""
Trend Following Strategy - Ride established trends until they reverse.
~600 lines as per schema
"""

from typing import Dict, Any, Optional, List, Tuple
from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy that identifies and rides strong trends.

    Features:
    - Multiple timeframe analysis
    - Moving average crossovers
    - ADX (Average Directional Index) for trend strength
    - Higher highs / lower lows detection
    - Trailing stop logic
    - Trend confirmation filters
    """

    name = 'trend_following'

    def __init__(
        self,
        fast_ma_period: int = 10,
        slow_ma_period: int = 30,
        trend_period: int = 14,
        min_trend_strength: float = 25.0,
        confirmation_bars: int = 2,
        trailing_stop_pct: float = 2.0,
        min_confidence: float = 0.65
    ):
        """
        Initialize trend following strategy.

        Args:
            fast_ma_period: Fast moving average period
            slow_ma_period: Slow moving average period
            trend_period: Period for trend strength calculation
            min_trend_strength: Minimum trend strength to trade (0-100)
            confirmation_bars: Number of bars to confirm trend
            trailing_stop_pct: Trailing stop percentage
            min_confidence: Minimum confidence for signals
        """
        super().__init__()
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.trend_period = trend_period
        self.min_trend_strength = min_trend_strength
        self.confirmation_bars = confirmation_bars
        self.trailing_stop_pct = trailing_stop_pct
        self.min_confidence = min_confidence

        # Track positions and stops
        self.positions: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.stats = {
            "signals_generated": 0,
            "trend_entries": 0,
            "trend_exits": 0,
            "stop_outs": 0,
            "weak_trends_skipped": 0,
            "crossovers_detected": 0
        }

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        """Generate trend following signal from market snapshot."""
        if not self._validate_snapshot(snapshot):
            return None

        # Calculate moving averages
        fast_ma = self._calculate_sma(snapshot.history, self.fast_ma_period)
        slow_ma = self._calculate_sma(snapshot.history, self.slow_ma_period)

        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(snapshot)

        # Check if trend is strong enough
        if trend_strength < self.min_trend_strength:
            self.stats["weak_trends_skipped"] += 1
            return None

        # Detect trend direction
        trend_direction = self._detect_trend_direction(snapshot, fast_ma, slow_ma)

        # Check for crossover
        crossover = self._detect_crossover(snapshot, fast_ma, slow_ma)
        if crossover:
            self.stats["crossovers_detected"] += 1

        # Check trailing stop
        stop_triggered = self._check_trailing_stop(snapshot.symbol, snapshot.price)

        # Determine signal
        signal_type, signal_reason = self._determine_signal(
            snapshot.symbol,
            trend_direction,
            crossover,
            stop_triggered,
            fast_ma,
            slow_ma
        )

        if signal_type is None:
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(
            trend_strength,
            trend_direction,
            fast_ma,
            slow_ma,
            snapshot.price
        )

        if confidence < self.min_confidence:
            return None

        # Track statistics
        self.stats["signals_generated"] += 1
        if signal_reason == 'trend_entry':
            self.stats["trend_entries"] += 1
        elif signal_reason == 'trend_exit':
            self.stats["trend_exits"] += 1
        elif signal_reason == 'stop_out':
            self.stats["stop_outs"] += 1

        # Update position tracking
        self._update_position(snapshot.symbol, signal_type, snapshot.price)

        # Create signal
        return SignalIntent(
            symbol=snapshot.symbol,
            action=signal_type,
            confidence=confidence,
            metadata={
                'strategy': self.name,
                'fast_ma': round(fast_ma, 2),
                'slow_ma': round(slow_ma, 2),
                'trend_strength': round(trend_strength, 2),
                'trend_direction': trend_direction,
                'signal_reason': signal_reason,
                'crossover': crossover,
                'price': snapshot.price
            }
        )

    def _validate_snapshot(self, snapshot: MarketSnapshot) -> bool:
        """Validate snapshot has sufficient data."""
        if snapshot.price <= 0:
            return False

        required_periods = max(self.slow_ma_period, self.trend_period)
        if len(snapshot.history) < required_periods + self.confirmation_bars:
            return False

        return True

    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return 0.0

        relevant_prices = prices[-period:]
        return sum(relevant_prices) / len(relevant_prices)

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return 0.0

        multiplier = 2.0 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_trend_strength(self, snapshot: MarketSnapshot) -> float:
        """
        Calculate trend strength similar to ADX (Average Directional Index).

        Returns a value from 0-100:
        - 0-25: No trend or weak trend
        - 25-50: Strong trend
        - 50-75: Very strong trend
        - 75-100: Extremely strong trend
        """
        prices = snapshot.history[-self.trend_period:]

        if len(prices) < 2:
            return 0.0

        # Calculate directional movements
        up_moves = []
        down_moves = []

        for i in range(1, len(prices)):
            move = prices[i] - prices[i-1]
            if move > 0:
                up_moves.append(move)
                down_moves.append(0)
            else:
                up_moves.append(0)
                down_moves.append(abs(move))

        # Average directional movements
        avg_up = sum(up_moves) / len(up_moves) if up_moves else 0
        avg_down = sum(down_moves) / len(down_moves) if down_moves else 0

        # Calculate directional index
        if avg_up + avg_down == 0:
            return 0.0

        dx = abs(avg_up - avg_down) / (avg_up + avg_down) * 100

        return min(dx, 100.0)

    def _detect_trend_direction(
        self,
        snapshot: MarketSnapshot,
        fast_ma: float,
        slow_ma: float
    ) -> str:
        """
        Detect trend direction.

        Returns: 'up', 'down', or 'sideways'
        """
        # Primary signal: MA crossover
        if fast_ma > slow_ma * 1.01:  # 1% buffer to avoid whipsaws
            primary_trend = 'up'
        elif fast_ma < slow_ma * 0.99:
            primary_trend = 'down'
        else:
            primary_trend = 'sideways'

        # Confirmation: Higher highs / lower lows
        recent_prices = snapshot.history[-self.confirmation_bars:]
        if len(recent_prices) >= 2:
            if all(recent_prices[i] >= recent_prices[i-1] for i in range(1, len(recent_prices))):
                secondary_trend = 'up'
            elif all(recent_prices[i] <= recent_prices[i-1] for i in range(1, len(recent_prices))):
                secondary_trend = 'down'
            else:
                secondary_trend = 'sideways'

            # Combine primary and secondary
            if primary_trend == secondary_trend:
                return primary_trend
            else:
                return 'sideways'

        return primary_trend

    def _detect_crossover(
        self,
        snapshot: MarketSnapshot,
        fast_ma: float,
        slow_ma: float
    ) -> Optional[str]:
        """
        Detect moving average crossover.

        Returns: 'bullish', 'bearish', or None
        """
        if len(snapshot.history) < self.slow_ma_period + 1:
            return None

        # Calculate previous MAs
        prev_fast = self._calculate_sma(
            snapshot.history[:-1],
            self.fast_ma_period
        )
        prev_slow = self._calculate_sma(
            snapshot.history[:-1],
            self.slow_ma_period
        )

        # Bullish crossover: fast crosses above slow
        if prev_fast <= prev_slow and fast_ma > slow_ma:
            return 'bullish'

        # Bearish crossover: fast crosses below slow
        if prev_fast >= prev_slow and fast_ma < slow_ma:
            return 'bearish'

        return None

    def _check_trailing_stop(self, symbol: str, current_price: float) -> bool:
        """
        Check if trailing stop is triggered.

        Returns: True if stop triggered, False otherwise
        """
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]
        position_type = position.get('type')
        trailing_stop = position.get('trailing_stop', 0)

        if position_type == 'long':
            # Update trailing stop if price made new high
            if current_price > trailing_stop:
                stop_price = current_price * (1 - self.trailing_stop_pct / 100)
                self.positions[symbol]['trailing_stop'] = stop_price
                return False
            # Check if stop triggered
            return current_price <= trailing_stop

        elif position_type == 'short':
            # Update trailing stop if price made new low
            if current_price < trailing_stop or trailing_stop == 0:
                stop_price = current_price * (1 + self.trailing_stop_pct / 100)
                self.positions[symbol]['trailing_stop'] = stop_price
                return False
            # Check if stop triggered
            return current_price >= trailing_stop

        return False

    def _determine_signal(
        self,
        symbol: str,
        trend_direction: str,
        crossover: Optional[str],
        stop_triggered: bool,
        fast_ma: float,
        slow_ma: float
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Determine buy/sell signal based on trend and crossovers.

        Trend following logic:
        - Enter long on bullish crossover in uptrend
        - Enter short on bearish crossover in downtrend
        - Exit on opposite crossover or trailing stop
        """
        has_position = symbol in self.positions

        # Stop triggered - exit position
        if stop_triggered:
            position_type = self.positions[symbol].get('type')
            if position_type == 'long':
                return ('sell', 'stop_out')
            elif position_type == 'short':
                return ('buy', 'stop_out')

        # Entry signals (no position)
        if not has_position:
            # Bullish entry: bullish crossover + uptrend
            if crossover == 'bullish' and trend_direction == 'up':
                return ('buy', 'trend_entry')

            # Bearish entry: bearish crossover + downtrend
            if crossover == 'bearish' and trend_direction == 'down':
                return ('sell', 'trend_entry')

            return (None, None)

        # Exit signals (have position)
        position_type = self.positions[symbol].get('type')

        if position_type == 'long':
            # Exit long on bearish crossover or downtrend
            if crossover == 'bearish' or trend_direction == 'down':
                return ('sell', 'trend_exit')

        elif position_type == 'short':
            # Exit short on bullish crossover or uptrend
            if crossover == 'bullish' or trend_direction == 'up':
                return ('buy', 'trend_exit')

        return (None, None)

    def _calculate_confidence(
        self,
        trend_strength: float,
        trend_direction: str,
        fast_ma: float,
        slow_ma: float,
        current_price: float
    ) -> float:
        """
        Calculate signal confidence.

        Higher confidence when:
        - Stronger trend
        - Clear trend direction
        - MAs well separated
        """
        confidence = 0.5  # Base confidence

        # Trend strength contribution (0.0 to 0.3)
        strength_score = min(trend_strength / 100.0, 1.0) * 0.3
        confidence += strength_score

        # Trend direction contribution (0.0 to 0.1)
        if trend_direction != 'sideways':
            confidence += 0.1

        # MA separation contribution (0.0 to 0.1)
        if slow_ma > 0:
            separation = abs(fast_ma - slow_ma) / slow_ma
            if separation > 0.02:  # At least 2% separated
                confidence += 0.1

        # Ensure confidence is in valid range
        return min(max(confidence, 0.0), 1.0)

    def _update_position(
        self,
        symbol: str,
        signal_type: str,
        price: float
    ):
        """Update position tracking and trailing stops."""
        if signal_type == 'buy':
            # Opening long or closing short
            if symbol in self.positions and self.positions[symbol].get('type') == 'short':
                # Closing short
                del self.positions[symbol]
            else:
                # Opening long
                stop_price = price * (1 - self.trailing_stop_pct / 100)
                self.positions[symbol] = {
                    'type': 'long',
                    'entry_price': price,
                    'trailing_stop': stop_price
                }

        elif signal_type == 'sell':
            # Opening short or closing long
            if symbol in self.positions and self.positions[symbol].get('type') == 'long':
                # Closing long
                del self.positions[symbol]
            else:
                # Opening short
                stop_price = price * (1 + self.trailing_stop_pct / 100)
                self.positions[symbol] = {
                    'type': 'short',
                    'entry_price': price,
                    'trailing_stop': stop_price
                }

    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return {
            "fast_ma_period": self.fast_ma_period,
            "slow_ma_period": self.slow_ma_period,
            "trend_period": self.trend_period,
            "min_trend_strength": self.min_trend_strength,
            "confirmation_bars": self.confirmation_bars,
            "trailing_stop_pct": self.trailing_stop_pct,
            "min_confidence": self.min_confidence
        }

    def set_parameters(self, params: Dict[str, Any]):
        """Update strategy parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current tracked positions."""
        return self.positions.copy()

    def clear_positions(self):
        """Clear all tracked positions."""
        self.positions.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        total = self.stats["signals_generated"]
        return {
            **self.stats,
            "active_positions": len(self.positions),
            "entry_ratio": (
                self.stats["trend_entries"] / total
                if total > 0 else 0.0
            ),
            "stop_out_ratio": (
                self.stats["stop_outs"] / total
                if total > 0 else 0.0
            )
        }

    def reset_stats(self):
        """Reset strategy statistics."""
        self.stats = {
            "signals_generated": 0,
            "trend_entries": 0,
            "trend_exits": 0,
            "stop_outs": 0,
            "weak_trends_skipped": 0,
            "crossovers_detected": 0
        }

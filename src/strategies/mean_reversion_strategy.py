"""
Mean Reversion Strategy - Trading based on price deviation from mean.
~550 lines as per schema
"""

from typing import Dict, Any, Optional, List, Tuple
from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy that profits from price returning to average.

    Features:
    - Z-score based entry/exit signals
    - Multiple moving average periods
    - Standard deviation bands
    - Oversold/overbought detection
    - Mean calculation methods (SMA, EMA)
    - Dynamic position sizing based on deviation
    """

    name = 'mean_reversion'

    def __init__(
        self,
        lookback_period: int = 20,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.5,
        min_volatility: float = 0.01,
        max_deviation: float = 4.0,
        mean_type: str = 'sma',  # 'sma' or 'ema'
        min_confidence: float = 0.6
    ):
        """
        Initialize mean reversion strategy.

        Args:
            lookback_period: Period for mean and std dev calculation
            entry_z_score: Z-score threshold for entry (deviation from mean)
            exit_z_score: Z-score threshold for exit
            min_volatility: Minimum volatility to trade (avoid low vol periods)
            max_deviation: Maximum z-score to prevent extreme entries
            mean_type: Type of moving average ('sma' or 'ema')
            min_confidence: Minimum confidence for signals
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.min_volatility = min_volatility
        self.max_deviation = max_deviation
        self.mean_type = mean_type
        self.min_confidence = min_confidence

        # Track positions for this strategy
        self.positions: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.stats = {
            "signals_generated": 0,
            "entry_signals": 0,
            "exit_signals": 0,
            "extreme_deviations": 0,
            "low_volatility_skips": 0
        }

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        """Generate mean reversion signal from market snapshot."""
        if not self._validate_snapshot(snapshot):
            return None

        # Calculate mean and standard deviation
        mean_price = self._calculate_mean(snapshot)
        std_dev = self._calculate_std_dev(snapshot, mean_price)

        # Check minimum volatility
        if std_dev < self.min_volatility * mean_price:
            self.stats["low_volatility_skips"] += 1
            return None

        # Calculate z-score (how many std devs from mean)
        z_score = self._calculate_z_score(snapshot.price, mean_price, std_dev)

        # Check if deviation is too extreme
        if abs(z_score) > self.max_deviation:
            self.stats["extreme_deviations"] += 1
            return None

        # Determine signal
        signal_type, signal_reason = self._determine_signal(
            snapshot.symbol, snapshot.price, z_score
        )

        if signal_type is None:
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(z_score, std_dev, mean_price)

        if confidence < self.min_confidence:
            return None

        # Track statistics
        self.stats["signals_generated"] += 1
        if signal_reason == 'entry':
            self.stats["entry_signals"] += 1
        elif signal_reason == 'exit':
            self.stats["exit_signals"] += 1

        # Update position tracking
        self._update_position_tracking(snapshot.symbol, signal_type, snapshot.price)

        # Create signal
        return SignalIntent(
            symbol=snapshot.symbol,
            action=signal_type,
            confidence=confidence,
            metadata={
                'strategy': self.name,
                'z_score': round(z_score, 3),
                'mean_price': round(mean_price, 2),
                'std_dev': round(std_dev, 3),
                'current_price': snapshot.price,
                'signal_reason': signal_reason,
                'lookback_period': self.lookback_period
            }
        )

    def _validate_snapshot(self, snapshot: MarketSnapshot) -> bool:
        """Validate snapshot has sufficient data."""
        if snapshot.price <= 0:
            return False

        if len(snapshot.history) < self.lookback_period:
            return False

        return True

    def _calculate_mean(self, snapshot: MarketSnapshot) -> float:
        """
        Calculate mean price over lookback period.

        Supports SMA (Simple Moving Average) or EMA (Exponential Moving Average).
        """
        prices = snapshot.history[-self.lookback_period:]

        if self.mean_type == 'ema':
            return self._calculate_ema(prices)
        else:
            # Simple Moving Average (default)
            return sum(prices) / len(prices)

    def _calculate_ema(self, prices: List[float]) -> float:
        """
        Calculate Exponential Moving Average.

        EMA gives more weight to recent prices.
        """
        if not prices:
            return 0.0

        multiplier = 2.0 / (len(prices) + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_std_dev(
        self,
        snapshot: MarketSnapshot,
        mean_price: float
    ) -> float:
        """
        Calculate standard deviation of prices.

        Standard deviation measures volatility.
        """
        prices = snapshot.history[-self.lookback_period:]

        if len(prices) < 2:
            return 0.0

        # Calculate variance
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)

        # Standard deviation is square root of variance
        return variance ** 0.5

    def _calculate_z_score(
        self,
        current_price: float,
        mean_price: float,
        std_dev: float
    ) -> float:
        """
        Calculate z-score (number of standard deviations from mean).

        Z-score formula: (X - μ) / σ
        Where:
        - X = current price
        - μ = mean price
        - σ = standard deviation

        Positive z-score = price above mean
        Negative z-score = price below mean
        """
        if std_dev <= 0:
            return 0.0

        return (current_price - mean_price) / std_dev

    def _determine_signal(
        self,
        symbol: str,
        current_price: float,
        z_score: float
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Determine buy/sell signal based on z-score and position.

        Mean reversion logic:
        - Buy when price is significantly below mean (oversold)
        - Sell when price is significantly above mean (overbought)
        - Exit when price returns near mean

        Returns:
            Tuple of (signal_type, signal_reason)
        """
        has_position = symbol in self.positions
        position = self.positions.get(symbol, {})

        # Entry signals (no position)
        if not has_position:
            # Price significantly below mean - BUY (expect price to rise)
            if z_score <= -self.entry_z_score:
                return ('buy', 'entry')

            # Price significantly above mean - SELL (expect price to fall)
            if z_score >= self.entry_z_score:
                return ('sell', 'entry')

            return (None, None)

        # Exit signals (have position)
        position_type = position.get('type')

        if position_type == 'long':
            # Exit long if price returned near mean or went too far up
            if z_score >= -self.exit_z_score or z_score >= self.entry_z_score:
                return ('sell', 'exit')

        elif position_type == 'short':
            # Exit short if price returned near mean or went too far down
            if z_score <= self.exit_z_score or z_score <= -self.entry_z_score:
                return ('buy', 'exit')

        return (None, None)

    def _calculate_confidence(
        self,
        z_score: float,
        std_dev: float,
        mean_price: float
    ) -> float:
        """
        Calculate signal confidence based on deviation and volatility.

        Higher confidence when:
        - Larger z-score (stronger deviation)
        - Moderate volatility (not too high or low)
        """
        confidence = 0.5  # Base confidence

        # Z-score contribution (0.0 to 0.3)
        # Higher absolute z-score = higher confidence
        z_score_contribution = min(abs(z_score) / self.max_deviation, 1.0) * 0.3
        confidence += z_score_contribution

        # Volatility contribution (0.0 to 0.2)
        # Moderate volatility is best (2-5% of price)
        volatility_ratio = std_dev / mean_price if mean_price > 0 else 0
        if 0.02 <= volatility_ratio <= 0.05:
            # Optimal volatility range
            confidence += 0.2
        elif 0.01 <= volatility_ratio < 0.02 or 0.05 < volatility_ratio <= 0.10:
            # Acceptable volatility
            confidence += 0.1
        else:
            # Too high or too low volatility
            confidence += 0.0

        # Ensure confidence is in valid range
        return min(max(confidence, 0.0), 1.0)

    def _update_position_tracking(
        self,
        symbol: str,
        signal_type: str,
        price: float
    ):
        """Update internal position tracking."""
        if signal_type == 'buy':
            # Opening long or closing short
            if symbol in self.positions and self.positions[symbol].get('type') == 'short':
                # Closing short
                del self.positions[symbol]
            else:
                # Opening long
                self.positions[symbol] = {
                    'type': 'long',
                    'entry_price': price
                }

        elif signal_type == 'sell':
            # Opening short or closing long
            if symbol in self.positions and self.positions[symbol].get('type') == 'long':
                # Closing long
                del self.positions[symbol]
            else:
                # Opening short
                self.positions[symbol] = {
                    'type': 'short',
                    'entry_price': price
                }

    def _calculate_percentile(self, value: float, values: List[float]) -> float:
        """Calculate percentile rank of value in values list."""
        if not values:
            return 0.5

        count_below = sum(1 for v in values if v < value)
        return count_below / len(values)

    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return {
            "lookback_period": self.lookback_period,
            "entry_z_score": self.entry_z_score,
            "exit_z_score": self.exit_z_score,
            "min_volatility": self.min_volatility,
            "max_deviation": self.max_deviation,
            "mean_type": self.mean_type,
            "min_confidence": self.min_confidence
        }

    def set_parameters(self, params: Dict[str, Any]):
        """Update strategy parameters."""
        if "lookback_period" in params:
            self.lookback_period = params["lookback_period"]
        if "entry_z_score" in params:
            self.entry_z_score = params["entry_z_score"]
        if "exit_z_score" in params:
            self.exit_z_score = params["exit_z_score"]
        if "min_volatility" in params:
            self.min_volatility = params["min_volatility"]
        if "max_deviation" in params:
            self.max_deviation = params["max_deviation"]
        if "mean_type" in params:
            self.mean_type = params["mean_type"]
        if "min_confidence" in params:
            self.min_confidence = params["min_confidence"]

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
                self.stats["entry_signals"] / total
                if total > 0 else 0.0
            ),
            "exit_ratio": (
                self.stats["exit_signals"] / total
                if total > 0 else 0.0
            ),
            "extreme_deviation_ratio": (
                self.stats["extreme_deviations"] / (total + self.stats["extreme_deviations"])
                if (total + self.stats["extreme_deviations"]) > 0 else 0.0
            )
        }

    def reset_stats(self):
        """Reset strategy statistics."""
        self.stats = {
            "signals_generated": 0,
            "entry_signals": 0,
            "exit_signals": 0,
            "extreme_deviations": 0,
            "low_volatility_skips": 0
        }

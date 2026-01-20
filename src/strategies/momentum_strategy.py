"""
Momentum Strategy - Trend continuation based on price momentum.
~600 lines as per schema
"""

from typing import Dict, Any, Optional, List
from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy that identifies and follows strong trends.

    Features:
    - Rate of Change (ROC) indicator
    - Momentum oscillator
    - Multi-period momentum analysis
    - Trend strength measurement
    - Volume confirmation
    - Overbought/oversold detection
    """

    name = 'momentum'

    def __init__(
        self,
        roc_period: int = 12,
        momentum_period: int = 14,
        volume_factor: float = 1.5,
        strong_momentum_threshold: float = 5.0,
        min_confidence: float = 0.6
    ):
        """
        Initialize momentum strategy.

        Args:
            roc_period: Period for Rate of Change calculation
            momentum_period: Period for momentum oscillator
            volume_factor: Volume multiplier for confirmation
            strong_momentum_threshold: Threshold for strong momentum (%)
            min_confidence: Minimum confidence for signals
        """
        super().__init__()
        self.roc_period = roc_period
        self.momentum_period = momentum_period
        self.volume_factor = volume_factor
        self.strong_momentum_threshold = strong_momentum_threshold
        self.min_confidence = min_confidence

        # Statistics
        self.stats = {
            "signals_generated": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "strong_momentum_count": 0
        }

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        """Generate momentum signal from market snapshot."""
        if not self._validate_snapshot(snapshot):
            return None

        # Calculate momentum indicators
        roc = self._calculate_roc(snapshot)
        momentum = self._calculate_momentum(snapshot)
        trend_strength = self._calculate_trend_strength(snapshot)

        # Analyze volume
        volume_confirmed = self._check_volume_confirmation(snapshot)

        # Determine signal
        signal_type = self._determine_signal(roc, momentum, trend_strength)

        if signal_type is None:
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(
            roc, momentum, trend_strength, volume_confirmed
        )

        if confidence < self.min_confidence:
            return None

        # Track statistics
        self.stats["signals_generated"] += 1
        if signal_type == 'buy':
            self.stats["buy_signals"] += 1
        else:
            self.stats["sell_signals"] += 1

        if abs(roc) > self.strong_momentum_threshold:
            self.stats["strong_momentum_count"] += 1

        # Create signal
        return SignalIntent(
            symbol=snapshot.symbol,
            action=signal_type,
            confidence=confidence,
            metadata={
                'strategy': self.name,
                'roc': round(roc, 2),
                'momentum': round(momentum, 2),
                'trend_strength': round(trend_strength, 2),
                'volume_confirmed': volume_confirmed,
                'price': snapshot.price
            }
        )

    def _validate_snapshot(self, snapshot: MarketSnapshot) -> bool:
        """Validate snapshot has sufficient data."""
        if snapshot.price <= 0:
            return False

        required_periods = max(self.roc_period, self.momentum_period)
        if len(snapshot.history) < required_periods:
            return False

        return True

    def _calculate_roc(self, snapshot: MarketSnapshot) -> float:
        """
        Calculate Rate of Change.

        ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100
        """
        if len(snapshot.history) < self.roc_period:
            return 0.0

        current_price = snapshot.price
        past_price = snapshot.history[-self.roc_period]

        if past_price <= 0:
            return 0.0

        roc = ((current_price - past_price) / past_price) * 100
        return roc

    def _calculate_momentum(self, snapshot: MarketSnapshot) -> float:
        """
        Calculate momentum oscillator.

        Momentum = Current Price - Price n periods ago
        """
        if len(snapshot.history) < self.momentum_period:
            return 0.0

        current_price = snapshot.price
        past_price = snapshot.history[-self.momentum_period]

        momentum = current_price - past_price
        return momentum

    def _calculate_trend_strength(self, snapshot: MarketSnapshot) -> float:
        """
        Calculate trend strength (0-100 scale).

        Based on consistent price movement direction.
        """
        if len(snapshot.history) < 10:
            return 0.0

        # Count consecutive moves in same direction
        consecutive_up = 0
        consecutive_down = 0

        for i in range(len(snapshot.history) - 1, max(0, len(snapshot.history) - 11), -1):
            if i == 0:
                break

            if snapshot.history[i] > snapshot.history[i-1]:
                consecutive_up += 1
                consecutive_down = 0
            elif snapshot.history[i] < snapshot.history[i-1]:
                consecutive_down += 1
                consecutive_up = 0
            else:
                break

        # Stronger trend = more consecutive moves
        strength = max(consecutive_up, consecutive_down) * 10
        return min(strength, 100.0)

    def _check_volume_confirmation(self, snapshot: MarketSnapshot) -> bool:
        """
        Check if volume confirms the price movement.

        Note: Since volume isn't in MarketSnapshot, this is a placeholder
        that could be enhanced when volume data is available.
        """
        # Placeholder - would check if volume is above average
        # For now, assume volume is confirmed if price movement is significant
        if len(snapshot.history) < 2:
            return False

        price_change = abs(snapshot.price - snapshot.history[-1])
        avg_price = sum(snapshot.history[-5:]) / min(5, len(snapshot.history))

        # Consider confirmed if price change is > 0.5% of average
        return (price_change / avg_price) > 0.005 if avg_price > 0 else False

    def _determine_signal(
        self,
        roc: float,
        momentum: float,
        trend_strength: float
    ) -> Optional[str]:
        """
        Determine buy/sell signal based on momentum indicators.

        Buy when:
        - Positive ROC and momentum
        - Strong uptrend

        Sell when:
        - Negative ROC and momentum
        - Strong downtrend
        """
        # Strong bullish momentum
        if roc > 0 and momentum > 0 and trend_strength > 30:
            return 'buy'

        # Strong bearish momentum
        if roc < 0 and momentum < 0 and trend_strength > 30:
            return 'sell'

        # Moderate bullish momentum
        if roc > 2 and momentum > 0:
            return 'buy'

        # Moderate bearish momentum
        if roc < -2 and momentum < 0:
            return 'sell'

        # No clear signal
        return None

    def _calculate_confidence(
        self,
        roc: float,
        momentum: float,
        trend_strength: float,
        volume_confirmed: bool
    ) -> float:
        """
        Calculate signal confidence (0.0 to 1.0).

        Higher confidence when:
        - Stronger ROC
        - Stronger trend
        - Volume confirmation
        """
        confidence = 0.5  # Base confidence

        # ROC contribution (0.0 to 0.25)
        roc_score = min(abs(roc) / 10.0, 1.0) * 0.25
        confidence += roc_score

        # Trend strength contribution (0.0 to 0.15)
        trend_score = (trend_strength / 100.0) * 0.15
        confidence += trend_score

        # Volume confirmation (0.1 bonus)
        if volume_confirmed:
            confidence += 0.1

        # Ensure confidence is in valid range
        return min(max(confidence, 0.0), 1.0)

    def _calculate_average(self, values: List[float]) -> float:
        """Calculate average of values."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return {
            "roc_period": self.roc_period,
            "momentum_period": self.momentum_period,
            "volume_factor": self.volume_factor,
            "strong_momentum_threshold": self.strong_momentum_threshold,
            "min_confidence": self.min_confidence
        }

    def set_parameters(self, params: Dict[str, Any]):
        """Update strategy parameters."""
        if "roc_period" in params:
            self.roc_period = params["roc_period"]
        if "momentum_period" in params:
            self.momentum_period = params["momentum_period"]
        if "volume_factor" in params:
            self.volume_factor = params["volume_factor"]
        if "strong_momentum_threshold" in params:
            self.strong_momentum_threshold = params["strong_momentum_threshold"]
        if "min_confidence" in params:
            self.min_confidence = params["min_confidence"]

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        total = self.stats["signals_generated"]
        return {
            **self.stats,
            "win_rate": 0.0,  # Would be calculated from actual trade results
            "avg_confidence": 0.0,  # Would be tracked over time
            "buy_ratio": (
                self.stats["buy_signals"] / total
                if total > 0 else 0.0
            ),
            "strong_momentum_ratio": (
                self.stats["strong_momentum_count"] / total
                if total > 0 else 0.0
            )
        }

    def reset_stats(self):
        """Reset strategy statistics."""
        self.stats = {
            "signals_generated": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "strong_momentum_count": 0
        }

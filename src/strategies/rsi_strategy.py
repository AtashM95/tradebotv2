"""
RSI (Relative Strength Index) Strategy

This strategy uses the RSI indicator to identify overbought and oversold conditions.
RSI values above 70 indicate overbought conditions (potential sell signal),
while values below 30 indicate oversold conditions (potential buy signal).
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import statistics

from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy


class RsiStrategy(BaseStrategy):
    """
    Relative Strength Index (RSI) trading strategy.

    The RSI is a momentum oscillator that measures the speed and magnitude of
    recent price changes to evaluate overbought or oversold conditions.

    Configuration parameters:
    - rsi_period: Number of periods for RSI calculation (default: 14)
    - oversold_threshold: RSI level indicating oversold condition (default: 30)
    - overbought_threshold: RSI level indicating overbought condition (default: 70)
    - min_confidence_oversold: Confidence when RSI crosses oversold (default: 0.75)
    - max_confidence_oversold: Max confidence at extreme oversold (default: 0.95)
    - min_confidence_overbought: Confidence when RSI crosses overbought (default: 0.75)
    - max_confidence_overbought: Max confidence at extreme overbought (default: 0.95)
    """

    name = 'rsi'

    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        min_confidence_oversold: float = 0.75,
        max_confidence_oversold: float = 0.95,
        min_confidence_overbought: float = 0.75,
        max_confidence_overbought: float = 0.95,
        extreme_oversold: float = 20.0,
        extreme_overbought: float = 80.0
    ):
        """
        Initialize RSI strategy with configuration parameters.

        Args:
            rsi_period: Number of periods for RSI calculation
            oversold_threshold: RSI level below which asset is considered oversold
            overbought_threshold: RSI level above which asset is considered overbought
            min_confidence_oversold: Base confidence for oversold signals
            max_confidence_oversold: Maximum confidence at extreme oversold levels
            min_confidence_overbought: Base confidence for overbought signals
            max_confidence_overbought: Maximum confidence at extreme overbought levels
            extreme_oversold: RSI level for extreme oversold (max confidence)
            extreme_overbought: RSI level for extreme overbought (max confidence)
        """
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.min_confidence_oversold = min_confidence_oversold
        self.max_confidence_oversold = max_confidence_oversold
        self.min_confidence_overbought = min_confidence_overbought
        self.max_confidence_overbought = max_confidence_overbought
        self.extreme_oversold = extreme_oversold
        self.extreme_overbought = extreme_overbought

        # State tracking
        self.rsi_history: Dict[str, List[float]] = {}
        self.last_signals: Dict[str, str] = {}

    def generate(self, snapshot: MarketSnapshot) -> Optional[SignalIntent]:
        """
        Generate trading signal based on RSI indicator.

        Args:
            snapshot: Current market data snapshot

        Returns:
            SignalIntent if a trading opportunity is identified, None otherwise
        """
        # Validate input data
        if not self._validate_snapshot(snapshot):
            return None

        # Calculate RSI
        rsi_value = self._calculate_rsi(snapshot)

        if rsi_value is None:
            return None

        # Store RSI in history
        if snapshot.symbol not in self.rsi_history:
            self.rsi_history[snapshot.symbol] = []
        self.rsi_history[snapshot.symbol].append(rsi_value)

        # Keep only recent history (100 periods)
        if len(self.rsi_history[snapshot.symbol]) > 100:
            self.rsi_history[snapshot.symbol] = self.rsi_history[snapshot.symbol][-100:]

        # Generate signal based on RSI level
        signal = self._generate_signal_from_rsi(snapshot.symbol, rsi_value)

        if signal:
            # Add RSI value to metadata
            signal.metadata.update({
                'strategy': self.name,
                'rsi_value': rsi_value,
                'oversold_threshold': self.oversold_threshold,
                'overbought_threshold': self.overbought_threshold,
                'signal_strength': self._calculate_signal_strength(rsi_value)
            })

            # Update last signal
            self.last_signals[snapshot.symbol] = signal.action

        return signal

    def _validate_snapshot(self, snapshot: MarketSnapshot) -> bool:
        """
        Validate that snapshot contains sufficient data for analysis.

        Args:
            snapshot: Market data snapshot to validate

        Returns:
            True if snapshot is valid, False otherwise
        """
        if snapshot.price <= 0:
            return False

        if not snapshot.history or len(snapshot.history) < self.rsi_period + 1:
            return False

        # Check for invalid prices in history
        if any(price <= 0 for price in snapshot.history):
            return False

        return True

    def _calculate_rsi(self, snapshot: MarketSnapshot) -> Optional[float]:
        """
        Calculate RSI value using the standard RSI formula.

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over the period

        Args:
            snapshot: Market data snapshot

        Returns:
            RSI value between 0 and 100, or None if calculation fails
        """
        try:
            # Get price history including current price
            prices = snapshot.history[-self.rsi_period-1:] + [snapshot.price]

            if len(prices) < self.rsi_period + 1:
                return None

            # Calculate price changes
            changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]

            # Separate gains and losses
            gains = [change if change > 0 else 0 for change in changes]
            losses = [-change if change < 0 else 0 for change in changes]

            # Calculate average gain and loss
            avg_gain = sum(gains) / self.rsi_period
            avg_loss = sum(losses) / self.rsi_period

            # Avoid division by zero
            if avg_loss == 0:
                return 100.0 if avg_gain > 0 else 50.0

            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

            return rsi

        except (ValueError, ZeroDivisionError, IndexError) as e:
            return None

    def _generate_signal_from_rsi(
        self,
        symbol: str,
        rsi_value: float
    ) -> Optional[SignalIntent]:
        """
        Generate trading signal based on RSI value.

        Args:
            symbol: Asset symbol
            rsi_value: Current RSI value

        Returns:
            SignalIntent if conditions are met, None otherwise
        """
        # Check for oversold condition (buy signal)
        if rsi_value <= self.oversold_threshold:
            # Calculate confidence based on how oversold it is
            confidence = self._calculate_oversold_confidence(rsi_value)

            # Avoid generating duplicate signals
            if self.last_signals.get(symbol) == 'buy':
                return None

            return SignalIntent(
                symbol=symbol,
                action='buy',
                confidence=confidence,
                metadata={'rsi_condition': 'oversold'}
            )

        # Check for overbought condition (sell signal)
        elif rsi_value >= self.overbought_threshold:
            # Calculate confidence based on how overbought it is
            confidence = self._calculate_overbought_confidence(rsi_value)

            # Avoid generating duplicate signals
            if self.last_signals.get(symbol) == 'sell':
                return None

            return SignalIntent(
                symbol=symbol,
                action='sell',
                confidence=confidence,
                metadata={'rsi_condition': 'overbought'}
            )

        # RSI is in neutral zone
        else:
            # Check for RSI divergence if we have history
            if symbol in self.rsi_history and len(self.rsi_history[symbol]) >= 5:
                divergence_signal = self._check_rsi_divergence(symbol, rsi_value)
                if divergence_signal:
                    return divergence_signal

        return None

    def _calculate_oversold_confidence(self, rsi_value: float) -> float:
        """
        Calculate confidence level for oversold conditions.

        Confidence increases as RSI gets more extremely oversold.

        Args:
            rsi_value: Current RSI value

        Returns:
            Confidence level between min and max confidence
        """
        if rsi_value <= self.extreme_oversold:
            return self.max_confidence_oversold

        # Linear interpolation between thresholds
        range_rsi = self.oversold_threshold - self.extreme_oversold
        range_confidence = self.max_confidence_oversold - self.min_confidence_oversold

        position = (self.oversold_threshold - rsi_value) / range_rsi
        confidence = self.min_confidence_oversold + (position * range_confidence)

        return min(max(confidence, self.min_confidence_oversold), self.max_confidence_oversold)

    def _calculate_overbought_confidence(self, rsi_value: float) -> float:
        """
        Calculate confidence level for overbought conditions.

        Confidence increases as RSI gets more extremely overbought.

        Args:
            rsi_value: Current RSI value

        Returns:
            Confidence level between min and max confidence
        """
        if rsi_value >= self.extreme_overbought:
            return self.max_confidence_overbought

        # Linear interpolation between thresholds
        range_rsi = self.extreme_overbought - self.overbought_threshold
        range_confidence = self.max_confidence_overbought - self.min_confidence_overbought

        position = (rsi_value - self.overbought_threshold) / range_rsi
        confidence = self.min_confidence_overbought + (position * range_confidence)

        return min(max(confidence, self.min_confidence_overbought), self.max_confidence_overbought)

    def _calculate_signal_strength(self, rsi_value: float) -> str:
        """
        Categorize signal strength based on RSI value.

        Args:
            rsi_value: Current RSI value

        Returns:
            Signal strength category: 'weak', 'moderate', or 'strong'
        """
        if rsi_value <= self.extreme_oversold or rsi_value >= self.extreme_overbought:
            return 'strong'
        elif rsi_value <= self.oversold_threshold or rsi_value >= self.overbought_threshold:
            return 'moderate'
        else:
            return 'weak'

    def _check_rsi_divergence(
        self,
        symbol: str,
        current_rsi: float
    ) -> Optional[SignalIntent]:
        """
        Check for bullish or bearish RSI divergence.

        Divergence occurs when price makes new highs/lows but RSI doesn't confirm.
        This can signal potential reversals.

        Args:
            symbol: Asset symbol
            current_rsi: Current RSI value

        Returns:
            SignalIntent if divergence is detected, None otherwise
        """
        # Need at least 5 RSI values to detect divergence
        if len(self.rsi_history[symbol]) < 5:
            return None

        recent_rsi = self.rsi_history[symbol][-5:]

        # Check for bullish divergence (price falling, RSI rising)
        # This is a simplified check - production version would correlate with price
        if current_rsi > 40 and current_rsi < 50:
            rsi_trend = recent_rsi[-1] - recent_rsi[0]
            if rsi_trend > 5:  # RSI trending up
                return SignalIntent(
                    symbol=symbol,
                    action='buy',
                    confidence=0.65,
                    metadata={'rsi_condition': 'bullish_divergence'}
                )

        # Check for bearish divergence (price rising, RSI falling)
        elif current_rsi > 50 and current_rsi < 60:
            rsi_trend = recent_rsi[-1] - recent_rsi[0]
            if rsi_trend < -5:  # RSI trending down
                return SignalIntent(
                    symbol=symbol,
                    action='sell',
                    confidence=0.65,
                    metadata={'rsi_condition': 'bearish_divergence'}
                )

        return None

    def get_indicator_value(self, snapshot: MarketSnapshot) -> Optional[float]:
        """
        Get the current RSI value for a given snapshot.

        This is a utility method for external analysis and debugging.

        Args:
            snapshot: Market data snapshot

        Returns:
            Current RSI value or None if calculation fails
        """
        if not self._validate_snapshot(snapshot):
            return None

        return self._calculate_rsi(snapshot)

    def reset_state(self, symbol: Optional[str] = None):
        """
        Reset strategy state for a symbol or all symbols.

        Args:
            symbol: Specific symbol to reset, or None to reset all
        """
        if symbol:
            self.rsi_history.pop(symbol, None)
            self.last_signals.pop(symbol, None)
        else:
            self.rsi_history.clear()
            self.last_signals.clear()

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current strategy configuration.

        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            'strategy': self.name,
            'rsi_period': self.rsi_period,
            'oversold_threshold': self.oversold_threshold,
            'overbought_threshold': self.overbought_threshold,
            'min_confidence_oversold': self.min_confidence_oversold,
            'max_confidence_oversold': self.max_confidence_oversold,
            'min_confidence_overbought': self.min_confidence_overbought,
            'max_confidence_overbought': self.max_confidence_overbought,
            'extreme_oversold': self.extreme_oversold,
            'extreme_overbought': self.extreme_overbought
        }

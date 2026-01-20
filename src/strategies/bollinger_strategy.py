"""
Bollinger Bands Strategy

This strategy uses Bollinger Bands to identify overbought and oversold conditions
and mean reversion opportunities. Generates buy signals when price touches lower
band and sell signals when price touches upper band.
"""

from typing import Optional, List, Dict, Any, Tuple
import statistics

from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy


class BollingerStrategy(BaseStrategy):
    """
    Bollinger Bands mean reversion trading strategy.

    Bollinger Bands consist of a middle band (SMA) and two outer bands
    (standard deviations away from the middle band).

    Configuration parameters:
    - period: Number of periods for SMA calculation (default: 20)
    - std_dev: Number of standard deviations for bands (default: 2.0)
    - min_confidence: Minimum confidence for signals (default: 0.70)
    - max_confidence: Maximum confidence for extreme moves (default: 0.95)
    - squeeze_detection: Enable Bollinger Squeeze detection (default: True)
    """

    name = 'bollinger'

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        min_confidence: float = 0.70,
        max_confidence: float = 0.95,
        squeeze_detection: bool = True,
        squeeze_threshold: float = 0.02
    ):
        """
        Initialize Bollinger Bands strategy.

        Args:
            period: Number of periods for moving average
            std_dev: Number of standard deviations for bands
            min_confidence: Base confidence level
            max_confidence: Maximum confidence level
            squeeze_detection: Whether to detect squeezes
            squeeze_threshold: Bandwidth ratio threshold for squeeze
        """
        self.period = period
        self.std_dev = std_dev
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.squeeze_detection = squeeze_detection
        self.squeeze_threshold = squeeze_threshold

        # State tracking
        self.bb_history: Dict[str, List[Tuple[float, float, float, float]]] = {}
        self.last_signal: Dict[str, str] = {}
        self.squeeze_state: Dict[str, bool] = {}

    def generate(self, snapshot: MarketSnapshot) -> Optional[SignalIntent]:
        """
        Generate trading signal based on Bollinger Bands.

        Args:
            snapshot: Current market data snapshot

        Returns:
            SignalIntent if a trading opportunity is identified, None otherwise
        """
        # Validate input data
        if not self._validate_snapshot(snapshot):
            return None

        # Calculate Bollinger Bands
        middle, upper, lower, bandwidth = self._calculate_bollinger_bands(snapshot)

        if middle is None:
            return None

        # Store values in history
        symbol = snapshot.symbol
        if symbol not in self.bb_history:
            self.bb_history[symbol] = []

        self.bb_history[symbol].append((middle, upper, lower, bandwidth))

        # Keep only recent history
        if len(self.bb_history[symbol]) > 100:
            self.bb_history[symbol] = self.bb_history[symbol][-100:]

        # Check for squeeze pattern
        if self.squeeze_detection:
            squeeze_signal = self._check_squeeze_pattern(
                symbol,
                snapshot.price,
                bandwidth
            )
            if squeeze_signal:
                return squeeze_signal

        # Generate signal based on band touches
        signal = self._generate_signal_from_bands(
            symbol,
            snapshot.price,
            middle,
            upper,
            lower,
            bandwidth
        )

        if signal:
            signal.metadata.update({
                'strategy': self.name,
                'middle_band': middle,
                'upper_band': upper,
                'lower_band': lower,
                'bandwidth': bandwidth,
                'percent_b': self._calculate_percent_b(snapshot.price, upper, lower)
            })

        return signal

    def _validate_snapshot(self, snapshot: MarketSnapshot) -> bool:
        """Validate snapshot has sufficient data."""
        if snapshot.price <= 0:
            return False

        if not snapshot.history or len(snapshot.history) < self.period:
            return False

        if any(price <= 0 for price in snapshot.history):
            return False

        return True

    def _calculate_bollinger_bands(
        self,
        snapshot: MarketSnapshot
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Calculate Bollinger Bands components.

        Args:
            snapshot: Market data snapshot

        Returns:
            Tuple of (middle_band, upper_band, lower_band, bandwidth)
        """
        try:
            prices = snapshot.history[-self.period:] + [snapshot.price]

            if len(prices) < self.period:
                return None, None, None, None

            # Calculate middle band (SMA)
            middle_band = sum(prices[-self.period:]) / self.period

            # Calculate standard deviation
            std = statistics.stdev(prices[-self.period:])

            # Calculate upper and lower bands
            upper_band = middle_band + (self.std_dev * std)
            lower_band = middle_band - (self.std_dev * std)

            # Calculate bandwidth (useful for squeeze detection)
            bandwidth = (upper_band - lower_band) / middle_band if middle_band > 0 else 0

            return middle_band, upper_band, lower_band, bandwidth

        except (ValueError, ZeroDivisionError, statistics.StatisticsError):
            return None, None, None, None

    def _generate_signal_from_bands(
        self,
        symbol: str,
        price: float,
        middle: float,
        upper: float,
        lower: float,
        bandwidth: float
    ) -> Optional[SignalIntent]:
        """
        Generate trading signal based on band touches.

        Args:
            symbol: Asset symbol
            price: Current price
            middle: Middle band value
            upper: Upper band value
            lower: Lower band value
            bandwidth: Band width ratio

        Returns:
            SignalIntent if conditions are met, None otherwise
        """
        # Calculate position within bands
        percent_b = self._calculate_percent_b(price, upper, lower)

        # Check for lower band touch (buy signal - oversold)
        if price <= lower:
            # Avoid duplicate signals
            if self.last_signal.get(symbol) == 'buy':
                return None

            # Calculate confidence based on how far below lower band
            distance_below = (lower - price) / lower
            confidence = self._calculate_band_touch_confidence(
                distance_below,
                'buy'
            )

            self.last_signal[symbol] = 'buy'

            return SignalIntent(
                symbol=symbol,
                action='buy',
                confidence=confidence,
                metadata={
                    'bollinger_condition': 'lower_band_touch',
                    'percent_b': percent_b,
                    'distance_from_band': distance_below
                }
            )

        # Check for upper band touch (sell signal - overbought)
        elif price >= upper:
            # Avoid duplicate signals
            if self.last_signal.get(symbol) == 'sell':
                return None

            # Calculate confidence based on how far above upper band
            distance_above = (price - upper) / upper
            confidence = self._calculate_band_touch_confidence(
                distance_above,
                'sell'
            )

            self.last_signal[symbol] = 'sell'

            return SignalIntent(
                symbol=symbol,
                action='sell',
                confidence=confidence,
                metadata={
                    'bollinger_condition': 'upper_band_touch',
                    'percent_b': percent_b,
                    'distance_from_band': distance_above
                }
            )

        # Check for mean reversion from extreme positions
        if len(self.bb_history[symbol]) >= 2:
            reversion_signal = self._check_mean_reversion(
                symbol,
                price,
                middle,
                percent_b
            )
            if reversion_signal:
                return reversion_signal

        return None

    def _calculate_percent_b(
        self,
        price: float,
        upper: float,
        lower: float
    ) -> float:
        """
        Calculate %B indicator.

        %B shows where price is relative to the bands.
        Values > 1 indicate price above upper band
        Values < 0 indicate price below lower band

        Args:
            price: Current price
            upper: Upper band value
            lower: Lower band value

        Returns:
            %B value
        """
        if upper == lower:
            return 0.5

        return (price - lower) / (upper - lower)

    def _calculate_band_touch_confidence(
        self,
        distance: float,
        direction: str
    ) -> float:
        """
        Calculate confidence based on distance from band.

        Args:
            distance: Distance ratio from band
            direction: 'buy' or 'sell'

        Returns:
            Confidence level
        """
        # Normalize distance (typical range 0-0.1)
        normalized_distance = min(distance / 0.1, 1.0)

        # Calculate confidence
        confidence_range = self.max_confidence - self.min_confidence
        confidence = self.min_confidence + (normalized_distance * confidence_range)

        return min(max(confidence, self.min_confidence), self.max_confidence)

    def _check_mean_reversion(
        self,
        symbol: str,
        current_price: float,
        middle_band: float,
        percent_b: float
    ) -> Optional[SignalIntent]:
        """
        Check for mean reversion signals.

        Args:
            symbol: Asset symbol
            current_price: Current price
            middle_band: Middle band value
            percent_b: Current %B value

        Returns:
            SignalIntent if reversion detected, None otherwise
        """
        # Get previous price position
        if len(self.bb_history[symbol]) < 2:
            return None

        prev_middle, prev_upper, prev_lower, _ = self.bb_history[symbol][-2]
        prev_percent_b = self._calculate_percent_b(
            current_price,
            prev_upper,
            prev_lower
        )

        # Check for reversion from oversold
        if prev_percent_b < 0.2 and percent_b > 0.3:
            return SignalIntent(
                symbol=symbol,
                action='buy',
                confidence=0.75,
                metadata={
                    'bollinger_condition': 'mean_reversion_bullish',
                    'percent_b': percent_b
                }
            )

        # Check for reversion from overbought
        elif prev_percent_b > 0.8 and percent_b < 0.7:
            return SignalIntent(
                symbol=symbol,
                action='sell',
                confidence=0.75,
                metadata={
                    'bollinger_condition': 'mean_reversion_bearish',
                    'percent_b': percent_b
                }
            )

        return None

    def _check_squeeze_pattern(
        self,
        symbol: str,
        price: float,
        bandwidth: float
    ) -> Optional[SignalIntent]:
        """
        Check for Bollinger Squeeze pattern.

        A squeeze occurs when bands narrow, indicating low volatility
        that often precedes significant price movement.

        Args:
            symbol: Asset symbol
            price: Current price
            bandwidth: Current bandwidth

        Returns:
            SignalIntent if squeeze breakout detected, None otherwise
        """
        # Need history to detect squeeze
        if len(self.bb_history[symbol]) < 10:
            return None

        # Check if currently in squeeze
        is_squeeze = bandwidth < self.squeeze_threshold

        # Update squeeze state
        was_in_squeeze = self.squeeze_state.get(symbol, False)
        self.squeeze_state[symbol] = is_squeeze

        # Detect squeeze breakout
        if was_in_squeeze and not is_squeeze:
            # Determine breakout direction from recent price movement
            recent_prices = [price] + [
                h[0] for h in self.bb_history[symbol][-5:]
            ]

            price_trend = recent_prices[-1] - recent_prices[0]
            middle_band = self.bb_history[symbol][-1][0]

            if price_trend > 0 and price > middle_band:
                return SignalIntent(
                    symbol=symbol,
                    action='buy',
                    confidence=0.80,
                    metadata={
                        'bollinger_condition': 'squeeze_breakout_bullish',
                        'bandwidth': bandwidth
                    }
                )
            elif price_trend < 0 and price < middle_band:
                return SignalIntent(
                    symbol=symbol,
                    action='sell',
                    confidence=0.80,
                    metadata={
                        'bollinger_condition': 'squeeze_breakout_bearish',
                        'bandwidth': bandwidth
                    }
                )

        return None

    def get_bollinger_values(
        self,
        snapshot: MarketSnapshot
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Get current Bollinger Bands values.

        Args:
            snapshot: Market data snapshot

        Returns:
            Tuple of (middle, upper, lower, bandwidth) or None
        """
        if not self._validate_snapshot(snapshot):
            return None

        return self._calculate_bollinger_bands(snapshot)

    def reset_state(self, symbol: Optional[str] = None):
        """
        Reset strategy state.

        Args:
            symbol: Specific symbol to reset, or None for all
        """
        if symbol:
            self.bb_history.pop(symbol, None)
            self.last_signal.pop(symbol, None)
            self.squeeze_state.pop(symbol, None)
        else:
            self.bb_history.clear()
            self.last_signal.clear()
            self.squeeze_state.clear()

    def get_configuration(self) -> Dict[str, Any]:
        """Get current strategy configuration."""
        return {
            'strategy': self.name,
            'period': self.period,
            'std_dev': self.std_dev,
            'min_confidence': self.min_confidence,
            'max_confidence': self.max_confidence,
            'squeeze_detection': self.squeeze_detection,
            'squeeze_threshold': self.squeeze_threshold
        }

"""
MACD (Moving Average Convergence Divergence) Strategy

This strategy uses MACD indicator to identify trend changes and momentum.
Generates buy signals on bullish MACD crossover (MACD crosses above signal line)
and sell signals on bearish crossover (MACD crosses below signal line).
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy


class MacdStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) trading strategy.

    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of prices.

    Configuration parameters:
    - fast_period: Fast EMA period (default: 12)
    - slow_period: Slow EMA period (default: 26)
    - signal_period: Signal line EMA period (default: 9)
    - min_confidence: Minimum confidence for weak signals (default: 0.65)
    - max_confidence: Maximum confidence for strong signals (default: 0.90)
    - histogram_threshold: Minimum histogram value for signal (default: 0.0)
    """

    name = 'macd'

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        min_confidence: float = 0.65,
        max_confidence: float = 0.90,
        histogram_threshold: float = 0.0,
        use_histogram_divergence: bool = True
    ):
        """
        Initialize MACD strategy with configuration parameters.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            min_confidence: Base confidence level for signals
            max_confidence: Maximum confidence for strong signals
            histogram_threshold: Minimum histogram value to generate signal
            use_histogram_divergence: Whether to use histogram divergence
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.histogram_threshold = histogram_threshold
        self.use_histogram_divergence = use_histogram_divergence

        # State tracking
        self.macd_history: Dict[str, List[Tuple[float, float, float]]] = {}
        self.last_crossover: Dict[str, str] = {}
        self.ema_fast_cache: Dict[str, float] = {}
        self.ema_slow_cache: Dict[str, float] = {}
        self.ema_signal_cache: Dict[str, float] = {}

    def generate(self, snapshot: MarketSnapshot) -> Optional[SignalIntent]:
        """
        Generate trading signal based on MACD indicator.

        Args:
            snapshot: Current market data snapshot

        Returns:
            SignalIntent if a trading opportunity is identified, None otherwise
        """
        # Validate input data
        if not self._validate_snapshot(snapshot):
            return None

        # Calculate MACD components
        macd_line, signal_line, histogram = self._calculate_macd(snapshot)

        if macd_line is None or signal_line is None:
            return None

        # Store MACD values in history
        symbol = snapshot.symbol
        if symbol not in self.macd_history:
            self.macd_history[symbol] = []

        self.macd_history[symbol].append((macd_line, signal_line, histogram))

        # Keep only recent history (100 periods)
        if len(self.macd_history[symbol]) > 100:
            self.macd_history[symbol] = self.macd_history[symbol][-100:]

        # Generate signal based on MACD crossover
        signal = self._generate_signal_from_macd(
            snapshot.symbol,
            macd_line,
            signal_line,
            histogram
        )

        if signal:
            # Add MACD values to metadata
            signal.metadata.update({
                'strategy': self.name,
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram,
                'signal_strength': self._calculate_signal_strength(histogram)
            })

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

        min_required = max(self.slow_period, self.fast_period) + self.signal_period
        if not snapshot.history or len(snapshot.history) < min_required:
            return False

        # Check for invalid prices in history
        if any(price <= 0 for price in snapshot.history):
            return False

        return True

    def _calculate_ema(
        self,
        prices: List[float],
        period: int,
        previous_ema: Optional[float] = None
    ) -> float:
        """
        Calculate Exponential Moving Average.

        Args:
            prices: List of price values
            period: EMA period
            previous_ema: Previous EMA value for continuous calculation

        Returns:
            EMA value
        """
        if len(prices) < period:
            return sum(prices) / len(prices)

        multiplier = 2.0 / (period + 1)

        if previous_ema is None:
            # Initial EMA is SMA
            ema = sum(prices[:period]) / period
            start_idx = period
        else:
            ema = previous_ema
            start_idx = 0

        # Calculate EMA for remaining prices
        for price in prices[start_idx:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_macd(
        self,
        snapshot: MarketSnapshot
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate MACD line, signal line, and histogram.

        MACD Line = 12-period EMA - 26-period EMA
        Signal Line = 9-period EMA of MACD Line
        Histogram = MACD Line - Signal Line

        Args:
            snapshot: Market data snapshot

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        try:
            prices = snapshot.history + [snapshot.price]
            symbol = snapshot.symbol

            # Calculate fast and slow EMAs
            fast_ema = self._calculate_ema(
                prices,
                self.fast_period,
                self.ema_fast_cache.get(symbol)
            )
            slow_ema = self._calculate_ema(
                prices,
                self.slow_period,
                self.ema_slow_cache.get(symbol)
            )

            # Cache EMAs for next calculation
            self.ema_fast_cache[symbol] = fast_ema
            self.ema_slow_cache[symbol] = slow_ema

            # Calculate MACD line
            macd_line = fast_ema - slow_ema

            # Calculate signal line (EMA of MACD)
            if symbol in self.macd_history and len(self.macd_history[symbol]) >= self.signal_period:
                # Get recent MACD values
                recent_macd = [m[0] for m in self.macd_history[symbol][-(self.signal_period-1):]]
                recent_macd.append(macd_line)
                signal_line = self._calculate_ema(
                    recent_macd,
                    self.signal_period,
                    self.ema_signal_cache.get(symbol)
                )
            else:
                # Not enough history, use simple moving average
                if symbol in self.macd_history:
                    recent_macd = [m[0] for m in self.macd_history[symbol]]
                    recent_macd.append(macd_line)
                    signal_line = sum(recent_macd) / len(recent_macd)
                else:
                    signal_line = macd_line

            # Cache signal EMA for next calculation
            self.ema_signal_cache[symbol] = signal_line

            # Calculate histogram
            histogram = macd_line - signal_line

            return macd_line, signal_line, histogram

        except (ValueError, ZeroDivisionError, IndexError):
            return None, None, None

    def _generate_signal_from_macd(
        self,
        symbol: str,
        macd_line: float,
        signal_line: float,
        histogram: float
    ) -> Optional[SignalIntent]:
        """
        Generate trading signal based on MACD crossover.

        Args:
            symbol: Asset symbol
            macd_line: Current MACD line value
            signal_line: Current signal line value
            histogram: Current histogram value

        Returns:
            SignalIntent if conditions are met, None otherwise
        """
        # Need at least 2 data points to detect crossover
        if symbol not in self.macd_history or len(self.macd_history[symbol]) < 2:
            return None

        # Get previous values
        prev_macd, prev_signal, prev_histogram = self.macd_history[symbol][-2]

        # Check for bullish crossover (MACD crosses above signal)
        if prev_macd <= prev_signal and macd_line > signal_line:
            # Avoid duplicate signals
            if self.last_crossover.get(symbol) == 'bullish':
                return None

            # Calculate confidence based on histogram strength
            confidence = self._calculate_crossover_confidence(
                histogram,
                prev_histogram,
                'bullish'
            )

            self.last_crossover[symbol] = 'bullish'

            return SignalIntent(
                symbol=symbol,
                action='buy',
                confidence=confidence,
                metadata={
                    'macd_condition': 'bullish_crossover',
                    'crossover_strength': abs(histogram)
                }
            )

        # Check for bearish crossover (MACD crosses below signal)
        elif prev_macd >= prev_signal and macd_line < signal_line:
            # Avoid duplicate signals
            if self.last_crossover.get(symbol) == 'bearish':
                return None

            # Calculate confidence based on histogram strength
            confidence = self._calculate_crossover_confidence(
                histogram,
                prev_histogram,
                'bearish'
            )

            self.last_crossover[symbol] = 'bearish'

            return SignalIntent(
                symbol=symbol,
                action='sell',
                confidence=confidence,
                metadata={
                    'macd_condition': 'bearish_crossover',
                    'crossover_strength': abs(histogram)
                }
            )

        # Check for histogram divergence if enabled
        if self.use_histogram_divergence:
            divergence_signal = self._check_histogram_divergence(
                symbol,
                histogram,
                macd_line
            )
            if divergence_signal:
                return divergence_signal

        return None

    def _calculate_crossover_confidence(
        self,
        histogram: float,
        prev_histogram: float,
        direction: str
    ) -> float:
        """
        Calculate confidence level for crossover signal.

        Stronger histogram values and faster crossovers indicate higher confidence.

        Args:
            histogram: Current histogram value
            prev_histogram: Previous histogram value
            direction: 'bullish' or 'bearish'

        Returns:
            Confidence level between min and max
        """
        # Calculate crossover momentum (rate of change)
        momentum = abs(histogram - prev_histogram)

        # Normalize momentum (typical range 0-5)
        normalized_momentum = min(momentum / 5.0, 1.0)

        # Calculate histogram strength (normalized to typical range)
        histogram_strength = min(abs(histogram) / 10.0, 1.0)

        # Combine momentum and strength for confidence
        strength_factor = (normalized_momentum + histogram_strength) / 2.0

        # Calculate final confidence
        confidence_range = self.max_confidence - self.min_confidence
        confidence = self.min_confidence + (strength_factor * confidence_range)

        return min(max(confidence, self.min_confidence), self.max_confidence)

    def _calculate_signal_strength(self, histogram: float) -> str:
        """
        Categorize signal strength based on histogram value.

        Args:
            histogram: Current histogram value

        Returns:
            Signal strength category: 'weak', 'moderate', or 'strong'
        """
        abs_histogram = abs(histogram)

        if abs_histogram >= 5.0:
            return 'strong'
        elif abs_histogram >= 2.0:
            return 'moderate'
        else:
            return 'weak'

    def _check_histogram_divergence(
        self,
        symbol: str,
        current_histogram: float,
        current_macd: float
    ) -> Optional[SignalIntent]:
        """
        Check for histogram divergence patterns.

        Divergence occurs when histogram peaks/troughs don't align with MACD peaks/troughs,
        indicating potential trend reversal.

        Args:
            symbol: Asset symbol
            current_histogram: Current histogram value
            current_macd: Current MACD line value

        Returns:
            SignalIntent if divergence is detected, None otherwise
        """
        # Need at least 10 data points to detect divergence
        if len(self.macd_history[symbol]) < 10:
            return None

        recent_data = self.macd_history[symbol][-10:]
        histograms = [h[2] for h in recent_data]
        macd_values = [m[0] for m in recent_data]

        # Check for bullish divergence (histogram making higher lows while MACD making lower lows)
        if self._detect_bullish_divergence(histograms, macd_values):
            return SignalIntent(
                symbol=symbol,
                action='buy',
                confidence=0.70,
                metadata={'macd_condition': 'bullish_divergence'}
            )

        # Check for bearish divergence (histogram making lower highs while MACD making higher highs)
        if self._detect_bearish_divergence(histograms, macd_values):
            return SignalIntent(
                symbol=symbol,
                action='sell',
                confidence=0.70,
                metadata={'macd_condition': 'bearish_divergence'}
            )

        return None

    def _detect_bullish_divergence(
        self,
        histograms: List[float],
        macd_values: List[float]
    ) -> bool:
        """
        Detect bullish divergence pattern.

        Args:
            histograms: Recent histogram values
            macd_values: Recent MACD values

        Returns:
            True if bullish divergence detected
        """
        # Find local minima in last 5 periods
        if len(histograms) < 5:
            return False

        # Simple detection: compare first half vs second half
        first_half_hist_min = min(histograms[:5])
        second_half_hist_min = min(histograms[5:])

        first_half_macd_min = min(macd_values[:5])
        second_half_macd_min = min(macd_values[5:])

        # Bullish divergence: histogram making higher lows, MACD making lower lows
        return (second_half_hist_min > first_half_hist_min and
                second_half_macd_min < first_half_macd_min)

    def _detect_bearish_divergence(
        self,
        histograms: List[float],
        macd_values: List[float]
    ) -> bool:
        """
        Detect bearish divergence pattern.

        Args:
            histograms: Recent histogram values
            macd_values: Recent MACD values

        Returns:
            True if bearish divergence detected
        """
        # Find local maxima in last 5 periods
        if len(histograms) < 5:
            return False

        # Simple detection: compare first half vs second half
        first_half_hist_max = max(histograms[:5])
        second_half_hist_max = max(histograms[5:])

        first_half_macd_max = max(macd_values[:5])
        second_half_macd_max = max(macd_values[5:])

        # Bearish divergence: histogram making lower highs, MACD making higher highs
        return (second_half_hist_max < first_half_hist_max and
                second_half_macd_max > first_half_macd_max)

    def get_macd_values(
        self,
        snapshot: MarketSnapshot
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get current MACD values for a snapshot.

        Utility method for external analysis and debugging.

        Args:
            snapshot: Market data snapshot

        Returns:
            Tuple of (macd_line, signal_line, histogram) or None
        """
        if not self._validate_snapshot(snapshot):
            return None

        return self._calculate_macd(snapshot)

    def reset_state(self, symbol: Optional[str] = None):
        """
        Reset strategy state for a symbol or all symbols.

        Args:
            symbol: Specific symbol to reset, or None to reset all
        """
        if symbol:
            self.macd_history.pop(symbol, None)
            self.last_crossover.pop(symbol, None)
            self.ema_fast_cache.pop(symbol, None)
            self.ema_slow_cache.pop(symbol, None)
            self.ema_signal_cache.pop(symbol, None)
        else:
            self.macd_history.clear()
            self.last_crossover.clear()
            self.ema_fast_cache.clear()
            self.ema_slow_cache.clear()
            self.ema_signal_cache.clear()

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current strategy configuration.

        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            'strategy': self.name,
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'min_confidence': self.min_confidence,
            'max_confidence': self.max_confidence,
            'histogram_threshold': self.histogram_threshold,
            'use_histogram_divergence': self.use_histogram_divergence
        }

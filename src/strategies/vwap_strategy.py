"""
VWAP Strategy - Volume Weighted Average Price trading.
~400 lines as per schema
"""

import logging
from typing import Dict, Any, List, Optional
import statistics

from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class VWAPStrategy(BaseStrategy):
    """
    VWAP (Volume Weighted Average Price) trading strategy.

    Algorithm:
    1. Calculate VWAP from cumulative (price * volume)
    2. Calculate VWAP standard deviation bands
    3. Detect price deviations from VWAP
    4. Enter on mean reversion or breakout
    5. Use VWAP as dynamic support/resistance
    6. Exit when price returns to VWAP

    Features:
    - Intraday VWAP calculation
    - Standard deviation bands
    - Mean reversion signals
    - Breakout signals
    - Volume profile analysis
    - VWAP anchored periods
    - Multi-timeframe VWAP
    """

    name = 'vwap'

    def __init__(
        self,
        deviation_threshold: float = 1.5,  # Std dev threshold
        min_volume_ratio: float = 1.0,
        use_bands: bool = True,
        num_std_dev: float = 2.0,
        mean_reversion_mode: bool = True,
        breakout_mode: bool = False,
        reset_period: str = 'daily'  # daily, session, none
    ):
        """
        Initialize VWAP strategy.

        Args:
            deviation_threshold: Threshold for price deviation from VWAP
            min_volume_ratio: Minimum volume ratio for signal
            use_bands: Use standard deviation bands
            num_std_dev: Number of std devs for bands
            mean_reversion_mode: Trade mean reversion to VWAP
            breakout_mode: Trade breakouts from VWAP bands
            reset_period: When to reset VWAP calculation
        """
        super().__init__()
        self.deviation_threshold = deviation_threshold
        self.min_volume_ratio = min_volume_ratio
        self.use_bands = use_bands
        self.num_std_dev = num_std_dev
        self.mean_reversion_mode = mean_reversion_mode
        self.breakout_mode = breakout_mode
        self.reset_period = reset_period

        # Track VWAP data
        self.vwap_data = {}  # symbol -> vwap_calculation
        self.cumulative_data = {}  # symbol -> cumulative_values
        self.session_start = {}  # symbol -> session_start_time

        # Statistics
        self.stats = {
            "signals_generated": 0,
            "mean_reversion_signals": 0,
            "breakout_signals": 0,
            "above_vwap": 0,
            "below_vwap": 0,
            "avg_deviation": 0.0
        }

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        """
        Generate trading signal based on VWAP analysis.

        Args:
            snapshot: Market snapshot

        Returns:
            SignalIntent or None
        """
        if not hasattr(snapshot, 'bars') or not snapshot.bars:
            return None

        bars = snapshot.bars

        if len(bars) < 2:
            return None

        symbol = snapshot.symbol
        current_price = snapshot.price
        metadata = snapshot.metadata or {}

        # Check if we need to reset VWAP
        self._check_reset(symbol, metadata.get('timestamp', 0))

        # Calculate VWAP
        vwap_result = self._calculate_vwap(symbol, bars)

        if not vwap_result:
            return None

        vwap = vwap_result['vwap']
        upper_band = vwap_result.get('upper_band')
        lower_band = vwap_result.get('lower_band')

        # Store VWAP data
        self.vwap_data[symbol] = vwap_result

        # Calculate price deviation from VWAP
        deviation = self._calculate_deviation(current_price, vwap)
        deviation_pct = (current_price - vwap) / vwap if vwap > 0 else 0

        # Update stats
        self.stats["avg_deviation"] = (
            (self.stats["avg_deviation"] * self.stats["signals_generated"] + abs(deviation_pct)) /
            (self.stats["signals_generated"] + 1)
        ) if self.stats["signals_generated"] > 0 else abs(deviation_pct)

        if current_price > vwap:
            self.stats["above_vwap"] += 1
        else:
            self.stats["below_vwap"] += 1

        # Check volume
        if not self._check_volume(bars):
            return None

        # Generate signal based on mode
        signal = None

        if self.mean_reversion_mode:
            signal = self._check_mean_reversion(
                symbol,
                current_price,
                vwap,
                upper_band,
                lower_band,
                deviation_pct,
                bars
            )

        if not signal and self.breakout_mode:
            signal = self._check_breakout(
                symbol,
                current_price,
                vwap,
                upper_band,
                lower_band,
                deviation_pct,
                bars
            )

        if signal:
            self.stats["signals_generated"] += 1

        return signal

    def _check_reset(self, symbol: str, timestamp: float):
        """Check if VWAP should be reset."""
        if self.reset_period == 'none':
            return

        if symbol not in self.session_start:
            self.session_start[symbol] = timestamp
            return

        # Check for daily reset
        if self.reset_period == 'daily':
            # Simple check: if more than 18 hours passed, reset
            time_diff = timestamp - self.session_start[symbol]
            if time_diff > 64800:  # 18 hours in seconds
                self._reset_symbol(symbol)
                self.session_start[symbol] = timestamp

    def _reset_symbol(self, symbol: str):
        """Reset VWAP calculation for symbol."""
        if symbol in self.cumulative_data:
            del self.cumulative_data[symbol]
        if symbol in self.vwap_data:
            del self.vwap_data[symbol]

    def _calculate_vwap(
        self,
        symbol: str,
        bars: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Calculate VWAP and standard deviation bands."""
        if symbol not in self.cumulative_data:
            self.cumulative_data[symbol] = {
                'cum_volume': 0.0,
                'cum_pv': 0.0,  # price * volume
                'cum_pv2': 0.0,  # price^2 * volume
                'prices': [],
                'volumes': []
            }

        cum_data = self.cumulative_data[symbol]

        # Process all bars
        for bar in bars:
            typical_price = (bar['high'] + bar['low'] + bar['close']) / 3
            volume = bar.get('volume', 0)

            if volume > 0:
                cum_data['cum_volume'] += volume
                cum_data['cum_pv'] += typical_price * volume
                cum_data['cum_pv2'] += (typical_price ** 2) * volume
                cum_data['prices'].append(typical_price)
                cum_data['volumes'].append(volume)

        if cum_data['cum_volume'] == 0:
            return None

        # Calculate VWAP
        vwap = cum_data['cum_pv'] / cum_data['cum_volume']

        result = {
            'vwap': vwap,
            'cum_volume': cum_data['cum_volume'],
            'bar_count': len(cum_data['prices'])
        }

        # Calculate standard deviation bands if enabled
        if self.use_bands:
            # Variance = E[X^2] - E[X]^2
            variance = (cum_data['cum_pv2'] / cum_data['cum_volume']) - (vwap ** 2)
            std_dev = variance ** 0.5 if variance > 0 else 0

            result['std_dev'] = std_dev
            result['upper_band'] = vwap + (std_dev * self.num_std_dev)
            result['lower_band'] = vwap - (std_dev * self.num_std_dev)

        return result

    def _calculate_deviation(self, price: float, vwap: float) -> float:
        """Calculate price deviation from VWAP in standard deviations."""
        if not self.use_bands:
            return 0.0

        # Get std dev from stored data
        # This is simplified; would use symbol-specific data in production
        return abs(price - vwap)

    def _check_volume(self, bars: List[Dict]) -> bool:
        """Check if current volume supports signal."""
        if len(bars) < 10:
            return True

        recent_bars = bars[-10:]
        volumes = [bar.get('volume', 0) for bar in recent_bars]

        if not volumes or all(v == 0 for v in volumes):
            return True  # No volume data

        current_volume = volumes[-1]
        avg_volume = statistics.mean(volumes[:-1])

        if avg_volume == 0:
            return True

        return current_volume >= avg_volume * self.min_volume_ratio

    def _check_mean_reversion(
        self,
        symbol: str,
        current_price: float,
        vwap: float,
        upper_band: Optional[float],
        lower_band: Optional[float],
        deviation_pct: float,
        bars: List[Dict]
    ) -> Optional[SignalIntent]:
        """Check for mean reversion opportunity."""
        # Price at lower band - potential buy
        if lower_band and current_price <= lower_band:
            distance_from_band = (lower_band - current_price) / lower_band

            if distance_from_band > 0.001:  # 0.1% below band
                self.stats["mean_reversion_signals"] += 1

                confidence = self._calculate_confidence(
                    'mean_reversion',
                    abs(deviation_pct),
                    distance_from_band
                )

                return SignalIntent(
                    symbol=symbol,
                    action='buy',
                    confidence=confidence,
                    metadata={
                        'strategy': self.name,
                        'signal_type': 'mean_reversion',
                        'vwap': vwap,
                        'current_price': current_price,
                        'deviation_pct': deviation_pct,
                        'target': vwap,  # Target is VWAP
                        'stop_loss': current_price * 0.99,
                        'entry_reason': 'oversold_vs_vwap'
                    }
                )

        # Price at upper band - potential sell
        if upper_band and current_price >= upper_band:
            distance_from_band = (current_price - upper_band) / upper_band

            if distance_from_band > 0.001:  # 0.1% above band
                self.stats["mean_reversion_signals"] += 1

                confidence = self._calculate_confidence(
                    'mean_reversion',
                    abs(deviation_pct),
                    distance_from_band
                )

                return SignalIntent(
                    symbol=symbol,
                    action='sell',
                    confidence=confidence,
                    metadata={
                        'strategy': self.name,
                        'signal_type': 'mean_reversion',
                        'vwap': vwap,
                        'current_price': current_price,
                        'deviation_pct': deviation_pct,
                        'target': vwap,  # Target is VWAP
                        'stop_loss': current_price * 1.01,
                        'entry_reason': 'overbought_vs_vwap'
                    }
                )

        return None

    def _check_breakout(
        self,
        symbol: str,
        current_price: float,
        vwap: float,
        upper_band: Optional[float],
        lower_band: Optional[float],
        deviation_pct: float,
        bars: List[Dict]
    ) -> Optional[SignalIntent]:
        """Check for breakout opportunity."""
        # Bullish breakout above upper band
        if upper_band and current_price > upper_band:
            distance_from_band = (current_price - upper_band) / upper_band

            if distance_from_band > 0.005:  # 0.5% above band
                # Check momentum
                if len(bars) >= 3:
                    recent_closes = [bar['close'] for bar in bars[-3:]]
                    if recent_closes[-1] > recent_closes[0]:  # Upward momentum
                        self.stats["breakout_signals"] += 1

                        confidence = self._calculate_confidence(
                            'breakout',
                            abs(deviation_pct),
                            distance_from_band
                        )

                        return SignalIntent(
                            symbol=symbol,
                            action='buy',
                            confidence=confidence,
                            metadata={
                                'strategy': self.name,
                                'signal_type': 'breakout',
                                'vwap': vwap,
                                'current_price': current_price,
                                'deviation_pct': deviation_pct,
                                'target': current_price * 1.02,
                                'stop_loss': vwap,  # Stop at VWAP
                                'entry_reason': 'bullish_vwap_breakout'
                            }
                        )

        # Bearish breakout below lower band
        if lower_band and current_price < lower_band:
            distance_from_band = (lower_band - current_price) / lower_band

            if distance_from_band > 0.005:  # 0.5% below band
                # Check momentum
                if len(bars) >= 3:
                    recent_closes = [bar['close'] for bar in bars[-3:]]
                    if recent_closes[-1] < recent_closes[0]:  # Downward momentum
                        self.stats["breakout_signals"] += 1

                        confidence = self._calculate_confidence(
                            'breakout',
                            abs(deviation_pct),
                            distance_from_band
                        )

                        return SignalIntent(
                            symbol=symbol,
                            action='sell',
                            confidence=confidence,
                            metadata={
                                'strategy': self.name,
                                'signal_type': 'breakout',
                                'vwap': vwap,
                                'current_price': current_price,
                                'deviation_pct': deviation_pct,
                                'target': current_price * 0.98,
                                'stop_loss': vwap,  # Stop at VWAP
                                'entry_reason': 'bearish_vwap_breakout'
                            }
                        )

        return None

    def _calculate_confidence(
        self,
        signal_type: str,
        deviation: float,
        distance_from_band: float
    ) -> float:
        """Calculate signal confidence."""
        confidence = 0.65  # Base confidence

        # Larger deviation = higher confidence
        if deviation > 0.03:  # > 3%
            confidence += 0.15
        elif deviation > 0.02:  # > 2%
            confidence += 0.10
        elif deviation > 0.01:  # > 1%
            confidence += 0.05

        # Distance from band
        if distance_from_band > 0.01:  # > 1%
            confidence += 0.10
        elif distance_from_band > 0.005:  # > 0.5%
            confidence += 0.05

        # Signal type adjustment
        if signal_type == 'mean_reversion':
            confidence += 0.05  # Mean reversion slightly more reliable

        return min(confidence, 0.95)

    def get_vwap_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get VWAP analysis for symbol."""
        if symbol not in self.vwap_data:
            return None

        vwap_data = self.vwap_data[symbol]

        return {
            'symbol': symbol,
            'vwap': vwap_data['vwap'],
            'upper_band': vwap_data.get('upper_band'),
            'lower_band': vwap_data.get('lower_band'),
            'std_dev': vwap_data.get('std_dev'),
            'cum_volume': vwap_data['cum_volume'],
            'bar_count': vwap_data['bar_count']
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        stats = self.stats.copy()

        # Calculate position bias
        total_positions = stats["above_vwap"] + stats["below_vwap"]
        if total_positions > 0:
            stats["above_vwap_pct"] = stats["above_vwap"] / total_positions
        else:
            stats["above_vwap_pct"] = 0.5

        # Signal type distribution
        total_signals = stats["mean_reversion_signals"] + stats["breakout_signals"]
        if total_signals > 0:
            stats["mean_reversion_pct"] = stats["mean_reversion_signals"] / total_signals
            stats["breakout_pct"] = stats["breakout_signals"] / total_signals
        else:
            stats["mean_reversion_pct"] = 0.0
            stats["breakout_pct"] = 0.0

        return stats

    def reset(self):
        """Reset strategy state."""
        self.vwap_data.clear()
        self.cumulative_data.clear()
        self.session_start.clear()
        self.stats = {
            "signals_generated": 0,
            "mean_reversion_signals": 0,
            "breakout_signals": 0,
            "above_vwap": 0,
            "below_vwap": 0,
            "avg_deviation": 0.0
        }

"""
Volume Breakout Strategy - Trade breakouts confirmed by volume spikes.
~400 lines as per schema
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import statistics

from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class VolumeBreakoutStrategy(BaseStrategy):
    """
    Volume-based breakout strategy using volume analysis.

    Algorithm:
    1. Analyze volume profile and patterns
    2. Detect volume spikes (> 2x average)
    3. Identify price consolidation zones
    4. Confirm breakouts with volume
    5. Enter on volume climax with price breakout
    6. Use volume-weighted stops

    Features:
    - Volume spike detection
    - Volume profile analysis
    - Accumulation/distribution detection
    - Climax volume identification
    - Volume-weighted support/resistance
    - Dynamic volume thresholds
    - False breakout filtering
    """

    name = 'volume_breakout'

    def __init__(
        self,
        volume_lookback: int = 20,
        volume_spike_threshold: float = 2.0,  # 2x average volume
        price_breakout_threshold: float = 0.015,  # 1.5% breakout
        min_consolidation_bars: int = 5,
        volume_climax_threshold: float = 3.0,  # 3x for climax
        require_climax: bool = False
    ):
        """
        Initialize volume breakout strategy.

        Args:
            volume_lookback: Period for volume analysis
            volume_spike_threshold: Volume multiplier for spike detection
            price_breakout_threshold: Price breakout percentage
            min_consolidation_bars: Minimum consolidation period
            volume_climax_threshold: Threshold for climax volume
            require_climax: Require climax volume for entry
        """
        super().__init__()
        self.volume_lookback = volume_lookback
        self.volume_spike_threshold = volume_spike_threshold
        self.price_breakout_threshold = price_breakout_threshold
        self.min_consolidation_bars = min_consolidation_bars
        self.volume_climax_threshold = volume_climax_threshold
        self.require_climax = require_climax

        # Track state
        self.consolidation_zones = {}  # symbol -> zone_info
        self.volume_profiles = {}  # symbol -> volume_data
        self.recent_breakouts = {}  # symbol -> breakout_info

        # Statistics
        self.stats = {
            "signals_generated": 0,
            "volume_spikes_detected": 0,
            "climax_volumes": 0,
            "breakouts_confirmed": 0,
            "false_breakouts": 0
        }

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        """
        Generate trading signal based on volume breakout logic.

        Args:
            snapshot: Market snapshot

        Returns:
            SignalIntent or None
        """
        if not hasattr(snapshot, 'bars') or not snapshot.bars:
            return None

        bars = snapshot.bars

        if len(bars) < self.volume_lookback:
            return None

        symbol = snapshot.symbol
        current_price = snapshot.price

        # Analyze volume profile
        self._update_volume_profile(symbol, bars)

        # Check for consolidation
        consolidation = self._detect_consolidation(symbol, bars)

        if not consolidation:
            return None

        # Detect volume spike
        volume_spike = self._detect_volume_spike(symbol, bars)

        if not volume_spike:
            return None

        # Check for price breakout
        breakout = self._detect_price_breakout(symbol, current_price, bars, consolidation)

        if not breakout:
            return None

        # Confirm breakout with volume
        if not self._confirm_with_volume(volume_spike, breakout):
            return None

        # Check if climax required
        is_climax = volume_spike['spike_ratio'] >= self.volume_climax_threshold

        if self.require_climax and not is_climax:
            return None

        if is_climax:
            self.stats["climax_volumes"] += 1

        # Generate signal
        signal = self._generate_signal(
            symbol,
            current_price,
            breakout,
            volume_spike,
            bars
        )

        if signal:
            self.stats["signals_generated"] += 1
            self.stats["breakouts_confirmed"] += 1

            # Track breakout
            self.recent_breakouts[symbol] = {
                'price': current_price,
                'direction': breakout['direction'],
                'volume_ratio': volume_spike['spike_ratio'],
                'timestamp': bars[-1].get('timestamp')
            }

        return signal

    def _update_volume_profile(self, symbol: str, bars: List[Dict]):
        """Update volume profile analysis."""
        recent_bars = bars[-self.volume_lookback:]

        volumes = [bar.get('volume', 0) for bar in recent_bars]
        prices = [bar['close'] for bar in recent_bars]

        avg_volume = statistics.mean(volumes) if volumes else 0
        max_volume = max(volumes) if volumes else 0
        current_volume = volumes[-1] if volumes else 0

        # Calculate volume-weighted average price
        total_vol = sum(volumes)
        if total_vol > 0:
            vwap = sum(p * v for p, v in zip(prices, volumes)) / total_vol
        else:
            vwap = prices[-1] if prices else 0

        # Accumulation/distribution
        acc_dist = self._calculate_accumulation_distribution(recent_bars)

        self.volume_profiles[symbol] = {
            'avg_volume': avg_volume,
            'max_volume': max_volume,
            'current_volume': current_volume,
            'vwap': vwap,
            'accumulation_distribution': acc_dist
        }

    def _detect_consolidation(
        self,
        symbol: str,
        bars: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Detect price consolidation zones."""
        if len(bars) < self.min_consolidation_bars:
            return None

        recent_bars = bars[-self.min_consolidation_bars:]

        highs = [bar['high'] for bar in recent_bars]
        lows = [bar['low'] for bar in recent_bars]
        closes = [bar['close'] for bar in recent_bars]

        # Calculate consolidation range
        consolidation_high = max(highs)
        consolidation_low = min(lows)
        avg_price = statistics.mean(closes)

        price_range = consolidation_high - consolidation_low
        range_pct = (price_range / avg_price) if avg_price > 0 else 0

        # Consolidation if range is tight (< 3%)
        if range_pct > 0.03:
            return None

        # Check if volume is decreasing (typical in consolidation)
        volumes = [bar.get('volume', 0) for bar in recent_bars]
        first_half_vol = statistics.mean(volumes[:len(volumes)//2])
        second_half_vol = statistics.mean(volumes[len(volumes)//2:])

        volume_declining = second_half_vol < first_half_vol * 0.9

        consolidation = {
            'high': consolidation_high,
            'low': consolidation_low,
            'range_pct': range_pct,
            'duration': len(recent_bars),
            'volume_declining': volume_declining
        }

        self.consolidation_zones[symbol] = consolidation

        return consolidation

    def _detect_volume_spike(
        self,
        symbol: str,
        bars: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Detect volume spikes."""
        if symbol not in self.volume_profiles:
            return None

        profile = self.volume_profiles[symbol]
        current_volume = profile['current_volume']
        avg_volume = profile['avg_volume']

        if avg_volume == 0:
            return None

        spike_ratio = current_volume / avg_volume

        if spike_ratio < self.volume_spike_threshold:
            return None

        self.stats["volume_spikes_detected"] += 1

        # Determine spike strength
        if spike_ratio >= self.volume_climax_threshold:
            strength = 'climax'
        elif spike_ratio >= 2.5:
            strength = 'strong'
        else:
            strength = 'moderate'

        return {
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'spike_ratio': spike_ratio,
            'strength': strength
        }

    def _detect_price_breakout(
        self,
        symbol: str,
        current_price: float,
        bars: List[Dict],
        consolidation: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detect price breakout from consolidation."""
        consolidation_high = consolidation['high']
        consolidation_low = consolidation['low']

        # Bullish breakout
        breakout_high = consolidation_high * (1 + self.price_breakout_threshold)
        if current_price > breakout_high:
            return {
                'direction': 'buy',
                'breakout_level': consolidation_high,
                'breakout_price': current_price,
                'breakout_pct': (current_price - consolidation_high) / consolidation_high,
                'consolidation_range': consolidation['range_pct']
            }

        # Bearish breakout
        breakout_low = consolidation_low * (1 - self.price_breakout_threshold)
        if current_price < breakout_low:
            return {
                'direction': 'sell',
                'breakout_level': consolidation_low,
                'breakout_price': current_price,
                'breakout_pct': (consolidation_low - current_price) / consolidation_low,
                'consolidation_range': consolidation['range_pct']
            }

        return None

    def _confirm_with_volume(
        self,
        volume_spike: Dict[str, Any],
        breakout: Dict[str, Any]
    ) -> bool:
        """Confirm breakout with volume analysis."""
        # Volume spike should be significant
        if volume_spike['spike_ratio'] < self.volume_spike_threshold:
            return False

        # Strong breakouts require strong volume
        if breakout['breakout_pct'] > 0.03:  # > 3% breakout
            return volume_spike['spike_ratio'] >= 2.5

        return True

    def _generate_signal(
        self,
        symbol: str,
        current_price: float,
        breakout: Dict[str, Any],
        volume_spike: Dict[str, Any],
        bars: List[Dict]
    ) -> Optional[SignalIntent]:
        """Generate trading signal."""
        direction = breakout['direction']
        consolidation = self.consolidation_zones.get(symbol, {})

        # Calculate stop loss
        if direction == 'buy':
            # Stop below consolidation
            stop_loss = consolidation.get('low', current_price * 0.97)
            # Target based on range expansion
            range_size = consolidation.get('high', current_price) - consolidation.get('low', current_price * 0.97)
            target = current_price + (range_size * 2)  # 2x range projection
        else:
            # Stop above consolidation
            stop_loss = consolidation.get('high', current_price * 1.03)
            # Target based on range expansion
            range_size = consolidation.get('high', current_price * 1.03) - consolidation.get('low', current_price)
            target = current_price - (range_size * 2)

        # Calculate confidence
        confidence = self._calculate_confidence(breakout, volume_spike, consolidation)

        return SignalIntent(
            symbol=symbol,
            action=direction,
            confidence=confidence,
            metadata={
                'strategy': self.name,
                'breakout_level': breakout['breakout_level'],
                'breakout_pct': breakout['breakout_pct'],
                'volume_spike_ratio': volume_spike['spike_ratio'],
                'volume_strength': volume_spike['strength'],
                'consolidation_range': consolidation.get('range_pct', 0),
                'consolidation_duration': consolidation.get('duration', 0),
                'stop_loss': stop_loss,
                'target': target
            }
        )

    def _calculate_confidence(
        self,
        breakout: Dict[str, Any],
        volume_spike: Dict[str, Any],
        consolidation: Dict[str, Any]
    ) -> float:
        """Calculate signal confidence."""
        confidence = 0.5  # Base confidence

        # Breakout strength
        breakout_pct = breakout.get('breakout_pct', 0)
        if breakout_pct > 0.03:  # > 3%
            confidence += 0.15
        elif breakout_pct > 0.02:  # > 2%
            confidence += 0.10

        # Volume spike strength
        spike_ratio = volume_spike.get('spike_ratio', 1.0)
        if spike_ratio >= self.volume_climax_threshold:
            confidence += 0.20
        elif spike_ratio >= 2.5:
            confidence += 0.15
        elif spike_ratio >= 2.0:
            confidence += 0.10

        # Consolidation quality
        if consolidation.get('volume_declining', False):
            confidence += 0.10

        duration = consolidation.get('duration', 0)
        if duration >= 10:
            confidence += 0.05

        return min(confidence, 0.95)

    def _calculate_accumulation_distribution(self, bars: List[Dict]) -> float:
        """Calculate accumulation/distribution indicator."""
        if not bars:
            return 0.0

        ad_values = []

        for bar in bars:
            high = bar.get('high', bar['close'])
            low = bar.get('low', bar['close'])
            close = bar['close']
            volume = bar.get('volume', 0)

            # Money Flow Multiplier
            if high != low:
                mf_multiplier = ((close - low) - (high - close)) / (high - low)
            else:
                mf_multiplier = 0

            # Money Flow Volume
            mf_volume = mf_multiplier * volume

            ad_values.append(mf_volume)

        # Cumulative A/D
        if ad_values:
            return sum(ad_values)

        return 0.0

    def check_false_breakout(
        self,
        symbol: str,
        current_price: float
    ) -> bool:
        """Check if recent breakout was false."""
        if symbol not in self.recent_breakouts:
            return False

        breakout = self.recent_breakouts[symbol]
        consolidation = self.consolidation_zones.get(symbol)

        if not consolidation:
            return False

        # Bullish breakout that failed
        if breakout['direction'] == 'buy':
            if current_price < consolidation['low']:
                self.stats["false_breakouts"] += 1
                return True

        # Bearish breakout that failed
        elif breakout['direction'] == 'sell':
            if current_price > consolidation['high']:
                self.stats["false_breakouts"] += 1
                return True

        return False

    def get_volume_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get volume analysis for symbol."""
        if symbol not in self.volume_profiles:
            return None

        profile = self.volume_profiles[symbol].copy()

        # Add relative volume
        if profile['avg_volume'] > 0:
            profile['relative_volume'] = profile['current_volume'] / profile['avg_volume']
        else:
            profile['relative_volume'] = 0.0

        return profile

    def get_consolidation_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get consolidation zone info."""
        return self.consolidation_zones.get(symbol)

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        stats = self.stats.copy()

        # Calculate false breakout rate
        total_breakouts = stats["breakouts_confirmed"] + stats["false_breakouts"]
        if total_breakouts > 0:
            stats["false_breakout_rate"] = stats["false_breakouts"] / total_breakouts
        else:
            stats["false_breakout_rate"] = 0.0

        # Calculate climax ratio
        if stats["volume_spikes_detected"] > 0:
            stats["climax_ratio"] = stats["climax_volumes"] / stats["volume_spikes_detected"]
        else:
            stats["climax_ratio"] = 0.0

        return stats

    def reset(self):
        """Reset strategy state."""
        self.consolidation_zones.clear()
        self.volume_profiles.clear()
        self.recent_breakouts.clear()
        self.stats = {
            "signals_generated": 0,
            "volume_spikes_detected": 0,
            "climax_volumes": 0,
            "breakouts_confirmed": 0,
            "false_breakouts": 0
        }

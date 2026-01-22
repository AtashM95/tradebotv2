"""
Ichimoku Cloud Strategy - Comprehensive trend-following system.
~400 lines as per schema
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import statistics

from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class IchimokuStrategy(BaseStrategy):
    """
    Ichimoku Kinko Hyo (Equilibrium Chart) trading strategy.

    Algorithm:
    1. Calculate Ichimoku Cloud components:
       - Tenkan-sen (Conversion Line)
       - Kijun-sen (Base Line)
       - Senkou Span A (Leading Span A)
       - Senkou Span B (Leading Span B)
       - Chikou Span (Lagging Span)
    2. Analyze cloud relationships
    3. Identify trend direction and strength
    4. Generate signals on key crossovers
    5. Confirm with cloud position

    Features:
    - Complete Ichimoku system
    - Cloud color and thickness analysis
    - TK cross signals
    - Price-cloud relationship
    - Kumo breakout detection
    - Multiple confirmation levels
    - Lagging span validation
    """

    name = 'ichimoku'

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_span_b_period: int = 52,
        displacement: int = 26,
        require_cloud_confirmation: bool = True,
        require_chikou_confirmation: bool = False
    ):
        """
        Initialize Ichimoku strategy.

        Args:
            tenkan_period: Conversion line period
            kijun_period: Base line period
            senkou_span_b_period: Leading Span B period
            displacement: Displacement for Senkou Spans
            require_cloud_confirmation: Require price above/below cloud
            require_chikou_confirmation: Require Chikou Span confirmation
        """
        super().__init__()
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_span_b_period = senkou_span_b_period
        self.displacement = displacement
        self.require_cloud_confirmation = require_cloud_confirmation
        self.require_chikou_confirmation = require_chikou_confirmation

        # Track Ichimoku components
        self.ichimoku_data = {}  # symbol -> ichimoku_components
        self.signal_history = {}  # symbol -> recent_signals

        # Statistics
        self.stats = {
            "signals_generated": 0,
            "tk_crosses": 0,
            "kumo_breakouts": 0,
            "cloud_confirmed_signals": 0,
            "chikou_confirmed_signals": 0
        }

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        """
        Generate trading signal based on Ichimoku analysis.

        Args:
            snapshot: Market snapshot

        Returns:
            SignalIntent or None
        """
        if not hasattr(snapshot, 'bars') or not snapshot.bars:
            if not snapshot.history or len(snapshot.history) < self.senkou_span_b_period + self.displacement:
                return None
            bars = self._create_bars_from_history(snapshot.history)
        else:
            bars = snapshot.bars

        if len(bars) < self.senkou_span_b_period + self.displacement:
            return None

        symbol = snapshot.symbol
        current_price = snapshot.price

        # Calculate Ichimoku components
        ichimoku = self._calculate_ichimoku(bars)

        if not ichimoku:
            return None

        # Store Ichimoku data
        self.ichimoku_data[symbol] = ichimoku

        # Check for TK cross
        tk_cross = self._detect_tk_cross(bars, ichimoku)

        # Check for Kumo breakout
        kumo_breakout = self._detect_kumo_breakout(current_price, ichimoku)

        # Analyze cloud position
        cloud_position = self._analyze_cloud_position(current_price, ichimoku)

        # Check Chikou Span
        chikou_clear = self._check_chikou_span(bars, ichimoku)

        # Generate signal based on conditions
        signal = None

        if tk_cross:
            signal = self._generate_tk_cross_signal(
                symbol,
                current_price,
                tk_cross,
                ichimoku,
                cloud_position,
                chikou_clear
            )

        elif kumo_breakout:
            signal = self._generate_kumo_breakout_signal(
                symbol,
                current_price,
                kumo_breakout,
                ichimoku,
                chikou_clear
            )

        if signal:
            self.stats["signals_generated"] += 1

        return signal

    def _create_bars_from_history(self, history: List[float]) -> List[Dict]:
        """Create OHLC bars from price history."""
        bars = []
        for price in history:
            bar = {
                'open': price,
                'high': price,
                'low': price,
                'close': price
            }
            bars.append(bar)
        return bars

    def _calculate_ichimoku(self, bars: List[Dict]) -> Optional[Dict[str, Any]]:
        """Calculate all Ichimoku components."""
        if len(bars) < self.senkou_span_b_period:
            return None

        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        tenkan_sen = self._calculate_midpoint(bars, self.tenkan_period)

        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        kijun_sen = self._calculate_midpoint(bars, self.kijun_period)

        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
        # Plotted 26 periods ahead
        senkou_span_a = (tenkan_sen + kijun_sen) / 2

        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
        # Plotted 26 periods ahead
        senkou_span_b = self._calculate_midpoint(bars, self.senkou_span_b_period)

        # Chikou Span (Lagging Span): Current closing price
        # Plotted 26 periods behind
        chikou_span = bars[-1]['close']

        # Cloud boundaries
        cloud_top = max(senkou_span_a, senkou_span_b)
        cloud_bottom = min(senkou_span_a, senkou_span_b)

        # Cloud color
        if senkou_span_a > senkou_span_b:
            cloud_color = 'bullish'  # Green cloud
        else:
            cloud_color = 'bearish'  # Red cloud

        # Cloud thickness
        cloud_thickness = abs(senkou_span_a - senkou_span_b)

        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span,
            'cloud_top': cloud_top,
            'cloud_bottom': cloud_bottom,
            'cloud_color': cloud_color,
            'cloud_thickness': cloud_thickness
        }

    def _calculate_midpoint(self, bars: List[Dict], period: int) -> float:
        """Calculate midpoint of high/low over period."""
        if len(bars) < period:
            period = len(bars)

        recent_bars = bars[-period:]
        highs = [bar['high'] for bar in recent_bars]
        lows = [bar['low'] for bar in recent_bars]

        highest_high = max(highs)
        lowest_low = min(lows)

        return (highest_high + lowest_low) / 2

    def _detect_tk_cross(
        self,
        bars: List[Dict],
        ichimoku: Dict[str, Any]
    ) -> Optional[str]:
        """Detect Tenkan-sen / Kijun-sen crossover."""
        if len(bars) < 2:
            return None

        current_tenkan = ichimoku['tenkan_sen']
        current_kijun = ichimoku['kijun_sen']

        # Calculate previous values
        prev_bars = bars[:-1]
        prev_tenkan = self._calculate_midpoint(prev_bars, self.tenkan_period)
        prev_kijun = self._calculate_midpoint(prev_bars, self.kijun_period)

        # Bullish cross: Tenkan crosses above Kijun
        if prev_tenkan <= prev_kijun and current_tenkan > current_kijun:
            self.stats["tk_crosses"] += 1
            return 'bullish_cross'

        # Bearish cross: Tenkan crosses below Kijun
        if prev_tenkan >= prev_kijun and current_tenkan < current_kijun:
            self.stats["tk_crosses"] += 1
            return 'bearish_cross'

        return None

    def _detect_kumo_breakout(
        self,
        current_price: float,
        ichimoku: Dict[str, Any]
    ) -> Optional[str]:
        """Detect breakout from Kumo (cloud)."""
        cloud_top = ichimoku['cloud_top']
        cloud_bottom = ichimoku['cloud_bottom']

        # Bullish breakout: price breaks above cloud
        if current_price > cloud_top:
            distance_pct = (current_price - cloud_top) / cloud_top
            if distance_pct > 0.005:  # 0.5% above cloud
                self.stats["kumo_breakouts"] += 1
                return 'bullish_breakout'

        # Bearish breakout: price breaks below cloud
        if current_price < cloud_bottom:
            distance_pct = (cloud_bottom - current_price) / cloud_bottom
            if distance_pct > 0.005:  # 0.5% below cloud
                self.stats["kumo_breakouts"] += 1
                return 'bearish_breakout'

        return None

    def _analyze_cloud_position(
        self,
        current_price: float,
        ichimoku: Dict[str, Any]
    ) -> str:
        """Analyze price position relative to cloud."""
        cloud_top = ichimoku['cloud_top']
        cloud_bottom = ichimoku['cloud_bottom']

        if current_price > cloud_top:
            return 'above_cloud'
        elif current_price < cloud_bottom:
            return 'below_cloud'
        else:
            return 'inside_cloud'

    def _check_chikou_span(
        self,
        bars: List[Dict],
        ichimoku: Dict[str, Any]
    ) -> bool:
        """Check if Chikou Span is clear of price action."""
        if len(bars) < self.displacement:
            return True  # Not enough data

        chikou_span = ichimoku['chikou_span']

        # Check price action 26 periods ago
        historical_bars = bars[-(self.displacement + 5):-self.displacement]

        for bar in historical_bars:
            # Chikou should be clear of price
            if bar['low'] <= chikou_span <= bar['high']:
                return False

        return True

    def _generate_tk_cross_signal(
        self,
        symbol: str,
        current_price: float,
        tk_cross: str,
        ichimoku: Dict[str, Any],
        cloud_position: str,
        chikou_clear: bool
    ) -> Optional[SignalIntent]:
        """Generate signal from TK cross."""
        # Determine action
        if tk_cross == 'bullish_cross':
            action = 'buy'

            # Check cloud confirmation if required
            if self.require_cloud_confirmation and cloud_position != 'above_cloud':
                return None

        elif tk_cross == 'bearish_cross':
            action = 'sell'

            # Check cloud confirmation if required
            if self.require_cloud_confirmation and cloud_position != 'below_cloud':
                return None

        else:
            return None

        # Check Chikou confirmation if required
        if self.require_chikou_confirmation and not chikou_clear:
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(
            'tk_cross',
            ichimoku,
            cloud_position,
            chikou_clear
        )

        # Update stats
        if cloud_position in ['above_cloud', 'below_cloud']:
            self.stats["cloud_confirmed_signals"] += 1

        if chikou_clear:
            self.stats["chikou_confirmed_signals"] += 1

        # Calculate targets
        if action == 'buy':
            stop_loss = ichimoku['kijun_sen']
            target = current_price + (current_price - stop_loss) * 2
        else:
            stop_loss = ichimoku['kijun_sen']
            target = current_price - (stop_loss - current_price) * 2

        return SignalIntent(
            symbol=symbol,
            action=action,
            confidence=confidence,
            metadata={
                'strategy': self.name,
                'signal_type': 'tk_cross',
                'cross_type': tk_cross,
                'cloud_position': cloud_position,
                'cloud_color': ichimoku['cloud_color'],
                'chikou_clear': chikou_clear,
                'tenkan_sen': ichimoku['tenkan_sen'],
                'kijun_sen': ichimoku['kijun_sen'],
                'stop_loss': stop_loss,
                'target': target
            }
        )

    def _generate_kumo_breakout_signal(
        self,
        symbol: str,
        current_price: float,
        kumo_breakout: str,
        ichimoku: Dict[str, Any],
        chikou_clear: bool
    ) -> Optional[SignalIntent]:
        """Generate signal from Kumo breakout."""
        # Determine action
        if kumo_breakout == 'bullish_breakout':
            action = 'buy'
            stop_loss = ichimoku['cloud_top']
        elif kumo_breakout == 'bearish_breakout':
            action = 'sell'
            stop_loss = ichimoku['cloud_bottom']
        else:
            return None

        # Check Chikou confirmation if required
        if self.require_chikou_confirmation and not chikou_clear:
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(
            'kumo_breakout',
            ichimoku,
            'breakout',
            chikou_clear
        )

        # Update stats
        if chikou_clear:
            self.stats["chikou_confirmed_signals"] += 1

        # Calculate target
        cloud_thickness = ichimoku['cloud_thickness']
        if action == 'buy':
            target = current_price + (cloud_thickness * 2)
        else:
            target = current_price - (cloud_thickness * 2)

        return SignalIntent(
            symbol=symbol,
            action=action,
            confidence=confidence,
            metadata={
                'strategy': self.name,
                'signal_type': 'kumo_breakout',
                'breakout_type': kumo_breakout,
                'cloud_color': ichimoku['cloud_color'],
                'cloud_thickness': cloud_thickness,
                'chikou_clear': chikou_clear,
                'stop_loss': stop_loss,
                'target': target
            }
        )

    def _calculate_confidence(
        self,
        signal_type: str,
        ichimoku: Dict[str, Any],
        cloud_position: str,
        chikou_clear: bool
    ) -> float:
        """Calculate signal confidence."""
        confidence = 0.6  # Base confidence

        # Cloud color alignment
        if signal_type == 'tk_cross':
            if ichimoku['cloud_color'] == 'bullish':
                confidence += 0.10
        else:  # kumo_breakout
            confidence += 0.10

        # Cloud position
        if cloud_position in ['above_cloud', 'below_cloud']:
            confidence += 0.15

        # Chikou Span confirmation
        if chikou_clear:
            confidence += 0.10

        # Cloud thickness (thicker = more reliable)
        if ichimoku['cloud_thickness'] > 0:
            thickness_score = min(ichimoku['cloud_thickness'] / ichimoku['kijun_sen'], 0.05)
            confidence += thickness_score * 2  # Max 0.10

        return min(confidence, 0.95)

    def get_ichimoku_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get Ichimoku analysis for symbol."""
        if symbol not in self.ichimoku_data:
            return None

        ichimoku = self.ichimoku_data[symbol]

        return {
            'symbol': symbol,
            'tenkan_sen': ichimoku['tenkan_sen'],
            'kijun_sen': ichimoku['kijun_sen'],
            'senkou_span_a': ichimoku['senkou_span_a'],
            'senkou_span_b': ichimoku['senkou_span_b'],
            'chikou_span': ichimoku['chikou_span'],
            'cloud_top': ichimoku['cloud_top'],
            'cloud_bottom': ichimoku['cloud_bottom'],
            'cloud_color': ichimoku['cloud_color'],
            'cloud_thickness': ichimoku['cloud_thickness']
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        stats = self.stats.copy()

        # Calculate confirmation rates
        if stats["signals_generated"] > 0:
            stats["cloud_confirmation_rate"] = (
                stats["cloud_confirmed_signals"] / stats["signals_generated"]
            )
            stats["chikou_confirmation_rate"] = (
                stats["chikou_confirmed_signals"] / stats["signals_generated"]
            )
        else:
            stats["cloud_confirmation_rate"] = 0.0
            stats["chikou_confirmation_rate"] = 0.0

        return stats

    def reset(self):
        """Reset strategy state."""
        self.ichimoku_data.clear()
        self.signal_history.clear()
        self.stats = {
            "signals_generated": 0,
            "tk_crosses": 0,
            "kumo_breakouts": 0,
            "cloud_confirmed_signals": 0,
            "chikou_confirmed_signals": 0
        }

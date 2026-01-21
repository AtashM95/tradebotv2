"""
Volatility Analysis module for measuring market volatility and risk.
~350 lines as per schema
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import statistics
import math

from ..core.contracts import MarketSnapshot

logger = logging.getLogger(__name__)


@dataclass
class VolatilityMetrics:
    """Volatility metrics container."""
    historical_volatility: float
    atr: float  # Average True Range
    realized_volatility: float
    volatility_percentile: float
    regime: str  # 'low', 'normal', 'high'


def analyze(snapshot: MarketSnapshot) -> dict:
    """
    Analyze market snapshot for volatility.

    Args:
        snapshot: Market snapshot with OHLCV data

    Returns:
        Dictionary with volatility analysis
    """
    analyzer = VolatilityAnalyzer()

    if hasattr(snapshot, 'bars') and snapshot.bars:
        metrics = analyzer.calculate_metrics(snapshot.bars)

        return {
            'symbol': snapshot.symbol,
            'price': snapshot.price,
            'volatility': {
                'historical_volatility': metrics.historical_volatility,
                'atr': metrics.atr,
                'regime': metrics.regime
            }
        }

    return {
        'symbol': snapshot.symbol,
        'price': snapshot.price,
        'volatility': None
    }


class VolatilityAnalyzer:
    """
    Comprehensive volatility analysis system.

    Features:
    - Historical volatility calculation
    - Average True Range (ATR)
    - Bollinger Bands width
    - Parkinson volatility
    - Garman-Klass volatility
    - Volatility regime detection
    - Implied vs realized volatility
    - Volatility clustering detection
    """

    def __init__(self, period: int = 20):
        """
        Initialize volatility analyzer.

        Args:
            period: Lookback period for calculations
        """
        self.period = period

        # Statistics
        self.stats = {
            "analyses_performed": 0,
            "high_volatility_periods": 0,
            "low_volatility_periods": 0,
            "volatility_spikes": 0
        }

    def calculate_metrics(self, bars: List[Dict]) -> VolatilityMetrics:
        """
        Calculate comprehensive volatility metrics.

        Args:
            bars: List of OHLCV bars

        Returns:
            VolatilityMetrics object
        """
        if len(bars) < self.period:
            return VolatilityMetrics(0.0, 0.0, 0.0, 0.0, 'unknown')

        # Calculate various volatility measures
        hist_vol = self.calculate_historical_volatility(bars)
        atr = self.calculate_atr(bars)
        realized_vol = self.calculate_realized_volatility(bars)

        # Calculate percentile
        vol_history = [self.calculate_historical_volatility(bars[i:i+self.period])
                      for i in range(len(bars) - self.period + 1)]
        percentile = self._calculate_percentile(hist_vol, vol_history)

        # Determine regime
        regime = self._determine_regime(hist_vol, vol_history)

        self.stats["analyses_performed"] += 1

        if regime == 'high':
            self.stats["high_volatility_periods"] += 1
        elif regime == 'low':
            self.stats["low_volatility_periods"] += 1

        return VolatilityMetrics(
            historical_volatility=hist_vol,
            atr=atr,
            realized_volatility=realized_vol,
            volatility_percentile=percentile,
            regime=regime
        )

    def calculate_historical_volatility(
        self,
        bars: List[Dict],
        annualize: bool = True
    ) -> float:
        """
        Calculate historical volatility (standard deviation of returns).

        Args:
            bars: List of OHLCV bars
            annualize: Annualize the volatility (assumes daily bars)

        Returns:
            Historical volatility
        """
        if len(bars) < 2:
            return 0.0

        # Calculate log returns
        closes = [bar['close'] for bar in bars]
        log_returns = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]

        if len(log_returns) < 2:
            return 0.0

        # Calculate standard deviation
        volatility = statistics.stdev(log_returns)

        # Annualize if requested (assuming 252 trading days)
        if annualize:
            volatility *= math.sqrt(252)

        return volatility

    def calculate_atr(self, bars: List[Dict], period: Optional[int] = None) -> float:
        """
        Calculate Average True Range.

        Args:
            bars: List of OHLCV bars
            period: Period for ATR calculation

        Returns:
            ATR value
        """
        if period is None:
            period = self.period

        if len(bars) < period + 1:
            return 0.0

        # Calculate True Range for each bar
        true_ranges = []

        for i in range(1, len(bars)):
            high = bars[i]['high']
            low = bars[i]['low']
            prev_close = bars[i-1]['close']

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        # Calculate ATR as average of recent TRs
        recent_trs = true_ranges[-period:]
        atr = statistics.mean(recent_trs) if recent_trs else 0.0

        return atr

    def calculate_realized_volatility(self, bars: List[Dict]) -> float:
        """
        Calculate realized volatility using intraday high-low range.

        Args:
            bars: List of OHLCV bars

        Returns:
            Realized volatility
        """
        if len(bars) < self.period:
            return 0.0

        # Use Parkinson's method (high-low range)
        recent_bars = bars[-self.period:]

        sum_squared = 0.0
        for bar in recent_bars:
            if bar['high'] > 0 and bar['low'] > 0:
                hl_ratio = math.log(bar['high'] / bar['low'])
                sum_squared += hl_ratio ** 2

        # Parkinson's formula
        volatility = math.sqrt(sum_squared / (4 * len(recent_bars) * math.log(2)))

        # Annualize
        volatility *= math.sqrt(252)

        return volatility

    def calculate_garman_klass_volatility(self, bars: List[Dict]) -> float:
        """
        Calculate Garman-Klass volatility estimator.

        Args:
            bars: List of OHLCV bars

        Returns:
            GK volatility
        """
        if len(bars) < self.period:
            return 0.0

        recent_bars = bars[-self.period:]

        sum_gk = 0.0
        for bar in recent_bars:
            if bar['open'] > 0 and bar['close'] > 0 and bar['high'] > 0 and bar['low'] > 0:
                hl = math.log(bar['high'] / bar['low'])
                co = math.log(bar['close'] / bar['open'])

                sum_gk += 0.5 * (hl ** 2) - (2 * math.log(2) - 1) * (co ** 2)

        volatility = math.sqrt(sum_gk / len(recent_bars))

        # Annualize
        volatility *= math.sqrt(252)

        return volatility

    def calculate_bollinger_band_width(
        self,
        bars: List[Dict],
        period: Optional[int] = None,
        num_std: float = 2.0
    ) -> float:
        """
        Calculate Bollinger Band width as volatility measure.

        Args:
            bars: List of OHLCV bars
            period: Period for moving average
            num_std: Number of standard deviations

        Returns:
            Band width as percentage
        """
        if period is None:
            period = self.period

        if len(bars) < period:
            return 0.0

        recent_bars = bars[-period:]
        closes = [bar['close'] for bar in recent_bars]

        sma = statistics.mean(closes)
        std = statistics.stdev(closes)

        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)

        band_width = ((upper_band - lower_band) / sma) * 100 if sma > 0 else 0.0

        return band_width

    def detect_volatility_regime_change(
        self,
        bars: List[Dict],
        lookback: int = 50
    ) -> Optional[str]:
        """
        Detect changes in volatility regime.

        Args:
            bars: List of OHLCV bars
            lookback: Lookback period

        Returns:
            'expanding', 'contracting', or None
        """
        if len(bars) < lookback + self.period:
            return None

        # Calculate current and historical volatility
        current_vol = self.calculate_historical_volatility(bars[-self.period:])
        historical_vol = self.calculate_historical_volatility(bars[-lookback:-self.period])

        # Detect regime change
        if current_vol > historical_vol * 1.5:
            return 'expanding'
        elif current_vol < historical_vol * 0.67:
            return 'contracting'

        return None

    def detect_volatility_spike(
        self,
        bars: List[Dict],
        threshold: float = 2.0
    ) -> bool:
        """
        Detect volatility spikes.

        Args:
            bars: List of OHLCV bars
            threshold: Multiple of average volatility to consider spike

        Returns:
            True if spike detected
        """
        if len(bars) < self.period + 5:
            return False

        # Recent volatility
        recent_vol = self.calculate_historical_volatility(bars[-5:], annualize=False)

        # Historical average volatility
        hist_vols = [
            self.calculate_historical_volatility(bars[i:i+self.period], annualize=False)
            for i in range(len(bars) - self.period - 5)
        ]

        if not hist_vols:
            return False

        avg_vol = statistics.mean(hist_vols)

        # Check for spike
        if recent_vol > avg_vol * threshold:
            self.stats["volatility_spikes"] += 1
            return True

        return False

    def calculate_volatility_cone(
        self,
        bars: List[Dict],
        periods: List[int] = [10, 20, 30, 60, 90]
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate volatility cone for different periods.

        Args:
            bars: List of OHLCV bars
            periods: List of periods to analyze

        Returns:
            Dictionary with volatility statistics for each period
        """
        cone = {}

        for period in periods:
            if len(bars) < period * 3:
                continue

            # Calculate rolling volatilities
            vols = []
            for i in range(len(bars) - period + 1):
                vol = self.calculate_historical_volatility(bars[i:i+period])
                vols.append(vol)

            if vols:
                cone[period] = {
                    'current': vols[-1],
                    'min': min(vols),
                    'max': max(vols),
                    'mean': statistics.mean(vols),
                    'median': statistics.median(vols),
                    'percentile_10': self._percentile(vols, 0.1),
                    'percentile_90': self._percentile(vols, 0.9)
                }

        return cone

    def _determine_regime(
        self,
        current_vol: float,
        vol_history: List[float]
    ) -> str:
        """Determine volatility regime."""
        if not vol_history:
            return 'unknown'

        percentile = self._calculate_percentile(current_vol, vol_history)

        if percentile > 75:
            return 'high'
        elif percentile < 25:
            return 'low'
        else:
            return 'normal'

    def _calculate_percentile(
        self,
        value: float,
        data: List[float]
    ) -> float:
        """Calculate percentile rank of value in data."""
        if not data:
            return 50.0

        sorted_data = sorted(data)
        rank = sum(1 for x in sorted_data if x < value)

        percentile = (rank / len(sorted_data)) * 100

        return percentile

    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate p-th percentile of data."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int(len(sorted_data) * p)
        index = min(index, len(sorted_data) - 1)

        return sorted_data[index]

    def get_stats(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "analyses_performed": 0,
            "high_volatility_periods": 0,
            "low_volatility_periods": 0,
            "volatility_spikes": 0
        }

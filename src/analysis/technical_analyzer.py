"""
Technical Analyzer - Main Orchestrator Module

Coordinates all technical analysis components and provides comprehensive
technical analysis results combining indicators, patterns, trends, and signals.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .indicator_calculator import IndicatorCalculator, IndicatorResult


class SignalStrength(Enum):
    """Signal strength enumeration."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class TrendDirection(Enum):
    """Trend direction enumeration."""
    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    SIDEWAYS = "SIDEWAYS"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"


@dataclass
class TechnicalSignal:
    """Technical analysis signal."""
    signal_type: str
    strength: SignalStrength
    confidence: float
    indicators: List[str]
    description: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TechnicalAnalysisResult:
    """Complete technical analysis result."""
    symbol: str
    timestamp: datetime
    price: float

    # Trend Analysis
    trend_direction: TrendDirection
    trend_strength: float

    # Signal Summary
    overall_signal: SignalStrength
    buy_signals: int
    sell_signals: int
    neutral_signals: int

    # Indicators
    indicators: Dict[str, Any]

    # Support/Resistance
    support_levels: List[float]
    resistance_levels: List[float]

    # Momentum
    momentum_score: float

    # Volatility
    volatility_score: float

    # Volume Analysis
    volume_trend: str
    volume_strength: float

    # Detailed Signals
    signals: List[TechnicalSignal]

    # Raw Data
    raw_data: Optional[Dict[str, Any]] = None


class TechnicalAnalyzer:
    """
    Main technical analyzer orchestrator.

    Coordinates all technical analysis components and provides
    comprehensive analysis results.
    """

    def __init__(self):
        """Initialize the technical analyzer."""
        self.indicator_calc = IndicatorCalculator()
        self.analysis_cache = {}

    def analyze(
        self,
        symbol: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: Optional[np.ndarray] = None,
        open_price: Optional[np.ndarray] = None
    ) -> TechnicalAnalysisResult:
        """
        Perform comprehensive technical analysis.

        Args:
            symbol: Symbol to analyze
            high: High prices array
            low: Low prices array
            close: Close prices array
            volume: Volume array (optional)
            open_price: Open prices array (optional)

        Returns:
            TechnicalAnalysisResult object
        """
        # Calculate all indicators
        indicators = self._calculate_indicators(high, low, close, volume)

        # Analyze trend
        trend_direction, trend_strength = self._analyze_trend(close, indicators)

        # Generate signals
        signals = self._generate_signals(indicators, high, low, close, volume)

        # Analyze momentum
        momentum_score = self._analyze_momentum(indicators)

        # Analyze volatility
        volatility_score = self._analyze_volatility(indicators, high, low, close)

        # Analyze volume
        volume_trend, volume_strength = self._analyze_volume(volume, close) if volume is not None else ("N/A", 0.0)

        # Detect support and resistance
        support_levels, resistance_levels = self._detect_support_resistance(high, low, close)

        # Calculate overall signal
        overall_signal = self._calculate_overall_signal(signals)

        # Count signals
        buy_signals = len([s for s in signals if s.strength in [SignalStrength.BUY, SignalStrength.STRONG_BUY]])
        sell_signals = len([s for s in signals if s.strength in [SignalStrength.SELL, SignalStrength.STRONG_SELL]])
        neutral_signals = len([s for s in signals if s.strength == SignalStrength.NEUTRAL])

        return TechnicalAnalysisResult(
            symbol=symbol,
            timestamp=datetime.now(),
            price=float(close[-1]),
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            overall_signal=overall_signal,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            neutral_signals=neutral_signals,
            indicators=indicators,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            volume_trend=volume_trend,
            volume_strength=volume_strength,
            signals=signals,
            raw_data={
                'high': high[-50:].tolist(),
                'low': low[-50:].tolist(),
                'close': close[-50:].tolist(),
                'volume': volume[-50:].tolist() if volume is not None else []
            }
        )

    def _calculate_indicators(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Calculate all technical indicators."""
        indicators = {}

        # Moving Averages
        indicators['sma_20'] = self.indicator_calc.sma(close, 20)
        indicators['sma_50'] = self.indicator_calc.sma(close, 50)
        indicators['sma_200'] = self.indicator_calc.sma(close, 200)
        indicators['ema_12'] = self.indicator_calc.ema(close, 12)
        indicators['ema_26'] = self.indicator_calc.ema(close, 26)
        indicators['ema_50'] = self.indicator_calc.ema(close, 50)

        # Momentum Indicators
        indicators['rsi'] = self.indicator_calc.rsi(close, 14)
        indicators['rsi_9'] = self.indicator_calc.rsi(close, 9)

        macd_line, signal_line, histogram = self.indicator_calc.macd(close)
        indicators['macd'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = histogram

        stoch_k, stoch_d = self.indicator_calc.stochastic(high, low, close)
        indicators['stoch_k'] = stoch_k
        indicators['stoch_d'] = stoch_d

        indicators['williams_r'] = self.indicator_calc.williams_r(high, low, close)
        indicators['cci'] = self.indicator_calc.cci(high, low, close)
        indicators['roc'] = self.indicator_calc.roc(close)

        # Trend Indicators
        adx, plus_di, minus_di = self.indicator_calc.adx(high, low, close)
        indicators['adx'] = adx
        indicators['plus_di'] = plus_di
        indicators['minus_di'] = minus_di

        aroon_up, aroon_down = self.indicator_calc.aroon(high, low)
        indicators['aroon_up'] = aroon_up
        indicators['aroon_down'] = aroon_down

        supertrend, st_direction = self.indicator_calc.supertrend(high, low, close)
        indicators['supertrend'] = supertrend
        indicators['supertrend_direction'] = st_direction

        # Volatility Indicators
        indicators['atr'] = self.indicator_calc.atr(high, low, close)

        bb_upper, bb_middle, bb_lower = self.indicator_calc.bollinger_bands(close)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower

        kc_upper, kc_middle, kc_lower = self.indicator_calc.keltner_channels(high, low, close)
        indicators['kc_upper'] = kc_upper
        indicators['kc_middle'] = kc_middle
        indicators['kc_lower'] = kc_lower

        dc_upper, dc_middle, dc_lower = self.indicator_calc.donchian_channels(high, low)
        indicators['dc_upper'] = dc_upper
        indicators['dc_middle'] = dc_middle
        indicators['dc_lower'] = dc_lower

        # Volume Indicators (if volume provided)
        if volume is not None:
            indicators['obv'] = self.indicator_calc.obv(close, volume)
            indicators['mfi'] = self.indicator_calc.mfi(high, low, close, volume)
            indicators['vwap'] = self.indicator_calc.vwap(high, low, close, volume)

        return indicators

    def _analyze_trend(
        self,
        close: np.ndarray,
        indicators: Dict[str, Any]
    ) -> Tuple[TrendDirection, float]:
        """Analyze trend direction and strength."""
        current_price = close[-1]

        # Check moving average alignment
        sma_20 = indicators['sma_20'][-1] if not np.isnan(indicators['sma_20'][-1]) else current_price
        sma_50 = indicators['sma_50'][-1] if not np.isnan(indicators['sma_50'][-1]) else current_price
        sma_200 = indicators['sma_200'][-1] if not np.isnan(indicators['sma_200'][-1]) else current_price

        # ADX for trend strength
        adx_value = indicators['adx'][-1] if not np.isnan(indicators['adx'][-1]) else 20
        plus_di = indicators['plus_di'][-1] if not np.isnan(indicators['plus_di'][-1]) else 20
        minus_di = indicators['minus_di'][-1] if not np.isnan(indicators['minus_di'][-1]) else 20

        # Aroon for trend confirmation
        aroon_up = indicators['aroon_up'][-1] if not np.isnan(indicators['aroon_up'][-1]) else 50
        aroon_down = indicators['aroon_down'][-1] if not np.isnan(indicators['aroon_down'][-1]) else 50

        # Calculate trend score (-100 to +100)
        trend_score = 0

        # Moving average alignment (max 40 points)
        if current_price > sma_20:
            trend_score += 10
        else:
            trend_score -= 10

        if current_price > sma_50:
            trend_score += 15
        else:
            trend_score -= 15

        if current_price > sma_200:
            trend_score += 15
        else:
            trend_score -= 15

        # ADX and DI (max 30 points)
        if plus_di > minus_di:
            trend_score += min(30, (plus_di - minus_di) / 2)
        else:
            trend_score -= min(30, (minus_di - plus_di) / 2)

        # Aroon (max 30 points)
        if aroon_up > aroon_down:
            trend_score += (aroon_up - aroon_down) / 100 * 30
        else:
            trend_score -= (aroon_down - aroon_up) / 100 * 30

        # Determine trend direction
        if trend_score > 60:
            direction = TrendDirection.STRONG_UPTREND
        elif trend_score > 20:
            direction = TrendDirection.UPTREND
        elif trend_score > -20:
            direction = TrendDirection.SIDEWAYS
        elif trend_score > -60:
            direction = TrendDirection.DOWNTREND
        else:
            direction = TrendDirection.STRONG_DOWNTREND

        # Trend strength (0-100)
        strength = min(100, adx_value * 1.5)

        return direction, strength

    def _generate_signals(
        self,
        indicators: Dict[str, Any],
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: Optional[np.ndarray]
    ) -> List[TechnicalSignal]:
        """Generate trading signals from indicators."""
        signals = []
        current_price = close[-1]

        # RSI Signal
        rsi = indicators['rsi'][-1]
        if not np.isnan(rsi):
            if rsi < 30:
                signals.append(TechnicalSignal(
                    signal_type="RSI_OVERSOLD",
                    strength=SignalStrength.BUY if rsi < 25 else SignalStrength.BUY,
                    confidence=min(1.0, (30 - rsi) / 30),
                    indicators=['RSI'],
                    description=f"RSI oversold at {rsi:.2f}"
                ))
            elif rsi > 70:
                signals.append(TechnicalSignal(
                    signal_type="RSI_OVERBOUGHT",
                    strength=SignalStrength.SELL if rsi > 75 else SignalStrength.SELL,
                    confidence=min(1.0, (rsi - 70) / 30),
                    indicators=['RSI'],
                    description=f"RSI overbought at {rsi:.2f}"
                ))

        # MACD Signal
        macd = indicators['macd'][-1]
        macd_signal = indicators['macd_signal'][-1]
        macd_hist = indicators['macd_histogram'][-1]

        if not (np.isnan(macd) or np.isnan(macd_signal)):
            if macd > macd_signal and macd_hist > 0:
                signals.append(TechnicalSignal(
                    signal_type="MACD_BULLISH",
                    strength=SignalStrength.BUY,
                    confidence=min(1.0, abs(macd_hist) / (abs(macd) + 0.01) * 2),
                    indicators=['MACD'],
                    description="MACD bullish crossover"
                ))
            elif macd < macd_signal and macd_hist < 0:
                signals.append(TechnicalSignal(
                    signal_type="MACD_BEARISH",
                    strength=SignalStrength.SELL,
                    confidence=min(1.0, abs(macd_hist) / (abs(macd) + 0.01) * 2),
                    indicators=['MACD'],
                    description="MACD bearish crossover"
                ))

        # Stochastic Signal
        stoch_k = indicators['stoch_k'][-1]
        stoch_d = indicators['stoch_d'][-1]

        if not (np.isnan(stoch_k) or np.isnan(stoch_d)):
            if stoch_k < 20 and stoch_k > stoch_d:
                signals.append(TechnicalSignal(
                    signal_type="STOCHASTIC_OVERSOLD",
                    strength=SignalStrength.BUY,
                    confidence=min(1.0, (20 - stoch_k) / 20),
                    indicators=['Stochastic'],
                    description=f"Stochastic oversold at {stoch_k:.2f}"
                ))
            elif stoch_k > 80 and stoch_k < stoch_d:
                signals.append(TechnicalSignal(
                    signal_type="STOCHASTIC_OVERBOUGHT",
                    strength=SignalStrength.SELL,
                    confidence=min(1.0, (stoch_k - 80) / 20),
                    indicators=['Stochastic'],
                    description=f"Stochastic overbought at {stoch_k:.2f}"
                ))

        # Bollinger Bands Signal
        bb_upper = indicators['bb_upper'][-1]
        bb_lower = indicators['bb_lower'][-1]
        bb_middle = indicators['bb_middle'][-1]

        if not (np.isnan(bb_upper) or np.isnan(bb_lower)):
            if current_price < bb_lower:
                signals.append(TechnicalSignal(
                    signal_type="BB_OVERSOLD",
                    strength=SignalStrength.BUY,
                    confidence=min(1.0, (bb_lower - current_price) / bb_lower * 10),
                    indicators=['Bollinger Bands'],
                    description="Price below lower Bollinger Band"
                ))
            elif current_price > bb_upper:
                signals.append(TechnicalSignal(
                    signal_type="BB_OVERBOUGHT",
                    strength=SignalStrength.SELL,
                    confidence=min(1.0, (current_price - bb_upper) / bb_upper * 10),
                    indicators=['Bollinger Bands'],
                    description="Price above upper Bollinger Band"
                ))

        # Moving Average Crossover
        sma_20 = indicators['sma_20'][-1]
        sma_50 = indicators['sma_50'][-1]
        sma_20_prev = indicators['sma_20'][-2] if len(indicators['sma_20']) > 1 else sma_20
        sma_50_prev = indicators['sma_50'][-2] if len(indicators['sma_50']) > 1 else sma_50

        if not (np.isnan(sma_20) or np.isnan(sma_50)):
            # Golden Cross
            if sma_20 > sma_50 and sma_20_prev <= sma_50_prev:
                signals.append(TechnicalSignal(
                    signal_type="GOLDEN_CROSS",
                    strength=SignalStrength.STRONG_BUY,
                    confidence=0.85,
                    indicators=['SMA_20', 'SMA_50'],
                    description="Golden cross: SMA 20 crossed above SMA 50"
                ))
            # Death Cross
            elif sma_20 < sma_50 and sma_20_prev >= sma_50_prev:
                signals.append(TechnicalSignal(
                    signal_type="DEATH_CROSS",
                    strength=SignalStrength.STRONG_SELL,
                    confidence=0.85,
                    indicators=['SMA_20', 'SMA_50'],
                    description="Death cross: SMA 20 crossed below SMA 50"
                ))

        # ADX Trend Strength
        adx = indicators['adx'][-1]
        plus_di = indicators['plus_di'][-1]
        minus_di = indicators['minus_di'][-1]

        if not (np.isnan(adx) or np.isnan(plus_di) or np.isnan(minus_di)):
            if adx > 25:
                if plus_di > minus_di:
                    signals.append(TechnicalSignal(
                        signal_type="ADX_STRONG_UPTREND",
                        strength=SignalStrength.BUY,
                        confidence=min(1.0, adx / 50),
                        indicators=['ADX', '+DI', '-DI'],
                        description=f"Strong uptrend detected (ADX: {adx:.2f})"
                    ))
                else:
                    signals.append(TechnicalSignal(
                        signal_type="ADX_STRONG_DOWNTREND",
                        strength=SignalStrength.SELL,
                        confidence=min(1.0, adx / 50),
                        indicators=['ADX', '+DI', '-DI'],
                        description=f"Strong downtrend detected (ADX: {adx:.2f})"
                    ))

        # SuperTrend Signal
        supertrend = indicators['supertrend'][-1]
        st_direction = indicators['supertrend_direction'][-1]

        if not (np.isnan(supertrend) or np.isnan(st_direction)):
            if st_direction > 0:
                signals.append(TechnicalSignal(
                    signal_type="SUPERTREND_BUY",
                    strength=SignalStrength.BUY,
                    confidence=0.75,
                    indicators=['SuperTrend'],
                    description="SuperTrend bullish signal"
                ))
            else:
                signals.append(TechnicalSignal(
                    signal_type="SUPERTREND_SELL",
                    strength=SignalStrength.SELL,
                    confidence=0.75,
                    indicators=['SuperTrend'],
                    description="SuperTrend bearish signal"
                ))

        return signals

    def _analyze_momentum(self, indicators: Dict[str, Any]) -> float:
        """Calculate momentum score (0-100)."""
        score = 50.0  # Neutral starting point

        # RSI contribution
        rsi = indicators['rsi'][-1]
        if not np.isnan(rsi):
            score += (rsi - 50) * 0.3

        # MACD histogram contribution
        macd_hist = indicators['macd_histogram'][-1]
        if not np.isnan(macd_hist):
            score += np.sign(macd_hist) * min(20, abs(macd_hist) * 10)

        # ROC contribution
        roc = indicators['roc'][-1]
        if not np.isnan(roc):
            score += np.sign(roc) * min(10, abs(roc) / 2)

        # Williams %R contribution
        williams = indicators['williams_r'][-1]
        if not np.isnan(williams):
            score += (williams + 50) * 0.2

        return np.clip(score, 0, 100)

    def _analyze_volatility(
        self,
        indicators: Dict[str, Any],
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> float:
        """Calculate volatility score (0-100)."""
        # ATR contribution
        atr = indicators['atr'][-1]
        current_price = close[-1]

        if not np.isnan(atr) and current_price > 0:
            atr_percent = (atr / current_price) * 100
            volatility_score = min(100, atr_percent * 5)
        else:
            volatility_score = 50.0

        # Bollinger Band width contribution
        bb_upper = indicators['bb_upper'][-1]
        bb_lower = indicators['bb_lower'][-1]

        if not (np.isnan(bb_upper) or np.isnan(bb_lower)) and current_price > 0:
            bb_width = ((bb_upper - bb_lower) / current_price) * 100
            volatility_score = (volatility_score + min(100, bb_width * 10)) / 2

        return volatility_score

    def _analyze_volume(
        self,
        volume: np.ndarray,
        close: np.ndarray
    ) -> Tuple[str, float]:
        """Analyze volume trend and strength."""
        if len(volume) < 20:
            return "INSUFFICIENT_DATA", 0.0

        # Calculate volume moving averages
        vol_sma_20 = np.mean(volume[-20:])
        current_volume = volume[-1]

        # Volume trend
        if current_volume > vol_sma_20 * 1.5:
            trend = "HIGH"
            strength = min(100, (current_volume / vol_sma_20 - 1) * 100)
        elif current_volume > vol_sma_20:
            trend = "ABOVE_AVERAGE"
            strength = (current_volume / vol_sma_20 - 1) * 100
        elif current_volume < vol_sma_20 * 0.5:
            trend = "LOW"
            strength = (1 - current_volume / vol_sma_20) * 100
        else:
            trend = "NORMAL"
            strength = 50.0

        return trend, strength

    def _detect_support_resistance(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """Detect support and resistance levels."""
        if len(close) < 50:
            return [], []

        # Use recent data (last 100 bars or available)
        lookback = min(100, len(close))
        recent_high = high[-lookback:]
        recent_low = low[-lookback:]
        recent_close = close[-lookback:]

        # Find local maxima (resistance)
        resistance_levels = []
        for i in range(2, len(recent_high) - 2):
            if (recent_high[i] > recent_high[i-1] and recent_high[i] > recent_high[i-2] and
                recent_high[i] > recent_high[i+1] and recent_high[i] > recent_high[i+2]):
                resistance_levels.append(float(recent_high[i]))

        # Find local minima (support)
        support_levels = []
        for i in range(2, len(recent_low) - 2):
            if (recent_low[i] < recent_low[i-1] and recent_low[i] < recent_low[i-2] and
                recent_low[i] < recent_low[i+1] and recent_low[i] < recent_low[i+2]):
                support_levels.append(float(recent_low[i]))

        # Cluster similar levels
        support_levels = self._cluster_levels(support_levels, tolerance=0.02)
        resistance_levels = self._cluster_levels(resistance_levels, tolerance=0.02)

        # Sort and return top 5 of each
        support_levels.sort(reverse=True)
        resistance_levels.sort()

        return support_levels[:5], resistance_levels[:5]

    def _cluster_levels(self, levels: List[float], tolerance: float = 0.02) -> List[float]:
        """Cluster similar price levels."""
        if not levels:
            return []

        clustered = []
        sorted_levels = sorted(levels)

        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]

        clustered.append(np.mean(current_cluster))

        return clustered

    def _calculate_overall_signal(self, signals: List[TechnicalSignal]) -> SignalStrength:
        """Calculate overall signal from all individual signals."""
        if not signals:
            return SignalStrength.NEUTRAL

        # Weight signals by confidence
        buy_score = sum(
            s.confidence for s in signals
            if s.strength in [SignalStrength.BUY, SignalStrength.STRONG_BUY]
        )
        sell_score = sum(
            s.confidence for s in signals
            if s.strength in [SignalStrength.SELL, SignalStrength.STRONG_SELL]
        )

        # Calculate net score
        net_score = buy_score - sell_score
        total_score = buy_score + sell_score

        if total_score == 0:
            return SignalStrength.NEUTRAL

        # Normalize to -1 to 1
        normalized = net_score / total_score

        # Determine overall signal
        if normalized > 0.5:
            return SignalStrength.STRONG_BUY
        elif normalized > 0.15:
            return SignalStrength.BUY
        elif normalized < -0.5:
            return SignalStrength.STRONG_SELL
        elif normalized < -0.15:
            return SignalStrength.SELL
        else:
            return SignalStrength.NEUTRAL

    def get_summary(self, result: TechnicalAnalysisResult) -> Dict[str, Any]:
        """Get a summary of the technical analysis."""
        return {
            'symbol': result.symbol,
            'price': result.price,
            'timestamp': result.timestamp.isoformat(),
            'overall_signal': result.overall_signal.value,
            'trend': result.trend_direction.value,
            'trend_strength': f"{result.trend_strength:.2f}%",
            'momentum': f"{result.momentum_score:.2f}/100",
            'volatility': f"{result.volatility_score:.2f}/100",
            'signals': {
                'buy': result.buy_signals,
                'sell': result.sell_signals,
                'neutral': result.neutral_signals,
                'total': len(result.signals)
            },
            'support_levels': [f"${level:.2f}" for level in result.support_levels],
            'resistance_levels': [f"${level:.2f}" for level in result.resistance_levels],
            'top_signals': [
                {
                    'type': s.signal_type,
                    'strength': s.strength.value,
                    'confidence': f"{s.confidence:.2%}",
                    'description': s.description
                }
                for s in sorted(result.signals, key=lambda x: x.confidence, reverse=True)[:5]
            ]
        }


# Convenience function
def quick_analysis(
    symbol: str,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Perform quick technical analysis and return summary."""
    analyzer = TechnicalAnalyzer()
    result = analyzer.analyze(symbol, high, low, close, volume)
    return analyzer.get_summary(result)

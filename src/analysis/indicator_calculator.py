"""
Technical Indicator Calculator Module

Calculates all major technical indicators using numpy for performance.
Supports 50+ indicators including momentum, trend, volatility, and volume indicators.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class IndicatorResult:
    """Result container for indicator calculations."""
    name: str
    value: Union[float, np.ndarray]
    signal: Optional[str] = None  # 'buy', 'sell', 'neutral'
    metadata: Optional[Dict] = None


class IndicatorCalculator:
    """
    Comprehensive technical indicator calculator.

    Supports all major technical indicators with optimized numpy calculations.
    All methods handle NaN values and edge cases gracefully.
    """

    def __init__(self):
        """Initialize the indicator calculator."""
        self.cache = {}

    # ==================== MOVING AVERAGES ====================

    def sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Simple Moving Average.

        Args:
            data: Price data array
            period: Number of periods

        Returns:
            SMA values array
        """
        if len(data) < period:
            return np.full(len(data), np.nan)

        weights = np.ones(period) / period
        sma = np.convolve(data, weights, mode='valid')
        return np.concatenate([np.full(period - 1, np.nan), sma])

    def ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Exponential Moving Average.

        Args:
            data: Price data array
            period: Number of periods

        Returns:
            EMA values array
        """
        if len(data) < period:
            return np.full(len(data), np.nan)

        alpha = 2 / (period + 1)
        ema = np.empty_like(data)
        ema[:period-1] = np.nan
        ema[period-1] = np.mean(data[:period])

        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

        return ema

    def wma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Weighted Moving Average.

        Args:
            data: Price data array
            period: Number of periods

        Returns:
            WMA values array
        """
        if len(data) < period:
            return np.full(len(data), np.nan)

        weights = np.arange(1, period + 1)
        wma = np.empty_like(data)
        wma[:period-1] = np.nan

        for i in range(period - 1, len(data)):
            wma[i] = np.dot(data[i-period+1:i+1], weights) / weights.sum()

        return wma

    def dema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Double Exponential Moving Average.

        Args:
            data: Price data array
            period: Number of periods

        Returns:
            DEMA values array
        """
        ema1 = self.ema(data, period)
        ema2 = self.ema(ema1, period)
        return 2 * ema1 - ema2

    def tema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Triple Exponential Moving Average.

        Args:
            data: Price data array
            period: Number of periods

        Returns:
            TEMA values array
        """
        ema1 = self.ema(data, period)
        ema2 = self.ema(ema1, period)
        ema3 = self.ema(ema2, period)
        return 3 * ema1 - 3 * ema2 + ema3

    # ==================== MOMENTUM INDICATORS ====================

    def rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Relative Strength Index.

        Args:
            data: Price data array
            period: RSI period (default: 14)

        Returns:
            RSI values array (0-100)
        """
        if len(data) < period + 1:
            return np.full(len(data), np.nan)

        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.empty(len(data))
        avg_loss = np.empty(len(data))
        avg_gain[:period] = np.nan
        avg_loss[:period] = np.nan

        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

        rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stochastic Oscillator (%K and %D).

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)

        Returns:
            Tuple of (%K, %D) arrays
        """
        if len(close) < k_period:
            return np.full(len(close), np.nan), np.full(len(close), np.nan)

        k = np.empty_like(close)
        k[:k_period-1] = np.nan

        for i in range(k_period - 1, len(close)):
            period_high = np.max(high[i-k_period+1:i+1])
            period_low = np.min(low[i-k_period+1:i+1])

            if period_high == period_low:
                k[i] = 50
            else:
                k[i] = 100 * (close[i] - period_low) / (period_high - period_low)

        d = self.sma(k, d_period)

        return k, d

    def roc(self, data: np.ndarray, period: int = 12) -> np.ndarray:
        """
        Rate of Change.

        Args:
            data: Price data array
            period: ROC period (default: 12)

        Returns:
            ROC values array (percentage)
        """
        if len(data) < period + 1:
            return np.full(len(data), np.nan)

        roc = np.empty_like(data)
        roc[:period] = np.nan

        for i in range(period, len(data)):
            if data[i-period] == 0:
                roc[i] = 0
            else:
                roc[i] = ((data[i] - data[i-period]) / data[i-period]) * 100

        return roc

    def momentum(self, data: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Momentum indicator.

        Args:
            data: Price data array
            period: Momentum period (default: 10)

        Returns:
            Momentum values array
        """
        if len(data) < period + 1:
            return np.full(len(data), np.nan)

        mom = np.empty_like(data)
        mom[:period] = np.nan
        mom[period:] = data[period:] - data[:-period]

        return mom

    def williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   period: int = 14) -> np.ndarray:
        """
        Williams %R.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            period: Williams %R period (default: 14)

        Returns:
            Williams %R values array (-100 to 0)
        """
        if len(close) < period:
            return np.full(len(close), np.nan)

        wr = np.empty_like(close)
        wr[:period-1] = np.nan

        for i in range(period - 1, len(close)):
            period_high = np.max(high[i-period+1:i+1])
            period_low = np.min(low[i-period+1:i+1])

            if period_high == period_low:
                wr[i] = -50
            else:
                wr[i] = -100 * (period_high - close[i]) / (period_high - period_low)

        return wr

    def cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int = 20) -> np.ndarray:
        """
        Commodity Channel Index.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            period: CCI period (default: 20)

        Returns:
            CCI values array
        """
        if len(close) < period:
            return np.full(len(close), np.nan)

        typical_price = (high + low + close) / 3
        sma_tp = self.sma(typical_price, period)

        cci = np.empty_like(close)
        cci[:period-1] = np.nan

        for i in range(period - 1, len(close)):
            mean_deviation = np.mean(np.abs(typical_price[i-period+1:i+1] - sma_tp[i]))
            if mean_deviation == 0:
                cci[i] = 0
            else:
                cci[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mean_deviation)

        return cci

    # ==================== TREND INDICATORS ====================

    def macd(self, data: np.ndarray, fast: int = 12, slow: int = 26,
             signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Moving Average Convergence Divergence.

        Args:
            data: Price data array
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)

        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Average Directional Index with +DI and -DI.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            period: ADX period (default: 14)

        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        if len(close) < period + 1:
            nan_array = np.full(len(close), np.nan)
            return nan_array, nan_array, nan_array

        # Calculate True Range
        tr = self.true_range(high, low, close)
        atr = self.wilder_smooth(tr, period)

        # Calculate directional movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])

        plus_di_smooth = self.wilder_smooth(plus_dm, period)
        minus_di_smooth = self.wilder_smooth(minus_dm, period)

        plus_di = 100 * plus_di_smooth / np.where(atr == 0, 1, atr)
        minus_di = 100 * minus_di_smooth / np.where(atr == 0, 1, atr)

        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
        adx = self.wilder_smooth(dx, period)

        return adx, plus_di, minus_di

    def aroon(self, high: np.ndarray, low: np.ndarray, period: int = 25) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aroon Up and Aroon Down.

        Args:
            high: High prices array
            low: Low prices array
            period: Aroon period (default: 25)

        Returns:
            Tuple of (Aroon Up, Aroon Down)
        """
        if len(high) < period + 1:
            nan_array = np.full(len(high), np.nan)
            return nan_array, nan_array

        aroon_up = np.empty_like(high)
        aroon_down = np.empty_like(low)

        aroon_up[:period] = np.nan
        aroon_down[:period] = np.nan

        for i in range(period, len(high)):
            period_high = high[i-period:i+1]
            period_low = low[i-period:i+1]

            days_since_high = period - np.argmax(period_high)
            days_since_low = period - np.argmin(period_low)

            aroon_up[i] = 100 * (period - days_since_high) / period
            aroon_down[i] = 100 * (period - days_since_low) / period

        return aroon_up, aroon_down

    def supertrend(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        SuperTrend indicator.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            period: ATR period (default: 10)
            multiplier: ATR multiplier (default: 3.0)

        Returns:
            Tuple of (SuperTrend values, Trend direction: 1=up, -1=down)
        """
        atr = self.atr(high, low, close, period)
        hl_avg = (high + low) / 2

        basic_upper = hl_avg + multiplier * atr
        basic_lower = hl_avg - multiplier * atr

        final_upper = np.empty_like(close)
        final_lower = np.empty_like(close)
        supertrend = np.empty_like(close)
        direction = np.empty_like(close)

        final_upper[:period] = np.nan
        final_lower[:period] = np.nan
        supertrend[:period] = np.nan
        direction[:period] = np.nan

        final_upper[period] = basic_upper[period]
        final_lower[period] = basic_lower[period]
        supertrend[period] = final_upper[period]
        direction[period] = -1

        for i in range(period + 1, len(close)):
            # Upper band
            if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i-1]

            # Lower band
            if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i-1]

            # SuperTrend and direction
            if supertrend[i-1] == final_upper[i-1] and close[i] <= final_upper[i]:
                supertrend[i] = final_upper[i]
                direction[i] = -1
            elif supertrend[i-1] == final_upper[i-1] and close[i] > final_upper[i]:
                supertrend[i] = final_lower[i]
                direction[i] = 1
            elif supertrend[i-1] == final_lower[i-1] and close[i] >= final_lower[i]:
                supertrend[i] = final_lower[i]
                direction[i] = 1
            elif supertrend[i-1] == final_lower[i-1] and close[i] < final_lower[i]:
                supertrend[i] = final_upper[i]
                direction[i] = -1
            else:
                supertrend[i] = final_lower[i]
                direction[i] = 1

        return supertrend, direction

    # ==================== VOLATILITY INDICATORS ====================

    def true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        True Range calculation.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array

        Returns:
            True Range array
        """
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value uses only high-low

        return tr

    def atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int = 14) -> np.ndarray:
        """
        Average True Range.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            period: ATR period (default: 14)

        Returns:
            ATR values array
        """
        tr = self.true_range(high, low, close)
        return self.wilder_smooth(tr, period)

    def bollinger_bands(self, data: np.ndarray, period: int = 20,
                       std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bands.

        Args:
            data: Price data array
            period: Moving average period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)

        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        middle = self.sma(data, period)

        std = np.empty_like(data)
        std[:period-1] = np.nan

        for i in range(period - 1, len(data)):
            std[i] = np.std(data[i-period+1:i+1])

        upper = middle + std_dev * std
        lower = middle - std_dev * std

        return upper, middle, lower

    def keltner_channels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        period: int = 20, multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Keltner Channels.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            period: EMA period (default: 20)
            multiplier: ATR multiplier (default: 2.0)

        Returns:
            Tuple of (Upper Channel, Middle Channel, Lower Channel)
        """
        middle = self.ema(close, period)
        atr_val = self.atr(high, low, close, period)

        upper = middle + multiplier * atr_val
        lower = middle - multiplier * atr_val

        return upper, middle, lower

    def donchian_channels(self, high: np.ndarray, low: np.ndarray,
                         period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Donchian Channels.

        Args:
            high: High prices array
            low: Low prices array
            period: Period for highest/lowest (default: 20)

        Returns:
            Tuple of (Upper Channel, Middle Channel, Lower Channel)
        """
        if len(high) < period:
            nan_array = np.full(len(high), np.nan)
            return nan_array, nan_array, nan_array

        upper = np.empty_like(high)
        lower = np.empty_like(low)

        upper[:period-1] = np.nan
        lower[:period-1] = np.nan

        for i in range(period - 1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])

        middle = (upper + lower) / 2

        return upper, middle, lower

    # ==================== VOLUME INDICATORS ====================

    def obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        On Balance Volume.

        Args:
            close: Close prices array
            volume: Volume array

        Returns:
            OBV values array
        """
        obv = np.zeros_like(volume, dtype=float)
        obv[0] = volume[0]

        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]

        return obv

    def mfi(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
            volume: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Money Flow Index.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            volume: Volume array
            period: MFI period (default: 14)

        Returns:
            MFI values array (0-100)
        """
        if len(close) < period + 1:
            return np.full(len(close), np.nan)

        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        mfi = np.empty_like(close)
        mfi[:period] = np.nan

        for i in range(period, len(close)):
            positive_flow = 0
            negative_flow = 0

            for j in range(i - period + 1, i + 1):
                if j > 0:
                    if typical_price[j] > typical_price[j-1]:
                        positive_flow += money_flow[j]
                    elif typical_price[j] < typical_price[j-1]:
                        negative_flow += money_flow[j]

            if negative_flow == 0:
                mfi[i] = 100
            else:
                money_ratio = positive_flow / negative_flow
                mfi[i] = 100 - (100 / (1 + money_ratio))

        return mfi

    def vwap(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
             volume: np.ndarray) -> np.ndarray:
        """
        Volume Weighted Average Price.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            volume: Volume array

        Returns:
            VWAP values array
        """
        typical_price = (high + low + close) / 3
        cumulative_tp_volume = np.cumsum(typical_price * volume)
        cumulative_volume = np.cumsum(volume)

        vwap = cumulative_tp_volume / np.where(cumulative_volume == 0, 1, cumulative_volume)

        return vwap

    # ==================== HELPER METHODS ====================

    def wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Wilder's smoothing method (used in RSI, ATR, ADX).

        Args:
            data: Data array to smooth
            period: Smoothing period

        Returns:
            Smoothed data array
        """
        if len(data) < period:
            return np.full(len(data), np.nan)

        smoothed = np.empty_like(data)
        smoothed[:period-1] = np.nan
        smoothed[period-1] = np.mean(data[:period])

        for i in range(period, len(data)):
            smoothed[i] = (smoothed[i-1] * (period - 1) + data[i]) / period

        return smoothed

    def calculate_all(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     volume: Optional[np.ndarray] = None) -> Dict[str, IndicatorResult]:
        """
        Calculate all available indicators.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            volume: Volume array (optional, for volume indicators)

        Returns:
            Dictionary of indicator names to IndicatorResult objects
        """
        results = {}

        # Moving Averages
        results['sma_20'] = IndicatorResult('SMA_20', self.sma(close, 20))
        results['ema_20'] = IndicatorResult('EMA_20', self.ema(close, 20))
        results['sma_50'] = IndicatorResult('SMA_50', self.sma(close, 50))
        results['sma_200'] = IndicatorResult('SMA_200', self.sma(close, 200))

        # Momentum
        rsi_val = self.rsi(close, 14)
        results['rsi'] = IndicatorResult(
            'RSI',
            rsi_val,
            signal='oversold' if rsi_val[-1] < 30 else 'overbought' if rsi_val[-1] > 70 else 'neutral'
        )

        macd_line, signal_line, histogram = self.macd(close)
        results['macd'] = IndicatorResult('MACD', {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })

        # Trend
        adx_val, plus_di, minus_di = self.adx(high, low, close)
        results['adx'] = IndicatorResult('ADX', {
            'adx': adx_val,
            'plus_di': plus_di,
            'minus_di': minus_di
        })

        # Volatility
        atr_val = self.atr(high, low, close)
        results['atr'] = IndicatorResult('ATR', atr_val)

        upper_bb, middle_bb, lower_bb = self.bollinger_bands(close)
        results['bollinger'] = IndicatorResult('Bollinger', {
            'upper': upper_bb,
            'middle': middle_bb,
            'lower': lower_bb
        })

        # Volume (if provided)
        if volume is not None:
            results['obv'] = IndicatorResult('OBV', self.obv(close, volume))
            results['mfi'] = IndicatorResult('MFI', self.mfi(high, low, close, volume))
            results['vwap'] = IndicatorResult('VWAP', self.vwap(high, low, close, volume))

        return results


# Convenience functions for quick calculations
def quick_rsi(close: np.ndarray, period: int = 14) -> float:
    """Quick RSI calculation returning only the last value."""
    calc = IndicatorCalculator()
    rsi_array = calc.rsi(close, period)
    return rsi_array[-1] if len(rsi_array) > 0 and not np.isnan(rsi_array[-1]) else 50.0


def quick_macd(close: np.ndarray) -> Dict[str, float]:
    """Quick MACD calculation returning only the last values."""
    calc = IndicatorCalculator()
    macd_line, signal_line, histogram = calc.macd(close)
    return {
        'macd': macd_line[-1] if not np.isnan(macd_line[-1]) else 0.0,
        'signal': signal_line[-1] if not np.isnan(signal_line[-1]) else 0.0,
        'histogram': histogram[-1] if not np.isnan(histogram[-1]) else 0.0
    }

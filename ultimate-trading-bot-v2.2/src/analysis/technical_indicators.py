"""
Technical Indicators Module for Ultimate Trading Bot v2.2.

This module provides comprehensive technical indicator calculations
including moving averages, oscillators, volatility, and volume indicators.
"""

import logging
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class MAType(str, Enum):
    """Moving average type enumeration."""

    SMA = "sma"
    EMA = "ema"
    WMA = "wma"
    DEMA = "dema"
    TEMA = "tema"
    KAMA = "kama"
    VWMA = "vwma"


class IndicatorResult(BaseModel):
    """Base indicator result model."""

    name: str
    value: float
    signal: Optional[str] = None
    params: dict = Field(default_factory=dict)


class MACDResult(BaseModel):
    """MACD indicator result."""

    macd_line: float
    signal_line: float
    histogram: float
    signal: str = Field(default="neutral")


class BollingerBandsResult(BaseModel):
    """Bollinger Bands result."""

    upper: float
    middle: float
    lower: float
    bandwidth: float
    percent_b: float


class IchimokuResult(BaseModel):
    """Ichimoku Cloud result."""

    tenkan_sen: float
    kijun_sen: float
    senkou_span_a: float
    senkou_span_b: float
    chikou_span: float
    cloud_color: str = Field(default="neutral")


class StochResult(BaseModel):
    """Stochastic result."""

    k: float
    d: float
    signal: str = Field(default="neutral")


class TechnicalIndicators:
    """
    Technical indicators calculator.

    Provides calculations for:
    - Trend indicators (MA, MACD, ADX)
    - Momentum indicators (RSI, Stochastic, CCI)
    - Volatility indicators (Bollinger, ATR)
    - Volume indicators (OBV, MFI, VWAP)
    """

    def __init__(self) -> None:
        """Initialize TechnicalIndicators."""
        logger.info("TechnicalIndicators initialized")

    def sma(
        self,
        data: list[float],
        period: int = 20,
    ) -> list[float]:
        """
        Calculate Simple Moving Average.

        Args:
            data: Price data
            period: SMA period

        Returns:
            List of SMA values
        """
        if len(data) < period:
            return [np.nan] * len(data)

        result = [np.nan] * (period - 1)
        for i in range(period - 1, len(data)):
            window = data[i - period + 1:i + 1]
            result.append(sum(window) / period)

        return result

    def ema(
        self,
        data: list[float],
        period: int = 20,
    ) -> list[float]:
        """
        Calculate Exponential Moving Average.

        Args:
            data: Price data
            period: EMA period

        Returns:
            List of EMA values
        """
        if len(data) < period:
            return [np.nan] * len(data)

        multiplier = 2.0 / (period + 1)
        result = [np.nan] * (period - 1)

        sma_value = sum(data[:period]) / period
        result.append(sma_value)

        for i in range(period, len(data)):
            ema_value = (data[i] - result[-1]) * multiplier + result[-1]
            result.append(ema_value)

        return result

    def wma(
        self,
        data: list[float],
        period: int = 20,
    ) -> list[float]:
        """
        Calculate Weighted Moving Average.

        Args:
            data: Price data
            period: WMA period

        Returns:
            List of WMA values
        """
        if len(data) < period:
            return [np.nan] * len(data)

        weights = list(range(1, period + 1))
        weight_sum = sum(weights)

        result = [np.nan] * (period - 1)
        for i in range(period - 1, len(data)):
            window = data[i - period + 1:i + 1]
            weighted_sum = sum(w * v for w, v in zip(weights, window))
            result.append(weighted_sum / weight_sum)

        return result

    def dema(
        self,
        data: list[float],
        period: int = 20,
    ) -> list[float]:
        """
        Calculate Double Exponential Moving Average.

        Args:
            data: Price data
            period: DEMA period

        Returns:
            List of DEMA values
        """
        ema1 = self.ema(data, period)
        ema1_clean = [v if not np.isnan(v) else 0 for v in ema1]
        ema2 = self.ema(ema1_clean, period)

        result = []
        for e1, e2 in zip(ema1, ema2):
            if np.isnan(e1) or np.isnan(e2):
                result.append(np.nan)
            else:
                result.append(2 * e1 - e2)

        return result

    def tema(
        self,
        data: list[float],
        period: int = 20,
    ) -> list[float]:
        """
        Calculate Triple Exponential Moving Average.

        Args:
            data: Price data
            period: TEMA period

        Returns:
            List of TEMA values
        """
        ema1 = self.ema(data, period)
        ema1_clean = [v if not np.isnan(v) else 0 for v in ema1]
        ema2 = self.ema(ema1_clean, period)
        ema2_clean = [v if not np.isnan(v) else 0 for v in ema2]
        ema3 = self.ema(ema2_clean, period)

        result = []
        for e1, e2, e3 in zip(ema1, ema2, ema3):
            if np.isnan(e1) or np.isnan(e2) or np.isnan(e3):
                result.append(np.nan)
            else:
                result.append(3 * e1 - 3 * e2 + e3)

        return result

    def vwma(
        self,
        prices: list[float],
        volumes: list[float],
        period: int = 20,
    ) -> list[float]:
        """
        Calculate Volume Weighted Moving Average.

        Args:
            prices: Price data
            volumes: Volume data
            period: VWMA period

        Returns:
            List of VWMA values
        """
        if len(prices) < period or len(volumes) < period:
            return [np.nan] * len(prices)

        result = [np.nan] * (period - 1)
        for i in range(period - 1, len(prices)):
            price_window = prices[i - period + 1:i + 1]
            volume_window = volumes[i - period + 1:i + 1]

            pv_sum = sum(p * v for p, v in zip(price_window, volume_window))
            v_sum = sum(volume_window)

            if v_sum > 0:
                result.append(pv_sum / v_sum)
            else:
                result.append(prices[i])

        return result

    def rsi(
        self,
        data: list[float],
        period: int = 14,
    ) -> list[float]:
        """
        Calculate Relative Strength Index.

        Args:
            data: Price data
            period: RSI period

        Returns:
            List of RSI values
        """
        if len(data) < period + 1:
            return [np.nan] * len(data)

        deltas = [data[i] - data[i - 1] for i in range(1, len(data))]

        gains = [max(d, 0) for d in deltas]
        losses = [abs(min(d, 0)) for d in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        result = [np.nan] * period

        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100 - (100 / (1 + rs)))

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                result.append(100.0)
            else:
                rs = avg_gain / avg_loss
                result.append(100 - (100 / (1 + rs)))

        return result

    def macd(
        self,
        data: list[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> list[MACDResult]:
        """
        Calculate MACD indicator.

        Args:
            data: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            List of MACDResult objects
        """
        fast_ema = self.ema(data, fast_period)
        slow_ema = self.ema(data, slow_period)

        macd_line = []
        for f, s in zip(fast_ema, slow_ema):
            if np.isnan(f) or np.isnan(s):
                macd_line.append(0.0)
            else:
                macd_line.append(f - s)

        signal_ema = self.ema(macd_line, signal_period)

        results = []
        for i, (m, s) in enumerate(zip(macd_line, signal_ema)):
            if i < slow_period + signal_period - 2:
                results.append(MACDResult(
                    macd_line=0.0,
                    signal_line=0.0,
                    histogram=0.0,
                ))
            else:
                histogram = m - s if not np.isnan(s) else 0.0

                if histogram > 0 and m > 0:
                    signal = "bullish"
                elif histogram < 0 and m < 0:
                    signal = "bearish"
                else:
                    signal = "neutral"

                results.append(MACDResult(
                    macd_line=m,
                    signal_line=s if not np.isnan(s) else 0.0,
                    histogram=histogram,
                    signal=signal,
                ))

        return results

    def bollinger_bands(
        self,
        data: list[float],
        period: int = 20,
        std_dev: float = 2.0,
    ) -> list[BollingerBandsResult]:
        """
        Calculate Bollinger Bands.

        Args:
            data: Price data
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            List of BollingerBandsResult objects
        """
        sma_values = self.sma(data, period)

        results = []
        for i, middle in enumerate(sma_values):
            if i < period - 1:
                results.append(BollingerBandsResult(
                    upper=0.0,
                    middle=0.0,
                    lower=0.0,
                    bandwidth=0.0,
                    percent_b=0.0,
                ))
            else:
                window = data[i - period + 1:i + 1]
                std = np.std(window)

                upper = middle + (std_dev * std)
                lower = middle - (std_dev * std)
                bandwidth = (upper - lower) / middle if middle != 0 else 0

                if upper != lower:
                    percent_b = (data[i] - lower) / (upper - lower)
                else:
                    percent_b = 0.5

                results.append(BollingerBandsResult(
                    upper=upper,
                    middle=middle,
                    lower=lower,
                    bandwidth=bandwidth,
                    percent_b=percent_b,
                ))

        return results

    def stochastic(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
        k_period: int = 14,
        d_period: int = 3,
    ) -> list[StochResult]:
        """
        Calculate Stochastic Oscillator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period
            d_period: %D period

        Returns:
            List of StochResult objects
        """
        n = len(close)
        k_values: list[float] = []

        for i in range(n):
            if i < k_period - 1:
                k_values.append(np.nan)
            else:
                highest_high = max(high[i - k_period + 1:i + 1])
                lowest_low = min(low[i - k_period + 1:i + 1])

                if highest_high != lowest_low:
                    k = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
                else:
                    k = 50.0

                k_values.append(k)

        d_values = self.sma(
            [v if not np.isnan(v) else 50 for v in k_values],
            d_period,
        )

        results = []
        for k, d in zip(k_values, d_values):
            if np.isnan(k) or np.isnan(d):
                results.append(StochResult(k=50.0, d=50.0))
            else:
                if k < 20:
                    signal = "oversold"
                elif k > 80:
                    signal = "overbought"
                elif k > d:
                    signal = "bullish"
                elif k < d:
                    signal = "bearish"
                else:
                    signal = "neutral"

                results.append(StochResult(k=k, d=d, signal=signal))

        return results

    def atr(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
        period: int = 14,
    ) -> list[float]:
        """
        Calculate Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            List of ATR values
        """
        n = len(close)
        if n < 2:
            return [np.nan] * n

        tr_values: list[float] = [high[0] - low[0]]

        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr_values.append(max(hl, hc, lc))

        result = [np.nan] * (period - 1)
        atr_value = sum(tr_values[:period]) / period
        result.append(atr_value)

        for i in range(period, n):
            atr_value = (atr_value * (period - 1) + tr_values[i]) / period
            result.append(atr_value)

        return result

    def adx(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
        period: int = 14,
    ) -> list[float]:
        """
        Calculate Average Directional Index.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period

        Returns:
            List of ADX values
        """
        n = len(close)
        if n < period * 2:
            return [np.nan] * n

        plus_dm: list[float] = [0.0]
        minus_dm: list[float] = [0.0]
        tr_values: list[float] = [high[0] - low[0]]

        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0.0)

            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0.0)

            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr_values.append(max(hl, hc, lc))

        smoothed_plus_dm = self.ema(plus_dm, period)
        smoothed_minus_dm = self.ema(minus_dm, period)
        smoothed_tr = self.ema(tr_values, period)

        plus_di: list[float] = []
        minus_di: list[float] = []
        dx_values: list[float] = []

        for i in range(n):
            if np.isnan(smoothed_tr[i]) or smoothed_tr[i] == 0:
                plus_di.append(0.0)
                minus_di.append(0.0)
                dx_values.append(0.0)
            else:
                pdi = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
                mdi = 100 * smoothed_minus_dm[i] / smoothed_tr[i]
                plus_di.append(pdi)
                minus_di.append(mdi)

                if pdi + mdi != 0:
                    dx = 100 * abs(pdi - mdi) / (pdi + mdi)
                else:
                    dx = 0.0
                dx_values.append(dx)

        adx_values = self.ema(dx_values, period)

        return adx_values

    def cci(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
        period: int = 20,
    ) -> list[float]:
        """
        Calculate Commodity Channel Index.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: CCI period

        Returns:
            List of CCI values
        """
        n = len(close)
        tp = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]

        tp_sma = self.sma(tp, period)

        result = [np.nan] * (period - 1)

        for i in range(period - 1, n):
            window = tp[i - period + 1:i + 1]
            mean_dev = sum(abs(x - tp_sma[i]) for x in window) / period

            if mean_dev != 0:
                cci = (tp[i] - tp_sma[i]) / (0.015 * mean_dev)
            else:
                cci = 0.0

            result.append(cci)

        return result

    def williams_r(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
        period: int = 14,
    ) -> list[float]:
        """
        Calculate Williams %R.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period

        Returns:
            List of Williams %R values
        """
        n = len(close)
        result = [np.nan] * (period - 1)

        for i in range(period - 1, n):
            highest_high = max(high[i - period + 1:i + 1])
            lowest_low = min(low[i - period + 1:i + 1])

            if highest_high != lowest_low:
                wr = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
            else:
                wr = -50.0

            result.append(wr)

        return result

    def mfi(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
        volume: list[float],
        period: int = 14,
    ) -> list[float]:
        """
        Calculate Money Flow Index.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            period: MFI period

        Returns:
            List of MFI values
        """
        n = len(close)
        if n < period + 1:
            return [np.nan] * n

        tp = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]
        mf = [t * v for t, v in zip(tp, volume)]

        positive_mf: list[float] = [0.0]
        negative_mf: list[float] = [0.0]

        for i in range(1, n):
            if tp[i] > tp[i - 1]:
                positive_mf.append(mf[i])
                negative_mf.append(0.0)
            elif tp[i] < tp[i - 1]:
                positive_mf.append(0.0)
                negative_mf.append(mf[i])
            else:
                positive_mf.append(0.0)
                negative_mf.append(0.0)

        result = [np.nan] * period

        for i in range(period, n):
            pos_sum = sum(positive_mf[i - period + 1:i + 1])
            neg_sum = sum(negative_mf[i - period + 1:i + 1])

            if neg_sum != 0:
                mf_ratio = pos_sum / neg_sum
                mfi_value = 100 - (100 / (1 + mf_ratio))
            else:
                mfi_value = 100.0

            result.append(mfi_value)

        return result

    def obv(
        self,
        close: list[float],
        volume: list[float],
    ) -> list[float]:
        """
        Calculate On Balance Volume.

        Args:
            close: Close prices
            volume: Volume data

        Returns:
            List of OBV values
        """
        n = len(close)
        result = [float(volume[0])]

        for i in range(1, n):
            if close[i] > close[i - 1]:
                result.append(result[-1] + volume[i])
            elif close[i] < close[i - 1]:
                result.append(result[-1] - volume[i])
            else:
                result.append(result[-1])

        return result

    def vwap(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
        volume: list[float],
    ) -> list[float]:
        """
        Calculate Volume Weighted Average Price.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data

        Returns:
            List of VWAP values
        """
        n = len(close)
        result: list[float] = []

        cumulative_tp_vol = 0.0
        cumulative_vol = 0.0

        for i in range(n):
            tp = (high[i] + low[i] + close[i]) / 3
            cumulative_tp_vol += tp * volume[i]
            cumulative_vol += volume[i]

            if cumulative_vol > 0:
                result.append(cumulative_tp_vol / cumulative_vol)
            else:
                result.append(close[i])

        return result

    def ichimoku(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
    ) -> list[IchimokuResult]:
        """
        Calculate Ichimoku Cloud.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            tenkan_period: Tenkan-sen period
            kijun_period: Kijun-sen period
            senkou_b_period: Senkou Span B period

        Returns:
            List of IchimokuResult objects
        """
        n = len(close)

        def period_mid(h: list[float], l: list[float], period: int, idx: int) -> float:
            if idx < period - 1:
                return np.nan
            hh = max(h[idx - period + 1:idx + 1])
            ll = min(l[idx - period + 1:idx + 1])
            return (hh + ll) / 2

        results = []
        for i in range(n):
            tenkan = period_mid(high, low, tenkan_period, i)
            kijun = period_mid(high, low, kijun_period, i)

            if not np.isnan(tenkan) and not np.isnan(kijun):
                senkou_a = (tenkan + kijun) / 2
            else:
                senkou_a = np.nan

            senkou_b = period_mid(high, low, senkou_b_period, i)

            chikou = close[i]

            if not np.isnan(senkou_a) and not np.isnan(senkou_b):
                if senkou_a > senkou_b:
                    cloud_color = "bullish"
                elif senkou_a < senkou_b:
                    cloud_color = "bearish"
                else:
                    cloud_color = "neutral"
            else:
                cloud_color = "neutral"

            results.append(IchimokuResult(
                tenkan_sen=tenkan if not np.isnan(tenkan) else 0.0,
                kijun_sen=kijun if not np.isnan(kijun) else 0.0,
                senkou_span_a=senkou_a if not np.isnan(senkou_a) else 0.0,
                senkou_span_b=senkou_b if not np.isnan(senkou_b) else 0.0,
                chikou_span=chikou,
                cloud_color=cloud_color,
            ))

        return results

    def pivot_points(
        self,
        high: float,
        low: float,
        close: float,
    ) -> dict:
        """
        Calculate pivot points for a period.

        Args:
            high: High price
            low: Low price
            close: Close price

        Returns:
            Dictionary with pivot levels
        """
        pivot = (high + low + close) / 3

        r1 = 2 * pivot - low
        s1 = 2 * pivot - high

        r2 = pivot + (high - low)
        s2 = pivot - (high - low)

        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)

        return {
            "pivot": pivot,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "s1": s1,
            "s2": s2,
            "s3": s3,
        }

    def fibonacci_retracement(
        self,
        high: float,
        low: float,
    ) -> dict:
        """
        Calculate Fibonacci retracement levels.

        Args:
            high: High price
            low: Low price

        Returns:
            Dictionary with Fibonacci levels
        """
        diff = high - low

        return {
            "level_0": low,
            "level_236": low + diff * 0.236,
            "level_382": low + diff * 0.382,
            "level_500": low + diff * 0.5,
            "level_618": low + diff * 0.618,
            "level_786": low + diff * 0.786,
            "level_100": high,
        }

    def keltner_channels(
        self,
        high: list[float],
        low: list[float],
        close: list[float],
        period: int = 20,
        multiplier: float = 2.0,
    ) -> list[dict]:
        """
        Calculate Keltner Channels.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: EMA period
            multiplier: ATR multiplier

        Returns:
            List of channel dictionaries
        """
        middle = self.ema(close, period)
        atr_values = self.atr(high, low, close, period)

        results = []
        for m, a in zip(middle, atr_values):
            if np.isnan(m) or np.isnan(a):
                results.append({
                    "upper": 0.0,
                    "middle": 0.0,
                    "lower": 0.0,
                })
            else:
                results.append({
                    "upper": m + multiplier * a,
                    "middle": m,
                    "lower": m - multiplier * a,
                })

        return results

    def donchian_channels(
        self,
        high: list[float],
        low: list[float],
        period: int = 20,
    ) -> list[dict]:
        """
        Calculate Donchian Channels.

        Args:
            high: High prices
            low: Low prices
            period: Channel period

        Returns:
            List of channel dictionaries
        """
        n = len(high)
        results: list[dict] = []

        for i in range(n):
            if i < period - 1:
                results.append({
                    "upper": high[i],
                    "lower": low[i],
                    "middle": (high[i] + low[i]) / 2,
                })
            else:
                upper = max(high[i - period + 1:i + 1])
                lower = min(low[i - period + 1:i + 1])
                results.append({
                    "upper": upper,
                    "lower": lower,
                    "middle": (upper + lower) / 2,
                })

        return results

    def __repr__(self) -> str:
        """String representation."""
        return "TechnicalIndicators()"

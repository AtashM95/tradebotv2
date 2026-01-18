"""
Volume Analysis Module for Ultimate Trading Bot v2.2.

This module provides comprehensive volume-based analysis
for confirming price movements and detecting trends.
"""

import logging
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from src.analysis.technical_indicators import TechnicalIndicators


logger = logging.getLogger(__name__)


class VolumeSignal(str, Enum):
    """Volume signal enumeration."""

    STRONG_BUYING = "strong_buying"
    BUYING = "buying"
    NEUTRAL = "neutral"
    SELLING = "selling"
    STRONG_SELLING = "strong_selling"


class VolumeProfile(str, Enum):
    """Volume profile type enumeration."""

    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    CLIMAX = "climax"
    DRYING_UP = "drying_up"
    NORMAL = "normal"


class VolumeAnalysisResult(BaseModel):
    """Volume analysis result model."""

    signal: VolumeSignal
    profile: VolumeProfile
    relative_volume: float = Field(default=1.0)
    volume_trend: str = Field(default="neutral")
    price_volume_correlation: float = Field(default=0.0)
    buying_pressure: float = Field(ge=0.0, le=1.0, default=0.5)
    selling_pressure: float = Field(ge=0.0, le=1.0, default=0.5)
    obv_trend: str = Field(default="neutral")
    mfi_value: float = Field(default=50.0)
    vwap: Optional[float] = None
    volume_nodes: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class VolumeZone(BaseModel):
    """Volume zone/node model."""

    price_level: float
    volume: float
    percentage: float = Field(ge=0.0, le=1.0)
    zone_type: str = Field(default="normal")


class VolumeAnalysisConfig(BaseModel):
    """Configuration for volume analysis."""

    short_period: int = Field(default=5, ge=2, le=20)
    medium_period: int = Field(default=20, ge=10, le=50)
    long_period: int = Field(default=50, ge=20, le=200)
    volume_spike_threshold: float = Field(default=2.0, ge=1.5, le=5.0)
    mfi_period: int = Field(default=14, ge=5, le=30)
    volume_profile_bins: int = Field(default=20, ge=10, le=100)


class VolumeAnalyzer:
    """
    Volume analyzer for trading.

    Provides:
    - Volume trend analysis
    - Buying/selling pressure
    - Volume profile analysis
    - On Balance Volume (OBV)
    - Money Flow Index (MFI)
    - VWAP analysis
    - Volume-price correlation
    """

    def __init__(
        self,
        config: Optional[VolumeAnalysisConfig] = None,
    ) -> None:
        """
        Initialize VolumeAnalyzer.

        Args:
            config: Volume analysis configuration
        """
        self._config = config or VolumeAnalysisConfig()
        self._indicators = TechnicalIndicators()

        logger.info("VolumeAnalyzer initialized")

    def analyze(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        opens: Optional[list[float]] = None,
    ) -> VolumeAnalysisResult:
        """
        Perform comprehensive volume analysis.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            volumes: Volume data
            opens: Optional open prices

        Returns:
            VolumeAnalysisResult with analysis
        """
        notes: list[str] = []

        relative_volume = self._calculate_relative_volume(volumes)
        notes.append(f"Relative volume: {relative_volume:.2f}x average")

        volume_trend = self._analyze_volume_trend(volumes)
        notes.append(f"Volume trend: {volume_trend}")

        buying_pressure, selling_pressure = self._calculate_pressure(
            highs, lows, closes, volumes,
        )

        signal = self._determine_signal(
            buying_pressure,
            selling_pressure,
            relative_volume,
        )

        profile = self._determine_profile(
            closes,
            volumes,
            relative_volume,
            volume_trend,
        )

        obv_values = self._indicators.obv(closes, volumes)
        obv_trend = self._analyze_obv_trend(obv_values)
        notes.append(f"OBV trend: {obv_trend}")

        mfi_values = self._indicators.mfi(highs, lows, closes, volumes, self._config.mfi_period)
        mfi_value = mfi_values[-1] if mfi_values else 50.0
        notes.append(f"MFI: {mfi_value:.1f}")

        vwap = self._indicators.vwap(highs, lows, closes, volumes)[-1] if volumes else None

        price_volume_corr = self._calculate_correlation(closes, volumes)

        volume_nodes = self._calculate_volume_profile(
            highs, lows, closes, volumes,
        )

        return VolumeAnalysisResult(
            signal=signal,
            profile=profile,
            relative_volume=relative_volume,
            volume_trend=volume_trend,
            price_volume_correlation=price_volume_corr,
            buying_pressure=buying_pressure,
            selling_pressure=selling_pressure,
            obv_trend=obv_trend,
            mfi_value=mfi_value,
            vwap=vwap,
            volume_nodes=volume_nodes,
            notes=notes,
        )

    def _calculate_relative_volume(self, volumes: list[float]) -> float:
        """Calculate relative volume vs average."""
        if len(volumes) < self._config.medium_period:
            return 1.0

        current_volume = volumes[-1]
        avg_volume = sum(volumes[-self._config.medium_period:-1]) / (self._config.medium_period - 1)

        if avg_volume == 0:
            return 1.0

        return current_volume / avg_volume

    def _analyze_volume_trend(self, volumes: list[float]) -> str:
        """Analyze volume trend."""
        if len(volumes) < self._config.short_period:
            return "insufficient_data"

        short_avg = sum(volumes[-self._config.short_period:]) / self._config.short_period

        if len(volumes) >= self._config.medium_period:
            medium_avg = sum(volumes[-self._config.medium_period:]) / self._config.medium_period
        else:
            medium_avg = short_avg

        if short_avg > medium_avg * 1.2:
            return "increasing"
        elif short_avg < medium_avg * 0.8:
            return "decreasing"
        else:
            return "stable"

    def _calculate_pressure(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
    ) -> tuple[float, float]:
        """Calculate buying and selling pressure."""
        if len(closes) < 2:
            return 0.5, 0.5

        buying_volume = 0.0
        selling_volume = 0.0
        total_volume = 0.0

        for i in range(1, len(closes)):
            price_range = highs[i] - lows[i]
            if price_range == 0:
                continue

            close_location = (closes[i] - lows[i]) / price_range

            buying_volume += volumes[i] * close_location
            selling_volume += volumes[i] * (1 - close_location)
            total_volume += volumes[i]

        if total_volume == 0:
            return 0.5, 0.5

        buying_pressure = buying_volume / total_volume
        selling_pressure = selling_volume / total_volume

        return buying_pressure, selling_pressure

    def _determine_signal(
        self,
        buying_pressure: float,
        selling_pressure: float,
        relative_volume: float,
    ) -> VolumeSignal:
        """Determine volume signal from pressure and relative volume."""
        pressure_diff = buying_pressure - selling_pressure

        if relative_volume > self._config.volume_spike_threshold:
            if pressure_diff > 0.2:
                return VolumeSignal.STRONG_BUYING
            elif pressure_diff < -0.2:
                return VolumeSignal.STRONG_SELLING
            elif pressure_diff > 0:
                return VolumeSignal.BUYING
            else:
                return VolumeSignal.SELLING
        else:
            if pressure_diff > 0.15:
                return VolumeSignal.BUYING
            elif pressure_diff < -0.15:
                return VolumeSignal.SELLING
            else:
                return VolumeSignal.NEUTRAL

    def _determine_profile(
        self,
        closes: list[float],
        volumes: list[float],
        relative_volume: float,
        volume_trend: str,
    ) -> VolumeProfile:
        """Determine volume profile type."""
        if len(closes) < 5:
            return VolumeProfile.NORMAL

        price_change = (closes[-1] - closes[-5]) / closes[-5]

        if relative_volume > self._config.volume_spike_threshold:
            if abs(price_change) > 0.03:
                return VolumeProfile.CLIMAX
            else:
                return VolumeProfile.NORMAL

        if volume_trend == "decreasing" and relative_volume < 0.5:
            return VolumeProfile.DRYING_UP

        if volume_trend == "increasing":
            if price_change > 0.01:
                return VolumeProfile.ACCUMULATION
            elif price_change < -0.01:
                return VolumeProfile.DISTRIBUTION

        return VolumeProfile.NORMAL

    def _analyze_obv_trend(self, obv_values: list[float]) -> str:
        """Analyze OBV trend."""
        if len(obv_values) < 10:
            return "insufficient_data"

        short_obv = obv_values[-5:]
        n = len(short_obv)
        x_mean = (n - 1) / 2
        y_mean = sum(short_obv) / n

        numerator = sum((i - x_mean) * (short_obv[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "flat"

        slope = numerator / denominator

        obv_range = max(abs(v) for v in obv_values) if obv_values else 1
        normalized_slope = slope / obv_range if obv_range != 0 else 0

        if normalized_slope > 0.01:
            return "bullish"
        elif normalized_slope < -0.01:
            return "bearish"
        else:
            return "flat"

    def _calculate_correlation(
        self,
        closes: list[float],
        volumes: list[float],
    ) -> float:
        """Calculate price-volume correlation."""
        if len(closes) < 10 or len(volumes) < 10:
            return 0.0

        price_changes = [
            closes[i] - closes[i - 1]
            for i in range(1, min(len(closes), 20))
        ]
        volume_slice = volumes[1:len(price_changes) + 1]

        if len(price_changes) != len(volume_slice):
            return 0.0

        n = len(price_changes)
        price_mean = sum(price_changes) / n
        vol_mean = sum(volume_slice) / n

        numerator = sum(
            (price_changes[i] - price_mean) * (volume_slice[i] - vol_mean)
            for i in range(n)
        )

        price_std = (sum((p - price_mean) ** 2 for p in price_changes) / n) ** 0.5
        vol_std = (sum((v - vol_mean) ** 2 for v in volume_slice) / n) ** 0.5

        if price_std == 0 or vol_std == 0:
            return 0.0

        correlation = numerator / (n * price_std * vol_std)

        return max(-1.0, min(1.0, correlation))

    def _calculate_volume_profile(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
    ) -> list[dict]:
        """Calculate volume profile (price levels with volume)."""
        if len(closes) < 10:
            return []

        price_min = min(lows)
        price_max = max(highs)
        price_range = price_max - price_min

        if price_range == 0:
            return []

        bins = self._config.volume_profile_bins
        bin_size = price_range / bins

        volume_at_price: dict[int, float] = {i: 0.0 for i in range(bins)}

        for i in range(len(closes)):
            h, l, v = highs[i], lows[i], volumes[i]

            low_bin = int((l - price_min) / bin_size)
            high_bin = int((h - price_min) / bin_size)

            low_bin = max(0, min(bins - 1, low_bin))
            high_bin = max(0, min(bins - 1, high_bin))

            bins_touched = high_bin - low_bin + 1
            volume_per_bin = v / bins_touched if bins_touched > 0 else 0

            for b in range(low_bin, high_bin + 1):
                volume_at_price[b] += volume_per_bin

        total_volume = sum(volume_at_price.values())
        if total_volume == 0:
            return []

        result: list[dict] = []
        max_volume = max(volume_at_price.values())

        for bin_idx, volume in sorted(volume_at_price.items(), key=lambda x: x[1], reverse=True)[:10]:
            price_level = price_min + (bin_idx + 0.5) * bin_size

            if volume > max_volume * 0.8:
                zone_type = "high_volume_node"
            elif volume < max_volume * 0.2:
                zone_type = "low_volume_node"
            else:
                zone_type = "normal"

            result.append({
                "price_level": round(price_level, 2),
                "volume": volume,
                "percentage": volume / total_volume,
                "zone_type": zone_type,
            })

        return result

    def detect_volume_divergence(
        self,
        closes: list[float],
        volumes: list[float],
        lookback: int = 20,
    ) -> Optional[str]:
        """
        Detect volume-price divergence.

        Args:
            closes: Close prices
            volumes: Volume data
            lookback: Lookback period

        Returns:
            Divergence type or None
        """
        if len(closes) < lookback or len(volumes) < lookback:
            return None

        recent_closes = closes[-lookback:]
        recent_volumes = volumes[-lookback:]

        price_trend = recent_closes[-1] - recent_closes[0]
        vol_trend = recent_volumes[-1] - sum(recent_volumes[:5]) / 5

        if price_trend > 0 and vol_trend < 0:
            return "bearish_divergence"
        elif price_trend < 0 and vol_trend > 0:
            return "bullish_divergence"

        return None

    def calculate_accumulation_distribution(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
    ) -> list[float]:
        """
        Calculate Accumulation/Distribution Line.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            volumes: Volume data

        Returns:
            A/D line values
        """
        result: list[float] = []
        ad = 0.0

        for i in range(len(closes)):
            high_low = highs[i] - lows[i]

            if high_low == 0:
                mf_multiplier = 0.0
            else:
                mf_multiplier = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / high_low

            mf_volume = mf_multiplier * volumes[i]
            ad += mf_volume
            result.append(ad)

        return result

    def calculate_chaikin_money_flow(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        period: int = 20,
    ) -> list[float]:
        """
        Calculate Chaikin Money Flow.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            volumes: Volume data
            period: CMF period

        Returns:
            CMF values
        """
        if len(closes) < period:
            return [0.0] * len(closes)

        mf_volumes: list[float] = []
        for i in range(len(closes)):
            high_low = highs[i] - lows[i]

            if high_low == 0:
                mf_multiplier = 0.0
            else:
                mf_multiplier = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / high_low

            mf_volumes.append(mf_multiplier * volumes[i])

        result = [0.0] * (period - 1)

        for i in range(period - 1, len(closes)):
            mf_sum = sum(mf_volumes[i - period + 1:i + 1])
            vol_sum = sum(volumes[i - period + 1:i + 1])

            if vol_sum != 0:
                cmf = mf_sum / vol_sum
            else:
                cmf = 0.0

            result.append(cmf)

        return result

    def analyze_volume_breakout(
        self,
        closes: list[float],
        volumes: list[float],
        threshold_multiplier: float = 2.0,
    ) -> dict:
        """
        Analyze potential volume breakout.

        Args:
            closes: Close prices
            volumes: Volume data
            threshold_multiplier: Volume spike threshold

        Returns:
            Breakout analysis dictionary
        """
        if len(volumes) < 20:
            return {"breakout": False}

        current_volume = volumes[-1]
        avg_volume = sum(volumes[-20:-1]) / 19

        is_spike = current_volume > avg_volume * threshold_multiplier

        price_change = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 else 0

        if is_spike:
            if price_change > 0:
                direction = "bullish"
            elif price_change < 0:
                direction = "bearish"
            else:
                direction = "neutral"

            return {
                "breakout": True,
                "direction": direction,
                "volume_ratio": current_volume / avg_volume,
                "price_change_pct": price_change,
                "confidence": min(1.0, (current_volume / avg_volume - 1) / 3),
            }

        return {"breakout": False}

    def __repr__(self) -> str:
        """String representation."""
        return "VolumeAnalyzer()"

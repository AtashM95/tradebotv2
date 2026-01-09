"""
Market Regime Module for Ultimate Trading Bot v2.2.

This module detects and classifies market regimes for
adaptive strategy selection.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from src.analysis.technical_indicators import TechnicalIndicators
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class RegimeType(str, Enum):
    """Market regime type enumeration."""

    BULL_QUIET = "bull_quiet"
    BULL_VOLATILE = "bull_volatile"
    BEAR_QUIET = "bear_quiet"
    BEAR_VOLATILE = "bear_volatile"
    SIDEWAYS_QUIET = "sideways_quiet"
    SIDEWAYS_VOLATILE = "sideways_volatile"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class VolatilityRegime(str, Enum):
    """Volatility regime enumeration."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class TrendRegime(str, Enum):
    """Trend regime enumeration."""

    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    NEUTRAL = "neutral"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class RegimeResult(BaseModel):
    """Market regime result model."""

    regime: RegimeType
    volatility_regime: VolatilityRegime
    trend_regime: TrendRegime
    confidence: float = Field(ge=0.0, le=1.0)
    vix_level: float = Field(default=20.0)
    trend_strength: float = Field(default=0.0)
    volatility_percentile: float = Field(default=50.0)
    regime_duration_days: int = Field(default=0)
    previous_regime: Optional[RegimeType] = None
    recommended_strategies: list[str] = Field(default_factory=list)
    risk_adjustment: float = Field(default=1.0)
    notes: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=now_utc)


class RegimeTransition(BaseModel):
    """Regime transition model."""

    from_regime: RegimeType
    to_regime: RegimeType
    transition_date: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    trigger: str = Field(default="")


class MarketRegimeConfig(BaseModel):
    """Configuration for market regime detection."""

    vix_low_threshold: float = Field(default=15.0, ge=5.0, le=25.0)
    vix_high_threshold: float = Field(default=25.0, ge=20.0, le=40.0)
    vix_extreme_threshold: float = Field(default=35.0, ge=30.0, le=60.0)
    trend_threshold: float = Field(default=0.02, ge=0.01, le=0.1)
    lookback_period: int = Field(default=20, ge=5, le=100)
    volatility_lookback: int = Field(default=50, ge=20, le=200)
    transition_confirmation_days: int = Field(default=3, ge=1, le=10)


class MarketRegimeDetector:
    """
    Market regime detector.

    Provides:
    - Regime classification (bull/bear/sideways)
    - Volatility regime detection
    - Regime transition tracking
    - Strategy recommendations per regime
    """

    def __init__(
        self,
        config: Optional[MarketRegimeConfig] = None,
    ) -> None:
        """
        Initialize MarketRegimeDetector.

        Args:
            config: Regime detection configuration
        """
        self._config = config or MarketRegimeConfig()
        self._indicators = TechnicalIndicators()

        self._regime_history: list[RegimeResult] = []
        self._transitions: list[RegimeTransition] = []
        self._current_regime: Optional[RegimeType] = None
        self._regime_start_date: Optional[datetime] = None

        logger.info("MarketRegimeDetector initialized")

    def detect_regime(
        self,
        closes: list[float],
        highs: list[float],
        lows: list[float],
        vix: Optional[float] = None,
        volumes: Optional[list[float]] = None,
    ) -> RegimeResult:
        """
        Detect current market regime.

        Args:
            closes: Close prices
            highs: High prices
            lows: Low prices
            vix: VIX level
            volumes: Optional volume data

        Returns:
            RegimeResult with regime classification
        """
        notes: list[str] = []

        volatility_regime = self._detect_volatility_regime(
            highs, lows, closes, vix,
        )
        notes.append(f"Volatility: {volatility_regime.value}")

        trend_regime, trend_strength = self._detect_trend_regime(closes)
        notes.append(f"Trend: {trend_regime.value} (strength: {trend_strength:.2f})")

        regime = self._classify_regime(
            volatility_regime, trend_regime, vix,
        )

        confidence = self._calculate_confidence(
            volatility_regime, trend_regime, closes,
        )

        volatility_pct = self._calculate_volatility_percentile(
            highs, lows, closes,
        )

        regime_duration = self._calculate_regime_duration(regime)

        previous_regime = self._current_regime

        if regime != self._current_regime:
            if self._current_regime is not None:
                self._transitions.append(RegimeTransition(
                    from_regime=self._current_regime,
                    to_regime=regime,
                    transition_date=now_utc(),
                    confidence=confidence,
                    trigger=f"Changed from {self._current_regime.value} to {regime.value}",
                ))
            self._current_regime = regime
            self._regime_start_date = now_utc()

        recommended = self._get_recommended_strategies(regime)
        risk_adj = self._calculate_risk_adjustment(regime, volatility_regime)

        result = RegimeResult(
            regime=regime,
            volatility_regime=volatility_regime,
            trend_regime=trend_regime,
            confidence=confidence,
            vix_level=vix or 20.0,
            trend_strength=trend_strength,
            volatility_percentile=volatility_pct,
            regime_duration_days=regime_duration,
            previous_regime=previous_regime,
            recommended_strategies=recommended,
            risk_adjustment=risk_adj,
            notes=notes,
        )

        self._regime_history.append(result)
        if len(self._regime_history) > 100:
            self._regime_history = self._regime_history[-100:]

        return result

    def _detect_volatility_regime(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        vix: Optional[float],
    ) -> VolatilityRegime:
        """Detect volatility regime."""
        if vix is not None:
            if vix >= self._config.vix_extreme_threshold:
                return VolatilityRegime.EXTREME
            elif vix >= self._config.vix_high_threshold:
                return VolatilityRegime.HIGH
            elif vix <= self._config.vix_low_threshold:
                return VolatilityRegime.LOW
            else:
                return VolatilityRegime.NORMAL

        if len(closes) < 20:
            return VolatilityRegime.NORMAL

        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
        ]

        recent_returns = returns[-20:]
        std_dev = np.std(recent_returns)
        annualized_vol = std_dev * np.sqrt(252) * 100

        if annualized_vol > 40:
            return VolatilityRegime.EXTREME
        elif annualized_vol > 25:
            return VolatilityRegime.HIGH
        elif annualized_vol < 12:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.NORMAL

    def _detect_trend_regime(
        self,
        closes: list[float],
    ) -> tuple[TrendRegime, float]:
        """Detect trend regime and strength."""
        if len(closes) < 50:
            return TrendRegime.NEUTRAL, 0.0

        sma_20 = sum(closes[-20:]) / 20
        sma_50 = sum(closes[-50:]) / 50
        current_price = closes[-1]

        price_vs_sma20 = (current_price - sma_20) / sma_20
        price_vs_sma50 = (current_price - sma_50) / sma_50
        sma20_vs_sma50 = (sma_20 - sma_50) / sma_50

        n = min(20, len(closes))
        recent = closes[-n:]
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0
        normalized_slope = slope / y_mean if y_mean != 0 else 0

        trend_score = (
            price_vs_sma20 * 0.3 +
            price_vs_sma50 * 0.3 +
            sma20_vs_sma50 * 0.2 +
            normalized_slope * 10 * 0.2
        )

        trend_strength = min(1.0, abs(trend_score) * 10)

        if trend_score > 0.05:
            if trend_strength > 0.7:
                return TrendRegime.STRONG_UPTREND, trend_strength
            return TrendRegime.UPTREND, trend_strength
        elif trend_score < -0.05:
            if trend_strength > 0.7:
                return TrendRegime.STRONG_DOWNTREND, trend_strength
            return TrendRegime.DOWNTREND, trend_strength
        else:
            return TrendRegime.NEUTRAL, trend_strength

    def _classify_regime(
        self,
        volatility: VolatilityRegime,
        trend: TrendRegime,
        vix: Optional[float],
    ) -> RegimeType:
        """Classify market regime from volatility and trend."""
        if volatility == VolatilityRegime.EXTREME:
            if trend in [TrendRegime.STRONG_DOWNTREND, TrendRegime.DOWNTREND]:
                return RegimeType.CRISIS
            else:
                return RegimeType.RECOVERY

        is_volatile = volatility in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]
        is_quiet = volatility == VolatilityRegime.LOW

        if trend in [TrendRegime.STRONG_UPTREND, TrendRegime.UPTREND]:
            if is_volatile:
                return RegimeType.BULL_VOLATILE
            else:
                return RegimeType.BULL_QUIET

        elif trend in [TrendRegime.STRONG_DOWNTREND, TrendRegime.DOWNTREND]:
            if is_volatile:
                return RegimeType.BEAR_VOLATILE
            else:
                return RegimeType.BEAR_QUIET

        else:
            if is_volatile:
                return RegimeType.SIDEWAYS_VOLATILE
            else:
                return RegimeType.SIDEWAYS_QUIET

    def _calculate_confidence(
        self,
        volatility: VolatilityRegime,
        trend: TrendRegime,
        closes: list[float],
    ) -> float:
        """Calculate confidence in regime classification."""
        confidence = 0.5

        if volatility in [VolatilityRegime.LOW, VolatilityRegime.EXTREME]:
            confidence += 0.15
        elif volatility == VolatilityRegime.NORMAL:
            confidence += 0.1

        if trend in [TrendRegime.STRONG_UPTREND, TrendRegime.STRONG_DOWNTREND]:
            confidence += 0.2
        elif trend in [TrendRegime.UPTREND, TrendRegime.DOWNTREND]:
            confidence += 0.1

        if len(closes) >= 50:
            sma_20 = sum(closes[-20:]) / 20
            sma_50 = sum(closes[-50:]) / 50
            current = closes[-1]

            if (current > sma_20 > sma_50) or (current < sma_20 < sma_50):
                confidence += 0.1

        return min(1.0, confidence)

    def _calculate_volatility_percentile(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> float:
        """Calculate current volatility percentile."""
        lookback = min(self._config.volatility_lookback, len(closes) - 1)

        if lookback < 10:
            return 50.0

        atr_values: list[float] = []
        for i in range(1, lookback + 1):
            idx = -lookback - 1 + i
            hl = highs[idx] - lows[idx]
            hc = abs(highs[idx] - closes[idx - 1])
            lc = abs(lows[idx] - closes[idx - 1])
            atr_values.append(max(hl, hc, lc))

        current_atr = atr_values[-1]
        sorted_atr = sorted(atr_values)

        rank = sorted_atr.index(current_atr) + 1
        percentile = rank / len(sorted_atr) * 100

        return percentile

    def _calculate_regime_duration(self, regime: RegimeType) -> int:
        """Calculate how long current regime has lasted."""
        if self._regime_start_date is None:
            return 0

        duration = now_utc() - self._regime_start_date
        return duration.days

    def _get_recommended_strategies(
        self,
        regime: RegimeType,
    ) -> list[str]:
        """Get recommended strategies for regime."""
        recommendations = {
            RegimeType.BULL_QUIET: [
                "trend_following",
                "momentum",
                "buy_dips",
                "covered_calls",
            ],
            RegimeType.BULL_VOLATILE: [
                "momentum",
                "breakout",
                "reduce_position_size",
            ],
            RegimeType.BEAR_QUIET: [
                "short_selling",
                "inverse_etfs",
                "defensive_stocks",
            ],
            RegimeType.BEAR_VOLATILE: [
                "cash_preservation",
                "hedging",
                "very_small_positions",
            ],
            RegimeType.SIDEWAYS_QUIET: [
                "mean_reversion",
                "range_trading",
                "iron_condors",
            ],
            RegimeType.SIDEWAYS_VOLATILE: [
                "straddles",
                "quick_scalping",
                "reduce_exposure",
            ],
            RegimeType.CRISIS: [
                "cash_only",
                "hedging",
                "safe_havens",
            ],
            RegimeType.RECOVERY: [
                "accumulation",
                "quality_stocks",
                "gradual_entry",
            ],
        }

        return recommendations.get(regime, ["balanced"])

    def _calculate_risk_adjustment(
        self,
        regime: RegimeType,
        volatility: VolatilityRegime,
    ) -> float:
        """Calculate risk adjustment factor for position sizing."""
        base_adjustments = {
            RegimeType.BULL_QUIET: 1.2,
            RegimeType.BULL_VOLATILE: 0.8,
            RegimeType.BEAR_QUIET: 0.7,
            RegimeType.BEAR_VOLATILE: 0.4,
            RegimeType.SIDEWAYS_QUIET: 1.0,
            RegimeType.SIDEWAYS_VOLATILE: 0.6,
            RegimeType.CRISIS: 0.2,
            RegimeType.RECOVERY: 0.5,
        }

        volatility_adjustments = {
            VolatilityRegime.LOW: 1.1,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: 0.7,
            VolatilityRegime.EXTREME: 0.4,
        }

        base = base_adjustments.get(regime, 1.0)
        vol_adj = volatility_adjustments.get(volatility, 1.0)

        return base * vol_adj

    def get_regime_statistics(self) -> dict:
        """Get regime detection statistics."""
        if not self._regime_history:
            return {}

        regime_counts: dict[str, int] = {}
        for result in self._regime_history:
            regime = result.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        total = len(self._regime_history)

        return {
            "total_observations": total,
            "current_regime": self._current_regime.value if self._current_regime else None,
            "regime_distribution": {
                k: v / total * 100 for k, v in regime_counts.items()
            },
            "total_transitions": len(self._transitions),
            "regime_duration_days": self._calculate_regime_duration(
                self._current_regime
            ) if self._current_regime else 0,
        }

    def get_transition_history(
        self,
        limit: int = 10,
    ) -> list[RegimeTransition]:
        """Get recent regime transitions."""
        return self._transitions[-limit:]

    def __repr__(self) -> str:
        """String representation."""
        return f"MarketRegimeDetector(current={self._current_regime})"

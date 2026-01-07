"""
AI Strategy Advisor Module for Ultimate Trading Bot v2.2.

This module provides AI-powered strategy recommendations
and optimization suggestions for trading.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.ai.openai_client import (
    OpenAIClient,
    OpenAIModel,
)
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime enumeration."""

    BULLISH_TRENDING = "bullish_trending"
    BEARISH_TRENDING = "bearish_trending"
    SIDEWAYS_RANGE = "sideways_range"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNCERTAIN = "uncertain"


class StrategyType(str, Enum):
    """Strategy type enumeration."""

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"
    POSITION = "position"
    ARBITRAGE = "arbitrage"


class TimeHorizon(str, Enum):
    """Trading time horizon."""

    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"
    LONG_TERM = "long_term"


class StrategyRecommendation(BaseModel):
    """Strategy recommendation model."""

    recommendation_id: str = Field(default_factory=generate_uuid)
    strategy_type: StrategyType
    confidence: float = Field(ge=0.0, le=1.0)
    time_horizon: TimeHorizon
    entry_criteria: list[str] = Field(default_factory=list)
    exit_criteria: list[str] = Field(default_factory=list)
    risk_parameters: dict = Field(default_factory=dict)
    expected_win_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_profit_factor: float = Field(default=0.0, ge=0.0)
    reasoning: str = Field(default="")
    warnings: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=now_utc)


class MarketAnalysis(BaseModel):
    """Market analysis for strategy selection."""

    analysis_id: str = Field(default_factory=generate_uuid)
    regime: MarketRegime
    regime_confidence: float = Field(ge=0.0, le=1.0)
    trend_strength: float = Field(ge=0.0, le=1.0)
    volatility_level: str = Field(default="normal")
    momentum_direction: str = Field(default="neutral")
    key_levels: dict = Field(default_factory=dict)
    sector_rotation: dict = Field(default_factory=dict)
    market_breadth: dict = Field(default_factory=dict)
    summary: str = Field(default="")
    timestamp: datetime = Field(default_factory=now_utc)


class StrategyOptimization(BaseModel):
    """Strategy optimization suggestion."""

    optimization_id: str = Field(default_factory=generate_uuid)
    strategy_name: str
    current_performance: dict = Field(default_factory=dict)
    suggested_changes: list[dict] = Field(default_factory=list)
    expected_improvement: dict = Field(default_factory=dict)
    backtesting_needed: bool = Field(default=True)
    reasoning: str = Field(default="")
    timestamp: datetime = Field(default_factory=now_utc)


class SymbolRecommendation(BaseModel):
    """Symbol recommendation for a strategy."""

    symbol: str
    score: float = Field(ge=0.0, le=1.0)
    setup_quality: str = Field(default="")
    entry_zone: tuple[float, float] = Field(default=(0.0, 0.0))
    target_zone: tuple[float, float] = Field(default=(0.0, 0.0))
    stop_zone: tuple[float, float] = Field(default=(0.0, 0.0))
    reasoning: str = Field(default="")


class AIStrategyAdvisorConfig(BaseModel):
    """Configuration for AI strategy advisor."""

    enable_ai_analysis: bool = Field(default=True)
    min_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_recommendations: int = Field(default=5, ge=1, le=20)
    include_reasoning: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=600, ge=60, le=3600)
    model: OpenAIModel = Field(default=OpenAIModel.GPT_4O)


class AIStrategyAdvisor:
    """
    AI-powered strategy advisor.

    Provides:
    - Market regime detection
    - Strategy recommendations
    - Strategy optimization suggestions
    - Symbol screening for strategies
    """

    def __init__(
        self,
        config: Optional[AIStrategyAdvisorConfig] = None,
        openai_client: Optional[OpenAIClient] = None,
    ) -> None:
        """
        Initialize AIStrategyAdvisor.

        Args:
            config: Advisor configuration
            openai_client: OpenAI client instance
        """
        self._config = config or AIStrategyAdvisorConfig()
        self._client = openai_client

        self._analysis_cache: dict[str, tuple[Any, datetime]] = {}
        self._total_recommendations = 0

        logger.info("AIStrategyAdvisor initialized")

    def set_client(self, client: OpenAIClient) -> None:
        """Set the OpenAI client."""
        self._client = client

    async def analyze_market(
        self,
        indices: dict[str, float],
        sectors: dict[str, float],
        vix: float,
        market_breadth: Optional[dict] = None,
        economic_data: Optional[dict] = None,
    ) -> MarketAnalysis:
        """
        Analyze current market conditions.

        Args:
            indices: Index changes (e.g., {"SPY": 0.5, "QQQ": 0.8})
            sectors: Sector performance
            vix: VIX level
            market_breadth: Advance/decline data
            economic_data: Economic indicators

        Returns:
            MarketAnalysis with regime and conditions
        """
        regime = self._detect_regime(indices, vix, market_breadth or {})
        regime_confidence = self._calculate_regime_confidence(indices, vix)

        trend_strength = self._calculate_trend_strength(indices)

        if vix > 30:
            volatility_level = "high"
        elif vix > 20:
            volatility_level = "elevated"
        elif vix > 15:
            volatility_level = "normal"
        else:
            volatility_level = "low"

        avg_index_change = sum(indices.values()) / len(indices) if indices else 0
        if avg_index_change > 0.5:
            momentum_direction = "bullish"
        elif avg_index_change < -0.5:
            momentum_direction = "bearish"
        else:
            momentum_direction = "neutral"

        sector_rotation = self._analyze_sector_rotation(sectors)

        summary = await self._generate_market_summary(
            regime=regime,
            indices=indices,
            vix=vix,
            sectors=sectors,
        )

        return MarketAnalysis(
            regime=regime,
            regime_confidence=regime_confidence,
            trend_strength=trend_strength,
            volatility_level=volatility_level,
            momentum_direction=momentum_direction,
            sector_rotation=sector_rotation,
            market_breadth=market_breadth or {},
            summary=summary,
        )

    async def recommend_strategies(
        self,
        market_analysis: MarketAnalysis,
        account_size: float,
        risk_tolerance: str = "moderate",
        preferred_timeframes: Optional[list[TimeHorizon]] = None,
    ) -> list[StrategyRecommendation]:
        """
        Recommend trading strategies based on market conditions.

        Args:
            market_analysis: Current market analysis
            account_size: Trading account size
            risk_tolerance: Risk tolerance level
            preferred_timeframes: Preferred trading timeframes

        Returns:
            List of strategy recommendations
        """
        self._total_recommendations += 1
        recommendations: list[StrategyRecommendation] = []

        regime = market_analysis.regime
        volatility = market_analysis.volatility_level

        strategy_scores = self._score_strategies(
            regime=regime,
            volatility=volatility,
            trend_strength=market_analysis.trend_strength,
            risk_tolerance=risk_tolerance,
        )

        sorted_strategies = sorted(
            strategy_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for strategy_type, score in sorted_strategies[:self._config.max_recommendations]:
            if score < self._config.min_confidence_threshold:
                continue

            time_horizon = self._get_strategy_timeframe(strategy_type)
            if preferred_timeframes and time_horizon not in preferred_timeframes:
                continue

            rec = await self._create_recommendation(
                strategy_type=strategy_type,
                confidence=score,
                time_horizon=time_horizon,
                regime=regime,
                account_size=account_size,
                risk_tolerance=risk_tolerance,
            )
            recommendations.append(rec)

        return recommendations

    async def optimize_strategy(
        self,
        strategy_name: str,
        current_params: dict,
        performance_metrics: dict,
        market_conditions: MarketAnalysis,
    ) -> StrategyOptimization:
        """
        Suggest optimizations for a strategy.

        Args:
            strategy_name: Name of the strategy
            current_params: Current strategy parameters
            performance_metrics: Recent performance data
            market_conditions: Current market analysis

        Returns:
            StrategyOptimization with suggestions
        """
        suggested_changes: list[dict] = []

        win_rate = performance_metrics.get("win_rate", 0.5)
        profit_factor = performance_metrics.get("profit_factor", 1.0)
        max_drawdown = performance_metrics.get("max_drawdown", 0.0)
        avg_win = performance_metrics.get("avg_win", 0.0)
        avg_loss = performance_metrics.get("avg_loss", 0.0)

        if win_rate < 0.4:
            suggested_changes.append({
                "parameter": "entry_criteria",
                "suggestion": "Tighten entry criteria to improve win rate",
                "priority": "high",
            })

        if profit_factor < 1.5 and win_rate > 0.5:
            suggested_changes.append({
                "parameter": "take_profit",
                "suggestion": "Consider wider take profit targets",
                "priority": "medium",
            })

        if max_drawdown > 0.15:
            suggested_changes.append({
                "parameter": "stop_loss",
                "suggestion": "Implement tighter stop losses",
                "priority": "high",
            })
            suggested_changes.append({
                "parameter": "position_size",
                "suggestion": "Reduce position sizing",
                "priority": "high",
            })

        if avg_loss > avg_win and win_rate < 0.6:
            suggested_changes.append({
                "parameter": "risk_reward",
                "suggestion": "Improve risk/reward ratio",
                "priority": "high",
            })

        regime = market_conditions.regime
        if regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.UNCERTAIN]:
            suggested_changes.append({
                "parameter": "position_size",
                "suggestion": "Reduce size in current market conditions",
                "priority": "medium",
            })

        reasoning = await self._generate_optimization_reasoning(
            strategy_name=strategy_name,
            metrics=performance_metrics,
            changes=suggested_changes,
        )

        return StrategyOptimization(
            strategy_name=strategy_name,
            current_performance=performance_metrics,
            suggested_changes=suggested_changes,
            expected_improvement={
                "win_rate": "+5-10%" if win_rate < 0.5 else "maintain",
                "profit_factor": "+0.2-0.5" if profit_factor < 2.0 else "maintain",
                "drawdown": "-20-30%" if max_drawdown > 0.1 else "maintain",
            },
            backtesting_needed=len(suggested_changes) > 0,
            reasoning=reasoning,
        )

    async def screen_symbols(
        self,
        strategy_type: StrategyType,
        symbols: list[str],
        symbol_data: dict[str, dict],
    ) -> list[SymbolRecommendation]:
        """
        Screen symbols for a specific strategy.

        Args:
            strategy_type: Type of strategy
            symbols: List of symbols to screen
            symbol_data: Market data for each symbol

        Returns:
            List of symbol recommendations
        """
        recommendations: list[SymbolRecommendation] = []

        for symbol in symbols:
            data = symbol_data.get(symbol, {})
            if not data:
                continue

            score = self._score_symbol_for_strategy(
                strategy_type=strategy_type,
                data=data,
            )

            if score < self._config.min_confidence_threshold:
                continue

            setup_quality = self._assess_setup_quality(score)
            price = data.get("price", 0.0)

            entry_zone = self._calculate_entry_zone(
                strategy_type=strategy_type,
                price=price,
                data=data,
            )

            target_zone = self._calculate_target_zone(
                strategy_type=strategy_type,
                price=price,
                data=data,
            )

            stop_zone = self._calculate_stop_zone(
                strategy_type=strategy_type,
                price=price,
                data=data,
            )

            rec = SymbolRecommendation(
                symbol=symbol,
                score=score,
                setup_quality=setup_quality,
                entry_zone=entry_zone,
                target_zone=target_zone,
                stop_zone=stop_zone,
            )
            recommendations.append(rec)

        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:self._config.max_recommendations]

    def _detect_regime(
        self,
        indices: dict[str, float],
        vix: float,
        breadth: dict,
    ) -> MarketRegime:
        """Detect current market regime."""
        avg_change = sum(indices.values()) / len(indices) if indices else 0

        if vix > 30:
            return MarketRegime.HIGH_VOLATILITY
        elif vix < 12:
            return MarketRegime.LOW_VOLATILITY

        advance_decline = breadth.get("advance_decline_ratio", 1.0)

        if avg_change > 0.3 and advance_decline > 1.5:
            return MarketRegime.BULLISH_TRENDING
        elif avg_change < -0.3 and advance_decline < 0.7:
            return MarketRegime.BEARISH_TRENDING
        elif abs(avg_change) < 0.2:
            return MarketRegime.SIDEWAYS_RANGE

        return MarketRegime.UNCERTAIN

    def _calculate_regime_confidence(
        self,
        indices: dict[str, float],
        vix: float,
    ) -> float:
        """Calculate confidence in regime detection."""
        changes = list(indices.values())
        if not changes:
            return 0.5

        avg = sum(changes) / len(changes)
        variance = sum((c - avg) ** 2 for c in changes) / len(changes)

        alignment = 1.0 - min(variance / 4.0, 1.0)

        if vix > 25 or vix < 15:
            vix_clarity = 0.8
        else:
            vix_clarity = 0.6

        return (alignment * 0.6 + vix_clarity * 0.4)

    def _calculate_trend_strength(self, indices: dict[str, float]) -> float:
        """Calculate overall trend strength."""
        if not indices:
            return 0.0

        avg_change = sum(indices.values()) / len(indices)
        same_direction = sum(
            1 for c in indices.values()
            if (c > 0) == (avg_change > 0)
        )

        alignment = same_direction / len(indices)
        magnitude = min(abs(avg_change) / 2.0, 1.0)

        return alignment * 0.5 + magnitude * 0.5

    def _analyze_sector_rotation(
        self,
        sectors: dict[str, float],
    ) -> dict:
        """Analyze sector rotation patterns."""
        if not sectors:
            return {}

        sorted_sectors = sorted(
            sectors.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        leaders = [s[0] for s in sorted_sectors[:3]]
        laggards = [s[0] for s in sorted_sectors[-3:]]

        defensive = ["XLU", "XLP", "XLRE", "XLV"]
        cyclical = ["XLY", "XLI", "XLB", "XLF"]

        defensive_avg = sum(
            sectors.get(s, 0) for s in defensive
        ) / len(defensive)
        cyclical_avg = sum(
            sectors.get(s, 0) for s in cyclical
        ) / len(cyclical)

        if cyclical_avg > defensive_avg + 0.5:
            rotation = "risk_on"
        elif defensive_avg > cyclical_avg + 0.5:
            rotation = "risk_off"
        else:
            rotation = "neutral"

        return {
            "leaders": leaders,
            "laggards": laggards,
            "rotation": rotation,
            "defensive_avg": defensive_avg,
            "cyclical_avg": cyclical_avg,
        }

    def _score_strategies(
        self,
        regime: MarketRegime,
        volatility: str,
        trend_strength: float,
        risk_tolerance: str,
    ) -> dict[StrategyType, float]:
        """Score strategies based on market conditions."""
        scores: dict[StrategyType, float] = {}

        regime_scores = {
            MarketRegime.BULLISH_TRENDING: {
                StrategyType.TREND_FOLLOWING: 0.9,
                StrategyType.MOMENTUM: 0.85,
                StrategyType.BREAKOUT: 0.7,
                StrategyType.SWING: 0.65,
                StrategyType.MEAN_REVERSION: 0.3,
            },
            MarketRegime.BEARISH_TRENDING: {
                StrategyType.TREND_FOLLOWING: 0.8,
                StrategyType.MOMENTUM: 0.75,
                StrategyType.MEAN_REVERSION: 0.4,
                StrategyType.SWING: 0.5,
            },
            MarketRegime.SIDEWAYS_RANGE: {
                StrategyType.MEAN_REVERSION: 0.85,
                StrategyType.SCALPING: 0.7,
                StrategyType.SWING: 0.6,
                StrategyType.TREND_FOLLOWING: 0.3,
            },
            MarketRegime.HIGH_VOLATILITY: {
                StrategyType.SCALPING: 0.6,
                StrategyType.MEAN_REVERSION: 0.5,
                StrategyType.TREND_FOLLOWING: 0.4,
            },
            MarketRegime.LOW_VOLATILITY: {
                StrategyType.BREAKOUT: 0.7,
                StrategyType.POSITION: 0.65,
                StrategyType.SWING: 0.6,
            },
        }

        base_scores = regime_scores.get(regime, {})
        for strategy in StrategyType:
            base = base_scores.get(strategy, 0.5)

            if strategy == StrategyType.TREND_FOLLOWING:
                base += trend_strength * 0.2

            if volatility == "high":
                if strategy in [StrategyType.SCALPING, StrategyType.BREAKOUT]:
                    base -= 0.15
            elif volatility == "low":
                if strategy == StrategyType.SCALPING:
                    base -= 0.2

            if risk_tolerance == "conservative":
                if strategy in [StrategyType.SCALPING, StrategyType.BREAKOUT]:
                    base -= 0.1
                if strategy == StrategyType.POSITION:
                    base += 0.1
            elif risk_tolerance == "aggressive":
                if strategy == StrategyType.MOMENTUM:
                    base += 0.1

            scores[strategy] = max(0.0, min(1.0, base))

        return scores

    def _get_strategy_timeframe(
        self,
        strategy_type: StrategyType,
    ) -> TimeHorizon:
        """Get typical timeframe for strategy."""
        timeframes = {
            StrategyType.SCALPING: TimeHorizon.INTRADAY,
            StrategyType.TREND_FOLLOWING: TimeHorizon.SWING,
            StrategyType.MOMENTUM: TimeHorizon.SWING,
            StrategyType.MEAN_REVERSION: TimeHorizon.SWING,
            StrategyType.BREAKOUT: TimeHorizon.SWING,
            StrategyType.SWING: TimeHorizon.SWING,
            StrategyType.POSITION: TimeHorizon.POSITION,
        }
        return timeframes.get(strategy_type, TimeHorizon.SWING)

    async def _create_recommendation(
        self,
        strategy_type: StrategyType,
        confidence: float,
        time_horizon: TimeHorizon,
        regime: MarketRegime,
        account_size: float,
        risk_tolerance: str,
    ) -> StrategyRecommendation:
        """Create a strategy recommendation."""
        entry_criteria = self._get_entry_criteria(strategy_type)
        exit_criteria = self._get_exit_criteria(strategy_type)
        risk_params = self._get_risk_parameters(
            strategy_type=strategy_type,
            account_size=account_size,
            risk_tolerance=risk_tolerance,
        )

        win_rate = self._estimate_win_rate(strategy_type, confidence)
        profit_factor = self._estimate_profit_factor(strategy_type, confidence)

        warnings = self._get_strategy_warnings(
            strategy_type=strategy_type,
            regime=regime,
            risk_tolerance=risk_tolerance,
        )

        reasoning = ""
        if self._config.include_reasoning and self._client:
            reasoning = await self._generate_strategy_reasoning(
                strategy_type=strategy_type,
                regime=regime,
                confidence=confidence,
            )

        return StrategyRecommendation(
            strategy_type=strategy_type,
            confidence=confidence,
            time_horizon=time_horizon,
            entry_criteria=entry_criteria,
            exit_criteria=exit_criteria,
            risk_parameters=risk_params,
            expected_win_rate=win_rate,
            expected_profit_factor=profit_factor,
            reasoning=reasoning,
            warnings=warnings,
        )

    def _get_entry_criteria(self, strategy_type: StrategyType) -> list[str]:
        """Get entry criteria for strategy."""
        criteria = {
            StrategyType.TREND_FOLLOWING: [
                "Price above 20 and 50 EMA",
                "ADX > 25 indicating strong trend",
                "Higher highs and higher lows pattern",
                "Volume confirmation on breakouts",
            ],
            StrategyType.MOMENTUM: [
                "RSI between 50-70 (bullish) or 30-50 (bearish)",
                "MACD crossover signal",
                "Price momentum positive over 5-10 days",
                "Relative strength vs market positive",
            ],
            StrategyType.MEAN_REVERSION: [
                "RSI below 30 (oversold) or above 70 (overbought)",
                "Price at Bollinger Band extremes",
                "Support/resistance level test",
                "Volume spike indicating potential reversal",
            ],
            StrategyType.BREAKOUT: [
                "Price consolidation period (lower volatility)",
                "Volume building near resistance",
                "Breakout above key resistance",
                "Volume surge on breakout",
            ],
            StrategyType.SCALPING: [
                "Tight bid-ask spread",
                "High intraday volume",
                "Clear short-term levels",
                "Low market impact",
            ],
            StrategyType.SWING: [
                "Clear support/resistance levels",
                "Trend aligned with higher timeframe",
                "Momentum indicator confirmation",
                "Risk/reward >= 2:1",
            ],
            StrategyType.POSITION: [
                "Strong fundamental catalyst",
                "Long-term trend aligned",
                "Major support level entry",
                "Sector rotation favorable",
            ],
        }
        return criteria.get(strategy_type, ["Price action confirmation"])

    def _get_exit_criteria(self, strategy_type: StrategyType) -> list[str]:
        """Get exit criteria for strategy."""
        criteria = {
            StrategyType.TREND_FOLLOWING: [
                "Price closes below 20 EMA",
                "ADX drops below 20",
                "Lower low formed",
                "Trailing stop hit",
            ],
            StrategyType.MOMENTUM: [
                "RSI reaches extreme (>80 or <20)",
                "MACD divergence forms",
                "Momentum slowing significantly",
                "Target reached",
            ],
            StrategyType.MEAN_REVERSION: [
                "Price returns to mean (20 EMA)",
                "RSI normalizes (40-60)",
                "Bollinger Band middle line test",
                "Time-based exit (3-5 days)",
            ],
            StrategyType.BREAKOUT: [
                "Failed breakout (price returns below level)",
                "Volume dries up after breakout",
                "Target percentage reached",
                "Next resistance level hit",
            ],
        }
        return criteria.get(strategy_type, ["Stop loss hit", "Target reached"])

    def _get_risk_parameters(
        self,
        strategy_type: StrategyType,
        account_size: float,
        risk_tolerance: str,
    ) -> dict:
        """Get risk parameters for strategy."""
        risk_per_trade = {
            "conservative": 0.005,
            "moderate": 0.01,
            "aggressive": 0.02,
        }.get(risk_tolerance, 0.01)

        max_position = {
            "conservative": 0.05,
            "moderate": 0.1,
            "aggressive": 0.15,
        }.get(risk_tolerance, 0.1)

        params = {
            "risk_per_trade": risk_per_trade,
            "max_position_size": max_position,
            "max_daily_loss": risk_per_trade * 3,
            "max_open_positions": 5 if risk_tolerance == "conservative" else 10,
        }

        if strategy_type == StrategyType.SCALPING:
            params["risk_per_trade"] = risk_per_trade * 0.5
            params["max_trades_per_day"] = 20
        elif strategy_type == StrategyType.POSITION:
            params["max_position_size"] = max_position * 1.5
            params["max_open_positions"] = 3

        return params

    def _estimate_win_rate(
        self,
        strategy_type: StrategyType,
        confidence: float,
    ) -> float:
        """Estimate win rate for strategy."""
        base_rates = {
            StrategyType.TREND_FOLLOWING: 0.45,
            StrategyType.MOMENTUM: 0.50,
            StrategyType.MEAN_REVERSION: 0.55,
            StrategyType.BREAKOUT: 0.40,
            StrategyType.SCALPING: 0.55,
            StrategyType.SWING: 0.50,
            StrategyType.POSITION: 0.45,
        }

        base = base_rates.get(strategy_type, 0.50)
        adjustment = (confidence - 0.5) * 0.2

        return max(0.3, min(0.7, base + adjustment))

    def _estimate_profit_factor(
        self,
        strategy_type: StrategyType,
        confidence: float,
    ) -> float:
        """Estimate profit factor for strategy."""
        base_factors = {
            StrategyType.TREND_FOLLOWING: 1.8,
            StrategyType.MOMENTUM: 1.6,
            StrategyType.MEAN_REVERSION: 1.4,
            StrategyType.BREAKOUT: 2.0,
            StrategyType.SCALPING: 1.3,
            StrategyType.SWING: 1.7,
            StrategyType.POSITION: 2.2,
        }

        base = base_factors.get(strategy_type, 1.5)
        adjustment = (confidence - 0.5) * 0.5

        return max(1.0, base + adjustment)

    def _get_strategy_warnings(
        self,
        strategy_type: StrategyType,
        regime: MarketRegime,
        risk_tolerance: str,
    ) -> list[str]:
        """Get warnings for strategy."""
        warnings: list[str] = []

        if regime == MarketRegime.HIGH_VOLATILITY:
            warnings.append("High volatility - consider reduced position size")

        if strategy_type == StrategyType.TREND_FOLLOWING:
            if regime == MarketRegime.SIDEWAYS_RANGE:
                warnings.append("Trend following may underperform in ranging markets")

        if strategy_type == StrategyType.MEAN_REVERSION:
            if regime in [MarketRegime.BULLISH_TRENDING, MarketRegime.BEARISH_TRENDING]:
                warnings.append("Mean reversion can fail in strong trends")

        if strategy_type == StrategyType.BREAKOUT:
            warnings.append("Many breakouts fail - use stop losses strictly")

        if risk_tolerance == "aggressive":
            warnings.append("Aggressive sizing increases drawdown risk")

        return warnings

    def _score_symbol_for_strategy(
        self,
        strategy_type: StrategyType,
        data: dict,
    ) -> float:
        """Score a symbol for a strategy."""
        score = 0.5

        rsi = data.get("rsi", 50)
        adx = data.get("adx", 20)
        volume_ratio = data.get("volume_ratio", 1.0)
        price_vs_sma20 = data.get("price_vs_sma20", 0.0)

        if strategy_type == StrategyType.TREND_FOLLOWING:
            if adx > 25:
                score += 0.2
            if price_vs_sma20 > 0:
                score += 0.15
            if volume_ratio > 1.2:
                score += 0.1

        elif strategy_type == StrategyType.MOMENTUM:
            if 50 < rsi < 70:
                score += 0.2
            if adx > 20:
                score += 0.1

        elif strategy_type == StrategyType.MEAN_REVERSION:
            if rsi < 30 or rsi > 70:
                score += 0.25
            if abs(price_vs_sma20) > 0.05:
                score += 0.15

        elif strategy_type == StrategyType.BREAKOUT:
            if volume_ratio > 1.5:
                score += 0.2
            if adx < 20:
                score += 0.1

        return max(0.0, min(1.0, score))

    def _assess_setup_quality(self, score: float) -> str:
        """Assess quality of a trading setup."""
        if score >= 0.8:
            return "A+ (Excellent)"
        elif score >= 0.7:
            return "A (Very Good)"
        elif score >= 0.6:
            return "B (Good)"
        elif score >= 0.5:
            return "C (Average)"
        else:
            return "D (Below Average)"

    def _calculate_entry_zone(
        self,
        strategy_type: StrategyType,
        price: float,
        data: dict,
    ) -> tuple[float, float]:
        """Calculate entry price zone."""
        atr = data.get("atr", price * 0.02)

        if strategy_type == StrategyType.BREAKOUT:
            return (price, price + atr * 0.5)
        elif strategy_type == StrategyType.MEAN_REVERSION:
            return (price - atr * 0.5, price + atr * 0.5)
        else:
            return (price - atr * 0.25, price + atr * 0.25)

    def _calculate_target_zone(
        self,
        strategy_type: StrategyType,
        price: float,
        data: dict,
    ) -> tuple[float, float]:
        """Calculate target price zone."""
        atr = data.get("atr", price * 0.02)

        multiplier = {
            StrategyType.TREND_FOLLOWING: 3.0,
            StrategyType.MOMENTUM: 2.5,
            StrategyType.MEAN_REVERSION: 1.5,
            StrategyType.BREAKOUT: 2.5,
            StrategyType.SCALPING: 1.0,
            StrategyType.SWING: 2.0,
            StrategyType.POSITION: 4.0,
        }.get(strategy_type, 2.0)

        target = price + (atr * multiplier)
        return (target - atr * 0.25, target + atr * 0.5)

    def _calculate_stop_zone(
        self,
        strategy_type: StrategyType,
        price: float,
        data: dict,
    ) -> tuple[float, float]:
        """Calculate stop loss zone."""
        atr = data.get("atr", price * 0.02)

        stop_multiplier = {
            StrategyType.TREND_FOLLOWING: 1.5,
            StrategyType.MOMENTUM: 1.0,
            StrategyType.MEAN_REVERSION: 1.0,
            StrategyType.BREAKOUT: 1.0,
            StrategyType.SCALPING: 0.5,
            StrategyType.SWING: 1.0,
            StrategyType.POSITION: 2.0,
        }.get(strategy_type, 1.0)

        stop = price - (atr * stop_multiplier)
        return (stop - atr * 0.25, stop)

    async def _generate_market_summary(
        self,
        regime: MarketRegime,
        indices: dict[str, float],
        vix: float,
        sectors: dict[str, float],
    ) -> str:
        """Generate AI market summary."""
        if not self._client:
            return f"Market regime: {regime.value}, VIX: {vix:.1f}"

        index_str = ", ".join([f"{k}: {v:+.1f}%" for k, v in indices.items()])

        prompt = f"""Provide a brief 2-3 sentence market summary:

Indices: {index_str}
VIX: {vix:.1f}
Regime: {regime.value}

Focus on actionable insights for traders."""

        try:
            return await self._client.simple_chat(
                prompt=prompt,
                system_prompt="You are a market analyst. Be concise.",
                model=OpenAIModel.GPT_4O_MINI,
            )
        except Exception as e:
            logger.error(f"Market summary error: {e}")
            return f"Market regime: {regime.value}, VIX: {vix:.1f}"

    async def _generate_strategy_reasoning(
        self,
        strategy_type: StrategyType,
        regime: MarketRegime,
        confidence: float,
    ) -> str:
        """Generate reasoning for strategy recommendation."""
        if not self._client:
            return ""

        prompt = f"""In 2-3 sentences, explain why {strategy_type.value} is recommended for a {regime.value} market with {confidence*100:.0f}% confidence."""

        try:
            return await self._client.simple_chat(
                prompt=prompt,
                system_prompt="You are a trading strategist. Be concise and practical.",
                model=OpenAIModel.GPT_4O_MINI,
            )
        except Exception as e:
            logger.error(f"Strategy reasoning error: {e}")
            return ""

    async def _generate_optimization_reasoning(
        self,
        strategy_name: str,
        metrics: dict,
        changes: list[dict],
    ) -> str:
        """Generate reasoning for optimization suggestions."""
        if not self._client or not changes:
            return ""

        changes_str = "\n".join([
            f"- {c['parameter']}: {c['suggestion']}"
            for c in changes
        ])

        prompt = f"""Briefly explain why these optimizations are suggested for {strategy_name}:

Current metrics: {metrics}

Suggested changes:
{changes_str}

Provide 2-3 sentences of reasoning."""

        try:
            return await self._client.simple_chat(
                prompt=prompt,
                system_prompt="You are a quantitative analyst. Be specific and data-driven.",
                model=OpenAIModel.GPT_4O_MINI,
            )
        except Exception as e:
            logger.error(f"Optimization reasoning error: {e}")
            return ""

    def get_statistics(self) -> dict:
        """Get advisor statistics."""
        return {
            "total_recommendations": self._total_recommendations,
            "cache_size": len(self._analysis_cache),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"AIStrategyAdvisor(recommendations={self._total_recommendations})"

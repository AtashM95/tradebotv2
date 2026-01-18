"""
Liquidity Risk Management for Ultimate Trading Bot v2.2.

This module provides comprehensive liquidity risk assessment including
market impact analysis, execution risk, and liquidity scoring.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.risk.base_risk import RiskAlert, RiskLevel, RiskType


logger = logging.getLogger(__name__)


class LiquidityLevel(str, Enum):
    """Liquidity classification levels."""

    HIGHLY_LIQUID = "highly_liquid"
    LIQUID = "liquid"
    MODERATELY_LIQUID = "moderately_liquid"
    ILLIQUID = "illiquid"
    HIGHLY_ILLIQUID = "highly_illiquid"


class MarketImpactModel(str, Enum):
    """Market impact estimation models."""

    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    ALMGREN_CHRISS = "almgren_chriss"
    KYLE = "kyle"
    CUSTOM = "custom"


class LiquidityRiskConfig(BaseModel):
    """Configuration for liquidity risk management."""

    model_config = {"arbitrary_types_allowed": True}

    min_avg_volume: float = Field(default=100000.0, description="Minimum average daily volume")
    max_participation_rate: float = Field(default=0.10, description="Maximum % of daily volume")
    max_spread_percent: float = Field(default=0.005, description="Maximum bid-ask spread %")
    volume_lookback_days: int = Field(default=20, description="Days for volume analysis")
    impact_model: MarketImpactModel = Field(
        default=MarketImpactModel.SQUARE_ROOT,
        description="Market impact model"
    )
    impact_coefficient: float = Field(default=0.1, description="Impact model coefficient")
    min_depth_ratio: float = Field(default=2.0, description="Min book depth to order ratio")
    slippage_tolerance: float = Field(default=0.002, description="Maximum slippage tolerance")
    urgency_penalty: float = Field(default=0.5, description="Urgency impact multiplier")
    enable_adaptive_sizing: bool = Field(default=True, description="Enable adaptive position sizing")
    illiquidity_premium: float = Field(default=0.01, description="Required return premium for illiquid")


class LiquidityMetrics(BaseModel):
    """Liquidity metrics for an asset."""

    symbol: str
    timestamp: datetime = Field(default_factory=datetime.now)

    avg_daily_volume: float = Field(default=0.0, description="Average daily volume")
    current_volume: float = Field(default=0.0, description="Current session volume")
    volume_ratio: float = Field(default=1.0, description="Current vs average volume")

    bid_ask_spread: float = Field(default=0.0, description="Current bid-ask spread")
    spread_percent: float = Field(default=0.0, description="Spread as % of price")
    avg_spread: float = Field(default=0.0, description="Average spread")

    bid_depth: float = Field(default=0.0, description="Total bid book depth")
    ask_depth: float = Field(default=0.0, description="Total ask book depth")
    book_imbalance: float = Field(default=0.0, description="Order book imbalance")

    turnover_ratio: float = Field(default=0.0, description="Volume to float ratio")
    amihud_illiquidity: float = Field(default=0.0, description="Amihud illiquidity ratio")
    kyle_lambda: float = Field(default=0.0, description="Kyle's lambda estimate")

    liquidity_score: float = Field(default=50.0, description="Composite liquidity score 0-100")
    liquidity_level: LiquidityLevel = Field(default=LiquidityLevel.MODERATELY_LIQUID)


class MarketImpactEstimate(BaseModel):
    """Estimated market impact for a trade."""

    symbol: str
    order_size: float
    side: str  # 'buy' or 'sell'

    temporary_impact: float = Field(default=0.0, description="Temporary price impact %")
    permanent_impact: float = Field(default=0.0, description="Permanent price impact %")
    total_impact: float = Field(default=0.0, description="Total expected impact %")

    expected_slippage: float = Field(default=0.0, description="Expected slippage $")
    expected_cost: float = Field(default=0.0, description="Total execution cost $")

    participation_rate: float = Field(default=0.0, description="% of daily volume")
    execution_time_estimate: timedelta = Field(
        default_factory=lambda: timedelta(minutes=5),
        description="Estimated execution time"
    )

    confidence: float = Field(default=0.8, description="Estimate confidence 0-1")
    model_used: MarketImpactModel = Field(default=MarketImpactModel.SQUARE_ROOT)


class ExecutionRiskAssessment(BaseModel):
    """Assessment of execution risk for an order."""

    symbol: str
    order_size: float
    timestamp: datetime = Field(default_factory=datetime.now)

    can_execute: bool = Field(default=True, description="Whether order can be executed")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)

    market_impact: MarketImpactEstimate | None = None
    liquidity_metrics: LiquidityMetrics | None = None

    recommended_size: float | None = Field(default=None, description="Recommended adjusted size")
    recommended_algo: str | None = Field(default=None, description="Recommended execution algo")
    slice_recommendation: int = Field(default=1, description="Recommended order slices")

    warnings: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)


@dataclass
class VolumeProfile:
    """Intraday volume profile for timing."""

    symbol: str
    date: datetime
    hourly_volumes: dict[int, float] = field(default_factory=dict)
    peak_hour: int = 10
    trough_hour: int = 12
    opening_volume_pct: float = 0.15
    closing_volume_pct: float = 0.20

    def get_expected_volume_pct(self, hour: int) -> float:
        """Get expected volume percentage for hour."""
        return self.hourly_volumes.get(hour, 1.0 / 24.0)


class LiquidityRiskManager:
    """
    Manages liquidity risk assessment and market impact estimation.

    Provides tools for analyzing liquidity conditions, estimating market
    impact, and adjusting order sizes to minimize execution costs.
    """

    def __init__(self, config: LiquidityRiskConfig | None = None):
        """
        Initialize liquidity risk manager.

        Args:
            config: Liquidity risk configuration
        """
        self.config = config or LiquidityRiskConfig()
        self._metrics_cache: dict[str, LiquidityMetrics] = {}
        self._volume_profiles: dict[str, VolumeProfile] = {}
        self._historical_data: dict[str, list[dict]] = {}
        self._lock = asyncio.Lock()

        logger.info("LiquidityRiskManager initialized")

    async def calculate_liquidity_metrics(
        self,
        symbol: str,
        market_data: dict[str, Any],
    ) -> LiquidityMetrics:
        """
        Calculate comprehensive liquidity metrics.

        Args:
            symbol: Asset symbol
            market_data: Current market data

        Returns:
            LiquidityMetrics object
        """
        try:
            current_price = market_data.get("price", 0.0)
            bid = market_data.get("bid", current_price * 0.999)
            ask = market_data.get("ask", current_price * 1.001)
            volume = market_data.get("volume", 0.0)
            avg_volume = market_data.get("avg_volume", volume)
            bid_size = market_data.get("bid_size", 0.0)
            ask_size = market_data.get("ask_size", 0.0)
            shares_outstanding = market_data.get("shares_outstanding", 1e9)
            daily_returns = market_data.get("daily_returns", [])
            daily_volumes = market_data.get("daily_volumes", [])

            spread = ask - bid if ask > bid else 0.0
            spread_pct = spread / current_price if current_price > 0 else 0.0

            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

            book_imbalance = 0.0
            total_depth = bid_size + ask_size
            if total_depth > 0:
                book_imbalance = (bid_size - ask_size) / total_depth

            turnover = volume / shares_outstanding if shares_outstanding > 0 else 0.0

            amihud = 0.0
            if daily_returns and daily_volumes:
                for ret, vol in zip(daily_returns, daily_volumes):
                    if vol > 0:
                        amihud += abs(ret) / (vol * current_price)
                if len(daily_returns) > 0:
                    amihud /= len(daily_returns)

            kyle_lambda = self._estimate_kyle_lambda(daily_returns, daily_volumes, current_price)

            liquidity_score = self._calculate_liquidity_score(
                avg_volume=avg_volume,
                spread_pct=spread_pct,
                amihud=amihud,
                volume_ratio=volume_ratio,
                bid_depth=bid_size,
                ask_depth=ask_size,
            )

            liquidity_level = self._classify_liquidity(liquidity_score)

            metrics = LiquidityMetrics(
                symbol=symbol,
                avg_daily_volume=avg_volume,
                current_volume=volume,
                volume_ratio=volume_ratio,
                bid_ask_spread=spread,
                spread_percent=spread_pct,
                avg_spread=spread,
                bid_depth=bid_size,
                ask_depth=ask_size,
                book_imbalance=book_imbalance,
                turnover_ratio=turnover,
                amihud_illiquidity=amihud,
                kyle_lambda=kyle_lambda,
                liquidity_score=liquidity_score,
                liquidity_level=liquidity_level,
            )

            async with self._lock:
                self._metrics_cache[symbol] = metrics

            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate liquidity metrics for {symbol}: {e}")
            return LiquidityMetrics(symbol=symbol)

    def _estimate_kyle_lambda(
        self,
        returns: list[float],
        volumes: list[float],
        price: float,
    ) -> float:
        """Estimate Kyle's lambda using regression."""
        if len(returns) < 5 or len(volumes) < 5:
            return 0.0

        try:
            returns_arr = np.array(returns[:len(volumes)])
            volumes_arr = np.array(volumes[:len(returns)])

            signed_volume = returns_arr * volumes_arr * price

            if np.std(signed_volume) > 0:
                correlation = np.corrcoef(returns_arr, signed_volume)[0, 1]
                if not np.isnan(correlation):
                    return abs(correlation) * 0.001

            return 0.0

        except Exception:
            return 0.0

    def _calculate_liquidity_score(
        self,
        avg_volume: float,
        spread_pct: float,
        amihud: float,
        volume_ratio: float,
        bid_depth: float,
        ask_depth: float,
    ) -> float:
        """Calculate composite liquidity score 0-100."""
        score = 50.0

        if avg_volume >= 10_000_000:
            score += 20
        elif avg_volume >= 1_000_000:
            score += 15
        elif avg_volume >= 100_000:
            score += 10
        elif avg_volume >= 10_000:
            score += 5
        else:
            score -= 10

        if spread_pct <= 0.001:
            score += 15
        elif spread_pct <= 0.005:
            score += 10
        elif spread_pct <= 0.01:
            score += 5
        elif spread_pct <= 0.02:
            score -= 5
        else:
            score -= 15

        if amihud > 0:
            if amihud < 0.0001:
                score += 10
            elif amihud < 0.001:
                score += 5
            elif amihud > 0.01:
                score -= 10

        if volume_ratio >= 1.5:
            score += 5
        elif volume_ratio <= 0.5:
            score -= 5

        total_depth = bid_depth + ask_depth
        if total_depth >= 100_000:
            score += 5
        elif total_depth <= 1_000:
            score -= 5

        return max(0.0, min(100.0, score))

    def _classify_liquidity(self, score: float) -> LiquidityLevel:
        """Classify liquidity level from score."""
        if score >= 80:
            return LiquidityLevel.HIGHLY_LIQUID
        elif score >= 60:
            return LiquidityLevel.LIQUID
        elif score >= 40:
            return LiquidityLevel.MODERATELY_LIQUID
        elif score >= 20:
            return LiquidityLevel.ILLIQUID
        else:
            return LiquidityLevel.HIGHLY_ILLIQUID

    async def estimate_market_impact(
        self,
        symbol: str,
        order_size: float,
        side: str,
        market_data: dict[str, Any],
        urgency: float = 0.5,
    ) -> MarketImpactEstimate:
        """
        Estimate market impact for a trade.

        Args:
            symbol: Asset symbol
            order_size: Order size in shares
            side: 'buy' or 'sell'
            market_data: Current market data
            urgency: Trade urgency 0-1 (higher = more urgent)

        Returns:
            MarketImpactEstimate object
        """
        try:
            price = market_data.get("price", 0.0)
            avg_volume = market_data.get("avg_volume", 1e6)
            volatility = market_data.get("volatility", 0.02)
            spread_pct = market_data.get("spread_percent", 0.001)

            participation_rate = order_size / avg_volume if avg_volume > 0 else 1.0

            if self.config.impact_model == MarketImpactModel.LINEAR:
                temp_impact = self.config.impact_coefficient * participation_rate * volatility
                perm_impact = temp_impact * 0.5

            elif self.config.impact_model == MarketImpactModel.SQUARE_ROOT:
                temp_impact = self.config.impact_coefficient * np.sqrt(participation_rate) * volatility
                perm_impact = temp_impact * 0.5

            elif self.config.impact_model == MarketImpactModel.ALMGREN_CHRISS:
                sigma = volatility
                eta = 0.01
                gamma = 0.1

                temp_impact = eta * sigma * np.sqrt(participation_rate)
                perm_impact = gamma * participation_rate * sigma

            elif self.config.impact_model == MarketImpactModel.KYLE:
                lambda_kyle = market_data.get("kyle_lambda", 0.0001)
                temp_impact = lambda_kyle * order_size / avg_volume
                perm_impact = temp_impact * 0.3

            else:
                temp_impact = self.config.impact_coefficient * participation_rate * volatility
                perm_impact = temp_impact * 0.5

            urgency_multiplier = 1.0 + urgency * self.config.urgency_penalty
            temp_impact *= urgency_multiplier

            total_impact = temp_impact + perm_impact + spread_pct / 2

            order_value = order_size * price
            expected_slippage = order_value * total_impact

            spread_cost = order_value * spread_pct / 2
            impact_cost = order_value * (temp_impact + perm_impact)
            total_cost = spread_cost + impact_cost

            exec_minutes = max(1, int(participation_rate * 390))
            exec_time = timedelta(minutes=min(exec_minutes, 390))

            confidence = 0.9 if participation_rate < 0.01 else 0.7 if participation_rate < 0.1 else 0.5

            return MarketImpactEstimate(
                symbol=symbol,
                order_size=order_size,
                side=side,
                temporary_impact=temp_impact,
                permanent_impact=perm_impact,
                total_impact=total_impact,
                expected_slippage=expected_slippage,
                expected_cost=total_cost,
                participation_rate=participation_rate,
                execution_time_estimate=exec_time,
                confidence=confidence,
                model_used=self.config.impact_model,
            )

        except Exception as e:
            logger.error(f"Failed to estimate market impact for {symbol}: {e}")
            return MarketImpactEstimate(
                symbol=symbol,
                order_size=order_size,
                side=side,
            )

    async def assess_execution_risk(
        self,
        symbol: str,
        order_size: float,
        side: str,
        market_data: dict[str, Any],
    ) -> ExecutionRiskAssessment:
        """
        Assess execution risk for an order.

        Args:
            symbol: Asset symbol
            order_size: Order size in shares
            side: 'buy' or 'sell'
            market_data: Current market data

        Returns:
            ExecutionRiskAssessment object
        """
        try:
            metrics = await self.calculate_liquidity_metrics(symbol, market_data)
            impact = await self.estimate_market_impact(
                symbol, order_size, side, market_data
            )

            warnings: list[str] = []
            blocking_reasons: list[str] = []

            if metrics.avg_daily_volume < self.config.min_avg_volume:
                blocking_reasons.append(
                    f"Volume {metrics.avg_daily_volume:.0f} below minimum {self.config.min_avg_volume:.0f}"
                )

            if impact.participation_rate > self.config.max_participation_rate:
                warnings.append(
                    f"High participation rate {impact.participation_rate:.1%} "
                    f"(max {self.config.max_participation_rate:.1%})"
                )

            if metrics.spread_percent > self.config.max_spread_percent:
                warnings.append(
                    f"Wide spread {metrics.spread_percent:.3%} "
                    f"(max {self.config.max_spread_percent:.3%})"
                )

            if impact.total_impact > self.config.slippage_tolerance:
                warnings.append(
                    f"High expected impact {impact.total_impact:.3%} "
                    f"(tolerance {self.config.slippage_tolerance:.3%})"
                )

            relevant_depth = metrics.bid_depth if side == "sell" else metrics.ask_depth
            if relevant_depth > 0 and order_size / relevant_depth > self.config.min_depth_ratio:
                warnings.append(
                    f"Order size {order_size:.0f} exceeds book depth {relevant_depth:.0f}"
                )

            risk_level = RiskLevel.LOW
            if len(warnings) >= 3 or len(blocking_reasons) > 0:
                risk_level = RiskLevel.HIGH
            elif len(warnings) >= 2:
                risk_level = RiskLevel.MEDIUM
            elif len(warnings) >= 1:
                risk_level = RiskLevel.LOW
            else:
                risk_level = RiskLevel.MINIMAL

            can_execute = len(blocking_reasons) == 0

            recommended_size = None
            if self.config.enable_adaptive_sizing and not can_execute:
                recommended_size = min(
                    order_size,
                    metrics.avg_daily_volume * self.config.max_participation_rate,
                )

            recommended_algo = self._recommend_execution_algo(metrics, impact, order_size)

            slice_count = max(1, int(np.ceil(impact.participation_rate / 0.05)))

            return ExecutionRiskAssessment(
                symbol=symbol,
                order_size=order_size,
                can_execute=can_execute,
                risk_level=risk_level,
                market_impact=impact,
                liquidity_metrics=metrics,
                recommended_size=recommended_size,
                recommended_algo=recommended_algo,
                slice_recommendation=slice_count,
                warnings=warnings,
                blocking_reasons=blocking_reasons,
            )

        except Exception as e:
            logger.error(f"Failed to assess execution risk for {symbol}: {e}")
            return ExecutionRiskAssessment(
                symbol=symbol,
                order_size=order_size,
                can_execute=False,
                risk_level=RiskLevel.HIGH,
                blocking_reasons=[f"Assessment failed: {str(e)}"],
            )

    def _recommend_execution_algo(
        self,
        metrics: LiquidityMetrics,
        impact: MarketImpactEstimate,
        order_size: float,
    ) -> str:
        """Recommend execution algorithm based on conditions."""
        if impact.participation_rate < 0.01:
            return "MARKET"
        elif metrics.liquidity_level == LiquidityLevel.HIGHLY_LIQUID:
            if impact.participation_rate < 0.05:
                return "VWAP"
            else:
                return "TWAP"
        elif metrics.liquidity_level in [LiquidityLevel.LIQUID, LiquidityLevel.MODERATELY_LIQUID]:
            return "IMPLEMENTATION_SHORTFALL"
        else:
            return "ARRIVAL_PRICE"

    async def check_liquidity_alerts(
        self,
        symbol: str,
        market_data: dict[str, Any],
    ) -> list[RiskAlert]:
        """
        Check for liquidity-related risk alerts.

        Args:
            symbol: Asset symbol
            market_data: Current market data

        Returns:
            List of liquidity risk alerts
        """
        alerts: list[RiskAlert] = []

        try:
            metrics = await self.calculate_liquidity_metrics(symbol, market_data)

            if metrics.liquidity_level == LiquidityLevel.HIGHLY_ILLIQUID:
                alerts.append(RiskAlert(
                    alert_type=RiskType.LIQUIDITY,
                    level=RiskLevel.HIGH,
                    message=f"{symbol} is highly illiquid (score: {metrics.liquidity_score:.1f})",
                    details={"liquidity_score": metrics.liquidity_score},
                ))
            elif metrics.liquidity_level == LiquidityLevel.ILLIQUID:
                alerts.append(RiskAlert(
                    alert_type=RiskType.LIQUIDITY,
                    level=RiskLevel.MEDIUM,
                    message=f"{symbol} has low liquidity (score: {metrics.liquidity_score:.1f})",
                    details={"liquidity_score": metrics.liquidity_score},
                ))

            if metrics.spread_percent > self.config.max_spread_percent * 2:
                alerts.append(RiskAlert(
                    alert_type=RiskType.LIQUIDITY,
                    level=RiskLevel.HIGH,
                    message=f"{symbol} has abnormally wide spread ({metrics.spread_percent:.2%})",
                    details={"spread_percent": metrics.spread_percent},
                ))

            if metrics.volume_ratio < 0.3:
                alerts.append(RiskAlert(
                    alert_type=RiskType.LIQUIDITY,
                    level=RiskLevel.MEDIUM,
                    message=f"{symbol} has unusually low volume ({metrics.volume_ratio:.1%} of average)",
                    details={"volume_ratio": metrics.volume_ratio},
                ))

            if abs(metrics.book_imbalance) > 0.8:
                direction = "bid" if metrics.book_imbalance > 0 else "ask"
                alerts.append(RiskAlert(
                    alert_type=RiskType.LIQUIDITY,
                    level=RiskLevel.MEDIUM,
                    message=f"{symbol} has significant {direction}-side order book imbalance",
                    details={"book_imbalance": metrics.book_imbalance},
                ))

        except Exception as e:
            logger.error(f"Failed to check liquidity alerts for {symbol}: {e}")
            alerts.append(RiskAlert(
                alert_type=RiskType.LIQUIDITY,
                level=RiskLevel.MEDIUM,
                message=f"Failed to assess liquidity for {symbol}: {str(e)}",
            ))

        return alerts

    async def get_optimal_execution_window(
        self,
        symbol: str,
        order_size: float,
    ) -> dict[str, Any]:
        """
        Get optimal execution window based on volume profile.

        Args:
            symbol: Asset symbol
            order_size: Order size in shares

        Returns:
            Dictionary with optimal execution timing
        """
        profile = self._volume_profiles.get(symbol)

        if not profile:
            return {
                "best_hours": [10, 11, 14, 15],
                "avoid_hours": [12, 13],
                "reason": "Using default profile - no historical data",
            }

        hour_rankings = sorted(
            profile.hourly_volumes.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        best_hours = [h for h, _ in hour_rankings[:4]]
        avoid_hours = [h for h, _ in hour_rankings[-4:]]

        return {
            "best_hours": best_hours,
            "avoid_hours": avoid_hours,
            "peak_hour": profile.peak_hour,
            "trough_hour": profile.trough_hour,
            "opening_volume_pct": profile.opening_volume_pct,
            "closing_volume_pct": profile.closing_volume_pct,
            "reason": "Based on historical volume profile",
        }

    async def update_volume_profile(
        self,
        symbol: str,
        intraday_volumes: dict[int, float],
    ) -> None:
        """
        Update volume profile for a symbol.

        Args:
            symbol: Asset symbol
            intraday_volumes: Hour -> volume mapping
        """
        try:
            total_volume = sum(intraday_volumes.values())
            if total_volume == 0:
                return

            normalized = {h: v / total_volume for h, v in intraday_volumes.items()}

            peak_hour = max(intraday_volumes.items(), key=lambda x: x[1])[0]
            trough_hour = min(intraday_volumes.items(), key=lambda x: x[1])[0]

            opening_pct = sum(normalized.get(h, 0) for h in [9, 10])
            closing_pct = sum(normalized.get(h, 0) for h in [15, 16])

            self._volume_profiles[symbol] = VolumeProfile(
                symbol=symbol,
                date=datetime.now(),
                hourly_volumes=normalized,
                peak_hour=peak_hour,
                trough_hour=trough_hour,
                opening_volume_pct=opening_pct,
                closing_volume_pct=closing_pct,
            )

            logger.debug(f"Updated volume profile for {symbol}")

        except Exception as e:
            logger.error(f"Failed to update volume profile for {symbol}: {e}")

    def get_cached_metrics(self, symbol: str) -> LiquidityMetrics | None:
        """Get cached liquidity metrics for a symbol."""
        return self._metrics_cache.get(symbol)

    async def get_liquidity_summary(self, symbols: list[str]) -> dict[str, Any]:
        """
        Get liquidity summary for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Summary dictionary
        """
        summary: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "symbols_analyzed": len(symbols),
            "by_level": {level.value: [] for level in LiquidityLevel},
            "average_score": 0.0,
            "warnings": [],
        }

        scores = []

        for symbol in symbols:
            metrics = self._metrics_cache.get(symbol)
            if metrics:
                summary["by_level"][metrics.liquidity_level.value].append(symbol)
                scores.append(metrics.liquidity_score)

                if metrics.liquidity_level in [
                    LiquidityLevel.ILLIQUID,
                    LiquidityLevel.HIGHLY_ILLIQUID,
                ]:
                    summary["warnings"].append(
                        f"{symbol}: Low liquidity (score: {metrics.liquidity_score:.1f})"
                    )

        if scores:
            summary["average_score"] = np.mean(scores)

        return summary

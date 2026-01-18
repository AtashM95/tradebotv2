"""
AI Risk Assessor Module for Ultimate Trading Bot v2.2.

This module provides AI-powered risk assessment and analysis
for trades, positions, and portfolio management.
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


class RiskLevel(str, Enum):
    """Risk level enumeration."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class RiskCategory(str, Enum):
    """Risk category enumeration."""

    MARKET = "market"
    POSITION = "position"
    PORTFOLIO = "portfolio"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    DRAWDOWN = "drawdown"


class RiskFactor(BaseModel):
    """Individual risk factor."""

    category: RiskCategory
    name: str
    level: RiskLevel
    score: float = Field(ge=0.0, le=1.0)
    description: str
    mitigation: Optional[str] = None


class TradeRiskAssessment(BaseModel):
    """Trade risk assessment result."""

    assessment_id: str = Field(default_factory=generate_uuid)
    symbol: str
    direction: str
    entry_price: float
    quantity: int
    overall_risk: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    factors: list[RiskFactor] = Field(default_factory=list)
    max_loss_amount: float = Field(default=0.0)
    max_loss_percent: float = Field(default=0.0)
    recommended_stop_loss: Optional[float] = None
    recommended_position_size: Optional[int] = None
    approval_status: str = Field(default="pending")
    reasoning: str = Field(default="")
    warnings: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=now_utc)


class PositionRiskAssessment(BaseModel):
    """Position risk assessment result."""

    assessment_id: str = Field(default_factory=generate_uuid)
    symbol: str
    current_quantity: int
    current_price: float
    avg_entry_price: float
    unrealized_pnl: float
    overall_risk: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    factors: list[RiskFactor] = Field(default_factory=list)
    exit_recommendation: Optional[str] = None
    hedge_recommendation: Optional[str] = None
    position_health: str = Field(default="healthy")
    timestamp: datetime = Field(default_factory=now_utc)


class PortfolioRiskAssessment(BaseModel):
    """Portfolio risk assessment result."""

    assessment_id: str = Field(default_factory=generate_uuid)
    total_value: float
    cash_balance: float
    positions_count: int
    overall_risk: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    factors: list[RiskFactor] = Field(default_factory=list)
    concentration_issues: list[str] = Field(default_factory=list)
    correlation_issues: list[str] = Field(default_factory=list)
    rebalance_recommendations: list[str] = Field(default_factory=list)
    hedging_suggestions: list[str] = Field(default_factory=list)
    max_portfolio_drawdown: float = Field(default=0.0)
    value_at_risk_95: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=now_utc)


class AIRiskAssessorConfig(BaseModel):
    """Configuration for AI risk assessor."""

    max_risk_score_for_approval: float = Field(default=0.7, ge=0.0, le=1.0)
    position_concentration_limit: float = Field(default=0.2, ge=0.01, le=0.5)
    sector_concentration_limit: float = Field(default=0.35, ge=0.05, le=0.6)
    correlation_threshold: float = Field(default=0.7, ge=0.3, le=1.0)
    max_drawdown_limit: float = Field(default=0.15, ge=0.05, le=0.5)
    volatility_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)
    enable_ai_analysis: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300, ge=60, le=3600)


class AIRiskAssessor:
    """
    AI-powered risk assessment system.

    Provides comprehensive risk analysis for:
    - Individual trades before execution
    - Open positions monitoring
    - Portfolio-level risk management
    """

    def __init__(
        self,
        config: Optional[AIRiskAssessorConfig] = None,
        openai_client: Optional[OpenAIClient] = None,
    ) -> None:
        """
        Initialize AIRiskAssessor.

        Args:
            config: Risk assessor configuration
            openai_client: OpenAI client instance
        """
        self._config = config or AIRiskAssessorConfig()
        self._client = openai_client

        self._assessment_cache: dict[str, tuple[Any, datetime]] = {}
        self._total_assessments = 0
        self._approved_trades = 0
        self._rejected_trades = 0

        logger.info("AIRiskAssessor initialized")

    def set_client(self, client: OpenAIClient) -> None:
        """Set the OpenAI client."""
        self._client = client

    async def assess_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: int,
        stop_loss: Optional[float] = None,
        account_value: float = 100000.0,
        current_positions: Optional[list[dict]] = None,
        market_conditions: Optional[dict] = None,
    ) -> TradeRiskAssessment:
        """
        Assess risk for a proposed trade.

        Args:
            symbol: Trading symbol
            direction: Trade direction (long/short)
            entry_price: Entry price
            quantity: Trade quantity
            stop_loss: Stop loss price
            account_value: Total account value
            current_positions: List of current positions
            market_conditions: Market condition data

        Returns:
            TradeRiskAssessment with risk analysis
        """
        self._total_assessments += 1
        factors: list[RiskFactor] = []

        position_value = entry_price * quantity
        position_pct = position_value / account_value if account_value > 0 else 1.0

        concentration_score = self._assess_concentration_risk(
            position_pct,
            current_positions or [],
            symbol,
        )
        factors.append(concentration_score)

        if stop_loss:
            loss_per_share = abs(entry_price - stop_loss)
            max_loss = loss_per_share * quantity
            max_loss_pct = max_loss / account_value if account_value > 0 else 1.0
        else:
            max_loss = position_value * 0.1
            max_loss_pct = max_loss / account_value if account_value > 0 else 0.1

        position_size_score = self._assess_position_size_risk(
            position_pct,
            max_loss_pct,
        )
        factors.append(position_size_score)

        volatility_score = self._assess_volatility_risk(
            market_conditions or {},
            symbol,
        )
        factors.append(volatility_score)

        market_score = self._assess_market_risk(
            market_conditions or {},
            direction,
        )
        factors.append(market_score)

        overall_score = sum(f.score for f in factors) / len(factors)
        overall_risk = self._score_to_risk_level(overall_score)

        warnings = self._generate_warnings(factors, position_pct, max_loss_pct)

        if overall_score <= self._config.max_risk_score_for_approval:
            approval_status = "approved"
            self._approved_trades += 1
        else:
            approval_status = "rejected"
            self._rejected_trades += 1

        reasoning = ""
        if self._config.enable_ai_analysis and self._client:
            reasoning = await self._get_ai_trade_analysis(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                quantity=quantity,
                position_pct=position_pct,
                max_loss_pct=max_loss_pct,
                factors=factors,
            )

        recommended_size = None
        if position_pct > self._config.position_concentration_limit:
            max_position = account_value * self._config.position_concentration_limit
            recommended_size = int(max_position / entry_price)

        return TradeRiskAssessment(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            overall_risk=overall_risk,
            risk_score=overall_score,
            factors=factors,
            max_loss_amount=max_loss,
            max_loss_percent=max_loss_pct * 100,
            recommended_stop_loss=stop_loss,
            recommended_position_size=recommended_size,
            approval_status=approval_status,
            reasoning=reasoning,
            warnings=warnings,
        )

    async def assess_position(
        self,
        symbol: str,
        quantity: int,
        current_price: float,
        avg_entry_price: float,
        days_held: int = 0,
        market_conditions: Optional[dict] = None,
    ) -> PositionRiskAssessment:
        """
        Assess risk for an open position.

        Args:
            symbol: Trading symbol
            quantity: Position quantity
            current_price: Current market price
            avg_entry_price: Average entry price
            days_held: Days position has been held
            market_conditions: Market condition data

        Returns:
            PositionRiskAssessment with risk analysis
        """
        factors: list[RiskFactor] = []

        unrealized_pnl = (current_price - avg_entry_price) * quantity
        pnl_pct = (current_price - avg_entry_price) / avg_entry_price if avg_entry_price > 0 else 0

        pnl_factor = self._assess_pnl_risk(pnl_pct)
        factors.append(pnl_factor)

        duration_factor = self._assess_duration_risk(days_held, pnl_pct)
        factors.append(duration_factor)

        volatility_factor = self._assess_volatility_risk(
            market_conditions or {},
            symbol,
        )
        factors.append(volatility_factor)

        overall_score = sum(f.score for f in factors) / len(factors)
        overall_risk = self._score_to_risk_level(overall_score)

        exit_recommendation = None
        if pnl_pct < -0.1:
            exit_recommendation = "Consider exiting to limit losses"
        elif pnl_pct > 0.2:
            exit_recommendation = "Consider taking partial profits"

        position_health = "healthy"
        if pnl_pct < -0.15:
            position_health = "critical"
        elif pnl_pct < -0.05:
            position_health = "warning"

        return PositionRiskAssessment(
            symbol=symbol,
            current_quantity=quantity,
            current_price=current_price,
            avg_entry_price=avg_entry_price,
            unrealized_pnl=unrealized_pnl,
            overall_risk=overall_risk,
            risk_score=overall_score,
            factors=factors,
            exit_recommendation=exit_recommendation,
            position_health=position_health,
        )

    async def assess_portfolio(
        self,
        positions: list[dict],
        cash_balance: float,
        account_value: float,
        market_conditions: Optional[dict] = None,
    ) -> PortfolioRiskAssessment:
        """
        Assess portfolio-level risk.

        Args:
            positions: List of position dictionaries
            cash_balance: Available cash
            account_value: Total account value
            market_conditions: Market condition data

        Returns:
            PortfolioRiskAssessment with analysis
        """
        factors: list[RiskFactor] = []

        concentration_issues: list[str] = []
        position_weights: dict[str, float] = {}

        for pos in positions:
            symbol = pos.get("symbol", "")
            value = pos.get("market_value", 0.0)
            weight = value / account_value if account_value > 0 else 0.0
            position_weights[symbol] = weight

            if weight > self._config.position_concentration_limit:
                concentration_issues.append(
                    f"{symbol}: {weight*100:.1f}% exceeds {self._config.position_concentration_limit*100:.0f}% limit"
                )

        concentration_factor = self._assess_portfolio_concentration(
            position_weights,
        )
        factors.append(concentration_factor)

        correlation_factor = self._assess_correlation_risk(positions)
        factors.append(correlation_factor)

        cash_pct = cash_balance / account_value if account_value > 0 else 1.0
        liquidity_factor = self._assess_liquidity_risk(cash_pct)
        factors.append(liquidity_factor)

        market_factor = self._assess_market_risk(
            market_conditions or {},
            "neutral",
        )
        factors.append(market_factor)

        overall_score = sum(f.score for f in factors) / len(factors)
        overall_risk = self._score_to_risk_level(overall_score)

        rebalance_recs: list[str] = []
        if concentration_issues:
            rebalance_recs.append("Reduce concentrated positions")
        if cash_pct < 0.05:
            rebalance_recs.append("Increase cash buffer")

        var_95 = account_value * overall_score * 0.1

        return PortfolioRiskAssessment(
            total_value=account_value,
            cash_balance=cash_balance,
            positions_count=len(positions),
            overall_risk=overall_risk,
            risk_score=overall_score,
            factors=factors,
            concentration_issues=concentration_issues,
            rebalance_recommendations=rebalance_recs,
            value_at_risk_95=var_95,
        )

    def _assess_concentration_risk(
        self,
        position_pct: float,
        current_positions: list[dict],
        symbol: str,
    ) -> RiskFactor:
        """Assess concentration risk for a trade."""
        existing_weight = 0.0
        for pos in current_positions:
            if pos.get("symbol") == symbol:
                existing_weight = pos.get("weight", 0.0)
                break

        total_weight = position_pct + existing_weight

        if total_weight > 0.3:
            level = RiskLevel.VERY_HIGH
            score = 0.9
        elif total_weight > 0.2:
            level = RiskLevel.HIGH
            score = 0.7
        elif total_weight > 0.15:
            level = RiskLevel.MODERATE
            score = 0.5
        elif total_weight > 0.1:
            level = RiskLevel.LOW
            score = 0.3
        else:
            level = RiskLevel.VERY_LOW
            score = 0.1

        return RiskFactor(
            category=RiskCategory.CONCENTRATION,
            name="Position Concentration",
            level=level,
            score=score,
            description=f"Position weight: {total_weight*100:.1f}%",
            mitigation="Reduce position size to meet concentration limits",
        )

    def _assess_position_size_risk(
        self,
        position_pct: float,
        max_loss_pct: float,
    ) -> RiskFactor:
        """Assess position size risk."""
        if max_loss_pct > 0.05:
            level = RiskLevel.VERY_HIGH
            score = 0.9
        elif max_loss_pct > 0.03:
            level = RiskLevel.HIGH
            score = 0.7
        elif max_loss_pct > 0.02:
            level = RiskLevel.MODERATE
            score = 0.5
        elif max_loss_pct > 0.01:
            level = RiskLevel.LOW
            score = 0.3
        else:
            level = RiskLevel.VERY_LOW
            score = 0.1

        return RiskFactor(
            category=RiskCategory.POSITION,
            name="Position Size Risk",
            level=level,
            score=score,
            description=f"Max loss: {max_loss_pct*100:.2f}% of account",
            mitigation="Use tighter stop loss or reduce position size",
        )

    def _assess_volatility_risk(
        self,
        market_conditions: dict,
        symbol: str,
    ) -> RiskFactor:
        """Assess volatility risk."""
        vix = market_conditions.get("vix", 20.0)
        symbol_volatility = market_conditions.get("symbol_volatility", 0.02)

        if vix > 35 or symbol_volatility > 0.05:
            level = RiskLevel.VERY_HIGH
            score = 0.9
        elif vix > 25 or symbol_volatility > 0.04:
            level = RiskLevel.HIGH
            score = 0.7
        elif vix > 20 or symbol_volatility > 0.03:
            level = RiskLevel.MODERATE
            score = 0.5
        elif vix > 15:
            level = RiskLevel.LOW
            score = 0.3
        else:
            level = RiskLevel.VERY_LOW
            score = 0.15

        return RiskFactor(
            category=RiskCategory.VOLATILITY,
            name="Volatility Risk",
            level=level,
            score=score,
            description=f"VIX: {vix:.1f}, Symbol volatility: {symbol_volatility*100:.1f}%",
            mitigation="Reduce position size in high volatility",
        )

    def _assess_market_risk(
        self,
        market_conditions: dict,
        direction: str,
    ) -> RiskFactor:
        """Assess market risk."""
        market_trend = market_conditions.get("trend", "neutral")
        spy_change = market_conditions.get("spy_change", 0.0)

        is_contrarian = (
            (direction == "long" and market_trend == "bearish") or
            (direction == "short" and market_trend == "bullish")
        )

        if is_contrarian:
            level = RiskLevel.HIGH
            score = 0.7
            description = f"Trading against market trend ({market_trend})"
        elif abs(spy_change) > 0.02:
            level = RiskLevel.MODERATE
            score = 0.5
            description = f"High market movement: SPY {spy_change*100:+.1f}%"
        else:
            level = RiskLevel.LOW
            score = 0.25
            description = "Normal market conditions"

        return RiskFactor(
            category=RiskCategory.MARKET,
            name="Market Risk",
            level=level,
            score=score,
            description=description,
            mitigation="Consider market direction when sizing",
        )

    def _assess_pnl_risk(self, pnl_pct: float) -> RiskFactor:
        """Assess P&L risk for a position."""
        if pnl_pct < -0.15:
            level = RiskLevel.EXTREME
            score = 0.95
        elif pnl_pct < -0.1:
            level = RiskLevel.VERY_HIGH
            score = 0.8
        elif pnl_pct < -0.05:
            level = RiskLevel.HIGH
            score = 0.6
        elif pnl_pct < 0:
            level = RiskLevel.MODERATE
            score = 0.4
        else:
            level = RiskLevel.LOW
            score = 0.2

        return RiskFactor(
            category=RiskCategory.POSITION,
            name="P&L Risk",
            level=level,
            score=score,
            description=f"Unrealized P&L: {pnl_pct*100:+.1f}%",
            mitigation="Review stop loss levels",
        )

    def _assess_duration_risk(
        self,
        days_held: int,
        pnl_pct: float,
    ) -> RiskFactor:
        """Assess holding duration risk."""
        if days_held > 30 and pnl_pct < 0:
            level = RiskLevel.HIGH
            score = 0.7
            description = f"Held {days_held} days with negative P&L"
        elif days_held > 60:
            level = RiskLevel.MODERATE
            score = 0.5
            description = f"Long holding period: {days_held} days"
        else:
            level = RiskLevel.LOW
            score = 0.2
            description = f"Normal holding period: {days_held} days"

        return RiskFactor(
            category=RiskCategory.POSITION,
            name="Duration Risk",
            level=level,
            score=score,
            description=description,
            mitigation="Review thesis if prolonged losing position",
        )

    def _assess_portfolio_concentration(
        self,
        position_weights: dict[str, float],
    ) -> RiskFactor:
        """Assess portfolio concentration."""
        if not position_weights:
            return RiskFactor(
                category=RiskCategory.CONCENTRATION,
                name="Portfolio Concentration",
                level=RiskLevel.VERY_LOW,
                score=0.1,
                description="No positions",
            )

        max_weight = max(position_weights.values())
        top_3_weight = sum(sorted(position_weights.values(), reverse=True)[:3])

        if max_weight > 0.3 or top_3_weight > 0.6:
            level = RiskLevel.VERY_HIGH
            score = 0.85
        elif max_weight > 0.2 or top_3_weight > 0.5:
            level = RiskLevel.HIGH
            score = 0.65
        elif max_weight > 0.15:
            level = RiskLevel.MODERATE
            score = 0.45
        else:
            level = RiskLevel.LOW
            score = 0.25

        return RiskFactor(
            category=RiskCategory.CONCENTRATION,
            name="Portfolio Concentration",
            level=level,
            score=score,
            description=f"Max position: {max_weight*100:.1f}%, Top 3: {top_3_weight*100:.1f}%",
            mitigation="Diversify across more positions",
        )

    def _assess_correlation_risk(
        self,
        positions: list[dict],
    ) -> RiskFactor:
        """Assess correlation risk between positions."""
        sectors = [pos.get("sector", "unknown") for pos in positions]

        if not sectors:
            return RiskFactor(
                category=RiskCategory.CORRELATION,
                name="Correlation Risk",
                level=RiskLevel.VERY_LOW,
                score=0.1,
                description="No positions to correlate",
            )

        sector_counts: dict[str, int] = {}
        for sector in sectors:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        max_sector_pct = max(sector_counts.values()) / len(sectors) if sectors else 0

        if max_sector_pct > 0.5:
            level = RiskLevel.HIGH
            score = 0.7
        elif max_sector_pct > 0.35:
            level = RiskLevel.MODERATE
            score = 0.5
        else:
            level = RiskLevel.LOW
            score = 0.25

        return RiskFactor(
            category=RiskCategory.CORRELATION,
            name="Correlation Risk",
            level=level,
            score=score,
            description=f"Sector concentration: {max_sector_pct*100:.1f}%",
            mitigation="Diversify across sectors",
        )

    def _assess_liquidity_risk(self, cash_pct: float) -> RiskFactor:
        """Assess liquidity risk."""
        if cash_pct < 0.02:
            level = RiskLevel.VERY_HIGH
            score = 0.85
        elif cash_pct < 0.05:
            level = RiskLevel.HIGH
            score = 0.65
        elif cash_pct < 0.1:
            level = RiskLevel.MODERATE
            score = 0.45
        else:
            level = RiskLevel.LOW
            score = 0.2

        return RiskFactor(
            category=RiskCategory.LIQUIDITY,
            name="Liquidity Risk",
            level=level,
            score=score,
            description=f"Cash: {cash_pct*100:.1f}%",
            mitigation="Maintain adequate cash buffer",
        )

    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level."""
        if score >= 0.85:
            return RiskLevel.EXTREME
        elif score >= 0.7:
            return RiskLevel.VERY_HIGH
        elif score >= 0.55:
            return RiskLevel.HIGH
        elif score >= 0.4:
            return RiskLevel.MODERATE
        elif score >= 0.25:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

    def _generate_warnings(
        self,
        factors: list[RiskFactor],
        position_pct: float,
        max_loss_pct: float,
    ) -> list[str]:
        """Generate warning messages."""
        warnings: list[str] = []

        for factor in factors:
            if factor.level in [RiskLevel.VERY_HIGH, RiskLevel.EXTREME]:
                warnings.append(f"{factor.name}: {factor.description}")

        if position_pct > self._config.position_concentration_limit:
            warnings.append(
                f"Position exceeds {self._config.position_concentration_limit*100:.0f}% limit"
            )

        if max_loss_pct > 0.02:
            warnings.append(
                f"Max loss ({max_loss_pct*100:.1f}%) exceeds 2% rule"
            )

        return warnings

    async def _get_ai_trade_analysis(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: int,
        position_pct: float,
        max_loss_pct: float,
        factors: list[RiskFactor],
    ) -> str:
        """Get AI analysis for trade risk."""
        if not self._client:
            return ""

        factor_summary = "\n".join([
            f"- {f.name}: {f.level.value} ({f.score:.2f})"
            for f in factors
        ])

        prompt = f"""Briefly assess this proposed trade from a risk perspective:

Symbol: {symbol}
Direction: {direction}
Entry Price: ${entry_price:.2f}
Quantity: {quantity}
Position Size: {position_pct*100:.1f}% of portfolio
Max Loss: {max_loss_pct*100:.2f}% of portfolio

Risk Factors:
{factor_summary}

Provide a 2-3 sentence risk assessment focusing on the most important considerations."""

        try:
            response = await self._client.simple_chat(
                prompt=prompt,
                system_prompt="You are a risk management expert. Be concise and specific.",
                model=OpenAIModel.GPT_4O_MINI,
            )
            return response

        except Exception as e:
            logger.error(f"AI risk analysis error: {e}")
            return ""

    def get_statistics(self) -> dict:
        """Get assessor statistics."""
        return {
            "total_assessments": self._total_assessments,
            "approved_trades": self._approved_trades,
            "rejected_trades": self._rejected_trades,
            "approval_rate": (
                self._approved_trades / self._total_assessments * 100
                if self._total_assessments > 0 else 0
            ),
            "cache_size": len(self._assessment_cache),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AIRiskAssessor(assessments={self._total_assessments}, "
            f"approved={self._approved_trades})"
        )

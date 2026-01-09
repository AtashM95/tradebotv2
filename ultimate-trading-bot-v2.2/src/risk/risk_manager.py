"""
Risk Manager Module for Ultimate Trading Bot v2.2.

This module provides comprehensive risk management with
real-time monitoring and automated responses.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.risk.base_risk import (
    BaseRiskManager,
    RiskConfig,
    RiskLevel,
    RiskType,
    RiskMetric,
    RiskAlert,
    RiskLimit,
    RiskAssessment,
    RiskContext,
    calculate_var,
    calculate_cvar,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)
from src.risk.position_sizer import PositionSizer, PositionSizerConfig
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class RiskManagerConfig(RiskConfig):
    """Configuration for the main risk manager."""

    name: str = Field(default="Main Risk Manager")

    position_sizer_config: PositionSizerConfig = Field(
        default_factory=PositionSizerConfig
    )

    max_position_size_pct: float = Field(default=0.10, ge=0.01, le=0.5)
    max_sector_exposure_pct: float = Field(default=0.30, ge=0.1, le=0.6)
    max_correlation_group_pct: float = Field(default=0.40, ge=0.1, le=0.7)

    max_total_exposure_pct: float = Field(default=1.0, ge=0.5, le=3.0)
    max_long_exposure_pct: float = Field(default=1.0, ge=0.5, le=2.0)
    max_short_exposure_pct: float = Field(default=0.5, ge=0.0, le=1.5)

    max_daily_loss_pct: float = Field(default=0.03, ge=0.01, le=0.1)
    max_weekly_loss_pct: float = Field(default=0.07, ge=0.02, le=0.2)
    max_monthly_loss_pct: float = Field(default=0.12, ge=0.05, le=0.3)
    max_drawdown_pct: float = Field(default=0.15, ge=0.05, le=0.5)

    var_confidence: float = Field(default=0.95, ge=0.9, le=0.99)
    max_var_pct: float = Field(default=0.03, ge=0.01, le=0.1)
    max_cvar_pct: float = Field(default=0.05, ge=0.02, le=0.15)

    min_liquidity_days: float = Field(default=1.0, ge=0.1, le=10.0)
    max_position_adv_pct: float = Field(default=0.05, ge=0.01, le=0.2)

    halt_on_daily_loss: bool = Field(default=True)
    halt_on_drawdown: bool = Field(default=True)
    auto_deleverage: bool = Field(default=True)
    deleverage_threshold_pct: float = Field(default=0.80, ge=0.5, le=0.95)


class TradingHalt(BaseModel):
    """Model for trading halt."""

    halt_id: str = Field(default_factory=generate_uuid)
    reason: str
    started_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = Field(default=True)
    affected_symbols: list[str] = Field(default_factory=list)


class RiskManager(BaseRiskManager):
    """
    Comprehensive risk management system.

    Features:
    - Real-time risk monitoring
    - Position sizing integration
    - Exposure management
    - Drawdown control
    - Trading halts
    - Automated deleveraging
    """

    def __init__(
        self,
        config: Optional[RiskManagerConfig] = None,
    ) -> None:
        """
        Initialize RiskManager.

        Args:
            config: Risk manager configuration
        """
        config = config or RiskManagerConfig()
        super().__init__(config)

        self._rm_config = config
        self._position_sizer = PositionSizer(config.position_sizer_config)

        self._trading_halts: list[TradingHalt] = []
        self._pnl_history: list[tuple[datetime, float]] = []
        self._equity_history: list[tuple[datetime, float]] = []
        self._peak_equity: float = 0.0

        self._initialize_limits()

        logger.info(f"RiskManager initialized: {config.name}")

    def _initialize_limits(self) -> None:
        """Initialize default risk limits."""
        limits = [
            RiskLimit(
                name="Daily Loss",
                limit_type="pnl",
                limit_value=self._rm_config.max_daily_loss_pct,
                warning_threshold=0.7,
                description="Maximum daily loss as percentage of equity",
            ),
            RiskLimit(
                name="Drawdown",
                limit_type="drawdown",
                limit_value=self._rm_config.max_drawdown_pct,
                warning_threshold=0.6,
                description="Maximum drawdown from peak equity",
            ),
            RiskLimit(
                name="Total Exposure",
                limit_type="exposure",
                limit_value=self._rm_config.max_total_exposure_pct,
                warning_threshold=0.8,
                description="Maximum total market exposure",
            ),
            RiskLimit(
                name="VaR",
                limit_type="var",
                limit_value=self._rm_config.max_var_pct,
                warning_threshold=0.8,
                description="Maximum Value at Risk",
            ),
            RiskLimit(
                name="Single Position",
                limit_type="position",
                limit_value=self._rm_config.max_position_size_pct,
                warning_threshold=0.9,
                description="Maximum single position size",
            ),
        ]

        for limit in limits:
            self.add_limit(limit)

    async def assess_risk(
        self,
        context: dict[str, Any],
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment.

        Args:
            context: Risk assessment context

        Returns:
            Risk assessment result
        """
        risk_context = self._build_risk_context(context)

        self._update_equity_history(risk_context)

        market_risk = await self._assess_market_risk(risk_context)
        position_risk = await self._assess_position_risk(risk_context)
        portfolio_risk = await self._assess_portfolio_risk(risk_context)
        liquidity_risk = await self._assess_liquidity_risk(risk_context)

        overall_score = (
            market_risk * 0.25 +
            position_risk * 0.25 +
            portfolio_risk * 0.35 +
            liquidity_risk * 0.15
        )

        overall_level = self.calculate_risk_level(overall_score)

        alerts = await self.check_limits(context)

        recommendations = self._generate_recommendations(
            overall_level, market_risk, position_risk, portfolio_risk, liquidity_risk
        )

        assessment = RiskAssessment(
            timestamp=now_utc(),
            overall_risk_level=overall_level,
            overall_risk_score=overall_score,
            market_risk_score=market_risk,
            position_risk_score=position_risk,
            portfolio_risk_score=portfolio_risk,
            liquidity_risk_score=liquidity_risk,
            alerts=alerts,
            limits_status=list(self._limits.values()),
            recommendations=recommendations,
            metadata={
                "account_value": risk_context.account_value,
                "positions_count": risk_context.position_count,
                "leverage": risk_context.leverage_ratio,
                "drawdown": risk_context.current_drawdown,
            },
        )

        return assessment

    async def check_limits(
        self,
        context: dict[str, Any],
    ) -> list[RiskAlert]:
        """
        Check all risk limits.

        Args:
            context: Context for limit checking

        Returns:
            List of risk alerts
        """
        alerts: list[RiskAlert] = []
        risk_context = self._build_risk_context(context)

        for limit in self._limits.values():
            current_value = self._get_limit_value(limit.limit_type, risk_context)

            alert = self.check_limit(limit, current_value)
            if alert:
                alerts.append(alert)

                if limit.is_breached:
                    await self._handle_limit_breach(limit, risk_context)

        return alerts

    def _build_risk_context(self, context: dict[str, Any]) -> RiskContext:
        """Build RiskContext from generic context."""
        return RiskContext(
            timestamp=context.get("timestamp", now_utc()),
            account_value=context.get("account_value", 0),
            cash_balance=context.get("cash_balance", 0),
            positions=context.get("positions", []),
            open_orders=context.get("open_orders", []),
            daily_pnl=context.get("daily_pnl", 0),
            unrealized_pnl=context.get("unrealized_pnl", 0),
            peak_value=self._peak_equity,
            current_drawdown=self._calculate_current_drawdown(
                context.get("account_value", 0)
            ),
            market_data=context.get("market_data", {}),
            metadata=context.get("metadata", {}),
        )

    def _update_equity_history(self, context: RiskContext) -> None:
        """Update equity history for drawdown tracking."""
        self._equity_history.append((context.timestamp, context.account_value))

        if context.account_value > self._peak_equity:
            self._peak_equity = context.account_value

        cutoff = context.timestamp - timedelta(days=365)
        self._equity_history = [
            (ts, val) for ts, val in self._equity_history
            if ts >= cutoff
        ]

    def _calculate_current_drawdown(self, current_equity: float) -> float:
        """Calculate current drawdown from peak."""
        if self._peak_equity <= 0:
            return 0.0

        drawdown = (self._peak_equity - current_equity) / self._peak_equity
        return max(0, drawdown)

    async def _assess_market_risk(self, context: RiskContext) -> float:
        """Assess market-wide risk factors."""
        score = 25.0

        market_data = context.market_data
        if market_data:
            vix = market_data.get("vix", 20)
            if vix > 30:
                score += 30
            elif vix > 25:
                score += 20
            elif vix > 20:
                score += 10

            market_return = market_data.get("market_return_1d", 0)
            if abs(market_return) > 0.03:
                score += 20
            elif abs(market_return) > 0.02:
                score += 10

        return min(100, score)

    async def _assess_position_risk(self, context: RiskContext) -> float:
        """Assess individual position risks."""
        if not context.positions:
            return 0.0

        score = 0.0
        account_value = context.account_value

        for position in context.positions:
            pos_value = abs(position.get("market_value", 0))
            pos_pct = pos_value / account_value if account_value > 0 else 0

            if pos_pct > self._rm_config.max_position_size_pct:
                score += 25
            elif pos_pct > self._rm_config.max_position_size_pct * 0.8:
                score += 15

            unrealized_pnl_pct = position.get("unrealized_pnl_pct", 0)
            if unrealized_pnl_pct < -0.1:
                score += 20
            elif unrealized_pnl_pct < -0.05:
                score += 10

        score = score / max(1, len(context.positions))

        return min(100, score)

    async def _assess_portfolio_risk(self, context: RiskContext) -> float:
        """Assess portfolio-level risks."""
        score = 0.0

        leverage = context.leverage_ratio
        if leverage > 2.0:
            score += 40
        elif leverage > 1.5:
            score += 25
        elif leverage > 1.0:
            score += 10

        drawdown = context.current_drawdown
        if drawdown > self._rm_config.max_drawdown_pct:
            score += 40
        elif drawdown > self._rm_config.max_drawdown_pct * 0.7:
            score += 25
        elif drawdown > self._rm_config.max_drawdown_pct * 0.5:
            score += 15

        account_value = context.account_value
        if account_value > 0:
            daily_loss_pct = -context.daily_pnl / account_value if context.daily_pnl < 0 else 0
            if daily_loss_pct > self._rm_config.max_daily_loss_pct:
                score += 30
            elif daily_loss_pct > self._rm_config.max_daily_loss_pct * 0.7:
                score += 15

        return min(100, score)

    async def _assess_liquidity_risk(self, context: RiskContext) -> float:
        """Assess liquidity risk."""
        score = 0.0

        cash_ratio = context.cash_balance / context.account_value if context.account_value > 0 else 0

        if cash_ratio < 0.05:
            score += 30
        elif cash_ratio < 0.10:
            score += 15

        return min(100, score)

    def _get_limit_value(self, limit_type: str, context: RiskContext) -> float:
        """Get current value for a limit type."""
        account_value = context.account_value

        if limit_type == "pnl":
            if context.daily_pnl >= 0:
                return 0.0
            return abs(context.daily_pnl) / account_value if account_value > 0 else 0

        elif limit_type == "drawdown":
            return context.current_drawdown

        elif limit_type == "exposure":
            return context.leverage_ratio

        elif limit_type == "var":
            returns = [0.01, -0.005, 0.008, -0.02, 0.015]
            return calculate_var(returns, self._rm_config.var_confidence)

        elif limit_type == "position":
            if not context.positions:
                return 0.0
            max_pos = max(
                abs(p.get("market_value", 0)) / account_value
                for p in context.positions
            ) if account_value > 0 else 0
            return max_pos

        return 0.0

    async def _handle_limit_breach(
        self,
        limit: RiskLimit,
        context: RiskContext,
    ) -> None:
        """Handle a limit breach."""
        logger.warning(f"Limit breached: {limit.name}")

        if limit.limit_type == "pnl" and self._rm_config.halt_on_daily_loss:
            await self._initiate_trading_halt(
                f"Daily loss limit breached: {limit.current_value:.2%}",
                duration_hours=24,
            )

        elif limit.limit_type == "drawdown" and self._rm_config.halt_on_drawdown:
            await self._initiate_trading_halt(
                f"Maximum drawdown breached: {limit.current_value:.2%}",
                duration_hours=None,
            )

        elif limit.limit_type == "exposure" and self._rm_config.auto_deleverage:
            await self._trigger_deleveraging(context)

    async def _initiate_trading_halt(
        self,
        reason: str,
        duration_hours: Optional[float] = None,
        symbols: Optional[list[str]] = None,
    ) -> TradingHalt:
        """Initiate a trading halt."""
        expires_at = None
        if duration_hours:
            expires_at = now_utc() + timedelta(hours=duration_hours)

        halt = TradingHalt(
            reason=reason,
            started_at=now_utc(),
            expires_at=expires_at,
            affected_symbols=symbols or [],
        )

        self._trading_halts.append(halt)

        self.create_alert(
            risk_type=RiskType.PORTFOLIO,
            risk_level=RiskLevel.CRITICAL,
            title="Trading Halt Initiated",
            message=reason,
            recommended_action="Review positions and risk parameters before resuming",
        )

        logger.critical(f"Trading halt initiated: {reason}")

        return halt

    def lift_trading_halt(self, halt_id: str) -> bool:
        """Lift a trading halt."""
        for halt in self._trading_halts:
            if halt.halt_id == halt_id:
                halt.is_active = False
                logger.info(f"Trading halt lifted: {halt.reason}")
                return True
        return False

    def is_trading_halted(self, symbol: Optional[str] = None) -> bool:
        """Check if trading is halted."""
        current_time = now_utc()

        for halt in self._trading_halts:
            if not halt.is_active:
                continue

            if halt.expires_at and current_time > halt.expires_at:
                halt.is_active = False
                continue

            if not halt.affected_symbols:
                return True

            if symbol and symbol in halt.affected_symbols:
                return True

        return False

    async def _trigger_deleveraging(self, context: RiskContext) -> list[dict]:
        """Trigger automated deleveraging."""
        if not self._rm_config.auto_deleverage:
            return []

        target_leverage = self._rm_config.max_total_exposure_pct * self._rm_config.deleverage_threshold_pct
        current_leverage = context.leverage_ratio

        if current_leverage <= target_leverage:
            return []

        reduction_needed = 1 - (target_leverage / current_leverage)

        positions_to_reduce: list[dict] = []
        sorted_positions = sorted(
            context.positions,
            key=lambda p: abs(p.get("unrealized_pnl_pct", 0)),
            reverse=True,
        )

        for position in sorted_positions:
            reduce_pct = min(reduction_needed, 0.5)

            positions_to_reduce.append({
                "symbol": position.get("symbol"),
                "current_qty": position.get("quantity", 0),
                "reduce_pct": reduce_pct,
                "reason": "auto_deleverage",
            })

        logger.warning(f"Deleveraging triggered: {len(positions_to_reduce)} positions")

        return positions_to_reduce

    def _generate_recommendations(
        self,
        overall_level: RiskLevel,
        market_risk: float,
        position_risk: float,
        portfolio_risk: float,
        liquidity_risk: float,
    ) -> list[str]:
        """Generate risk management recommendations."""
        recommendations: list[str] = []

        if overall_level in [RiskLevel.EXTREME, RiskLevel.CRITICAL]:
            recommendations.append("Consider reducing overall exposure immediately")
            recommendations.append("Review all stop-loss levels")

        if market_risk > 60:
            recommendations.append("Market conditions elevated - consider defensive positioning")

        if position_risk > 50:
            recommendations.append("Review concentrated positions for potential reduction")

        if portfolio_risk > 50:
            recommendations.append("Portfolio risk elevated - evaluate hedging strategies")

        if liquidity_risk > 40:
            recommendations.append("Increase cash reserves for improved liquidity")

        return recommendations

    def can_open_position(
        self,
        symbol: str,
        size: float,
        context: RiskContext,
    ) -> tuple[bool, str]:
        """
        Check if a new position can be opened.

        Args:
            symbol: Trading symbol
            size: Position size (notional value)
            context: Risk context

        Returns:
            Tuple of (can_open, reason)
        """
        if self.is_trading_halted(symbol):
            return False, "Trading is currently halted"

        account_value = context.account_value
        if account_value <= 0:
            return False, "Invalid account value"

        position_pct = size / account_value
        if position_pct > self._rm_config.max_position_size_pct:
            return False, f"Position size {position_pct:.2%} exceeds limit {self._rm_config.max_position_size_pct:.2%}"

        new_exposure = context.leverage_ratio + (size / account_value)
        if new_exposure > self._rm_config.max_total_exposure_pct:
            return False, f"Total exposure {new_exposure:.2%} would exceed limit"

        if context.current_drawdown > self._rm_config.max_drawdown_pct * 0.9:
            return False, "Drawdown near maximum - reduce risk"

        return True, "OK"

    @property
    def position_sizer(self) -> PositionSizer:
        """Get position sizer instance."""
        return self._position_sizer

    def get_active_halts(self) -> list[TradingHalt]:
        """Get active trading halts."""
        return [h for h in self._trading_halts if h.is_active]

    def get_risk_summary(self) -> dict:
        """Get risk management summary."""
        return {
            "enabled": self._is_enabled,
            "peak_equity": self._peak_equity,
            "current_drawdown": self._calculate_current_drawdown(
                self._equity_history[-1][1] if self._equity_history else 0
            ),
            "active_halts": len([h for h in self._trading_halts if h.is_active]),
            "breached_limits": len([l for l in self._limits.values() if l.is_breached]),
            "active_alerts": len([a for a in self._alerts if not a.resolved]),
            "equity_history_days": len(self._equity_history),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"RiskManager(limits={len(self._limits)}, alerts={len(self._alerts)})"

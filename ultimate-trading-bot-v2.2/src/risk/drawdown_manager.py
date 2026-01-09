"""
Drawdown Manager Module for Ultimate Trading Bot v2.2.

This module provides drawdown monitoring, analysis, and
automated response mechanisms.
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
    RiskAssessment,
    RiskContext,
    calculate_max_drawdown,
)
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class DrawdownPeriod(BaseModel):
    """Model for a drawdown period."""

    period_id: str = Field(default_factory=generate_uuid)
    start_date: datetime
    end_date: Optional[datetime] = None
    peak_value: float
    trough_value: float = Field(default=0.0)
    current_value: float = Field(default=0.0)
    max_drawdown: float = Field(default=0.0)
    duration_days: int = Field(default=0)
    is_active: bool = Field(default=True)
    recovery_date: Optional[datetime] = None
    recovery_days: Optional[int] = None


class DrawdownStats(BaseModel):
    """Model for drawdown statistics."""

    max_drawdown: float = Field(default=0.0)
    current_drawdown: float = Field(default=0.0)
    avg_drawdown: float = Field(default=0.0)
    drawdown_count: int = Field(default=0)
    avg_recovery_days: float = Field(default=0.0)
    longest_drawdown_days: int = Field(default=0)
    time_in_drawdown_pct: float = Field(default=0.0)
    current_drawdown_days: int = Field(default=0)
    peak_equity: float = Field(default=0.0)
    current_equity: float = Field(default=0.0)


class DrawdownManagerConfig(RiskConfig):
    """Configuration for drawdown manager."""

    warning_threshold_pct: float = Field(default=0.05, ge=0.01, le=0.2)
    critical_threshold_pct: float = Field(default=0.10, ge=0.05, le=0.3)
    max_drawdown_pct: float = Field(default=0.15, ge=0.05, le=0.5)
    emergency_threshold_pct: float = Field(default=0.20, ge=0.1, le=0.5)

    reduction_at_warning: float = Field(default=0.0, ge=0.0, le=0.5)
    reduction_at_critical: float = Field(default=0.25, ge=0.0, le=0.5)
    reduction_at_max: float = Field(default=0.50, ge=0.0, le=0.75)
    halt_at_emergency: bool = Field(default=True)

    recovery_threshold_pct: float = Field(default=0.02, ge=0.005, le=0.1)
    min_recovery_days: int = Field(default=1, ge=0, le=30)

    lookback_days: int = Field(default=252, ge=30, le=1000)
    significant_drawdown_pct: float = Field(default=0.03, ge=0.01, le=0.1)

    use_time_based_recovery: bool = Field(default=True)
    time_recovery_factor: float = Field(default=0.1, ge=0.01, le=0.5)


class DrawdownManager(BaseRiskManager):
    """
    Drawdown monitoring and management.

    Features:
    - Real-time drawdown tracking
    - Multi-threshold alerts
    - Automated position reduction
    - Recovery monitoring
    - Historical analysis
    """

    def __init__(
        self,
        config: Optional[DrawdownManagerConfig] = None,
    ) -> None:
        """
        Initialize DrawdownManager.

        Args:
            config: Drawdown manager configuration
        """
        config = config or DrawdownManagerConfig()
        super().__init__(config)

        self._dd_config = config

        self._equity_history: list[tuple[datetime, float]] = []
        self._peak_equity: float = 0.0
        self._peak_date: Optional[datetime] = None
        self._trough_equity: float = float("inf")
        self._trough_date: Optional[datetime] = None

        self._drawdown_periods: list[DrawdownPeriod] = []
        self._current_period: Optional[DrawdownPeriod] = None

        self._last_reduction_level: Optional[str] = None
        self._position_reductions: list[dict] = []

        logger.info("DrawdownManager initialized")

    async def assess_risk(
        self,
        context: dict[str, Any],
    ) -> RiskAssessment:
        """
        Assess drawdown risk.

        Args:
            context: Risk assessment context

        Returns:
            Risk assessment result
        """
        risk_context = self._build_risk_context(context)

        self._update_equity_tracking(risk_context)

        stats = self.get_drawdown_stats()
        current_dd = stats.current_drawdown

        risk_score = self._calculate_drawdown_risk_score(stats)
        risk_level = self.calculate_risk_level(risk_score)

        metrics = [
            RiskMetric(
                name="current_drawdown",
                value=current_dd,
                unit="percent",
                timestamp=now_utc(),
                risk_level=self._get_drawdown_level(current_dd),
                threshold_warning=self._dd_config.warning_threshold_pct,
                threshold_critical=self._dd_config.critical_threshold_pct,
            ),
            RiskMetric(
                name="max_drawdown",
                value=stats.max_drawdown,
                unit="percent",
                timestamp=now_utc(),
                risk_level=self._get_drawdown_level(stats.max_drawdown),
            ),
            RiskMetric(
                name="drawdown_days",
                value=stats.current_drawdown_days,
                unit="days",
                timestamp=now_utc(),
                risk_level=RiskLevel.HIGH if stats.current_drawdown_days > 30 else RiskLevel.MEDIUM,
            ),
        ]

        for metric in metrics:
            self.record_metric(metric)

        recommendations = self._generate_drawdown_recommendations(stats)

        return RiskAssessment(
            timestamp=now_utc(),
            overall_risk_level=risk_level,
            overall_risk_score=risk_score,
            portfolio_risk_score=risk_score,
            metrics=metrics,
            recommendations=recommendations,
            metadata={
                "peak_equity": stats.peak_equity,
                "current_equity": stats.current_equity,
                "drawdown_count": stats.drawdown_count,
                "time_in_drawdown_pct": stats.time_in_drawdown_pct,
            },
        )

    async def check_limits(
        self,
        context: dict[str, Any],
    ) -> list[RiskAlert]:
        """
        Check drawdown limits.

        Args:
            context: Context for limit checking

        Returns:
            List of risk alerts
        """
        risk_context = self._build_risk_context(context)
        alerts: list[RiskAlert] = []

        self._update_equity_tracking(risk_context)

        stats = self.get_drawdown_stats()
        current_dd = stats.current_drawdown

        if current_dd >= self._dd_config.emergency_threshold_pct:
            alert = self.create_alert(
                risk_type=RiskType.DRAWDOWN,
                risk_level=RiskLevel.CRITICAL,
                title="EMERGENCY: Maximum Drawdown Exceeded",
                message=f"Drawdown at {current_dd:.2%} exceeds emergency threshold",
                current_value=current_dd,
                threshold_value=self._dd_config.emergency_threshold_pct,
                recommended_action="Halt all trading and review positions",
            )
            alerts.append(alert)

            if self._last_reduction_level != "emergency":
                self._last_reduction_level = "emergency"
                if self._dd_config.halt_at_emergency:
                    self._position_reductions.append({
                        "timestamp": now_utc(),
                        "level": "emergency",
                        "action": "halt",
                        "drawdown": current_dd,
                    })

        elif current_dd >= self._dd_config.max_drawdown_pct:
            alert = self.create_alert(
                risk_type=RiskType.DRAWDOWN,
                risk_level=RiskLevel.EXTREME,
                title="Maximum Drawdown Reached",
                message=f"Drawdown at {current_dd:.2%} reached maximum threshold",
                current_value=current_dd,
                threshold_value=self._dd_config.max_drawdown_pct,
                recommended_action=f"Reduce positions by {self._dd_config.reduction_at_max:.0%}",
            )
            alerts.append(alert)

            if self._last_reduction_level not in ["emergency", "max"]:
                self._last_reduction_level = "max"
                self._position_reductions.append({
                    "timestamp": now_utc(),
                    "level": "max",
                    "reduction_pct": self._dd_config.reduction_at_max,
                    "drawdown": current_dd,
                })

        elif current_dd >= self._dd_config.critical_threshold_pct:
            alert = self.create_alert(
                risk_type=RiskType.DRAWDOWN,
                risk_level=RiskLevel.HIGH,
                title="Critical Drawdown Level",
                message=f"Drawdown at {current_dd:.2%} above critical threshold",
                current_value=current_dd,
                threshold_value=self._dd_config.critical_threshold_pct,
                recommended_action=f"Consider reducing positions by {self._dd_config.reduction_at_critical:.0%}",
            )
            alerts.append(alert)

            if self._last_reduction_level not in ["emergency", "max", "critical"]:
                self._last_reduction_level = "critical"
                if self._dd_config.reduction_at_critical > 0:
                    self._position_reductions.append({
                        "timestamp": now_utc(),
                        "level": "critical",
                        "reduction_pct": self._dd_config.reduction_at_critical,
                        "drawdown": current_dd,
                    })

        elif current_dd >= self._dd_config.warning_threshold_pct:
            if self.config.alert_on_warning:
                alert = self.create_alert(
                    risk_type=RiskType.DRAWDOWN,
                    risk_level=RiskLevel.MEDIUM,
                    title="Drawdown Warning",
                    message=f"Drawdown at {current_dd:.2%} above warning threshold",
                    current_value=current_dd,
                    threshold_value=self._dd_config.warning_threshold_pct,
                    recommended_action="Monitor positions closely",
                )
                alerts.append(alert)

            if self._last_reduction_level not in ["emergency", "max", "critical", "warning"]:
                self._last_reduction_level = "warning"

        else:
            if self._last_reduction_level is not None:
                alert = self.create_alert(
                    risk_type=RiskType.DRAWDOWN,
                    risk_level=RiskLevel.LOW,
                    title="Drawdown Recovery",
                    message=f"Drawdown recovered to {current_dd:.2%}",
                    current_value=current_dd,
                    threshold_value=self._dd_config.warning_threshold_pct,
                    recommended_action="Continue monitoring",
                )
                alerts.append(alert)
                self._last_reduction_level = None

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
        )

    def _update_equity_tracking(self, context: RiskContext) -> None:
        """Update equity history and drawdown periods."""
        current_equity = context.account_value
        current_time = context.timestamp

        self._equity_history.append((current_time, current_equity))

        cutoff = current_time - timedelta(days=self._dd_config.lookback_days)
        self._equity_history = [
            (ts, val) for ts, val in self._equity_history
            if ts >= cutoff
        ]

        if current_equity > self._peak_equity:
            if self._current_period and self._current_period.is_active:
                self._current_period.is_active = False
                self._current_period.end_date = current_time
                self._current_period.recovery_date = current_time

                if self._current_period.start_date:
                    recovery_days = (current_time - self._current_period.start_date).days
                    self._current_period.recovery_days = recovery_days

                self._drawdown_periods.append(self._current_period)
                self._current_period = None

            self._peak_equity = current_equity
            self._peak_date = current_time
            self._trough_equity = current_equity
            self._trough_date = current_time

        elif current_equity < self._peak_equity:
            if current_equity < self._trough_equity:
                self._trough_equity = current_equity
                self._trough_date = current_time

            if self._current_period is None:
                self._current_period = DrawdownPeriod(
                    start_date=self._peak_date or current_time,
                    peak_value=self._peak_equity,
                    trough_value=current_equity,
                    current_value=current_equity,
                    max_drawdown=self._calculate_current_drawdown(current_equity),
                )
            else:
                current_dd = self._calculate_current_drawdown(current_equity)
                if current_dd > self._current_period.max_drawdown:
                    self._current_period.max_drawdown = current_dd
                    self._current_period.trough_value = current_equity

                self._current_period.current_value = current_equity
                self._current_period.duration_days = (
                    current_time - self._current_period.start_date
                ).days

    def _calculate_current_drawdown(self, current_equity: float) -> float:
        """Calculate current drawdown percentage."""
        if self._peak_equity <= 0:
            return 0.0

        return max(0, (self._peak_equity - current_equity) / self._peak_equity)

    def _calculate_drawdown_risk_score(self, stats: DrawdownStats) -> float:
        """Calculate risk score from drawdown stats."""
        score = 0.0

        dd_ratio = stats.current_drawdown / self._dd_config.max_drawdown_pct
        score += min(50, dd_ratio * 50)

        if stats.current_drawdown_days > 30:
            score += 20
        elif stats.current_drawdown_days > 14:
            score += 10

        if stats.time_in_drawdown_pct > 0.5:
            score += 15
        elif stats.time_in_drawdown_pct > 0.3:
            score += 8

        if stats.max_drawdown > self._dd_config.max_drawdown_pct:
            score += 15

        return min(100, score)

    def _get_drawdown_level(self, drawdown: float) -> RiskLevel:
        """Get risk level for a drawdown value."""
        if drawdown >= self._dd_config.emergency_threshold_pct:
            return RiskLevel.CRITICAL
        elif drawdown >= self._dd_config.max_drawdown_pct:
            return RiskLevel.EXTREME
        elif drawdown >= self._dd_config.critical_threshold_pct:
            return RiskLevel.HIGH
        elif drawdown >= self._dd_config.warning_threshold_pct:
            return RiskLevel.MEDIUM
        elif drawdown > 0:
            return RiskLevel.LOW
        return RiskLevel.MINIMAL

    def _generate_drawdown_recommendations(
        self,
        stats: DrawdownStats,
    ) -> list[str]:
        """Generate recommendations based on drawdown analysis."""
        recommendations: list[str] = []

        if stats.current_drawdown >= self._dd_config.critical_threshold_pct:
            recommendations.append("Consider reducing position sizes until recovery")
            recommendations.append("Review and tighten stop-loss levels")

        if stats.current_drawdown_days > 20:
            recommendations.append("Extended drawdown period - review strategy performance")

        if stats.avg_recovery_days > 30:
            recommendations.append("Historical recovery times are long - consider risk reduction")

        if stats.time_in_drawdown_pct > 0.4:
            recommendations.append("High time in drawdown suggests strategy review needed")

        if stats.drawdown_count > 5 and stats.avg_drawdown > 0.05:
            recommendations.append("Frequent significant drawdowns indicate high volatility")

        return recommendations

    def get_drawdown_stats(self) -> DrawdownStats:
        """Get comprehensive drawdown statistics."""
        current_equity = self._equity_history[-1][1] if self._equity_history else 0
        current_dd = self._calculate_current_drawdown(current_equity)

        max_dd = current_dd
        drawdowns = [p.max_drawdown for p in self._drawdown_periods]
        if drawdowns:
            max_dd = max(max_dd, max(drawdowns))

        avg_dd = sum(drawdowns) / len(drawdowns) if drawdowns else current_dd

        recovery_days = [
            p.recovery_days for p in self._drawdown_periods
            if p.recovery_days is not None
        ]
        avg_recovery = sum(recovery_days) / len(recovery_days) if recovery_days else 0

        longest_dd = 0
        if self._current_period:
            longest_dd = self._current_period.duration_days
        for period in self._drawdown_periods:
            if period.duration_days > longest_dd:
                longest_dd = period.duration_days

        total_dd_days = sum(p.duration_days for p in self._drawdown_periods)
        if self._current_period:
            total_dd_days += self._current_period.duration_days

        total_days = len(self._equity_history)
        time_in_dd = total_dd_days / total_days if total_days > 0 else 0

        current_dd_days = 0
        if self._current_period:
            current_dd_days = self._current_period.duration_days

        return DrawdownStats(
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            avg_drawdown=avg_dd,
            drawdown_count=len(self._drawdown_periods) + (1 if self._current_period else 0),
            avg_recovery_days=avg_recovery,
            longest_drawdown_days=longest_dd,
            time_in_drawdown_pct=time_in_dd,
            current_drawdown_days=current_dd_days,
            peak_equity=self._peak_equity,
            current_equity=current_equity,
        )

    def get_drawdown_history(self, limit: int = 20) -> list[DrawdownPeriod]:
        """Get drawdown period history."""
        periods = self._drawdown_periods.copy()
        if self._current_period:
            periods.append(self._current_period)

        return periods[-limit:]

    def get_pending_reductions(self) -> list[dict]:
        """Get pending position reductions."""
        cutoff = now_utc() - timedelta(hours=1)
        return [
            r for r in self._position_reductions
            if r["timestamp"] >= cutoff and r.get("reduction_pct", 0) > 0
        ]

    def reset_peak(self) -> None:
        """Reset peak equity tracking."""
        if self._equity_history:
            self._peak_equity = self._equity_history[-1][1]
            self._peak_date = self._equity_history[-1][0]
            self._trough_equity = self._peak_equity
            self._trough_date = self._peak_date

        logger.info(f"Peak reset to {self._peak_equity}")

    def get_drawdown_summary(self) -> dict:
        """Get drawdown management summary."""
        stats = self.get_drawdown_stats()

        return {
            "current_drawdown": stats.current_drawdown,
            "max_drawdown": stats.max_drawdown,
            "peak_equity": stats.peak_equity,
            "current_equity": stats.current_equity,
            "in_drawdown": self._current_period is not None,
            "drawdown_days": stats.current_drawdown_days,
            "reduction_level": self._last_reduction_level,
            "total_periods": stats.drawdown_count,
        }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_drawdown_stats()
        return f"DrawdownManager(current_dd={stats.current_drawdown:.2%}, max_dd={stats.max_drawdown:.2%})"

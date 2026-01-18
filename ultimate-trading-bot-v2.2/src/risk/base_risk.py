"""
Base Risk Module for Ultimate Trading Bot v2.2.

This module provides the foundational risk management classes and models
used throughout the risk management system.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.utils.helpers import generate_uuid


logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk level enumeration."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"
    CRITICAL = "critical"


class RiskType(str, Enum):
    """Risk type enumeration."""

    MARKET = "market"
    POSITION = "position"
    PORTFOLIO = "portfolio"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    LEVERAGE = "leverage"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    COUNTERPARTY = "counterparty"


class RiskMetric(BaseModel):
    """Model for a risk metric."""

    metric_id: str = Field(default_factory=generate_uuid)
    name: str
    value: float
    unit: str = Field(default="")
    timestamp: datetime
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    description: str = Field(default="")
    metadata: dict = Field(default_factory=dict)


class RiskAlert(BaseModel):
    """Model for a risk alert."""

    alert_id: str = Field(default_factory=generate_uuid)
    risk_type: RiskType
    risk_level: RiskLevel
    title: str
    message: str
    timestamp: datetime
    symbol: Optional[str] = None
    current_value: float = Field(default=0.0)
    threshold_value: float = Field(default=0.0)
    recommended_action: str = Field(default="")
    acknowledged: bool = Field(default=False)
    resolved: bool = Field(default=False)
    metadata: dict = Field(default_factory=dict)


class RiskLimit(BaseModel):
    """Model for a risk limit."""

    limit_id: str = Field(default_factory=generate_uuid)
    name: str
    limit_type: str
    current_value: float = Field(default=0.0)
    limit_value: float
    warning_threshold: float = Field(default=0.8)
    is_breached: bool = Field(default=False)
    is_warning: bool = Field(default=False)
    last_checked: Optional[datetime] = None
    description: str = Field(default="")


class RiskAssessment(BaseModel):
    """Model for comprehensive risk assessment."""

    assessment_id: str = Field(default_factory=generate_uuid)
    timestamp: datetime
    overall_risk_level: RiskLevel
    overall_risk_score: float = Field(ge=0.0, le=100.0)

    market_risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    position_risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    portfolio_risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    liquidity_risk_score: float = Field(default=0.0, ge=0.0, le=100.0)

    metrics: list[RiskMetric] = Field(default_factory=list)
    alerts: list[RiskAlert] = Field(default_factory=list)
    limits_status: list[RiskLimit] = Field(default_factory=list)

    recommendations: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class RiskConfig(BaseModel):
    """Base configuration for risk management."""

    enabled: bool = Field(default=True)
    check_interval_seconds: int = Field(default=60, ge=1, le=3600)

    max_position_size_pct: float = Field(default=0.10, ge=0.01, le=1.0)
    max_portfolio_risk_pct: float = Field(default=0.20, ge=0.01, le=1.0)
    max_single_loss_pct: float = Field(default=0.02, ge=0.001, le=0.1)
    max_daily_loss_pct: float = Field(default=0.05, ge=0.01, le=0.2)
    max_drawdown_pct: float = Field(default=0.15, ge=0.05, le=0.5)

    stop_loss_required: bool = Field(default=True)
    default_stop_loss_pct: float = Field(default=0.02, ge=0.005, le=0.1)

    alert_on_warning: bool = Field(default=True)
    alert_on_breach: bool = Field(default=True)

    auto_reduce_on_breach: bool = Field(default=False)
    auto_close_on_critical: bool = Field(default=False)


class BaseRiskManager(ABC):
    """
    Abstract base class for risk managers.

    Provides common functionality and interface for all
    risk management components.
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
    ) -> None:
        """
        Initialize BaseRiskManager.

        Args:
            config: Risk configuration
        """
        self.config = config or RiskConfig()
        self._is_enabled = self.config.enabled
        self._alerts: list[RiskAlert] = []
        self._metrics_history: list[RiskMetric] = []
        self._limits: dict[str, RiskLimit] = {}

        logger.info(f"{self.__class__.__name__} initialized")

    @property
    def is_enabled(self) -> bool:
        """Check if risk manager is enabled."""
        return self._is_enabled

    def enable(self) -> None:
        """Enable risk manager."""
        self._is_enabled = True
        logger.info(f"{self.__class__.__name__} enabled")

    def disable(self) -> None:
        """Disable risk manager."""
        self._is_enabled = False
        logger.info(f"{self.__class__.__name__} disabled")

    @abstractmethod
    async def assess_risk(
        self,
        context: dict[str, Any],
    ) -> RiskAssessment:
        """
        Assess current risk levels.

        Args:
            context: Risk assessment context

        Returns:
            Risk assessment result
        """
        pass

    @abstractmethod
    async def check_limits(
        self,
        context: dict[str, Any],
    ) -> list[RiskAlert]:
        """
        Check all risk limits.

        Args:
            context: Context for limit checking

        Returns:
            List of risk alerts if limits breached
        """
        pass

    def add_limit(self, limit: RiskLimit) -> None:
        """Add a risk limit."""
        self._limits[limit.limit_id] = limit
        logger.debug(f"Risk limit added: {limit.name}")

    def remove_limit(self, limit_id: str) -> bool:
        """Remove a risk limit."""
        if limit_id in self._limits:
            del self._limits[limit_id]
            return True
        return False

    def get_limits(self) -> list[RiskLimit]:
        """Get all risk limits."""
        return list(self._limits.values())

    def create_alert(
        self,
        risk_type: RiskType,
        risk_level: RiskLevel,
        title: str,
        message: str,
        symbol: Optional[str] = None,
        current_value: float = 0.0,
        threshold_value: float = 0.0,
        recommended_action: str = "",
        metadata: Optional[dict] = None,
    ) -> RiskAlert:
        """Create and store a risk alert."""
        from src.utils.date_utils import now_utc

        alert = RiskAlert(
            risk_type=risk_type,
            risk_level=risk_level,
            title=title,
            message=message,
            timestamp=now_utc(),
            symbol=symbol,
            current_value=current_value,
            threshold_value=threshold_value,
            recommended_action=recommended_action,
            metadata=metadata or {},
        )

        self._alerts.append(alert)

        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-1000:]

        logger.warning(f"Risk alert created: {title} - {risk_level.value}")

        return alert

    def get_alerts(
        self,
        risk_level: Optional[RiskLevel] = None,
        risk_type: Optional[RiskType] = None,
        unresolved_only: bool = False,
        limit: int = 100,
    ) -> list[RiskAlert]:
        """Get risk alerts with optional filtering."""
        alerts = self._alerts

        if risk_level:
            alerts = [a for a in alerts if a.risk_level == risk_level]

        if risk_type:
            alerts = [a for a in alerts if a.risk_type == risk_type]

        if unresolved_only:
            alerts = [a for a in alerts if not a.resolved]

        return alerts[-limit:]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
        return False

    def record_metric(self, metric: RiskMetric) -> None:
        """Record a risk metric."""
        self._metrics_history.append(metric)

        if len(self._metrics_history) > 5000:
            self._metrics_history = self._metrics_history[-5000:]

    def get_metrics_history(
        self,
        metric_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[RiskMetric]:
        """Get metrics history with optional filtering."""
        metrics = self._metrics_history

        if metric_name:
            metrics = [m for m in metrics if m.name == metric_name]

        return metrics[-limit:]

    def calculate_risk_level(self, score: float) -> RiskLevel:
        """Calculate risk level from score (0-100)."""
        if score <= 10:
            return RiskLevel.MINIMAL
        elif score <= 25:
            return RiskLevel.LOW
        elif score <= 50:
            return RiskLevel.MEDIUM
        elif score <= 75:
            return RiskLevel.HIGH
        elif score <= 90:
            return RiskLevel.EXTREME
        else:
            return RiskLevel.CRITICAL

    def check_limit(self, limit: RiskLimit, current_value: float) -> RiskAlert | None:
        """Check a single limit and return alert if breached."""
        from src.utils.date_utils import now_utc

        limit.current_value = current_value
        limit.last_checked = now_utc()

        warning_value = limit.limit_value * limit.warning_threshold

        if current_value >= limit.limit_value:
            limit.is_breached = True
            limit.is_warning = True

            return self.create_alert(
                risk_type=RiskType.PORTFOLIO,
                risk_level=RiskLevel.HIGH,
                title=f"Limit Breached: {limit.name}",
                message=f"{limit.name} limit breached. Current: {current_value:.2f}, Limit: {limit.limit_value:.2f}",
                current_value=current_value,
                threshold_value=limit.limit_value,
                recommended_action=f"Reduce exposure to bring {limit.name} below limit",
            )

        elif current_value >= warning_value:
            limit.is_breached = False
            limit.is_warning = True

            if self.config.alert_on_warning:
                return self.create_alert(
                    risk_type=RiskType.PORTFOLIO,
                    risk_level=RiskLevel.MEDIUM,
                    title=f"Limit Warning: {limit.name}",
                    message=f"{limit.name} approaching limit. Current: {current_value:.2f}, Limit: {limit.limit_value:.2f}",
                    current_value=current_value,
                    threshold_value=limit.limit_value,
                    recommended_action=f"Monitor {limit.name} closely",
                )

        else:
            limit.is_breached = False
            limit.is_warning = False

        return None

    def get_statistics(self) -> dict:
        """Get risk manager statistics."""
        active_alerts = [a for a in self._alerts if not a.resolved]
        breached_limits = [l for l in self._limits.values() if l.is_breached]

        return {
            "enabled": self._is_enabled,
            "total_alerts": len(self._alerts),
            "active_alerts": len(active_alerts),
            "total_limits": len(self._limits),
            "breached_limits": len(breached_limits),
            "metrics_recorded": len(self._metrics_history),
            "alerts_by_level": {
                level.value: sum(1 for a in self._alerts if a.risk_level == level)
                for level in RiskLevel
            },
        }


class RiskContext(BaseModel):
    """Context for risk calculations."""

    timestamp: datetime
    account_value: float = Field(ge=0)
    cash_balance: float = Field(ge=0)
    positions: list[dict] = Field(default_factory=list)
    open_orders: list[dict] = Field(default_factory=list)
    daily_pnl: float = Field(default=0.0)
    unrealized_pnl: float = Field(default=0.0)
    peak_value: float = Field(default=0.0)
    current_drawdown: float = Field(default=0.0)
    market_data: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    @property
    def total_position_value(self) -> float:
        """Calculate total position value."""
        return sum(
            abs(p.get("market_value", 0))
            for p in self.positions
        )

    @property
    def leverage_ratio(self) -> float:
        """Calculate leverage ratio."""
        if self.account_value == 0:
            return 0.0
        return self.total_position_value / self.account_value

    @property
    def position_count(self) -> int:
        """Get number of positions."""
        return len(self.positions)


def calculate_var(
    returns: list[float],
    confidence_level: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: List of historical returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: Calculation method (historical, parametric)

    Returns:
        VaR value as positive number
    """
    import numpy as np

    if not returns:
        return 0.0

    returns_array = np.array(returns)

    if method == "historical":
        var = np.percentile(returns_array, (1 - confidence_level) * 100)
    else:
        mean = np.mean(returns_array)
        std = np.std(returns_array)
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence_level)
        var = mean + z_score * std

    return abs(var)


def calculate_cvar(
    returns: list[float],
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

    Args:
        returns: List of historical returns
        confidence_level: Confidence level

    Returns:
        CVaR value as positive number
    """
    import numpy as np

    if not returns:
        return 0.0

    returns_array = np.array(returns)
    var = calculate_var(returns, confidence_level, "historical")

    tail_losses = returns_array[returns_array <= -var]

    if len(tail_losses) == 0:
        return var

    return abs(np.mean(tail_losses))


def calculate_max_drawdown(equity_curve: list[float]) -> tuple[float, int, int]:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: List of equity values

    Returns:
        Tuple of (max_drawdown, peak_index, trough_index)
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0, 0

    peak = equity_curve[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_peak_idx = 0
    max_dd_trough_idx = 0

    for i, value in enumerate(equity_curve):
        if value > peak:
            peak = value
            peak_idx = i

        drawdown = (peak - value) / peak if peak > 0 else 0

        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_peak_idx = peak_idx
            max_dd_trough_idx = i

    return max_dd, max_dd_peak_idx, max_dd_trough_idx


def calculate_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sharpe ratio
    """
    import numpy as np

    if not returns or len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)

    if std_return == 0:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess_return = mean_return - rf_per_period

    sharpe = (excess_return / std_return) * np.sqrt(periods_per_year)

    return sharpe


def calculate_sortino_ratio(
    returns: list[float],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sortino ratio (downside risk adjusted).

    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sortino ratio
    """
    import numpy as np

    if not returns or len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)
    mean_return = np.mean(returns_array)

    negative_returns = returns_array[returns_array < 0]
    if len(negative_returns) == 0:
        return float("inf")

    downside_std = np.std(negative_returns)
    if downside_std == 0:
        return float("inf")

    rf_per_period = risk_free_rate / periods_per_year
    excess_return = mean_return - rf_per_period

    sortino = (excess_return / downside_std) * np.sqrt(periods_per_year)

    return sortino

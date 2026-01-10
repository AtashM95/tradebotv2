"""
Risk Limits Management for Ultimate Trading Bot v2.2.

This module provides comprehensive risk limit definition, monitoring,
and enforcement including position limits, exposure limits, and loss limits.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field

from src.risk.base_risk import RiskAlert, RiskLevel, RiskType


logger = logging.getLogger(__name__)


class LimitType(str, Enum):
    """Types of risk limits."""

    POSITION_SIZE = "position_size"
    POSITION_COUNT = "position_count"
    SECTOR_EXPOSURE = "sector_exposure"
    ASSET_CLASS_EXPOSURE = "asset_class_exposure"
    GEOGRAPHIC_EXPOSURE = "geographic_exposure"
    SINGLE_NAME_EXPOSURE = "single_name_exposure"
    GROSS_EXPOSURE = "gross_exposure"
    NET_EXPOSURE = "net_exposure"
    LEVERAGE = "leverage"
    DAILY_LOSS = "daily_loss"
    WEEKLY_LOSS = "weekly_loss"
    MONTHLY_LOSS = "monthly_loss"
    DRAWDOWN = "drawdown"
    VAR = "var"
    VOLATILITY = "volatility"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    ORDER_SIZE = "order_size"
    DAILY_TURNOVER = "daily_turnover"
    BETA = "beta"


class LimitAction(str, Enum):
    """Actions to take when limit is breached."""

    WARN = "warn"
    BLOCK = "block"
    REDUCE = "reduce"
    LIQUIDATE = "liquidate"
    HALT_TRADING = "halt_trading"
    NOTIFY = "notify"


class LimitStatus(str, Enum):
    """Current status of a limit."""

    OK = "ok"
    WARNING = "warning"
    SOFT_BREACH = "soft_breach"
    HARD_BREACH = "hard_breach"
    CRITICAL = "critical"


class RiskLimit(BaseModel):
    """Definition of a risk limit."""

    name: str
    limit_type: LimitType
    description: str = ""

    soft_limit: float = Field(description="Soft limit threshold")
    hard_limit: float = Field(description="Hard limit threshold")
    critical_limit: float | None = Field(default=None, description="Critical threshold")

    warning_threshold: float = Field(default=0.8, description="Warning at % of soft limit")

    soft_action: LimitAction = Field(default=LimitAction.WARN)
    hard_action: LimitAction = Field(default=LimitAction.BLOCK)
    critical_action: LimitAction = Field(default=LimitAction.HALT_TRADING)

    is_percentage: bool = Field(default=False, description="Whether values are percentages")
    is_enabled: bool = Field(default=True, description="Whether limit is active")

    applies_to: list[str] = Field(
        default_factory=list,
        description="Symbols/sectors this limit applies to"
    )
    exemptions: list[str] = Field(
        default_factory=list,
        description="Exempt symbols/sectors"
    )

    cooldown_minutes: int = Field(default=0, description="Cooldown after breach")
    auto_reset: bool = Field(default=True, description="Auto reset after period")
    reset_period_hours: int = Field(default=24, description="Reset period")


class LimitCheck(BaseModel):
    """Result of checking a limit."""

    limit: RiskLimit
    timestamp: datetime = Field(default_factory=datetime.now)

    current_value: float
    status: LimitStatus

    utilization_pct: float = Field(default=0.0, description="% of soft limit used")
    distance_to_soft: float = Field(default=0.0, description="Distance to soft limit")
    distance_to_hard: float = Field(default=0.0, description="Distance to hard limit")

    is_breached: bool = Field(default=False)
    action_required: LimitAction | None = None

    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass
class LimitBreach:
    """Record of a limit breach."""

    limit_name: str
    limit_type: LimitType
    timestamp: datetime
    breach_value: float
    limit_value: float
    status: LimitStatus
    action_taken: LimitAction
    resolved: bool = False
    resolution_time: datetime | None = None


@dataclass
class LimitProfile:
    """Collection of limits for a profile."""

    name: str
    description: str = ""
    limits: dict[str, RiskLimit] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


class RiskLimitsManager:
    """
    Manages risk limits definition, monitoring, and enforcement.

    Provides comprehensive limit checking, breach handling,
    and limit management functionality.
    """

    def __init__(self):
        """Initialize risk limits manager."""
        self._limits: dict[str, RiskLimit] = {}
        self._profiles: dict[str, LimitProfile] = {}
        self._active_profile: str | None = None
        self._breach_history: list[LimitBreach] = []
        self._cooldowns: dict[str, datetime] = {}
        self._callbacks: dict[LimitAction, list[Callable]] = {
            action: [] for action in LimitAction
        }
        self._lock = asyncio.Lock()

        self._initialize_default_limits()

        logger.info("RiskLimitsManager initialized")

    def _initialize_default_limits(self) -> None:
        """Initialize default risk limits."""
        default_limits = [
            RiskLimit(
                name="max_position_size",
                limit_type=LimitType.POSITION_SIZE,
                description="Maximum single position size",
                soft_limit=50000.0,
                hard_limit=100000.0,
                critical_limit=150000.0,
            ),
            RiskLimit(
                name="max_position_count",
                limit_type=LimitType.POSITION_COUNT,
                description="Maximum number of positions",
                soft_limit=20,
                hard_limit=30,
            ),
            RiskLimit(
                name="max_sector_exposure",
                limit_type=LimitType.SECTOR_EXPOSURE,
                description="Maximum exposure to single sector",
                soft_limit=0.30,
                hard_limit=0.40,
                is_percentage=True,
            ),
            RiskLimit(
                name="max_single_name",
                limit_type=LimitType.SINGLE_NAME_EXPOSURE,
                description="Maximum single name exposure",
                soft_limit=0.10,
                hard_limit=0.15,
                critical_limit=0.20,
                is_percentage=True,
            ),
            RiskLimit(
                name="max_gross_exposure",
                limit_type=LimitType.GROSS_EXPOSURE,
                description="Maximum gross exposure",
                soft_limit=1.5,
                hard_limit=2.0,
                is_percentage=True,
            ),
            RiskLimit(
                name="max_leverage",
                limit_type=LimitType.LEVERAGE,
                description="Maximum leverage",
                soft_limit=1.5,
                hard_limit=2.0,
                critical_limit=3.0,
            ),
            RiskLimit(
                name="max_daily_loss",
                limit_type=LimitType.DAILY_LOSS,
                description="Maximum daily loss",
                soft_limit=0.02,
                hard_limit=0.03,
                critical_limit=0.05,
                is_percentage=True,
                hard_action=LimitAction.REDUCE,
                critical_action=LimitAction.HALT_TRADING,
            ),
            RiskLimit(
                name="max_weekly_loss",
                limit_type=LimitType.WEEKLY_LOSS,
                description="Maximum weekly loss",
                soft_limit=0.05,
                hard_limit=0.08,
                is_percentage=True,
            ),
            RiskLimit(
                name="max_drawdown",
                limit_type=LimitType.DRAWDOWN,
                description="Maximum drawdown",
                soft_limit=0.10,
                hard_limit=0.15,
                critical_limit=0.20,
                is_percentage=True,
                critical_action=LimitAction.HALT_TRADING,
            ),
            RiskLimit(
                name="max_var",
                limit_type=LimitType.VAR,
                description="Maximum daily VaR (95%)",
                soft_limit=0.02,
                hard_limit=0.03,
                is_percentage=True,
            ),
            RiskLimit(
                name="max_volatility",
                limit_type=LimitType.VOLATILITY,
                description="Maximum portfolio volatility",
                soft_limit=0.20,
                hard_limit=0.30,
                is_percentage=True,
            ),
            RiskLimit(
                name="max_concentration",
                limit_type=LimitType.CONCENTRATION,
                description="Maximum concentration (HHI)",
                soft_limit=0.20,
                hard_limit=0.30,
            ),
            RiskLimit(
                name="max_order_size",
                limit_type=LimitType.ORDER_SIZE,
                description="Maximum single order size",
                soft_limit=25000.0,
                hard_limit=50000.0,
            ),
            RiskLimit(
                name="max_daily_turnover",
                limit_type=LimitType.DAILY_TURNOVER,
                description="Maximum daily turnover",
                soft_limit=0.50,
                hard_limit=1.0,
                is_percentage=True,
            ),
            RiskLimit(
                name="max_beta",
                limit_type=LimitType.BETA,
                description="Maximum portfolio beta",
                soft_limit=1.2,
                hard_limit=1.5,
            ),
        ]

        for limit in default_limits:
            self._limits[limit.name] = limit

        default_profile = LimitProfile(
            name="default",
            description="Default risk limits profile",
            limits={limit.name: limit for limit in default_limits},
        )
        self._profiles["default"] = default_profile
        self._active_profile = "default"

    async def check_limit(
        self,
        limit_name: str,
        current_value: float,
        context: dict[str, Any] | None = None,
    ) -> LimitCheck:
        """
        Check a specific limit against current value.

        Args:
            limit_name: Name of limit to check
            current_value: Current value to check
            context: Additional context

        Returns:
            LimitCheck result
        """
        limit = self._limits.get(limit_name)
        if not limit:
            return LimitCheck(
                limit=RiskLimit(
                    name=limit_name,
                    limit_type=LimitType.POSITION_SIZE,
                    soft_limit=0,
                    hard_limit=0,
                ),
                current_value=current_value,
                status=LimitStatus.OK,
                message=f"Limit {limit_name} not found",
            )

        if not limit.is_enabled:
            return LimitCheck(
                limit=limit,
                current_value=current_value,
                status=LimitStatus.OK,
                message="Limit is disabled",
            )

        if limit_name in self._cooldowns:
            if datetime.now() < self._cooldowns[limit_name]:
                return LimitCheck(
                    limit=limit,
                    current_value=current_value,
                    status=LimitStatus.OK,
                    message="Limit in cooldown period",
                )

        status = LimitStatus.OK
        action_required = None
        message = ""

        utilization = current_value / limit.soft_limit if limit.soft_limit > 0 else 0
        dist_to_soft = limit.soft_limit - current_value
        dist_to_hard = limit.hard_limit - current_value

        warning_level = limit.soft_limit * limit.warning_threshold

        if limit.critical_limit and current_value >= limit.critical_limit:
            status = LimitStatus.CRITICAL
            action_required = limit.critical_action
            message = f"CRITICAL: {limit.name} at {current_value:.2f} (limit: {limit.critical_limit:.2f})"
        elif current_value >= limit.hard_limit:
            status = LimitStatus.HARD_BREACH
            action_required = limit.hard_action
            message = f"HARD BREACH: {limit.name} at {current_value:.2f} (limit: {limit.hard_limit:.2f})"
        elif current_value >= limit.soft_limit:
            status = LimitStatus.SOFT_BREACH
            action_required = limit.soft_action
            message = f"SOFT BREACH: {limit.name} at {current_value:.2f} (limit: {limit.soft_limit:.2f})"
        elif current_value >= warning_level:
            status = LimitStatus.WARNING
            action_required = LimitAction.WARN
            message = f"WARNING: {limit.name} at {utilization:.1%} of limit"
        else:
            message = f"{limit.name} OK: {current_value:.2f} ({utilization:.1%} utilized)"

        is_breached = status in [
            LimitStatus.SOFT_BREACH,
            LimitStatus.HARD_BREACH,
            LimitStatus.CRITICAL,
        ]

        check_result = LimitCheck(
            limit=limit,
            current_value=current_value,
            status=status,
            utilization_pct=utilization,
            distance_to_soft=dist_to_soft,
            distance_to_hard=dist_to_hard,
            is_breached=is_breached,
            action_required=action_required,
            message=message,
            details=context or {},
        )

        if is_breached:
            await self._handle_breach(check_result)

        return check_result

    async def check_all_limits(
        self,
        values: dict[str, float],
        context: dict[str, Any] | None = None,
    ) -> list[LimitCheck]:
        """
        Check all limits against provided values.

        Args:
            values: Dictionary of limit names to current values
            context: Additional context

        Returns:
            List of LimitCheck results
        """
        results: list[LimitCheck] = []

        for limit_name, value in values.items():
            result = await self.check_limit(limit_name, value, context)
            results.append(result)

        return results

    async def _handle_breach(self, check: LimitCheck) -> None:
        """Handle a limit breach."""
        async with self._lock:
            breach = LimitBreach(
                limit_name=check.limit.name,
                limit_type=check.limit.limit_type,
                timestamp=datetime.now(),
                breach_value=check.current_value,
                limit_value=check.limit.soft_limit
                if check.status == LimitStatus.SOFT_BREACH
                else check.limit.hard_limit,
                status=check.status,
                action_taken=check.action_required or LimitAction.WARN,
            )
            self._breach_history.append(breach)

            if check.limit.cooldown_minutes > 0:
                self._cooldowns[check.limit.name] = datetime.now() + timedelta(
                    minutes=check.limit.cooldown_minutes
                )

        if check.action_required:
            callbacks = self._callbacks.get(check.action_required, [])
            for callback in callbacks:
                try:
                    await callback(check)
                except Exception as e:
                    logger.error(f"Callback error for {check.action_required}: {e}")

        logger.warning(
            f"Limit breach: {check.limit.name} - {check.status.value} "
            f"(value: {check.current_value}, action: {check.action_required})"
        )

    def add_limit(self, limit: RiskLimit) -> None:
        """
        Add a new risk limit.

        Args:
            limit: Risk limit to add
        """
        self._limits[limit.name] = limit
        logger.info(f"Added limit: {limit.name}")

    def update_limit(
        self,
        name: str,
        soft_limit: float | None = None,
        hard_limit: float | None = None,
        critical_limit: float | None = None,
        is_enabled: bool | None = None,
    ) -> bool:
        """
        Update an existing limit.

        Args:
            name: Limit name
            soft_limit: New soft limit
            hard_limit: New hard limit
            critical_limit: New critical limit
            is_enabled: Enable/disable

        Returns:
            Whether update was successful
        """
        if name not in self._limits:
            return False

        limit = self._limits[name]

        if soft_limit is not None:
            limit.soft_limit = soft_limit
        if hard_limit is not None:
            limit.hard_limit = hard_limit
        if critical_limit is not None:
            limit.critical_limit = critical_limit
        if is_enabled is not None:
            limit.is_enabled = is_enabled

        logger.info(f"Updated limit: {name}")
        return True

    def remove_limit(self, name: str) -> bool:
        """
        Remove a limit.

        Args:
            name: Limit name

        Returns:
            Whether removal was successful
        """
        if name in self._limits:
            del self._limits[name]
            logger.info(f"Removed limit: {name}")
            return True
        return False

    def enable_limit(self, name: str) -> bool:
        """Enable a limit."""
        return self.update_limit(name, is_enabled=True)

    def disable_limit(self, name: str) -> bool:
        """Disable a limit."""
        return self.update_limit(name, is_enabled=False)

    def create_profile(
        self,
        name: str,
        description: str = "",
        base_profile: str | None = None,
    ) -> LimitProfile:
        """
        Create a new limit profile.

        Args:
            name: Profile name
            description: Profile description
            base_profile: Base profile to copy from

        Returns:
            Created LimitProfile
        """
        if base_profile and base_profile in self._profiles:
            limits = dict(self._profiles[base_profile].limits)
        else:
            limits = {}

        profile = LimitProfile(
            name=name,
            description=description,
            limits=limits,
        )

        self._profiles[name] = profile
        logger.info(f"Created profile: {name}")

        return profile

    def activate_profile(self, name: str) -> bool:
        """
        Activate a limit profile.

        Args:
            name: Profile name

        Returns:
            Whether activation was successful
        """
        if name not in self._profiles:
            return False

        self._active_profile = name
        self._limits = dict(self._profiles[name].limits)
        logger.info(f"Activated profile: {name}")

        return True

    def register_callback(
        self,
        action: LimitAction,
        callback: Callable,
    ) -> None:
        """
        Register callback for limit action.

        Args:
            action: Limit action to trigger on
            callback: Async callback function
        """
        self._callbacks[action].append(callback)

    def get_limit(self, name: str) -> RiskLimit | None:
        """Get a limit by name."""
        return self._limits.get(name)

    def get_all_limits(self) -> dict[str, RiskLimit]:
        """Get all limits."""
        return dict(self._limits)

    def get_limits_by_type(self, limit_type: LimitType) -> list[RiskLimit]:
        """Get limits by type."""
        return [l for l in self._limits.values() if l.limit_type == limit_type]

    def get_breach_history(
        self,
        limit_name: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[LimitBreach]:
        """
        Get breach history.

        Args:
            limit_name: Filter by limit name
            since: Filter by time
            limit: Maximum records

        Returns:
            List of breaches
        """
        breaches = self._breach_history

        if limit_name:
            breaches = [b for b in breaches if b.limit_name == limit_name]

        if since:
            breaches = [b for b in breaches if b.timestamp >= since]

        return breaches[-limit:]

    async def get_limit_alerts(self) -> list[RiskAlert]:
        """
        Get current limit-related alerts.

        Returns:
            List of RiskAlerts
        """
        alerts: list[RiskAlert] = []

        for breach in self._breach_history[-10:]:
            if not breach.resolved:
                level = RiskLevel.MEDIUM
                if breach.status == LimitStatus.HARD_BREACH:
                    level = RiskLevel.HIGH
                elif breach.status == LimitStatus.CRITICAL:
                    level = RiskLevel.CRITICAL

                alerts.append(RiskAlert(
                    alert_type=RiskType.LIMIT_BREACH,
                    level=level,
                    message=f"{breach.limit_name} breach: {breach.breach_value:.2f}",
                    details={
                        "limit_value": breach.limit_value,
                        "breach_value": breach.breach_value,
                        "action": breach.action_taken.value,
                    },
                    requires_action=breach.status in [
                        LimitStatus.HARD_BREACH,
                        LimitStatus.CRITICAL,
                    ],
                ))

        return alerts

    async def get_limits_summary(self) -> dict[str, Any]:
        """
        Get limits summary.

        Returns:
            Summary dictionary
        """
        enabled_count = sum(1 for l in self._limits.values() if l.is_enabled)
        breaches_24h = [
            b for b in self._breach_history
            if b.timestamp > datetime.now() - timedelta(hours=24)
        ]
        unresolved = [b for b in self._breach_history if not b.resolved]

        by_type: dict[str, int] = {}
        for limit in self._limits.values():
            t = limit.limit_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "timestamp": datetime.now().isoformat(),
            "active_profile": self._active_profile,
            "total_limits": len(self._limits),
            "enabled_limits": enabled_count,
            "limits_by_type": by_type,
            "breaches_24h": len(breaches_24h),
            "unresolved_breaches": len(unresolved),
            "limits_in_cooldown": len(self._cooldowns),
            "profiles_available": list(self._profiles.keys()),
        }

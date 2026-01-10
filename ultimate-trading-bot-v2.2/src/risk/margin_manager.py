"""
Margin and Leverage Management for Ultimate Trading Bot v2.2.

This module provides comprehensive margin management including
leverage control, margin call prevention, and maintenance margin monitoring.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.risk.base_risk import RiskAlert, RiskLevel, RiskType


logger = logging.getLogger(__name__)


class MarginType(str, Enum):
    """Types of margin accounts."""

    CASH = "cash"
    REG_T = "reg_t"
    PORTFOLIO = "portfolio"
    PRIME = "prime"


class MarginCallType(str, Enum):
    """Types of margin calls."""

    HOUSE_CALL = "house_call"
    FED_CALL = "fed_call"
    MAINTENANCE_CALL = "maintenance_call"
    DAY_TRADE_CALL = "day_trade_call"
    EXCHANGE_CALL = "exchange_call"


class LeverageLevel(str, Enum):
    """Leverage level classifications."""

    NO_LEVERAGE = "no_leverage"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class MarginManagerConfig(BaseModel):
    """Configuration for margin management."""

    model_config = {"arbitrary_types_allowed": True}

    margin_type: MarginType = Field(default=MarginType.REG_T, description="Account margin type")
    max_leverage: float = Field(default=2.0, description="Maximum allowed leverage")
    target_leverage: float = Field(default=1.5, description="Target leverage level")
    min_margin_buffer: float = Field(default=0.25, description="Minimum margin buffer %")
    maintenance_margin_pct: float = Field(default=0.25, description="Maintenance margin %")
    initial_margin_pct: float = Field(default=0.50, description="Initial margin %")
    day_trade_margin_pct: float = Field(default=0.25, description="Day trade margin %")
    overnight_margin_pct: float = Field(default=0.50, description="Overnight margin %")
    concentrated_position_threshold: float = Field(default=0.25, description="Concentration threshold")
    concentrated_margin_multiplier: float = Field(default=1.5, description="Extra margin for concentration")
    margin_call_warning_pct: float = Field(default=0.10, description="Margin call warning threshold")
    auto_delever_enabled: bool = Field(default=True, description="Enable auto-deleverage")
    auto_delever_threshold: float = Field(default=0.05, description="Auto-delever trigger threshold")
    check_interval_seconds: int = Field(default=60, description="Margin check interval")


class MarginRequirement(BaseModel):
    """Margin requirement for a position or order."""

    symbol: str
    position_value: float = Field(default=0.0, description="Position market value")

    initial_margin: float = Field(default=0.0, description="Initial margin required")
    maintenance_margin: float = Field(default=0.0, description="Maintenance margin required")
    day_trade_margin: float = Field(default=0.0, description="Day trade margin required")
    overnight_margin: float = Field(default=0.0, description="Overnight margin required")

    margin_pct_used: float = Field(default=0.0, description="Margin percentage used")
    is_concentrated: bool = Field(default=False, description="Is concentrated position")
    concentration_multiplier: float = Field(default=1.0, description="Concentration multiplier applied")

    special_requirements: dict[str, float] = Field(default_factory=dict)


class AccountMarginStatus(BaseModel):
    """Current margin status for the account."""

    timestamp: datetime = Field(default_factory=datetime.now)

    equity: float = Field(default=0.0, description="Account equity")
    cash: float = Field(default=0.0, description="Cash balance")
    market_value_long: float = Field(default=0.0, description="Long position value")
    market_value_short: float = Field(default=0.0, description="Short position value")
    total_market_value: float = Field(default=0.0, description="Total market value")

    initial_margin_used: float = Field(default=0.0, description="Initial margin used")
    maintenance_margin_used: float = Field(default=0.0, description="Maintenance margin used")
    available_margin: float = Field(default=0.0, description="Available margin")
    buying_power: float = Field(default=0.0, description="Current buying power")
    day_trade_buying_power: float = Field(default=0.0, description="Day trade buying power")

    leverage: float = Field(default=1.0, description="Current leverage ratio")
    leverage_level: LeverageLevel = Field(default=LeverageLevel.NO_LEVERAGE)

    margin_utilization: float = Field(default=0.0, description="Margin utilization %")
    margin_buffer: float = Field(default=1.0, description="Margin buffer %")
    distance_to_margin_call: float = Field(default=1.0, description="Distance to margin call %")

    is_margin_call: bool = Field(default=False, description="Currently in margin call")
    margin_call_type: MarginCallType | None = None
    margin_call_amount: float = Field(default=0.0, description="Amount needed to cure call")

    warnings: list[str] = Field(default_factory=list)


@dataclass
class MarginCallEvent:
    """Margin call event record."""

    timestamp: datetime
    call_type: MarginCallType
    amount_required: float
    equity_at_call: float
    maintenance_required: float
    is_resolved: bool = False
    resolution_time: datetime | None = None
    resolution_action: str | None = None


@dataclass
class LeverageHistory:
    """Historical leverage tracking."""

    timestamps: list[datetime] = field(default_factory=list)
    leverage_values: list[float] = field(default_factory=list)
    max_leverage: float = 0.0
    avg_leverage: float = 0.0
    time_over_target: timedelta = field(default_factory=lambda: timedelta(0))


class MarginManager:
    """
    Manages margin requirements and leverage control.

    Provides real-time margin monitoring, margin call prevention,
    and automatic deleverage functionality.
    """

    def __init__(self, config: MarginManagerConfig | None = None):
        """
        Initialize margin manager.

        Args:
            config: Margin management configuration
        """
        self.config = config or MarginManagerConfig()
        self._current_status: AccountMarginStatus | None = None
        self._position_requirements: dict[str, MarginRequirement] = {}
        self._margin_call_history: list[MarginCallEvent] = []
        self._leverage_history = LeverageHistory()
        self._lock = asyncio.Lock()
        self._monitoring_task: asyncio.Task | None = None

        logger.info(f"MarginManager initialized with {self.config.margin_type.value} margin")

    async def calculate_margin_requirement(
        self,
        symbol: str,
        quantity: float,
        price: float,
        is_short: bool = False,
        is_day_trade: bool = False,
        portfolio_value: float = 0.0,
    ) -> MarginRequirement:
        """
        Calculate margin requirement for a position.

        Args:
            symbol: Asset symbol
            quantity: Position quantity
            price: Current price
            is_short: Whether position is short
            is_day_trade: Whether this is a day trade
            portfolio_value: Total portfolio value for concentration calc

        Returns:
            MarginRequirement object
        """
        try:
            position_value = abs(quantity * price)

            base_initial = position_value * self.config.initial_margin_pct
            base_maintenance = position_value * self.config.maintenance_margin_pct

            is_concentrated = False
            concentration_mult = 1.0

            if portfolio_value > 0:
                concentration = position_value / portfolio_value
                if concentration > self.config.concentrated_position_threshold:
                    is_concentrated = True
                    concentration_mult = self.config.concentrated_margin_multiplier

            if is_short:
                base_initial *= 1.5
                base_maintenance *= 1.3

            initial_margin = base_initial * concentration_mult
            maintenance_margin = base_maintenance * concentration_mult

            if is_day_trade:
                day_trade_margin = position_value * self.config.day_trade_margin_pct
            else:
                day_trade_margin = 0.0

            overnight_margin = position_value * self.config.overnight_margin_pct

            margin_pct = initial_margin / position_value if position_value > 0 else 0.0

            requirement = MarginRequirement(
                symbol=symbol,
                position_value=position_value,
                initial_margin=initial_margin,
                maintenance_margin=maintenance_margin,
                day_trade_margin=day_trade_margin,
                overnight_margin=overnight_margin,
                margin_pct_used=margin_pct,
                is_concentrated=is_concentrated,
                concentration_multiplier=concentration_mult,
            )

            async with self._lock:
                self._position_requirements[symbol] = requirement

            return requirement

        except Exception as e:
            logger.error(f"Failed to calculate margin for {symbol}: {e}")
            return MarginRequirement(symbol=symbol)

    async def update_account_margin_status(
        self,
        account_data: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> AccountMarginStatus:
        """
        Update and return current account margin status.

        Args:
            account_data: Account information from broker
            positions: Current positions

        Returns:
            AccountMarginStatus object
        """
        try:
            equity = float(account_data.get("equity", 0.0))
            cash = float(account_data.get("cash", 0.0))
            buying_power = float(account_data.get("buying_power", 0.0))
            day_trade_bp = float(account_data.get("daytrading_buying_power", buying_power))

            long_value = 0.0
            short_value = 0.0
            total_initial = 0.0
            total_maintenance = 0.0

            for pos in positions:
                symbol = pos.get("symbol", "")
                qty = float(pos.get("qty", 0))
                price = float(pos.get("current_price", 0))
                is_short = qty < 0

                pos_value = abs(qty * price)

                if is_short:
                    short_value += pos_value
                else:
                    long_value += pos_value

                req = await self.calculate_margin_requirement(
                    symbol=symbol,
                    quantity=qty,
                    price=price,
                    is_short=is_short,
                    portfolio_value=equity,
                )

                total_initial += req.initial_margin
                total_maintenance += req.maintenance_margin

            total_market_value = long_value + short_value

            leverage = total_market_value / equity if equity > 0 else 0.0
            leverage_level = self._classify_leverage(leverage)

            available_margin = equity - total_initial
            margin_utilization = total_initial / equity if equity > 0 else 0.0

            margin_buffer = (equity - total_maintenance) / equity if equity > 0 else 0.0
            distance_to_call = margin_buffer - self.config.min_margin_buffer

            is_margin_call = equity < total_maintenance
            margin_call_type = None
            margin_call_amount = 0.0

            if is_margin_call:
                margin_call_type = MarginCallType.MAINTENANCE_CALL
                margin_call_amount = total_maintenance - equity

            warnings = self._generate_margin_warnings(
                margin_buffer=margin_buffer,
                leverage=leverage,
                distance_to_call=distance_to_call,
                margin_utilization=margin_utilization,
            )

            status = AccountMarginStatus(
                equity=equity,
                cash=cash,
                market_value_long=long_value,
                market_value_short=short_value,
                total_market_value=total_market_value,
                initial_margin_used=total_initial,
                maintenance_margin_used=total_maintenance,
                available_margin=available_margin,
                buying_power=buying_power,
                day_trade_buying_power=day_trade_bp,
                leverage=leverage,
                leverage_level=leverage_level,
                margin_utilization=margin_utilization,
                margin_buffer=margin_buffer,
                distance_to_margin_call=distance_to_call,
                is_margin_call=is_margin_call,
                margin_call_type=margin_call_type,
                margin_call_amount=margin_call_amount,
                warnings=warnings,
            )

            async with self._lock:
                self._current_status = status
                self._update_leverage_history(leverage)

                if is_margin_call:
                    self._record_margin_call(status)

            return status

        except Exception as e:
            logger.error(f"Failed to update margin status: {e}")
            return AccountMarginStatus(warnings=[f"Status update failed: {str(e)}"])

    def _classify_leverage(self, leverage: float) -> LeverageLevel:
        """Classify leverage level."""
        if leverage <= 1.0:
            return LeverageLevel.NO_LEVERAGE
        elif leverage <= 1.5:
            return LeverageLevel.LOW
        elif leverage <= 2.0:
            return LeverageLevel.MODERATE
        elif leverage <= 4.0:
            return LeverageLevel.HIGH
        else:
            return LeverageLevel.EXTREME

    def _generate_margin_warnings(
        self,
        margin_buffer: float,
        leverage: float,
        distance_to_call: float,
        margin_utilization: float,
    ) -> list[str]:
        """Generate margin warning messages."""
        warnings: list[str] = []

        if distance_to_call < self.config.margin_call_warning_pct:
            warnings.append(
                f"CRITICAL: Only {distance_to_call:.1%} buffer to margin call"
            )

        if leverage > self.config.max_leverage:
            warnings.append(
                f"Leverage {leverage:.2f}x exceeds maximum {self.config.max_leverage:.2f}x"
            )
        elif leverage > self.config.target_leverage:
            warnings.append(
                f"Leverage {leverage:.2f}x above target {self.config.target_leverage:.2f}x"
            )

        if margin_utilization > 0.9:
            warnings.append(f"High margin utilization: {margin_utilization:.1%}")
        elif margin_utilization > 0.75:
            warnings.append(f"Elevated margin utilization: {margin_utilization:.1%}")

        if margin_buffer < self.config.min_margin_buffer:
            warnings.append(
                f"Margin buffer {margin_buffer:.1%} below minimum {self.config.min_margin_buffer:.1%}"
            )

        return warnings

    def _update_leverage_history(self, leverage: float) -> None:
        """Update leverage history tracking."""
        now = datetime.now()
        self._leverage_history.timestamps.append(now)
        self._leverage_history.leverage_values.append(leverage)

        cutoff = now - timedelta(days=30)
        while (
            self._leverage_history.timestamps and
            self._leverage_history.timestamps[0] < cutoff
        ):
            self._leverage_history.timestamps.pop(0)
            self._leverage_history.leverage_values.pop(0)

        if self._leverage_history.leverage_values:
            self._leverage_history.max_leverage = max(self._leverage_history.leverage_values)
            self._leverage_history.avg_leverage = sum(self._leverage_history.leverage_values) / len(
                self._leverage_history.leverage_values
            )

    def _record_margin_call(self, status: AccountMarginStatus) -> None:
        """Record a margin call event."""
        if status.margin_call_type:
            event = MarginCallEvent(
                timestamp=datetime.now(),
                call_type=status.margin_call_type,
                amount_required=status.margin_call_amount,
                equity_at_call=status.equity,
                maintenance_required=status.maintenance_margin_used,
            )
            self._margin_call_history.append(event)
            logger.warning(
                f"Margin call recorded: {status.margin_call_type.value}, "
                f"amount required: ${status.margin_call_amount:,.2f}"
            )

    async def check_order_margin(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        is_day_trade: bool = False,
    ) -> dict[str, Any]:
        """
        Check if an order can be executed within margin limits.

        Args:
            symbol: Asset symbol
            quantity: Order quantity
            side: 'buy' or 'sell'
            price: Order price
            is_day_trade: Whether this is a day trade

        Returns:
            Dictionary with approval status and details
        """
        try:
            if not self._current_status:
                return {
                    "approved": False,
                    "reason": "No margin status available",
                }

            is_short = side == "sell"

            requirement = await self.calculate_margin_requirement(
                symbol=symbol,
                quantity=quantity,
                price=price,
                is_short=is_short,
                is_day_trade=is_day_trade,
                portfolio_value=self._current_status.equity,
            )

            margin_needed = requirement.initial_margin

            if is_day_trade:
                can_execute = self._current_status.day_trade_buying_power >= requirement.position_value
                bp_used = "day_trade_buying_power"
                bp_available = self._current_status.day_trade_buying_power
            else:
                can_execute = self._current_status.available_margin >= margin_needed
                bp_used = "available_margin"
                bp_available = self._current_status.available_margin

            new_leverage = (
                self._current_status.total_market_value + requirement.position_value
            ) / self._current_status.equity

            if new_leverage > self.config.max_leverage:
                can_execute = False

            warnings: list[str] = []

            if new_leverage > self.config.target_leverage:
                warnings.append(f"Order would increase leverage to {new_leverage:.2f}x")

            if requirement.is_concentrated:
                warnings.append("Position is concentrated - higher margin applied")

            post_order_buffer = (
                self._current_status.equity - self._current_status.maintenance_margin_used - margin_needed
            ) / self._current_status.equity

            if post_order_buffer < self.config.min_margin_buffer:
                warnings.append(
                    f"Order would reduce margin buffer to {post_order_buffer:.1%}"
                )

            return {
                "approved": can_execute,
                "margin_required": margin_needed,
                f"{bp_used}": bp_available,
                "buying_power_remaining": bp_available - margin_needed if can_execute else 0,
                "new_leverage": new_leverage,
                "post_order_margin_buffer": post_order_buffer,
                "warnings": warnings,
                "reason": None if can_execute else "Insufficient margin or leverage limit exceeded",
            }

        except Exception as e:
            logger.error(f"Failed to check order margin: {e}")
            return {
                "approved": False,
                "reason": f"Margin check failed: {str(e)}",
            }

    async def get_deleverage_recommendations(self) -> list[dict[str, Any]]:
        """
        Get recommendations for reducing leverage.

        Returns:
            List of deleverage recommendations
        """
        recommendations: list[dict[str, Any]] = []

        if not self._current_status:
            return recommendations

        if self._current_status.leverage <= self.config.target_leverage:
            return recommendations

        excess_leverage = self._current_status.leverage - self.config.target_leverage
        value_to_reduce = excess_leverage * self._current_status.equity

        sorted_positions = sorted(
            self._position_requirements.items(),
            key=lambda x: x[1].position_value,
            reverse=True,
        )

        remaining_reduction = value_to_reduce

        for symbol, req in sorted_positions:
            if remaining_reduction <= 0:
                break

            reduce_value = min(req.position_value * 0.5, remaining_reduction)
            reduce_pct = reduce_value / req.position_value if req.position_value > 0 else 0

            recommendations.append({
                "symbol": symbol,
                "current_value": req.position_value,
                "reduce_value": reduce_value,
                "reduce_percentage": reduce_pct,
                "margin_freed": reduce_value * self.config.initial_margin_pct,
                "is_concentrated": req.is_concentrated,
                "priority": "high" if req.is_concentrated else "medium",
            })

            remaining_reduction -= reduce_value

        return recommendations

    async def check_margin_alerts(self) -> list[RiskAlert]:
        """
        Check for margin-related risk alerts.

        Returns:
            List of margin risk alerts
        """
        alerts: list[RiskAlert] = []

        if not self._current_status:
            return alerts

        if self._current_status.is_margin_call:
            alerts.append(RiskAlert(
                alert_type=RiskType.MARGIN,
                level=RiskLevel.CRITICAL,
                message=f"MARGIN CALL: ${self._current_status.margin_call_amount:,.2f} required",
                details={
                    "call_type": self._current_status.margin_call_type.value if self._current_status.margin_call_type else "unknown",
                    "amount": self._current_status.margin_call_amount,
                },
                requires_action=True,
            ))

        if self._current_status.distance_to_margin_call < self.config.auto_delever_threshold:
            alerts.append(RiskAlert(
                alert_type=RiskType.MARGIN,
                level=RiskLevel.HIGH,
                message=f"Near margin call - only {self._current_status.distance_to_margin_call:.1%} buffer",
                details={"buffer": self._current_status.distance_to_margin_call},
                requires_action=True,
            ))
        elif self._current_status.distance_to_margin_call < self.config.margin_call_warning_pct:
            alerts.append(RiskAlert(
                alert_type=RiskType.MARGIN,
                level=RiskLevel.MEDIUM,
                message=f"Margin buffer low - {self._current_status.distance_to_margin_call:.1%}",
                details={"buffer": self._current_status.distance_to_margin_call},
            ))

        if self._current_status.leverage > self.config.max_leverage:
            alerts.append(RiskAlert(
                alert_type=RiskType.LEVERAGE,
                level=RiskLevel.HIGH,
                message=f"Leverage {self._current_status.leverage:.2f}x exceeds max {self.config.max_leverage:.2f}x",
                details={
                    "current": self._current_status.leverage,
                    "max": self.config.max_leverage,
                },
                requires_action=True,
            ))
        elif self._current_status.leverage > self.config.target_leverage:
            alerts.append(RiskAlert(
                alert_type=RiskType.LEVERAGE,
                level=RiskLevel.LOW,
                message=f"Leverage {self._current_status.leverage:.2f}x above target {self.config.target_leverage:.2f}x",
                details={
                    "current": self._current_status.leverage,
                    "target": self.config.target_leverage,
                },
            ))

        concentrated = [
            symbol for symbol, req in self._position_requirements.items()
            if req.is_concentrated
        ]
        if concentrated:
            alerts.append(RiskAlert(
                alert_type=RiskType.CONCENTRATION,
                level=RiskLevel.MEDIUM,
                message=f"Concentrated positions: {', '.join(concentrated)}",
                details={"symbols": concentrated},
            ))

        return alerts

    async def calculate_max_position_size(
        self,
        symbol: str,
        price: float,
        is_short: bool = False,
        is_day_trade: bool = False,
    ) -> dict[str, Any]:
        """
        Calculate maximum position size within margin constraints.

        Args:
            symbol: Asset symbol
            price: Current price
            is_short: Whether position would be short
            is_day_trade: Whether this is a day trade

        Returns:
            Dictionary with max size calculations
        """
        if not self._current_status:
            return {
                "max_shares": 0,
                "max_value": 0.0,
                "limiting_factor": "No margin status available",
            }

        try:
            if is_day_trade:
                available_bp = self._current_status.day_trade_buying_power
                margin_pct = self.config.day_trade_margin_pct
            else:
                available_bp = self._current_status.available_margin
                margin_pct = self.config.initial_margin_pct

            if is_short:
                margin_pct *= 1.5

            max_value_margin = available_bp / margin_pct

            max_new_market_value = (
                self.config.max_leverage * self._current_status.equity -
                self._current_status.total_market_value
            )
            max_value_leverage = max(0.0, max_new_market_value)

            concentration_limit = (
                self._current_status.equity * self.config.concentrated_position_threshold
            )

            max_value = min(max_value_margin, max_value_leverage, concentration_limit)
            max_shares = int(max_value / price) if price > 0 else 0

            if max_value == max_value_margin:
                limiting_factor = "Available margin"
            elif max_value == max_value_leverage:
                limiting_factor = "Leverage limit"
            else:
                limiting_factor = "Concentration limit"

            return {
                "max_shares": max_shares,
                "max_value": max_value,
                "limiting_factor": limiting_factor,
                "available_margin": available_bp,
                "leverage_capacity": max_value_leverage,
                "concentration_limit": concentration_limit,
            }

        except Exception as e:
            logger.error(f"Failed to calculate max position size: {e}")
            return {
                "max_shares": 0,
                "max_value": 0.0,
                "limiting_factor": f"Calculation error: {str(e)}",
            }

    def get_current_status(self) -> AccountMarginStatus | None:
        """Get current margin status."""
        return self._current_status

    def get_leverage_history(self) -> LeverageHistory:
        """Get leverage history."""
        return self._leverage_history

    def get_margin_call_history(self) -> list[MarginCallEvent]:
        """Get margin call history."""
        return self._margin_call_history

    async def get_margin_summary(self) -> dict[str, Any]:
        """
        Get comprehensive margin summary.

        Returns:
            Margin summary dictionary
        """
        if not self._current_status:
            return {"error": "No margin status available"}

        return {
            "timestamp": datetime.now().isoformat(),
            "equity": self._current_status.equity,
            "leverage": {
                "current": self._current_status.leverage,
                "level": self._current_status.leverage_level.value,
                "target": self.config.target_leverage,
                "max": self.config.max_leverage,
            },
            "margin": {
                "initial_used": self._current_status.initial_margin_used,
                "maintenance_used": self._current_status.maintenance_margin_used,
                "available": self._current_status.available_margin,
                "utilization": self._current_status.margin_utilization,
                "buffer": self._current_status.margin_buffer,
            },
            "buying_power": {
                "regular": self._current_status.buying_power,
                "day_trade": self._current_status.day_trade_buying_power,
            },
            "margin_call": {
                "is_active": self._current_status.is_margin_call,
                "type": self._current_status.margin_call_type.value if self._current_status.margin_call_type else None,
                "amount": self._current_status.margin_call_amount,
                "distance": self._current_status.distance_to_margin_call,
            },
            "history": {
                "max_leverage_30d": self._leverage_history.max_leverage,
                "avg_leverage_30d": self._leverage_history.avg_leverage,
                "margin_calls_30d": len([
                    mc for mc in self._margin_call_history
                    if mc.timestamp > datetime.now() - timedelta(days=30)
                ]),
            },
            "warnings": self._current_status.warnings,
        }

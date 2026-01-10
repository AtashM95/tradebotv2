"""
Trade Validation for Ultimate Trading Bot v2.2.

This module provides comprehensive pre-trade validation including
order validation, risk checks, and compliance verification.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.risk.base_risk import RiskLevel


logger = logging.getLogger(__name__)


class ValidationResult(str, Enum):
    """Trade validation result."""

    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"
    MODIFIED = "modified"


class RejectionReason(str, Enum):
    """Reasons for trade rejection."""

    INSUFFICIENT_BUYING_POWER = "insufficient_buying_power"
    EXCEEDS_POSITION_LIMIT = "exceeds_position_limit"
    EXCEEDS_ORDER_LIMIT = "exceeds_order_limit"
    EXCEEDS_DAILY_LOSS_LIMIT = "exceeds_daily_loss_limit"
    EXCEEDS_LEVERAGE_LIMIT = "exceeds_leverage_limit"
    MARKET_CLOSED = "market_closed"
    SYMBOL_RESTRICTED = "symbol_restricted"
    DUPLICATE_ORDER = "duplicate_order"
    INVALID_PRICE = "invalid_price"
    INVALID_QUANTITY = "invalid_quantity"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    CONCENTRATION_LIMIT = "concentration_limit"
    VOLATILITY_TOO_HIGH = "volatility_too_high"
    LIQUIDITY_TOO_LOW = "liquidity_too_low"
    COMPLIANCE_VIOLATION = "compliance_violation"
    MANUAL_OVERRIDE = "manual_override"
    TRADING_HALTED = "trading_halted"


class OrderType(str, Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(str, Enum):
    """Order sides."""

    BUY = "buy"
    SELL = "sell"


class TradeValidatorConfig(BaseModel):
    """Configuration for trade validation."""

    model_config = {"arbitrary_types_allowed": True}

    max_order_value: float = Field(default=100000.0, description="Maximum single order value")
    max_position_value: float = Field(default=250000.0, description="Maximum position value")
    max_position_pct: float = Field(default=0.25, description="Max position as % of portfolio")
    max_daily_orders: int = Field(default=100, description="Maximum daily orders")
    max_daily_volume: float = Field(default=500000.0, description="Maximum daily trading volume")
    max_daily_loss: float = Field(default=5000.0, description="Maximum daily loss")
    max_daily_loss_pct: float = Field(default=0.02, description="Maximum daily loss %")
    min_order_value: float = Field(default=1.0, description="Minimum order value")
    max_slippage_pct: float = Field(default=0.02, description="Maximum slippage tolerance")
    price_deviation_limit: float = Field(default=0.05, description="Max price deviation from last")
    duplicate_order_window_seconds: int = Field(default=5, description="Window for duplicate detection")
    market_hours_only: bool = Field(default=False, description="Only allow market hours trading")
    market_open: dt_time = Field(default_factory=lambda: dt_time(9, 30))
    market_close: dt_time = Field(default_factory=lambda: dt_time(16, 0))
    allow_extended_hours: bool = Field(default=True, description="Allow extended hours trading")
    restricted_symbols: list[str] = Field(default_factory=list, description="Restricted symbols")
    max_volatility: float = Field(default=0.10, description="Max daily volatility for trading")
    min_liquidity_volume: float = Field(default=10000.0, description="Minimum average volume")
    require_stop_loss: bool = Field(default=False, description="Require stop loss on entries")
    max_leverage: float = Field(default=2.0, description="Maximum leverage allowed")
    allow_short_selling: bool = Field(default=True, description="Allow short selling")


class OrderRequest(BaseModel):
    """Order request to validate."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: float | None = None
    stop_price: float | None = None

    time_in_force: str = Field(default="day")
    extended_hours: bool = Field(default=False)

    strategy_id: str | None = None
    signal_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResponse(BaseModel):
    """Response from trade validation."""

    request: OrderRequest
    timestamp: datetime = Field(default_factory=datetime.now)

    result: ValidationResult
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)

    approved: bool = Field(default=False)
    rejection_reasons: list[RejectionReason] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    modified_quantity: float | None = None
    modified_price: float | None = None

    estimated_cost: float = Field(default=0.0)
    estimated_slippage: float = Field(default=0.0)

    checks_passed: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)

    validation_time_ms: float = Field(default=0.0)


@dataclass
class DailyStats:
    """Daily trading statistics."""

    date: datetime
    order_count: int = 0
    total_volume: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class RecentOrder:
    """Recent order for duplicate detection."""

    symbol: str
    side: OrderSide
    quantity: float
    timestamp: datetime


class TradeValidator:
    """
    Validates trades before execution.

    Performs comprehensive pre-trade checks including risk limits,
    compliance rules, and market conditions.
    """

    def __init__(self, config: TradeValidatorConfig | None = None):
        """
        Initialize trade validator.

        Args:
            config: Validation configuration
        """
        self.config = config or TradeValidatorConfig()
        self._daily_stats = DailyStats(date=datetime.now())
        self._recent_orders: list[RecentOrder] = []
        self._portfolio_value: float = 0.0
        self._positions: dict[str, dict[str, Any]] = {}
        self._current_leverage: float = 1.0
        self._trading_halted: bool = False
        self._halt_reason: str | None = None
        self._lock = asyncio.Lock()

        logger.info("TradeValidator initialized")

    async def validate_order(
        self,
        order: OrderRequest,
        market_data: dict[str, Any],
    ) -> ValidationResponse:
        """
        Validate an order request.

        Args:
            order: Order request to validate
            market_data: Current market data for the symbol

        Returns:
            ValidationResponse with validation results
        """
        start_time = datetime.now()
        checks_passed: list[str] = []
        checks_failed: list[str] = []
        rejection_reasons: list[RejectionReason] = []
        warnings: list[str] = []

        try:
            current_price = market_data.get("price", 0.0)
            order_value = order.quantity * (order.limit_price or current_price)

            if self._trading_halted:
                checks_failed.append("trading_halted")
                rejection_reasons.append(RejectionReason.TRADING_HALTED)
            else:
                checks_passed.append("trading_active")

            quantity_valid, quantity_reason = self._validate_quantity(order, order_value)
            if quantity_valid:
                checks_passed.append("quantity_valid")
            else:
                checks_failed.append("quantity_invalid")
                rejection_reasons.append(quantity_reason)

            price_valid, price_reason, price_warning = self._validate_price(
                order, current_price
            )
            if price_valid:
                checks_passed.append("price_valid")
            else:
                checks_failed.append("price_invalid")
                rejection_reasons.append(price_reason)
            if price_warning:
                warnings.append(price_warning)

            if self._is_symbol_restricted(order.symbol):
                checks_failed.append("symbol_restricted")
                rejection_reasons.append(RejectionReason.SYMBOL_RESTRICTED)
            else:
                checks_passed.append("symbol_allowed")

            hours_valid, hours_reason = self._validate_market_hours(order)
            if hours_valid:
                checks_passed.append("market_hours_valid")
            else:
                checks_failed.append("market_hours_invalid")
                rejection_reasons.append(hours_reason)

            position_valid, position_reason = await self._validate_position_limits(
                order, order_value
            )
            if position_valid:
                checks_passed.append("position_limits_valid")
            else:
                checks_failed.append("position_limits_exceeded")
                rejection_reasons.append(position_reason)

            order_valid, order_reason = self._validate_order_limits(order_value)
            if order_valid:
                checks_passed.append("order_limits_valid")
            else:
                checks_failed.append("order_limits_exceeded")
                rejection_reasons.append(order_reason)

            daily_valid, daily_reason = self._validate_daily_limits(order_value)
            if daily_valid:
                checks_passed.append("daily_limits_valid")
            else:
                checks_failed.append("daily_limits_exceeded")
                rejection_reasons.append(daily_reason)

            leverage_valid, leverage_reason = self._validate_leverage(order_value)
            if leverage_valid:
                checks_passed.append("leverage_valid")
            else:
                checks_failed.append("leverage_exceeded")
                rejection_reasons.append(leverage_reason)

            if order.side == OrderSide.SELL:
                short_valid, short_reason = self._validate_short_selling(order)
                if short_valid:
                    checks_passed.append("short_selling_valid")
                else:
                    checks_failed.append("short_selling_invalid")
                    rejection_reasons.append(short_reason)

            duplicate, dup_reason = self._check_duplicate_order(order)
            if not duplicate:
                checks_passed.append("not_duplicate")
            else:
                checks_failed.append("duplicate_detected")
                rejection_reasons.append(dup_reason)

            volatility_ok, vol_warning = self._validate_volatility(market_data)
            if volatility_ok:
                checks_passed.append("volatility_acceptable")
            else:
                warnings.append(vol_warning)
                if market_data.get("volatility", 0) > self.config.max_volatility * 1.5:
                    rejection_reasons.append(RejectionReason.VOLATILITY_TOO_HIGH)
                    checks_failed.append("volatility_too_high")

            liquidity_ok, liq_warning = self._validate_liquidity(market_data, order.quantity)
            if liquidity_ok:
                checks_passed.append("liquidity_sufficient")
            else:
                warnings.append(liq_warning)
                if market_data.get("volume", float("inf")) < self.config.min_liquidity_volume * 0.5:
                    rejection_reasons.append(RejectionReason.LIQUIDITY_TOO_LOW)
                    checks_failed.append("liquidity_too_low")

            approved = len(rejection_reasons) == 0
            result = ValidationResult.APPROVED if approved else ValidationResult.REJECTED

            if approved and len(warnings) > 0:
                result = ValidationResult.APPROVED

            risk_level = self._assess_risk_level(
                rejection_reasons, warnings, order_value
            )

            slippage = self._estimate_slippage(order, market_data)

            validation_time = (datetime.now() - start_time).total_seconds() * 1000

            response = ValidationResponse(
                request=order,
                result=result,
                risk_level=risk_level,
                approved=approved,
                rejection_reasons=rejection_reasons,
                warnings=warnings,
                estimated_cost=order_value,
                estimated_slippage=slippage,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                validation_time_ms=validation_time,
            )

            if approved:
                await self._record_order(order, order_value)

            return response

        except Exception as e:
            logger.error(f"Validation error for {order.symbol}: {e}")
            return ValidationResponse(
                request=order,
                result=ValidationResult.REJECTED,
                risk_level=RiskLevel.HIGH,
                approved=False,
                rejection_reasons=[RejectionReason.COMPLIANCE_VIOLATION],
                warnings=[f"Validation error: {str(e)}"],
                checks_passed=checks_passed,
                checks_failed=checks_failed + ["validation_error"],
                validation_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    def _validate_quantity(
        self,
        order: OrderRequest,
        order_value: float,
    ) -> tuple[bool, RejectionReason | None]:
        """Validate order quantity."""
        if order.quantity <= 0:
            return False, RejectionReason.INVALID_QUANTITY

        if order_value < self.config.min_order_value:
            return False, RejectionReason.INVALID_QUANTITY

        if order_value > self.config.max_order_value:
            return False, RejectionReason.EXCEEDS_ORDER_LIMIT

        return True, None

    def _validate_price(
        self,
        order: OrderRequest,
        current_price: float,
    ) -> tuple[bool, RejectionReason | None, str | None]:
        """Validate order price."""
        warning = None

        if order.order_type == OrderType.MARKET:
            return True, None, None

        if order.limit_price is not None:
            if order.limit_price <= 0:
                return False, RejectionReason.INVALID_PRICE, None

            if current_price > 0:
                deviation = abs(order.limit_price - current_price) / current_price
                if deviation > self.config.price_deviation_limit:
                    warning = f"Price deviates {deviation:.1%} from current"
                    if deviation > self.config.price_deviation_limit * 2:
                        return False, RejectionReason.INVALID_PRICE, warning

        if order.stop_price is not None:
            if order.stop_price <= 0:
                return False, RejectionReason.INVALID_PRICE, warning

        return True, None, warning

    def _is_symbol_restricted(self, symbol: str) -> bool:
        """Check if symbol is restricted."""
        return symbol.upper() in [s.upper() for s in self.config.restricted_symbols]

    def _validate_market_hours(
        self,
        order: OrderRequest,
    ) -> tuple[bool, RejectionReason | None]:
        """Validate market hours."""
        if not self.config.market_hours_only:
            return True, None

        now = datetime.now().time()

        is_market_hours = self.config.market_open <= now <= self.config.market_close

        if is_market_hours:
            return True, None

        if order.extended_hours and self.config.allow_extended_hours:
            return True, None

        return False, RejectionReason.MARKET_CLOSED

    async def _validate_position_limits(
        self,
        order: OrderRequest,
        order_value: float,
    ) -> tuple[bool, RejectionReason | None]:
        """Validate position limits."""
        current_position = self._positions.get(order.symbol, {})
        current_value = current_position.get("market_value", 0.0)

        if order.side == OrderSide.BUY:
            new_value = current_value + order_value
        else:
            new_value = current_value - order_value

        if abs(new_value) > self.config.max_position_value:
            return False, RejectionReason.EXCEEDS_POSITION_LIMIT

        if self._portfolio_value > 0:
            position_pct = abs(new_value) / self._portfolio_value
            if position_pct > self.config.max_position_pct:
                return False, RejectionReason.CONCENTRATION_LIMIT

        return True, None

    def _validate_order_limits(
        self,
        order_value: float,
    ) -> tuple[bool, RejectionReason | None]:
        """Validate order limits."""
        if order_value > self.config.max_order_value:
            return False, RejectionReason.EXCEEDS_ORDER_LIMIT

        if self._daily_stats.order_count >= self.config.max_daily_orders:
            return False, RejectionReason.EXCEEDS_ORDER_LIMIT

        return True, None

    def _validate_daily_limits(
        self,
        order_value: float,
    ) -> tuple[bool, RejectionReason | None]:
        """Validate daily trading limits."""
        self._reset_daily_stats_if_needed()

        if self._daily_stats.total_volume + order_value > self.config.max_daily_volume:
            return False, RejectionReason.EXCEEDS_ORDER_LIMIT

        total_loss = -self._daily_stats.realized_pnl - self._daily_stats.unrealized_pnl
        if total_loss > self.config.max_daily_loss:
            return False, RejectionReason.EXCEEDS_DAILY_LOSS_LIMIT

        if self._portfolio_value > 0:
            loss_pct = total_loss / self._portfolio_value
            if loss_pct > self.config.max_daily_loss_pct:
                return False, RejectionReason.EXCEEDS_DAILY_LOSS_LIMIT

        return True, None

    def _validate_leverage(
        self,
        order_value: float,
    ) -> tuple[bool, RejectionReason | None]:
        """Validate leverage limits."""
        if self._portfolio_value <= 0:
            return True, None

        current_exposure = sum(
            abs(p.get("market_value", 0)) for p in self._positions.values()
        )
        new_exposure = current_exposure + order_value
        new_leverage = new_exposure / self._portfolio_value

        if new_leverage > self.config.max_leverage:
            return False, RejectionReason.EXCEEDS_LEVERAGE_LIMIT

        return True, None

    def _validate_short_selling(
        self,
        order: OrderRequest,
    ) -> tuple[bool, RejectionReason | None]:
        """Validate short selling."""
        if not self.config.allow_short_selling:
            current_position = self._positions.get(order.symbol, {})
            current_qty = current_position.get("quantity", 0)

            if order.quantity > current_qty:
                return False, RejectionReason.COMPLIANCE_VIOLATION

        return True, None

    def _check_duplicate_order(
        self,
        order: OrderRequest,
    ) -> tuple[bool, RejectionReason | None]:
        """Check for duplicate orders."""
        cutoff = datetime.now() - timedelta(
            seconds=self.config.duplicate_order_window_seconds
        )

        self._recent_orders = [
            o for o in self._recent_orders if o.timestamp > cutoff
        ]

        for recent in self._recent_orders:
            if (
                recent.symbol == order.symbol and
                recent.side == order.side and
                recent.quantity == order.quantity
            ):
                return True, RejectionReason.DUPLICATE_ORDER

        return False, None

    def _validate_volatility(
        self,
        market_data: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Validate market volatility."""
        volatility = market_data.get("volatility", 0.0)

        if volatility > self.config.max_volatility:
            return False, f"High volatility: {volatility:.1%}"

        return True, None

    def _validate_liquidity(
        self,
        market_data: dict[str, Any],
        quantity: float,
    ) -> tuple[bool, str | None]:
        """Validate market liquidity."""
        volume = market_data.get("volume", float("inf"))
        avg_volume = market_data.get("avg_volume", volume)

        if avg_volume < self.config.min_liquidity_volume:
            return False, f"Low liquidity: avg volume {avg_volume:,.0f}"

        participation = quantity / avg_volume if avg_volume > 0 else 1.0
        if participation > 0.1:
            return False, f"Order is {participation:.1%} of avg volume"

        return True, None

    def _assess_risk_level(
        self,
        rejection_reasons: list[RejectionReason],
        warnings: list[str],
        order_value: float,
    ) -> RiskLevel:
        """Assess overall risk level."""
        if len(rejection_reasons) > 0:
            return RiskLevel.HIGH

        risk_score = len(warnings)

        if order_value > self.config.max_order_value * 0.8:
            risk_score += 2

        if self._current_leverage > self.config.max_leverage * 0.8:
            risk_score += 2

        if risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        elif risk_score >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def _estimate_slippage(
        self,
        order: OrderRequest,
        market_data: dict[str, Any],
    ) -> float:
        """Estimate expected slippage."""
        if order.order_type != OrderType.MARKET:
            return 0.0

        spread = market_data.get("spread", 0.0)
        price = market_data.get("price", 0.0)
        volume = market_data.get("volume", 1e6)

        spread_slippage = spread / 2 if spread > 0 else price * 0.001

        participation = order.quantity / volume if volume > 0 else 0.0
        impact_slippage = price * participation * 0.1

        return spread_slippage + impact_slippage

    def _reset_daily_stats_if_needed(self) -> None:
        """Reset daily stats if new day."""
        today = datetime.now().date()
        if self._daily_stats.date.date() != today:
            self._daily_stats = DailyStats(date=datetime.now())
            logger.info("Daily stats reset for new trading day")

    async def _record_order(
        self,
        order: OrderRequest,
        order_value: float,
    ) -> None:
        """Record an approved order."""
        async with self._lock:
            self._daily_stats.order_count += 1
            self._daily_stats.total_volume += order_value

            self._recent_orders.append(RecentOrder(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                timestamp=datetime.now(),
            ))

    async def update_portfolio_state(
        self,
        portfolio_value: float,
        positions: dict[str, dict[str, Any]],
        leverage: float,
    ) -> None:
        """
        Update portfolio state for validation.

        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            leverage: Current leverage
        """
        async with self._lock:
            self._portfolio_value = portfolio_value
            self._positions = positions
            self._current_leverage = leverage

    async def update_daily_pnl(
        self,
        realized_pnl: float,
        unrealized_pnl: float,
    ) -> None:
        """
        Update daily P&L.

        Args:
            realized_pnl: Realized P&L
            unrealized_pnl: Unrealized P&L
        """
        async with self._lock:
            self._reset_daily_stats_if_needed()
            self._daily_stats.realized_pnl = realized_pnl
            self._daily_stats.unrealized_pnl = unrealized_pnl

    def halt_trading(self, reason: str) -> None:
        """
        Halt all trading.

        Args:
            reason: Reason for halt
        """
        self._trading_halted = True
        self._halt_reason = reason
        logger.warning(f"Trading halted: {reason}")

    def resume_trading(self) -> None:
        """Resume trading."""
        self._trading_halted = False
        self._halt_reason = None
        logger.info("Trading resumed")

    def add_restricted_symbol(self, symbol: str) -> None:
        """
        Add symbol to restricted list.

        Args:
            symbol: Symbol to restrict
        """
        if symbol.upper() not in [s.upper() for s in self.config.restricted_symbols]:
            self.config.restricted_symbols.append(symbol.upper())
            logger.info(f"Added {symbol} to restricted symbols")

    def remove_restricted_symbol(self, symbol: str) -> None:
        """
        Remove symbol from restricted list.

        Args:
            symbol: Symbol to unrestrict
        """
        self.config.restricted_symbols = [
            s for s in self.config.restricted_symbols
            if s.upper() != symbol.upper()
        ]
        logger.info(f"Removed {symbol} from restricted symbols")

    def get_daily_stats(self) -> DailyStats:
        """Get current daily statistics."""
        self._reset_daily_stats_if_needed()
        return self._daily_stats

    def is_trading_halted(self) -> tuple[bool, str | None]:
        """Check if trading is halted."""
        return self._trading_halted, self._halt_reason

    async def get_validation_summary(self) -> dict[str, Any]:
        """
        Get validation summary.

        Returns:
            Summary dictionary
        """
        self._reset_daily_stats_if_needed()

        return {
            "timestamp": datetime.now().isoformat(),
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "daily_stats": {
                "date": self._daily_stats.date.isoformat(),
                "order_count": self._daily_stats.order_count,
                "total_volume": self._daily_stats.total_volume,
                "realized_pnl": self._daily_stats.realized_pnl,
                "unrealized_pnl": self._daily_stats.unrealized_pnl,
                "orders_remaining": self.config.max_daily_orders - self._daily_stats.order_count,
                "volume_remaining": self.config.max_daily_volume - self._daily_stats.total_volume,
            },
            "limits": {
                "max_order_value": self.config.max_order_value,
                "max_position_value": self.config.max_position_value,
                "max_daily_orders": self.config.max_daily_orders,
                "max_daily_volume": self.config.max_daily_volume,
                "max_daily_loss": self.config.max_daily_loss,
                "max_leverage": self.config.max_leverage,
            },
            "portfolio": {
                "value": self._portfolio_value,
                "leverage": self._current_leverage,
                "position_count": len(self._positions),
            },
            "restricted_symbols": self.config.restricted_symbols,
        }

"""
Stop Loss Manager Module for Ultimate Trading Bot v2.2.

This module manages stop-loss orders including placement,
trailing stops, and dynamic adjustment.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.risk.base_risk import RiskConfig, RiskLevel, RiskType, RiskAlert
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class StopType(str, Enum):
    """Stop loss type enumeration."""

    FIXED = "fixed"
    PERCENTAGE = "percentage"
    ATR = "atr"
    TRAILING = "trailing"
    TRAILING_ATR = "trailing_atr"
    TIME_BASED = "time_based"
    BREAKEVEN = "breakeven"
    CHANDELIER = "chandelier"


class StopLossOrder(BaseModel):
    """Model for a stop loss order."""

    stop_id: str = Field(default_factory=generate_uuid)
    symbol: str
    stop_type: StopType
    entry_price: float
    stop_price: float
    current_price: float = Field(default=0.0)
    quantity: float
    side: str
    created_at: datetime
    updated_at: datetime
    triggered: bool = Field(default=False)
    triggered_at: Optional[datetime] = None

    initial_stop: float = Field(default=0.0)
    highest_price: float = Field(default=0.0)
    lowest_price: float = Field(default=float("inf"))
    trail_amount: float = Field(default=0.0)
    trail_percent: float = Field(default=0.0)

    breakeven_triggered: bool = Field(default=False)
    breakeven_buffer: float = Field(default=0.0)

    metadata: dict = Field(default_factory=dict)


class StopLossConfig(RiskConfig):
    """Configuration for stop loss manager."""

    default_stop_type: StopType = Field(default=StopType.PERCENTAGE)
    default_stop_pct: float = Field(default=0.02, ge=0.005, le=0.1)

    use_atr_stops: bool = Field(default=True)
    atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    atr_period: int = Field(default=14, ge=5, le=30)

    trailing_enabled: bool = Field(default=True)
    trailing_activation_pct: float = Field(default=0.01, ge=0.005, le=0.1)
    trailing_distance_pct: float = Field(default=0.015, ge=0.005, le=0.1)

    breakeven_enabled: bool = Field(default=True)
    breakeven_activation_pct: float = Field(default=0.02, ge=0.01, le=0.1)
    breakeven_buffer_pct: float = Field(default=0.002, ge=0.0, le=0.01)

    time_stop_enabled: bool = Field(default=False)
    time_stop_days: int = Field(default=5, ge=1, le=30)

    max_stop_distance_pct: float = Field(default=0.10, ge=0.02, le=0.3)
    min_stop_distance_pct: float = Field(default=0.005, ge=0.001, le=0.02)

    tighten_on_profit: bool = Field(default=True)
    tighten_threshold_pct: float = Field(default=0.05, ge=0.02, le=0.2)
    tighten_amount_pct: float = Field(default=0.005, ge=0.001, le=0.02)


class StopLossManager:
    """
    Stop loss order management.

    Features:
    - Multiple stop loss types
    - Trailing stops
    - ATR-based stops
    - Breakeven stops
    - Dynamic adjustment
    """

    def __init__(
        self,
        config: Optional[StopLossConfig] = None,
    ) -> None:
        """
        Initialize StopLossManager.

        Args:
            config: Stop loss configuration
        """
        self.config = config or StopLossConfig()
        self._stops: dict[str, StopLossOrder] = {}
        self._triggered_stops: list[StopLossOrder] = []
        self._alerts: list[RiskAlert] = []

        logger.info("StopLossManager initialized")

    def create_stop(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        side: str = "long",
        stop_type: Optional[StopType] = None,
        stop_price: Optional[float] = None,
        stop_pct: Optional[float] = None,
        atr: Optional[float] = None,
        trail_pct: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> StopLossOrder:
        """
        Create a new stop loss order.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position quantity
            side: Position side (long/short)
            stop_type: Type of stop loss
            stop_price: Explicit stop price
            stop_pct: Stop percentage from entry
            atr: ATR value for ATR-based stops
            trail_pct: Trailing percentage
            metadata: Additional metadata

        Returns:
            Created stop loss order
        """
        stop_type = stop_type or self.config.default_stop_type
        current_time = now_utc()

        if stop_price:
            calculated_stop = stop_price
        elif stop_type == StopType.ATR and atr:
            distance = atr * self.config.atr_multiplier
            if side == "long":
                calculated_stop = entry_price - distance
            else:
                calculated_stop = entry_price + distance
        elif stop_type in [StopType.TRAILING, StopType.TRAILING_ATR]:
            trail_distance = trail_pct or self.config.trailing_distance_pct
            if side == "long":
                calculated_stop = entry_price * (1 - trail_distance)
            else:
                calculated_stop = entry_price * (1 + trail_distance)
        else:
            pct = stop_pct or self.config.default_stop_pct
            if side == "long":
                calculated_stop = entry_price * (1 - pct)
            else:
                calculated_stop = entry_price * (1 + pct)

        calculated_stop = self._apply_stop_constraints(
            calculated_stop, entry_price, side
        )

        stop = StopLossOrder(
            symbol=symbol,
            stop_type=stop_type,
            entry_price=entry_price,
            stop_price=calculated_stop,
            current_price=entry_price,
            quantity=quantity,
            side=side,
            created_at=current_time,
            updated_at=current_time,
            initial_stop=calculated_stop,
            highest_price=entry_price,
            lowest_price=entry_price,
            trail_amount=abs(entry_price - calculated_stop) if stop_type in [StopType.TRAILING, StopType.TRAILING_ATR] else 0,
            trail_percent=trail_pct or self.config.trailing_distance_pct,
            metadata=metadata or {},
        )

        self._stops[symbol] = stop

        logger.info(
            f"Stop created for {symbol}: {stop_type.value} at {calculated_stop:.4f}"
        )

        return stop

    def update_price(
        self,
        symbol: str,
        current_price: float,
        atr: Optional[float] = None,
    ) -> Optional[StopLossOrder]:
        """
        Update price and adjust stop if necessary.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            atr: Current ATR value

        Returns:
            Updated stop order or None if triggered
        """
        if symbol not in self._stops:
            return None

        stop = self._stops[symbol]
        stop.current_price = current_price
        stop.updated_at = now_utc()

        if stop.side == "long":
            if current_price > stop.highest_price:
                stop.highest_price = current_price
            if current_price < stop.lowest_price:
                stop.lowest_price = current_price
        else:
            if current_price < stop.lowest_price:
                stop.lowest_price = current_price
            if current_price > stop.highest_price:
                stop.highest_price = current_price

        if self._is_stop_triggered(stop, current_price):
            return self._trigger_stop(stop, current_price)

        if stop.stop_type in [StopType.TRAILING, StopType.TRAILING_ATR]:
            self._update_trailing_stop(stop, current_price, atr)

        if self.config.breakeven_enabled and not stop.breakeven_triggered:
            self._check_breakeven(stop, current_price)

        if self.config.tighten_on_profit:
            self._check_tighten(stop, current_price)

        return stop

    def _is_stop_triggered(
        self,
        stop: StopLossOrder,
        current_price: float,
    ) -> bool:
        """Check if stop loss is triggered."""
        if stop.side == "long":
            return current_price <= stop.stop_price
        else:
            return current_price >= stop.stop_price

    def _trigger_stop(
        self,
        stop: StopLossOrder,
        trigger_price: float,
    ) -> StopLossOrder:
        """Trigger a stop loss order."""
        stop.triggered = True
        stop.triggered_at = now_utc()
        stop.current_price = trigger_price

        self._triggered_stops.append(stop)

        if stop.symbol in self._stops:
            del self._stops[stop.symbol]

        logger.warning(
            f"Stop triggered for {stop.symbol} at {trigger_price:.4f} "
            f"(stop was {stop.stop_price:.4f})"
        )

        return stop

    def _update_trailing_stop(
        self,
        stop: StopLossOrder,
        current_price: float,
        atr: Optional[float] = None,
    ) -> None:
        """Update trailing stop."""
        if stop.stop_type == StopType.TRAILING_ATR and atr:
            trail_distance = atr * self.config.atr_multiplier
        else:
            trail_distance = stop.trail_percent * current_price

        if stop.side == "long":
            new_stop = current_price - trail_distance

            if new_stop > stop.stop_price:
                stop.stop_price = new_stop
                stop.updated_at = now_utc()

        else:
            new_stop = current_price + trail_distance

            if new_stop < stop.stop_price:
                stop.stop_price = new_stop
                stop.updated_at = now_utc()

    def _check_breakeven(
        self,
        stop: StopLossOrder,
        current_price: float,
    ) -> None:
        """Check and apply breakeven stop."""
        activation_pct = self.config.breakeven_activation_pct

        if stop.side == "long":
            profit_pct = (current_price - stop.entry_price) / stop.entry_price

            if profit_pct >= activation_pct:
                breakeven_price = stop.entry_price * (1 + self.config.breakeven_buffer_pct)

                if breakeven_price > stop.stop_price:
                    stop.stop_price = breakeven_price
                    stop.breakeven_triggered = True
                    stop.breakeven_buffer = self.config.breakeven_buffer_pct
                    stop.updated_at = now_utc()

                    logger.info(f"Breakeven stop set for {stop.symbol} at {breakeven_price:.4f}")

        else:
            profit_pct = (stop.entry_price - current_price) / stop.entry_price

            if profit_pct >= activation_pct:
                breakeven_price = stop.entry_price * (1 - self.config.breakeven_buffer_pct)

                if breakeven_price < stop.stop_price:
                    stop.stop_price = breakeven_price
                    stop.breakeven_triggered = True
                    stop.breakeven_buffer = self.config.breakeven_buffer_pct
                    stop.updated_at = now_utc()

                    logger.info(f"Breakeven stop set for {stop.symbol} at {breakeven_price:.4f}")

    def _check_tighten(
        self,
        stop: StopLossOrder,
        current_price: float,
    ) -> None:
        """Check and tighten stop on significant profit."""
        threshold = self.config.tighten_threshold_pct
        tighten_amount = self.config.tighten_amount_pct

        if stop.side == "long":
            profit_pct = (current_price - stop.entry_price) / stop.entry_price

            if profit_pct >= threshold:
                current_stop_distance = (current_price - stop.stop_price) / current_price
                target_distance = current_stop_distance - tighten_amount

                if target_distance > self.config.min_stop_distance_pct:
                    new_stop = current_price * (1 - target_distance)
                    if new_stop > stop.stop_price:
                        stop.stop_price = new_stop
                        stop.updated_at = now_utc()

        else:
            profit_pct = (stop.entry_price - current_price) / stop.entry_price

            if profit_pct >= threshold:
                current_stop_distance = (stop.stop_price - current_price) / current_price
                target_distance = current_stop_distance - tighten_amount

                if target_distance > self.config.min_stop_distance_pct:
                    new_stop = current_price * (1 + target_distance)
                    if new_stop < stop.stop_price:
                        stop.stop_price = new_stop
                        stop.updated_at = now_utc()

    def _apply_stop_constraints(
        self,
        stop_price: float,
        entry_price: float,
        side: str,
    ) -> float:
        """Apply min/max stop distance constraints."""
        if side == "long":
            max_stop = entry_price * (1 - self.config.min_stop_distance_pct)
            min_stop = entry_price * (1 - self.config.max_stop_distance_pct)
            return max(min_stop, min(max_stop, stop_price))
        else:
            min_stop = entry_price * (1 + self.config.min_stop_distance_pct)
            max_stop = entry_price * (1 + self.config.max_stop_distance_pct)
            return min(max_stop, max(min_stop, stop_price))

    def modify_stop(
        self,
        symbol: str,
        new_stop_price: Optional[float] = None,
        new_stop_type: Optional[StopType] = None,
        new_trail_pct: Optional[float] = None,
    ) -> Optional[StopLossOrder]:
        """
        Modify an existing stop loss.

        Args:
            symbol: Trading symbol
            new_stop_price: New stop price
            new_stop_type: New stop type
            new_trail_pct: New trailing percentage

        Returns:
            Modified stop order
        """
        if symbol not in self._stops:
            return None

        stop = self._stops[symbol]

        if new_stop_price:
            stop.stop_price = self._apply_stop_constraints(
                new_stop_price, stop.entry_price, stop.side
            )

        if new_stop_type:
            stop.stop_type = new_stop_type

        if new_trail_pct:
            stop.trail_percent = new_trail_pct
            stop.trail_amount = stop.current_price * new_trail_pct

        stop.updated_at = now_utc()

        return stop

    def cancel_stop(self, symbol: str) -> bool:
        """Cancel a stop loss order."""
        if symbol in self._stops:
            del self._stops[symbol]
            logger.info(f"Stop cancelled for {symbol}")
            return True
        return False

    def get_stop(self, symbol: str) -> Optional[StopLossOrder]:
        """Get stop loss order for a symbol."""
        return self._stops.get(symbol)

    def get_all_stops(self) -> dict[str, StopLossOrder]:
        """Get all active stop loss orders."""
        return self._stops.copy()

    def get_triggered_stops(self, limit: int = 50) -> list[StopLossOrder]:
        """Get recently triggered stops."""
        return self._triggered_stops[-limit:]

    def calculate_stop_price(
        self,
        entry_price: float,
        side: str = "long",
        stop_type: Optional[StopType] = None,
        stop_pct: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> float:
        """
        Calculate stop price without creating order.

        Args:
            entry_price: Entry price
            side: Position side
            stop_type: Type of stop
            stop_pct: Stop percentage
            atr: ATR value

        Returns:
            Calculated stop price
        """
        stop_type = stop_type or self.config.default_stop_type

        if stop_type == StopType.ATR and atr:
            distance = atr * self.config.atr_multiplier
            if side == "long":
                return entry_price - distance
            else:
                return entry_price + distance

        pct = stop_pct or self.config.default_stop_pct
        if side == "long":
            return entry_price * (1 - pct)
        else:
            return entry_price * (1 + pct)

    def get_risk_amount(
        self,
        symbol: str,
        current_price: Optional[float] = None,
    ) -> float:
        """Get risk amount for a position."""
        if symbol not in self._stops:
            return 0.0

        stop = self._stops[symbol]
        price = current_price or stop.current_price

        risk_per_share = abs(price - stop.stop_price)
        return risk_per_share * stop.quantity

    def get_stop_statistics(self) -> dict:
        """Get stop loss statistics."""
        total_stops = len(self._stops)
        trailing_stops = sum(
            1 for s in self._stops.values()
            if s.stop_type in [StopType.TRAILING, StopType.TRAILING_ATR]
        )
        breakeven_stops = sum(
            1 for s in self._stops.values()
            if s.breakeven_triggered
        )

        return {
            "active_stops": total_stops,
            "trailing_stops": trailing_stops,
            "breakeven_stops": breakeven_stops,
            "triggered_total": len(self._triggered_stops),
            "stop_types": {
                stype.value: sum(1 for s in self._stops.values() if s.stop_type == stype)
                for stype in StopType
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"StopLossManager(active={len(self._stops)}, triggered={len(self._triggered_stops)})"

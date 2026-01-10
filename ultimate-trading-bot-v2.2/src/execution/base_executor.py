"""
Base Execution Classes for Ultimate Trading Bot v2.2.

This module provides the foundational classes for order execution
including order types, execution states, and base executor interface.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    MARKET_ON_CLOSE = "market_on_close"
    LIMIT_ON_CLOSE = "limit_on_close"


class TimeInForce(str, Enum):
    """Time in force options."""

    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    OPG = "opg"
    CLS = "cls"


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "pending"
    NEW = "new"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    ERROR = "error"


class ExecutionState(str, Enum):
    """Execution engine state."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class FillType(str, Enum):
    """Fill type."""

    FULL = "full"
    PARTIAL = "partial"


class Order(BaseModel):
    """Order representation."""

    model_config = {"arbitrary_types_allowed": True}

    order_id: str = Field(default_factory=lambda: str(uuid4()))
    client_order_id: str = Field(default_factory=lambda: str(uuid4()))
    broker_order_id: str | None = None

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: float | None = None
    stop_price: float | None = None
    trail_percent: float | None = None
    trail_amount: float | None = None

    time_in_force: TimeInForce = Field(default=TimeInForce.DAY)
    extended_hours: bool = Field(default=False)

    status: OrderStatus = Field(default=OrderStatus.PENDING)
    filled_qty: float = Field(default=0.0)
    remaining_qty: float = Field(default=0.0)
    avg_fill_price: float | None = None

    created_at: datetime = Field(default_factory=datetime.now)
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    canceled_at: datetime | None = None
    expired_at: datetime | None = None

    strategy_id: str | None = None
    signal_id: str | None = None
    parent_order_id: str | None = None
    linked_orders: list[str] = Field(default_factory=list)

    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    error_message: str | None = None

    @property
    def is_active(self) -> bool:
        """Check if order is active."""
        return self.status in [
            OrderStatus.NEW,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.PENDING_REPLACE,
        ]

    @property
    def is_complete(self) -> bool:
        """Check if order is complete."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]

    @property
    def is_open(self) -> bool:
        """Check if order is open for fills."""
        return self.status in [
            OrderStatus.NEW,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
        ]

    def update_remaining(self) -> None:
        """Update remaining quantity."""
        self.remaining_qty = self.quantity - self.filled_qty


class Fill(BaseModel):
    """Order fill representation."""

    fill_id: str = Field(default_factory=lambda: str(uuid4()))
    order_id: str
    symbol: str
    side: OrderSide

    quantity: float
    price: float
    fill_type: FillType

    commission: float = Field(default=0.0)
    fees: float = Field(default=0.0)

    executed_at: datetime = Field(default_factory=datetime.now)
    venue: str | None = None
    liquidity: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class Position(BaseModel):
    """Position representation."""

    symbol: str
    quantity: float = Field(default=0.0)
    avg_entry_price: float = Field(default=0.0)
    market_value: float = Field(default=0.0)
    cost_basis: float = Field(default=0.0)
    unrealized_pnl: float = Field(default=0.0)
    unrealized_pnl_pct: float = Field(default=0.0)
    realized_pnl: float = Field(default=0.0)
    current_price: float = Field(default=0.0)
    last_updated: datetime = Field(default_factory=datetime.now)

    side: str = Field(default="long")
    asset_class: str = Field(default="equity")
    exchange: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0


@dataclass
class ExecutionContext:
    """Context for order execution."""

    order: Order
    market_data: dict[str, Any] = field(default_factory=dict)
    account_info: dict[str, Any] = field(default_factory=dict)
    position: Position | None = None
    risk_approval: bool = True
    execution_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of an execution attempt."""

    order: Order
    success: bool
    fills: list[Fill] = field(default_factory=list)
    error_message: str | None = None
    broker_response: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


class ExecutorConfig(BaseModel):
    """Configuration for executor."""

    model_config = {"arbitrary_types_allowed": True}

    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay_seconds: float = Field(default=1.0, description="Delay between retries")
    timeout_seconds: float = Field(default=30.0, description="Execution timeout")
    max_concurrent_orders: int = Field(default=10, description="Max concurrent orders")
    enable_paper_trading: bool = Field(default=False, description="Paper trading mode")
    validate_orders: bool = Field(default=True, description="Validate before execution")
    log_all_orders: bool = Field(default=True, description="Log all order activity")
    default_time_in_force: TimeInForce = Field(default=TimeInForce.DAY)


class BaseExecutor(ABC):
    """
    Abstract base class for order executors.

    Provides the interface for all execution implementations.
    """

    def __init__(self, config: ExecutorConfig | None = None):
        """
        Initialize base executor.

        Args:
            config: Executor configuration
        """
        self.config = config or ExecutorConfig()
        self._state = ExecutionState.IDLE
        self._active_orders: dict[str, Order] = {}
        self._order_history: list[Order] = []
        self._fill_history: list[Fill] = []
        self._callbacks: dict[str, list[Callable]] = {
            "order_submitted": [],
            "order_filled": [],
            "order_canceled": [],
            "order_rejected": [],
            "fill_received": [],
            "error": [],
        }
        self._lock = asyncio.Lock()

        logger.info(f"{self.__class__.__name__} initialized")

    @abstractmethod
    async def submit_order(self, order: Order) -> ExecutionResult:
        """
        Submit an order for execution.

        Args:
            order: Order to submit

        Returns:
            ExecutionResult
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> ExecutionResult:
        """
        Cancel an active order.

        Args:
            order_id: Order ID to cancel

        Returns:
            ExecutionResult
        """
        pass

    @abstractmethod
    async def modify_order(
        self,
        order_id: str,
        new_quantity: float | None = None,
        new_limit_price: float | None = None,
        new_stop_price: float | None = None,
    ) -> ExecutionResult:
        """
        Modify an active order.

        Args:
            order_id: Order ID to modify
            new_quantity: New quantity
            new_limit_price: New limit price
            new_stop_price: New stop price

        Returns:
            ExecutionResult
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order | None:
        """
        Get current order status.

        Args:
            order_id: Order ID

        Returns:
            Order if found
        """
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """
        Get all current positions.

        Returns:
            List of positions
        """
        pass

    @abstractmethod
    async def get_account_info(self) -> dict[str, Any]:
        """
        Get account information.

        Returns:
            Account information dictionary
        """
        pass

    async def start(self) -> None:
        """Start the executor."""
        self._state = ExecutionState.RUNNING
        logger.info(f"{self.__class__.__name__} started")

    async def stop(self) -> None:
        """Stop the executor."""
        self._state = ExecutionState.STOPPING

        for order_id in list(self._active_orders.keys()):
            try:
                await self.cancel_order(order_id)
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")

        self._state = ExecutionState.STOPPED
        logger.info(f"{self.__class__.__name__} stopped")

    async def pause(self) -> None:
        """Pause the executor."""
        self._state = ExecutionState.PAUSED
        logger.info(f"{self.__class__.__name__} paused")

    async def resume(self) -> None:
        """Resume the executor."""
        self._state = ExecutionState.RUNNING
        logger.info(f"{self.__class__.__name__} resumed")

    def register_callback(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """
        Register callback for execution events.

        Args:
            event: Event name
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def _trigger_callbacks(
        self,
        event: str,
        data: Any,
    ) -> None:
        """Trigger callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    def validate_order(self, order: Order) -> tuple[bool, str | None]:
        """
        Validate an order before submission.

        Args:
            order: Order to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not order.symbol:
            return False, "Symbol is required"

        if order.quantity <= 0:
            return False, "Quantity must be positive"

        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            return False, "Limit price required for limit orders"

        if order.order_type == OrderType.STOP and order.stop_price is None:
            return False, "Stop price required for stop orders"

        if order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.limit_price is None:
                return False, "Both stop and limit prices required for stop-limit orders"

        if order.order_type == OrderType.TRAILING_STOP:
            if order.trail_percent is None and order.trail_amount is None:
                return False, "Trail percent or amount required for trailing stop"

        return True, None

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce | None = None,
        extended_hours: bool = False,
        **kwargs: Any,
    ) -> Order:
        """
        Create an order object.

        Args:
            symbol: Asset symbol
            side: Order side
            quantity: Order quantity
            order_type: Order type
            limit_price: Limit price
            stop_price: Stop price
            time_in_force: Time in force
            extended_hours: Allow extended hours
            **kwargs: Additional order parameters

        Returns:
            Order object
        """
        order = Order(
            symbol=symbol.upper(),
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force or self.config.default_time_in_force,
            extended_hours=extended_hours,
            remaining_qty=quantity,
            **kwargs,
        )

        return order

    async def execute_with_retry(
        self,
        order: Order,
    ) -> ExecutionResult:
        """
        Execute order with retry logic.

        Args:
            order: Order to execute

        Returns:
            ExecutionResult
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                result = await self.submit_order(order)

                if result.success:
                    return result

                last_error = result.error_message

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Execution attempt {attempt + 1} failed for {order.order_id}: {e}"
                )

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay_seconds)

        order.status = OrderStatus.ERROR
        order.error_message = f"Failed after {self.config.max_retries} attempts: {last_error}"

        return ExecutionResult(
            order=order,
            success=False,
            error_message=order.error_message,
        )

    def get_active_orders(self) -> list[Order]:
        """Get all active orders."""
        return [o for o in self._active_orders.values() if o.is_active]

    def get_order_history(self, limit: int = 100) -> list[Order]:
        """Get order history."""
        return self._order_history[-limit:]

    def get_fill_history(self, limit: int = 100) -> list[Fill]:
        """Get fill history."""
        return self._fill_history[-limit:]

    def get_state(self) -> ExecutionState:
        """Get current executor state."""
        return self._state

    async def get_execution_stats(self) -> dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Statistics dictionary
        """
        total_orders = len(self._order_history)
        filled_orders = sum(
            1 for o in self._order_history if o.status == OrderStatus.FILLED
        )
        canceled_orders = sum(
            1 for o in self._order_history if o.status == OrderStatus.CANCELED
        )
        rejected_orders = sum(
            1 for o in self._order_history if o.status == OrderStatus.REJECTED
        )

        total_fills = len(self._fill_history)
        total_volume = sum(f.quantity * f.price for f in self._fill_history)
        total_commission = sum(f.commission for f in self._fill_history)

        return {
            "state": self._state.value,
            "active_orders": len(self._active_orders),
            "total_orders": total_orders,
            "filled_orders": filled_orders,
            "canceled_orders": canceled_orders,
            "rejected_orders": rejected_orders,
            "fill_rate": filled_orders / total_orders if total_orders > 0 else 0,
            "total_fills": total_fills,
            "total_volume": total_volume,
            "total_commission": total_commission,
        }


class PaperExecutor(BaseExecutor):
    """
    Paper trading executor for simulation.

    Simulates order execution without real broker connection.
    """

    def __init__(self, config: ExecutorConfig | None = None):
        """
        Initialize paper executor.

        Args:
            config: Executor configuration
        """
        if config is None:
            config = ExecutorConfig()
        config.enable_paper_trading = True

        super().__init__(config)

        self._positions: dict[str, Position] = {}
        self._cash = 100000.0
        self._equity = 100000.0
        self._fill_probability = 0.95
        self._partial_fill_probability = 0.1
        self._slippage_bps = 5

        logger.info("PaperExecutor initialized")

    async def submit_order(self, order: Order) -> ExecutionResult:
        """Submit order for paper execution."""
        start_time = datetime.now()

        if self.config.validate_orders:
            valid, error = self.validate_order(order)
            if not valid:
                order.status = OrderStatus.REJECTED
                order.error_message = error
                return ExecutionResult(order=order, success=False, error_message=error)

        order.status = OrderStatus.ACCEPTED
        order.submitted_at = datetime.now()

        async with self._lock:
            self._active_orders[order.order_id] = order

        await self._trigger_callbacks("order_submitted", order)

        import random
        if random.random() > self._fill_probability:
            order.status = OrderStatus.REJECTED
            order.error_message = "Simulated rejection"
            return ExecutionResult(order=order, success=False, error_message="Simulated rejection")

        fill_price = order.limit_price
        if order.order_type == OrderType.MARKET or fill_price is None:
            fill_price = 100.0

        slippage = fill_price * (self._slippage_bps / 10000)
        if order.side == OrderSide.BUY:
            fill_price += slippage
        else:
            fill_price -= slippage

        if random.random() < self._partial_fill_probability:
            fill_qty = order.quantity * random.uniform(0.3, 0.7)
        else:
            fill_qty = order.quantity

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=fill_price,
            fill_type=FillType.FULL if fill_qty == order.quantity else FillType.PARTIAL,
            commission=fill_qty * fill_price * 0.0001,
        )

        order.filled_qty = fill_qty
        order.remaining_qty = order.quantity - fill_qty
        order.avg_fill_price = fill_price

        if order.remaining_qty <= 0:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        async with self._lock:
            self._fill_history.append(fill)
            if order.status == OrderStatus.FILLED:
                if order.order_id in self._active_orders:
                    del self._active_orders[order.order_id]
                self._order_history.append(order)

            await self._update_position(order, fill)

        await self._trigger_callbacks("fill_received", fill)
        if order.status == OrderStatus.FILLED:
            await self._trigger_callbacks("order_filled", order)

        exec_time = (datetime.now() - start_time).total_seconds() * 1000

        return ExecutionResult(
            order=order,
            success=True,
            fills=[fill],
            execution_time_ms=exec_time,
        )

    async def _update_position(self, order: Order, fill: Fill) -> None:
        """Update position after fill."""
        symbol = order.symbol

        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)

        pos = self._positions[symbol]

        if order.side == OrderSide.BUY:
            new_qty = pos.quantity + fill.quantity
            if new_qty != 0:
                pos.avg_entry_price = (
                    (pos.quantity * pos.avg_entry_price + fill.quantity * fill.price) /
                    new_qty
                )
            pos.quantity = new_qty
            self._cash -= fill.quantity * fill.price + fill.commission
        else:
            pos.quantity -= fill.quantity
            self._cash += fill.quantity * fill.price - fill.commission

        pos.cost_basis = abs(pos.quantity) * pos.avg_entry_price
        pos.market_value = pos.quantity * fill.price
        pos.unrealized_pnl = pos.market_value - pos.cost_basis
        pos.current_price = fill.price
        pos.last_updated = datetime.now()

        if pos.quantity == 0:
            del self._positions[symbol]

    async def cancel_order(self, order_id: str) -> ExecutionResult:
        """Cancel a paper order."""
        async with self._lock:
            order = self._active_orders.get(order_id)

            if not order:
                return ExecutionResult(
                    order=Order(order_id=order_id, symbol="", side=OrderSide.BUY,
                               order_type=OrderType.MARKET, quantity=0),
                    success=False,
                    error_message="Order not found",
                )

            order.status = OrderStatus.CANCELED
            order.canceled_at = datetime.now()

            del self._active_orders[order_id]
            self._order_history.append(order)

        await self._trigger_callbacks("order_canceled", order)

        return ExecutionResult(order=order, success=True)

    async def modify_order(
        self,
        order_id: str,
        new_quantity: float | None = None,
        new_limit_price: float | None = None,
        new_stop_price: float | None = None,
    ) -> ExecutionResult:
        """Modify a paper order."""
        async with self._lock:
            order = self._active_orders.get(order_id)

            if not order:
                return ExecutionResult(
                    order=Order(order_id=order_id, symbol="", side=OrderSide.BUY,
                               order_type=OrderType.MARKET, quantity=0),
                    success=False,
                    error_message="Order not found",
                )

            if new_quantity is not None:
                order.quantity = new_quantity
                order.remaining_qty = new_quantity - order.filled_qty

            if new_limit_price is not None:
                order.limit_price = new_limit_price

            if new_stop_price is not None:
                order.stop_price = new_stop_price

            order.status = OrderStatus.REPLACED

        return ExecutionResult(order=order, success=True)

    async def get_order_status(self, order_id: str) -> Order | None:
        """Get paper order status."""
        if order_id in self._active_orders:
            return self._active_orders[order_id]

        for order in self._order_history:
            if order.order_id == order_id:
                return order

        return None

    async def get_positions(self) -> list[Position]:
        """Get paper positions."""
        return list(self._positions.values())

    async def get_account_info(self) -> dict[str, Any]:
        """Get paper account info."""
        positions_value = sum(p.market_value for p in self._positions.values())
        self._equity = self._cash + positions_value

        return {
            "cash": self._cash,
            "equity": self._equity,
            "buying_power": self._cash * 2,
            "portfolio_value": self._equity,
            "position_count": len(self._positions),
            "is_paper": True,
        }

    def set_fill_probability(self, probability: float) -> None:
        """Set fill probability for simulation."""
        self._fill_probability = max(0.0, min(1.0, probability))

    def set_slippage(self, slippage_bps: float) -> None:
        """Set slippage in basis points."""
        self._slippage_bps = slippage_bps

    def reset_account(self, starting_cash: float = 100000.0) -> None:
        """Reset paper account."""
        self._cash = starting_cash
        self._equity = starting_cash
        self._positions.clear()
        self._active_orders.clear()
        self._order_history.clear()
        self._fill_history.clear()
        logger.info(f"Paper account reset with ${starting_cash:,.2f}")

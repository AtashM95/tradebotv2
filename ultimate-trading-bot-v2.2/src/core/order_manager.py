"""
Order Manager Module for Ultimate Trading Bot v2.2.

This module provides comprehensive order management including order creation,
submission, tracking, modification, and cancellation.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
import logging
from collections import defaultdict

from pydantic import BaseModel, Field

from src.core.event_bus import EventBus, EventType, Event, get_event_bus
from src.core.models import (
    Order, OrderFill, OrderSide, OrderType, OrderStatus, TimeInForce
)


logger = logging.getLogger(__name__)


class OrderManagerConfig(BaseModel):
    """Order manager configuration."""

    max_pending_orders: int = Field(default=100, ge=1, description="Max pending orders")
    max_orders_per_symbol: int = Field(default=10, ge=1, description="Max orders per symbol")
    order_timeout_seconds: int = Field(default=300, ge=60, description="Order timeout")
    auto_cancel_on_close: bool = Field(default=True, description="Cancel orders at market close")
    track_order_history: bool = Field(default=True, description="Track order history")
    max_history_size: int = Field(default=1000, ge=100, description="Max history size")
    validate_orders: bool = Field(default=True, description="Validate orders before submission")
    allow_fractional: bool = Field(default=True, description="Allow fractional shares")
    default_extended_hours: bool = Field(default=False, description="Default extended hours")


class OrderManager:
    """
    Manages all order-related operations.

    Provides functionality for:
    - Order creation and validation
    - Order submission and tracking
    - Order modification and cancellation
    - Order status updates
    - Order history management
    """

    def __init__(
        self,
        config: Optional[OrderManagerConfig] = None,
        event_bus: Optional[EventBus] = None,
        executor: Optional[Any] = None
    ) -> None:
        """
        Initialize the order manager.

        Args:
            config: Order manager configuration
            event_bus: Event bus instance
            executor: Order execution component
        """
        self.config = config or OrderManagerConfig()
        self.event_bus = event_bus or get_event_bus()
        self._executor = executor

        # Order storage
        self._orders: Dict[str, Order] = {}
        self._orders_by_symbol: Dict[str, Set[str]] = defaultdict(set)
        self._pending_orders: Dict[str, Order] = {}
        self._filled_orders: Dict[str, Order] = {}
        self._cancelled_orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []
        self._order_fills: Dict[str, List[OrderFill]] = defaultdict(list)

        # State
        self._lock = asyncio.Lock()
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Set up event handlers
        self._setup_event_handlers()

        logger.info("OrderManager initialized")

    def _setup_event_handlers(self) -> None:
        """Set up event handlers."""
        # Listen for external order updates (from broker)
        self.event_bus.subscribe(
            EventType.ORDER_FILLED,
            self._on_order_filled_event
        )
        self.event_bus.subscribe(
            EventType.ORDER_CANCELLED,
            self._on_order_cancelled_event
        )
        self.event_bus.subscribe(
            EventType.ORDER_REJECTED,
            self._on_order_rejected_event
        )

    def set_executor(self, executor: Any) -> None:
        """Set the order executor component."""
        self._executor = executor
        logger.debug("Order executor set")

    # =========================================================================
    # ORDER CREATION
    # =========================================================================

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: Optional[str] = None,
        extended_hours: Optional[bool] = None,
        strategy_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            order_type: Order type
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force
            client_order_id: Optional client order ID
            extended_hours: Enable extended hours trading
            strategy_id: Associated strategy ID
            tags: Custom tags

        Returns:
            Created Order object
        """
        if extended_hours is None:
            extended_hours = self.config.default_extended_hours

        order = Order(
            symbol=symbol.upper(),
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            extended_hours=extended_hours,
            strategy_id=strategy_id,
            tags=tags or {}
        )

        logger.debug(f"Created order: {order.order_id} - {symbol} {side.value} {quantity}")
        return order

    def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        **kwargs: Any
    ) -> Order:
        """Create a market order."""
        return self.create_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            **kwargs
        )

    def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        limit_price: float,
        **kwargs: Any
    ) -> Order:
        """Create a limit order."""
        return self.create_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            **kwargs
        )

    def create_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        **kwargs: Any
    ) -> Order:
        """Create a stop order."""
        return self.create_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_price,
            **kwargs
        )

    def create_stop_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        limit_price: float,
        **kwargs: Any
    ) -> Order:
        """Create a stop-limit order."""
        return self.create_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.STOP_LIMIT,
            stop_price=stop_price,
            limit_price=limit_price,
            **kwargs
        )

    # =========================================================================
    # ORDER SUBMISSION
    # =========================================================================

    async def submit_order(self, order: Order) -> Optional[Order]:
        """
        Submit an order for execution.

        Args:
            order: Order to submit

        Returns:
            Submitted order with updated status, or None on failure
        """
        async with self._lock:
            # Validate order
            if self.config.validate_orders:
                validation = self._validate_order(order)
                if not validation["valid"]:
                    logger.error(f"Order validation failed: {validation['reason']}")
                    order.status = OrderStatus.REJECTED
                    order.rejected_at = datetime.now(timezone.utc)
                    order.reject_reason = validation["reason"]
                    await self._emit_order_event(EventType.ORDER_REJECTED, order)
                    return None

            # Check limits
            if len(self._pending_orders) >= self.config.max_pending_orders:
                logger.error("Maximum pending orders reached")
                order.status = OrderStatus.REJECTED
                order.reject_reason = "Maximum pending orders reached"
                await self._emit_order_event(EventType.ORDER_REJECTED, order)
                return None

            symbol_orders = self._orders_by_symbol.get(order.symbol, set())
            if len(symbol_orders) >= self.config.max_orders_per_symbol:
                logger.error(f"Maximum orders for {order.symbol} reached")
                order.status = OrderStatus.REJECTED
                order.reject_reason = "Maximum orders for symbol reached"
                await self._emit_order_event(EventType.ORDER_REJECTED, order)
                return None

            # Submit to executor
            if self._executor:
                try:
                    result = await self._executor.submit_order(order)
                    if result:
                        order.status = OrderStatus.PENDING_NEW
                        order.submitted_at = datetime.now(timezone.utc)
                    else:
                        order.status = OrderStatus.REJECTED
                        order.reject_reason = "Executor rejected order"
                        await self._emit_order_event(EventType.ORDER_REJECTED, order)
                        return None
                except Exception as e:
                    logger.error(f"Error submitting order: {e}")
                    order.status = OrderStatus.REJECTED
                    order.reject_reason = str(e)
                    await self._emit_order_event(EventType.ORDER_REJECTED, order)
                    return None
            else:
                # Simulation mode - auto-accept
                order.status = OrderStatus.ACCEPTED
                order.submitted_at = datetime.now(timezone.utc)

            # Track order
            self._orders[order.order_id] = order
            self._orders_by_symbol[order.symbol].add(order.order_id)
            self._pending_orders[order.order_id] = order

            # Emit event
            await self._emit_order_event(EventType.ORDER_SUBMITTED, order)

            logger.info(f"Order submitted: {order.order_id} - {order.symbol} {order.side.value} {order.quantity}")
            return order

    def _validate_order(self, order: Order) -> Dict[str, Any]:
        """Validate an order before submission."""
        # Check required fields
        if not order.symbol:
            return {"valid": False, "reason": "Symbol is required"}

        if order.quantity <= 0:
            return {"valid": False, "reason": "Quantity must be positive"}

        # Check fractional shares
        if not self.config.allow_fractional:
            if order.quantity != int(order.quantity):
                return {"valid": False, "reason": "Fractional shares not allowed"}

        # Check prices for limit/stop orders
        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if order.limit_price is None or order.limit_price <= 0:
                return {"valid": False, "reason": "Valid limit price required"}

        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if order.stop_price is None or order.stop_price <= 0:
                return {"valid": False, "reason": "Valid stop price required"}

        return {"valid": True}

    # =========================================================================
    # ORDER MODIFICATION
    # =========================================================================

    async def modify_order(
        self,
        order_id: str,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None
    ) -> Optional[Order]:
        """
        Modify an existing order.

        Args:
            order_id: Order ID to modify
            quantity: New quantity
            limit_price: New limit price
            stop_price: New stop price
            time_in_force: New time in force

        Returns:
            Modified order or None
        """
        async with self._lock:
            order = self._orders.get(order_id)
            if not order:
                logger.error(f"Order not found: {order_id}")
                return None

            if not order.is_active:
                logger.error(f"Cannot modify inactive order: {order_id}")
                return None

            # Update fields
            if quantity is not None:
                order.quantity = quantity
            if limit_price is not None:
                order.limit_price = limit_price
            if stop_price is not None:
                order.stop_price = stop_price
            if time_in_force is not None:
                order.time_in_force = time_in_force

            order.status = OrderStatus.PENDING_REPLACE
            order.update_timestamp()

            # Submit modification to executor
            if self._executor:
                try:
                    await self._executor.modify_order(order)
                except Exception as e:
                    logger.error(f"Error modifying order: {e}")
                    return None

            logger.info(f"Order modified: {order_id}")
            return order

    # =========================================================================
    # ORDER CANCELLATION
    # =========================================================================

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation was submitted
        """
        async with self._lock:
            order = self._orders.get(order_id)
            if not order:
                logger.error(f"Order not found: {order_id}")
                return False

            if not order.is_active:
                logger.warning(f"Order already inactive: {order_id}")
                return False

            order.status = OrderStatus.PENDING_CANCEL

            # Submit cancellation to executor
            if self._executor:
                try:
                    await self._executor.cancel_order(order_id)
                except Exception as e:
                    logger.error(f"Error cancelling order: {e}")
                    return False
            else:
                # Simulation mode - auto-cancel
                await self._process_cancellation(order)

            logger.info(f"Order cancellation submitted: {order_id}")
            return True

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all orders, optionally filtered by symbol.

        Args:
            symbol: Optional symbol filter

        Returns:
            Number of cancellation requests submitted
        """
        cancelled = 0

        orders_to_cancel = list(self._pending_orders.values())
        if symbol:
            orders_to_cancel = [o for o in orders_to_cancel if o.symbol == symbol.upper()]

        for order in orders_to_cancel:
            if await self.cancel_order(order.order_id):
                cancelled += 1

        logger.info(f"Cancelled {cancelled} orders")
        return cancelled

    async def _process_cancellation(self, order: Order) -> None:
        """Process order cancellation."""
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now(timezone.utc)
        order.update_timestamp()

        # Move to cancelled
        if order.order_id in self._pending_orders:
            del self._pending_orders[order.order_id]
        self._cancelled_orders[order.order_id] = order

        await self._emit_order_event(EventType.ORDER_CANCELLED, order)

    # =========================================================================
    # ORDER QUERIES
    # =========================================================================

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID."""
        return self._orders.get(order_id)

    def get_order_by_client_id(self, client_order_id: str) -> Optional[Order]:
        """Get an order by client order ID."""
        for order in self._orders.values():
            if order.client_order_id == client_order_id:
                return order
        return None

    def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        side: Optional[OrderSide] = None,
        strategy_id: Optional[str] = None
    ) -> List[Order]:
        """
        Get orders with optional filters.

        Args:
            symbol: Filter by symbol
            status: Filter by status
            side: Filter by side
            strategy_id: Filter by strategy

        Returns:
            List of matching orders
        """
        orders = list(self._orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]

        if status:
            orders = [o for o in orders if o.status == status]

        if side:
            orders = [o for o in orders if o.side == side]

        if strategy_id:
            orders = [o for o in orders if o.strategy_id == strategy_id]

        return orders

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all pending orders."""
        orders = list(self._pending_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]
        return orders

    def get_filled_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all filled orders."""
        orders = list(self._filled_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]
        return orders

    def get_order_fills(self, order_id: str) -> List[OrderFill]:
        """Get fills for an order."""
        return self._order_fills.get(order_id, [])

    def get_order_history(self, limit: int = 100) -> List[Order]:
        """Get order history."""
        return self._order_history[-limit:]

    # =========================================================================
    # ORDER EVENTS
    # =========================================================================

    async def _on_order_filled_event(self, event: Event) -> None:
        """Handle order filled event from external source."""
        data = event.data
        if isinstance(data, dict):
            order_id = data.get("order_id")
            if order_id and order_id in self._orders:
                await self._process_fill(order_id, data)

    async def _on_order_cancelled_event(self, event: Event) -> None:
        """Handle order cancelled event from external source."""
        data = event.data
        if isinstance(data, dict):
            order_id = data.get("order_id")
            if order_id and order_id in self._orders:
                order = self._orders[order_id]
                await self._process_cancellation(order)

    async def _on_order_rejected_event(self, event: Event) -> None:
        """Handle order rejected event from external source."""
        data = event.data
        if isinstance(data, dict):
            order_id = data.get("order_id")
            if order_id and order_id in self._orders:
                order = self._orders[order_id]
                order.status = OrderStatus.REJECTED
                order.rejected_at = datetime.now(timezone.utc)
                order.reject_reason = data.get("reason", "Unknown")

                if order.order_id in self._pending_orders:
                    del self._pending_orders[order.order_id]

    async def _process_fill(self, order_id: str, fill_data: Dict[str, Any]) -> None:
        """Process an order fill."""
        order = self._orders.get(order_id)
        if not order:
            return

        # Create fill record
        fill = OrderFill(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_data.get("quantity", 0),
            price=fill_data.get("price", 0),
            commission=fill_data.get("commission", 0)
        )
        self._order_fills[order_id].append(fill)

        # Update order
        order.filled_quantity += fill.quantity
        if fill_data.get("avg_fill_price"):
            order.avg_fill_price = fill_data["avg_fill_price"]
        else:
            order.avg_fill_price = fill.price

        # Check if fully filled
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now(timezone.utc)

            # Move to filled
            if order.order_id in self._pending_orders:
                del self._pending_orders[order.order_id]
            self._filled_orders[order.order_id] = order

            # Add to history
            if self.config.track_order_history:
                self._add_to_history(order)

            await self._emit_order_event(EventType.ORDER_FILLED, order)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
            await self._emit_order_event(EventType.ORDER_PARTIALLY_FILLED, order)

        order.update_timestamp()

    def _add_to_history(self, order: Order) -> None:
        """Add order to history."""
        self._order_history.append(order)

        # Trim history if needed
        if len(self._order_history) > self.config.max_history_size:
            self._order_history = self._order_history[-self.config.max_history_size:]

    async def _emit_order_event(self, event_type: EventType, order: Order) -> None:
        """Emit an order event."""
        await self.event_bus.emit(
            event_type,
            data=order,
            source="order_manager"
        )

        # Call registered callbacks
        for callback in self._callbacks.get(event_type.name, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_order_filled(self, callback: Callable[[Order], Any]) -> None:
        """Register callback for order filled events."""
        self._callbacks["ORDER_FILLED"].append(callback)

    def on_order_cancelled(self, callback: Callable[[Order], Any]) -> None:
        """Register callback for order cancelled events."""
        self._callbacks["ORDER_CANCELLED"].append(callback)

    def on_order_rejected(self, callback: Callable[[Order], Any]) -> None:
        """Register callback for order rejected events."""
        self._callbacks["ORDER_REJECTED"].append(callback)

    # =========================================================================
    # SYNC METHODS
    # =========================================================================

    async def sync_orders(self) -> None:
        """Synchronize orders with broker."""
        if self._executor:
            try:
                broker_orders = await self._executor.get_orders()
                for order_data in broker_orders:
                    order_id = order_data.get("order_id")
                    if order_id and order_id not in self._orders:
                        # Create order from broker data
                        order = Order(**order_data)
                        self._orders[order_id] = order
                        self._orders_by_symbol[order.symbol].add(order_id)

                        if order.is_active:
                            self._pending_orders[order_id] = order

                logger.info(f"Synced {len(broker_orders)} orders from broker")
            except Exception as e:
                logger.error(f"Error syncing orders: {e}")

    async def cleanup_old_orders(self, max_age_hours: int = 24) -> int:
        """Clean up old completed orders."""
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        removed = 0

        async with self._lock:
            # Clean filled orders
            to_remove = [
                order_id for order_id, order in self._filled_orders.items()
                if order.filled_at and order.filled_at < cutoff
            ]
            for order_id in to_remove:
                del self._filled_orders[order_id]
                removed += 1

            # Clean cancelled orders
            to_remove = [
                order_id for order_id, order in self._cancelled_orders.items()
                if order.cancelled_at and order.cancelled_at < cutoff
            ]
            for order_id in to_remove:
                del self._cancelled_orders[order_id]
                removed += 1

        logger.info(f"Cleaned up {removed} old orders")
        return removed

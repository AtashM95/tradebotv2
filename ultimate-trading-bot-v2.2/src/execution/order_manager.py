"""
Order Management for Ultimate Trading Bot v2.2.

This module provides comprehensive order lifecycle management
including order tracking, state management, and order grouping.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from src.execution.base_executor import (
    BaseExecutor,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
    Fill,
    ExecutionResult,
)


logger = logging.getLogger(__name__)


class OrderGroupType(str, Enum):
    """Types of order groups."""

    BRACKET = "bracket"
    OCO = "oco"
    OTO = "oto"
    BASKET = "basket"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class OrderPriority(str, Enum):
    """Order execution priority."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class OrderManagerConfig(BaseModel):
    """Configuration for order manager."""

    model_config = {"arbitrary_types_allowed": True}

    max_pending_orders: int = Field(default=100, description="Max pending orders")
    max_active_orders: int = Field(default=50, description="Max active orders")
    order_timeout_seconds: int = Field(default=300, description="Order timeout")
    enable_order_grouping: bool = Field(default=True, description="Enable order groups")
    auto_cancel_orphans: bool = Field(default=True, description="Auto-cancel orphaned orders")
    persist_orders: bool = Field(default=True, description="Persist orders to storage")
    max_retries: int = Field(default=3, description="Max retry attempts")
    retry_delay_seconds: float = Field(default=2.0, description="Retry delay")


class OrderGroup(BaseModel):
    """Group of related orders."""

    group_id: str = Field(default_factory=lambda: str(uuid4()))
    group_type: OrderGroupType
    name: str = ""

    order_ids: list[str] = Field(default_factory=list)
    parent_order_id: str | None = None

    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None

    is_active: bool = Field(default=True)
    status: str = Field(default="pending")

    cancel_on_fill: bool = Field(default=False)
    fill_triggers_next: bool = Field(default=False)

    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class OrderQueueItem:
    """Item in the order queue."""

    order: Order
    priority: OrderPriority
    queued_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    last_attempt: datetime | None = None


@dataclass
class OrderMetrics:
    """Metrics for order tracking."""

    total_submitted: int = 0
    total_filled: int = 0
    total_canceled: int = 0
    total_rejected: int = 0
    partial_fills: int = 0
    avg_fill_time_ms: float = 0.0
    avg_fill_rate: float = 0.0
    total_commission: float = 0.0
    total_volume: float = 0.0


class OrderManager:
    """
    Manages order lifecycle and tracking.

    Provides order queuing, grouping, and comprehensive
    order state management.
    """

    def __init__(
        self,
        executor: BaseExecutor,
        config: OrderManagerConfig | None = None,
    ):
        """
        Initialize order manager.

        Args:
            executor: Order executor to use
            config: Order manager configuration
        """
        self.executor = executor
        self.config = config or OrderManagerConfig()

        self._order_queue: list[OrderQueueItem] = []
        self._active_orders: dict[str, Order] = {}
        self._order_history: dict[str, Order] = {}
        self._fills: dict[str, list[Fill]] = defaultdict(list)

        self._order_groups: dict[str, OrderGroup] = {}
        self._order_to_group: dict[str, str] = {}

        self._metrics = OrderMetrics()
        self._callbacks: dict[str, list[Callable]] = {
            "order_queued": [],
            "order_submitted": [],
            "order_filled": [],
            "order_canceled": [],
            "order_rejected": [],
            "group_completed": [],
        }

        self._processing = False
        self._process_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        logger.info("OrderManager initialized")

    async def start(self) -> None:
        """Start order processing."""
        self._processing = True
        self._process_task = asyncio.create_task(self._process_queue())
        logger.info("OrderManager started")

    async def stop(self) -> None:
        """Stop order processing."""
        self._processing = False

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        logger.info("OrderManager stopped")

    async def queue_order(
        self,
        order: Order,
        priority: OrderPriority = OrderPriority.NORMAL,
    ) -> str:
        """
        Queue an order for execution.

        Args:
            order: Order to queue
            priority: Execution priority

        Returns:
            Order ID
        """
        if len(self._order_queue) >= self.config.max_pending_orders:
            raise ValueError("Order queue is full")

        item = OrderQueueItem(order=order, priority=priority)

        async with self._lock:
            self._order_queue.append(item)
            self._order_queue.sort(
                key=lambda x: (
                    list(OrderPriority).index(x.priority),
                    x.queued_at,
                )
            )

        await self._trigger_callbacks("order_queued", order)

        logger.debug(f"Order queued: {order.order_id} (priority: {priority.value})")

        return order.order_id

    async def submit_order(
        self,
        order: Order,
    ) -> ExecutionResult:
        """
        Submit order immediately.

        Args:
            order: Order to submit

        Returns:
            ExecutionResult
        """
        if len(self._active_orders) >= self.config.max_active_orders:
            return ExecutionResult(
                order=order,
                success=False,
                error_message="Maximum active orders reached",
            )

        result = await self.executor.submit_order(order)

        if result.success:
            async with self._lock:
                self._active_orders[order.order_id] = order
                self._metrics.total_submitted += 1

            await self._trigger_callbacks("order_submitted", order)

            if order.status == OrderStatus.FILLED:
                await self._handle_fill(order, result.fills)

        return result

    async def cancel_order(
        self,
        order_id: str,
        reason: str | None = None,
    ) -> ExecutionResult:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason

        Returns:
            ExecutionResult
        """
        async with self._lock:
            for i, item in enumerate(self._order_queue):
                if item.order.order_id == order_id:
                    self._order_queue.pop(i)
                    item.order.status = OrderStatus.CANCELED
                    item.order.canceled_at = datetime.now()
                    item.order.metadata["cancel_reason"] = reason
                    self._order_history[order_id] = item.order
                    self._metrics.total_canceled += 1

                    await self._trigger_callbacks("order_canceled", item.order)

                    return ExecutionResult(order=item.order, success=True)

        result = await self.executor.cancel_order(order_id)

        if result.success:
            async with self._lock:
                if order_id in self._active_orders:
                    order = self._active_orders.pop(order_id)
                    order.metadata["cancel_reason"] = reason
                    self._order_history[order_id] = order
                    self._metrics.total_canceled += 1

                    await self._handle_group_cancel(order_id)

            await self._trigger_callbacks("order_canceled", result.order)

        return result

    async def cancel_all_orders(
        self,
        symbol: str | None = None,
        side: OrderSide | None = None,
    ) -> int:
        """
        Cancel all orders, optionally filtered.

        Args:
            symbol: Filter by symbol
            side: Filter by side

        Returns:
            Number of orders canceled
        """
        canceled = 0

        async with self._lock:
            to_remove = []
            for i, item in enumerate(self._order_queue):
                if symbol and item.order.symbol != symbol:
                    continue
                if side and item.order.side != side:
                    continue
                to_remove.append(i)

            for i in reversed(to_remove):
                item = self._order_queue.pop(i)
                item.order.status = OrderStatus.CANCELED
                self._order_history[item.order.order_id] = item.order
                canceled += 1

        for order_id, order in list(self._active_orders.items()):
            if symbol and order.symbol != symbol:
                continue
            if side and order.side != side:
                continue

            result = await self.cancel_order(order_id)
            if result.success:
                canceled += 1

        logger.info(f"Canceled {canceled} orders")
        return canceled

    async def modify_order(
        self,
        order_id: str,
        new_quantity: float | None = None,
        new_limit_price: float | None = None,
        new_stop_price: float | None = None,
    ) -> ExecutionResult:
        """
        Modify an order.

        Args:
            order_id: Order ID to modify
            new_quantity: New quantity
            new_limit_price: New limit price
            new_stop_price: New stop price

        Returns:
            ExecutionResult
        """
        async with self._lock:
            for item in self._order_queue:
                if item.order.order_id == order_id:
                    if new_quantity is not None:
                        item.order.quantity = new_quantity
                        item.order.remaining_qty = new_quantity
                    if new_limit_price is not None:
                        item.order.limit_price = new_limit_price
                    if new_stop_price is not None:
                        item.order.stop_price = new_stop_price

                    return ExecutionResult(order=item.order, success=True)

        return await self.executor.modify_order(
            order_id=order_id,
            new_quantity=new_quantity,
            new_limit_price=new_limit_price,
            new_stop_price=new_stop_price,
        )

    async def create_bracket_order(
        self,
        entry_order: Order,
        take_profit_price: float,
        stop_loss_price: float,
        take_profit_qty: float | None = None,
        stop_loss_qty: float | None = None,
    ) -> OrderGroup:
        """
        Create a bracket order group.

        Args:
            entry_order: Entry order
            take_profit_price: Take profit price
            stop_loss_price: Stop loss price
            take_profit_qty: Take profit quantity (defaults to entry qty)
            stop_loss_qty: Stop loss quantity (defaults to entry qty)

        Returns:
            OrderGroup
        """
        group = OrderGroup(
            group_type=OrderGroupType.BRACKET,
            name=f"Bracket_{entry_order.symbol}",
            parent_order_id=entry_order.order_id,
            cancel_on_fill=True,
        )

        exit_side = OrderSide.SELL if entry_order.side == OrderSide.BUY else OrderSide.BUY

        tp_order = Order(
            symbol=entry_order.symbol,
            side=exit_side,
            order_type=OrderType.LIMIT,
            quantity=take_profit_qty or entry_order.quantity,
            limit_price=take_profit_price,
            parent_order_id=entry_order.order_id,
        )

        sl_order = Order(
            symbol=entry_order.symbol,
            side=exit_side,
            order_type=OrderType.STOP,
            quantity=stop_loss_qty or entry_order.quantity,
            stop_price=stop_loss_price,
            parent_order_id=entry_order.order_id,
        )

        group.order_ids = [entry_order.order_id, tp_order.order_id, sl_order.order_id]

        async with self._lock:
            self._order_groups[group.group_id] = group
            self._order_to_group[entry_order.order_id] = group.group_id
            self._order_to_group[tp_order.order_id] = group.group_id
            self._order_to_group[sl_order.order_id] = group.group_id

        await self.queue_order(entry_order, OrderPriority.HIGH)

        group.metadata["tp_order"] = tp_order
        group.metadata["sl_order"] = sl_order

        logger.info(f"Created bracket order group: {group.group_id}")

        return group

    async def create_oco_order(
        self,
        order1: Order,
        order2: Order,
    ) -> OrderGroup:
        """
        Create an OCO (One-Cancels-Other) order group.

        Args:
            order1: First order
            order2: Second order

        Returns:
            OrderGroup
        """
        group = OrderGroup(
            group_type=OrderGroupType.OCO,
            name=f"OCO_{order1.symbol}",
            cancel_on_fill=True,
        )

        order1.linked_orders.append(order2.order_id)
        order2.linked_orders.append(order1.order_id)

        group.order_ids = [order1.order_id, order2.order_id]

        async with self._lock:
            self._order_groups[group.group_id] = group
            self._order_to_group[order1.order_id] = group.group_id
            self._order_to_group[order2.order_id] = group.group_id

        await self.queue_order(order1, OrderPriority.NORMAL)
        await self.queue_order(order2, OrderPriority.NORMAL)

        logger.info(f"Created OCO order group: {group.group_id}")

        return group

    async def create_basket_order(
        self,
        orders: list[Order],
        name: str = "",
    ) -> OrderGroup:
        """
        Create a basket order group.

        Args:
            orders: List of orders in basket
            name: Basket name

        Returns:
            OrderGroup
        """
        group = OrderGroup(
            group_type=OrderGroupType.BASKET,
            name=name or f"Basket_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        group.order_ids = [o.order_id for o in orders]

        async with self._lock:
            self._order_groups[group.group_id] = group
            for order in orders:
                self._order_to_group[order.order_id] = group.group_id

        for order in orders:
            await self.queue_order(order, OrderPriority.NORMAL)

        logger.info(f"Created basket order group: {group.group_id} ({len(orders)} orders)")

        return group

    async def _process_queue(self) -> None:
        """Process the order queue."""
        while self._processing:
            try:
                async with self._lock:
                    if not self._order_queue:
                        await asyncio.sleep(0.1)
                        continue

                    if len(self._active_orders) >= self.config.max_active_orders:
                        await asyncio.sleep(0.5)
                        continue

                    item = self._order_queue[0]

                    if item.attempts >= self.config.max_retries:
                        self._order_queue.pop(0)
                        item.order.status = OrderStatus.REJECTED
                        item.order.error_message = "Max retries exceeded"
                        self._order_history[item.order.order_id] = item.order
                        self._metrics.total_rejected += 1
                        await self._trigger_callbacks("order_rejected", item.order)
                        continue

                    self._order_queue.pop(0)

                item.attempts += 1
                item.last_attempt = datetime.now()

                result = await self.executor.submit_order(item.order)

                if result.success:
                    async with self._lock:
                        self._active_orders[item.order.order_id] = item.order
                        self._metrics.total_submitted += 1

                    await self._trigger_callbacks("order_submitted", item.order)

                    if item.order.status == OrderStatus.FILLED:
                        await self._handle_fill(item.order, result.fills)

                else:
                    if item.attempts < self.config.max_retries:
                        async with self._lock:
                            self._order_queue.append(item)
                        await asyncio.sleep(self.config.retry_delay_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing order queue: {e}")
                await asyncio.sleep(1.0)

    async def _handle_fill(
        self,
        order: Order,
        fills: list[Fill],
    ) -> None:
        """Handle order fill."""
        async with self._lock:
            self._fills[order.order_id].extend(fills)

            if order.status == OrderStatus.FILLED:
                if order.order_id in self._active_orders:
                    del self._active_orders[order.order_id]
                self._order_history[order.order_id] = order
                self._metrics.total_filled += 1

                for fill in fills:
                    self._metrics.total_volume += fill.quantity * fill.price
                    self._metrics.total_commission += fill.commission

                if order.submitted_at:
                    fill_time = (datetime.now() - order.submitted_at).total_seconds() * 1000
                    self._update_avg_fill_time(fill_time)

            elif order.status == OrderStatus.PARTIALLY_FILLED:
                self._metrics.partial_fills += 1

        await self._trigger_callbacks("order_filled", order)

        await self._handle_group_fill(order.order_id)

    async def _handle_group_fill(self, order_id: str) -> None:
        """Handle fill in context of order group."""
        group_id = self._order_to_group.get(order_id)
        if not group_id:
            return

        group = self._order_groups.get(group_id)
        if not group:
            return

        if group.cancel_on_fill:
            for linked_id in group.order_ids:
                if linked_id != order_id and linked_id in self._active_orders:
                    await self.cancel_order(linked_id, reason="OCO/Bracket fill")

        if group.group_type == OrderGroupType.BRACKET:
            if order_id == group.parent_order_id:
                tp_order = group.metadata.get("tp_order")
                sl_order = group.metadata.get("sl_order")

                if tp_order:
                    await self.queue_order(tp_order, OrderPriority.HIGH)
                if sl_order:
                    await self.queue_order(sl_order, OrderPriority.HIGH)

        all_filled = all(
            self._order_history.get(oid, Order(
                order_id="", symbol="", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=0
            )).status == OrderStatus.FILLED
            for oid in group.order_ids
        )

        if all_filled:
            group.is_active = False
            group.status = "completed"
            group.completed_at = datetime.now()
            await self._trigger_callbacks("group_completed", group)

    async def _handle_group_cancel(self, order_id: str) -> None:
        """Handle cancel in context of order group."""
        group_id = self._order_to_group.get(order_id)
        if not group_id:
            return

        group = self._order_groups.get(group_id)
        if not group:
            return

        if group.cancel_on_fill:
            for linked_id in group.order_ids:
                if linked_id != order_id:
                    if linked_id in self._active_orders:
                        await self.cancel_order(linked_id, reason="Group cancel")
                    else:
                        for i, item in enumerate(self._order_queue):
                            if item.order.order_id == linked_id:
                                async with self._lock:
                                    self._order_queue.pop(i)
                                break

    def _update_avg_fill_time(self, fill_time_ms: float) -> None:
        """Update average fill time."""
        n = self._metrics.total_filled
        if n == 1:
            self._metrics.avg_fill_time_ms = fill_time_ms
        else:
            self._metrics.avg_fill_time_ms = (
                (self._metrics.avg_fill_time_ms * (n - 1) + fill_time_ms) / n
            )

    def register_callback(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """Register callback for order events."""
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

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        if order_id in self._active_orders:
            return self._active_orders[order_id]
        return self._order_history.get(order_id)

    def get_active_orders(self) -> list[Order]:
        """Get all active orders."""
        return list(self._active_orders.values())

    def get_pending_orders(self) -> list[Order]:
        """Get all pending orders in queue."""
        return [item.order for item in self._order_queue]

    def get_order_group(self, group_id: str) -> OrderGroup | None:
        """Get order group by ID."""
        return self._order_groups.get(group_id)

    def get_fills(self, order_id: str) -> list[Fill]:
        """Get fills for an order."""
        return self._fills.get(order_id, [])

    def get_metrics(self) -> OrderMetrics:
        """Get order metrics."""
        return self._metrics

    async def get_status_summary(self) -> dict[str, Any]:
        """Get order manager status summary."""
        return {
            "queued_orders": len(self._order_queue),
            "active_orders": len(self._active_orders),
            "completed_orders": len(self._order_history),
            "active_groups": len([g for g in self._order_groups.values() if g.is_active]),
            "metrics": {
                "total_submitted": self._metrics.total_submitted,
                "total_filled": self._metrics.total_filled,
                "total_canceled": self._metrics.total_canceled,
                "total_rejected": self._metrics.total_rejected,
                "avg_fill_time_ms": self._metrics.avg_fill_time_ms,
                "total_volume": self._metrics.total_volume,
                "total_commission": self._metrics.total_commission,
            },
            "processing": self._processing,
        }

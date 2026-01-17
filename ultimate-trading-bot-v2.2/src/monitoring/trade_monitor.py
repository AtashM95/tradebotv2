"""
Trade Monitor for Ultimate Trading Bot v2.2.

Real-time monitoring of trading activity, orders, and positions.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TradeEventType(str, Enum):
    """Types of trade events."""
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL = "order_partial"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    STOP_TRIGGERED = "stop_triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"


@dataclass
class TradeMonitorConfig:
    """Configuration for trade monitoring."""

    # Monitoring settings
    update_interval: float = 1.0  # seconds
    max_events: int = 1000
    max_orders: int = 500

    # Alert thresholds
    large_order_threshold: float = 10000.0  # USD
    slippage_alert_threshold: float = 0.01  # 1%
    fill_time_alert_threshold: float = 60.0  # seconds


@dataclass
class OrderEvent:
    """Order event record."""

    event_id: str
    event_type: TradeEventType
    timestamp: datetime
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float | None = None
    filled_quantity: float = 0.0
    filled_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "status": self.status.value,
            "message": self.message,
        }


@dataclass
class TrackedOrder:
    """Tracked order with history."""

    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    limit_price: float | None = None
    stop_price: float | None = None

    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None

    # Metrics
    slippage: float = 0.0
    fill_time: float = 0.0  # seconds

    # Strategy info
    strategy: str | None = None
    signal_id: str | None = None

    # Events
    events: list[OrderEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "slippage": self.slippage,
            "fill_time": self.fill_time,
            "strategy": self.strategy,
        }


@dataclass
class TradingStats:
    """Trading statistics."""

    # Order counts
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    pending_orders: int = 0

    # Fill metrics
    avg_fill_time: float = 0.0
    avg_slippage: float = 0.0
    fill_rate: float = 0.0

    # Volume
    total_volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0

    # By symbol
    volume_by_symbol: dict[str, float] = field(default_factory=dict)
    orders_by_symbol: dict[str, int] = field(default_factory=dict)

    # By strategy
    orders_by_strategy: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "orders": {
                "total": self.total_orders,
                "filled": self.filled_orders,
                "cancelled": self.cancelled_orders,
                "rejected": self.rejected_orders,
                "pending": self.pending_orders,
            },
            "metrics": {
                "avg_fill_time": self.avg_fill_time,
                "avg_slippage": self.avg_slippage,
                "fill_rate": self.fill_rate,
            },
            "volume": {
                "total": self.total_volume,
                "buy": self.buy_volume,
                "sell": self.sell_volume,
            },
            "by_symbol": self.volume_by_symbol,
            "by_strategy": self.orders_by_strategy,
        }


TradeEventCallback = Callable[[OrderEvent], None]


class TradeMonitor:
    """
    Real-time trade monitoring system.

    Tracks orders, fills, and trading activity.
    """

    def __init__(self, config: TradeMonitorConfig | None = None) -> None:
        """
        Initialize trade monitor.

        Args:
            config: Trade monitoring configuration
        """
        self.config = config or TradeMonitorConfig()

        # Order tracking
        self._orders: dict[str, TrackedOrder] = {}
        self._events: list[OrderEvent] = []

        # Stats
        self._fill_times: list[float] = []
        self._slippages: list[float] = []

        # Callbacks
        self._event_callbacks: list[TradeEventCallback] = []

        # Event counter
        self._event_counter = 0

        logger.info("TradeMonitor initialized")

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        self._event_counter += 1
        return f"evt_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._event_counter}"

    def add_callback(self, callback: TradeEventCallback) -> None:
        """Add event callback."""
        self._event_callbacks.append(callback)

    def _notify_event(self, event: OrderEvent) -> None:
        """Notify callbacks of event."""
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    async def track_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        limit_price: float | None = None,
        stop_price: float | None = None,
        strategy: str | None = None,
        signal_id: str | None = None,
    ) -> TrackedOrder:
        """
        Start tracking a new order.

        Args:
            order_id: Order ID
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_type: Order type (market/limit/stop)
            quantity: Order quantity
            limit_price: Limit price
            stop_price: Stop price
            strategy: Strategy name
            signal_id: Signal ID

        Returns:
            Tracked order
        """
        order = TrackedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            strategy=strategy,
            signal_id=signal_id,
            created_at=datetime.now(),
        )

        self._orders[order_id] = order

        # Create event
        event = OrderEvent(
            event_id=self._generate_event_id(),
            event_type=TradeEventType.ORDER_SUBMITTED,
            timestamp=datetime.now(),
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=limit_price,
            status=OrderStatus.PENDING,
        )

        self._add_event(event, order)

        # Check for large order
        if limit_price and quantity * limit_price > self.config.large_order_threshold:
            logger.warning(
                f"Large order submitted: {symbol} {side} "
                f"{quantity} @ ${limit_price:.2f}"
            )

        return order

    async def order_submitted(self, order_id: str) -> None:
        """Mark order as submitted to broker."""
        if order_id not in self._orders:
            return

        order = self._orders[order_id]
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()

        event = OrderEvent(
            event_id=self._generate_event_id(),
            event_type=TradeEventType.ORDER_SUBMITTED,
            timestamp=datetime.now(),
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            status=OrderStatus.SUBMITTED,
        )

        self._add_event(event, order)

    async def order_filled(
        self,
        order_id: str,
        filled_quantity: float,
        fill_price: float,
        partial: bool = False,
    ) -> None:
        """
        Record order fill.

        Args:
            order_id: Order ID
            filled_quantity: Quantity filled
            fill_price: Fill price
            partial: Whether this is a partial fill
        """
        if order_id not in self._orders:
            return

        order = self._orders[order_id]

        # Update fill info
        prev_filled = order.filled_quantity
        order.filled_quantity += filled_quantity

        if order.avg_fill_price == 0:
            order.avg_fill_price = fill_price
        else:
            # Weighted average
            total_filled = prev_filled + filled_quantity
            order.avg_fill_price = (
                (prev_filled * order.avg_fill_price + filled_quantity * fill_price) /
                total_filled
            )

        # Check if fully filled
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()

            # Calculate fill time
            if order.submitted_at:
                order.fill_time = (order.filled_at - order.submitted_at).total_seconds()
                self._fill_times.append(order.fill_time)

                # Alert on slow fill
                if order.fill_time > self.config.fill_time_alert_threshold:
                    logger.warning(
                        f"Slow fill: {order.symbol} took {order.fill_time:.1f}s"
                    )

            # Calculate slippage
            if order.limit_price:
                if order.side == "buy":
                    order.slippage = (order.avg_fill_price - order.limit_price) / order.limit_price
                else:
                    order.slippage = (order.limit_price - order.avg_fill_price) / order.limit_price

                self._slippages.append(order.slippage)

                # Alert on high slippage
                if abs(order.slippage) > self.config.slippage_alert_threshold:
                    logger.warning(
                        f"High slippage: {order.symbol} {order.slippage:.2%}"
                    )

            event_type = TradeEventType.ORDER_FILLED
        else:
            order.status = OrderStatus.PARTIAL
            event_type = TradeEventType.ORDER_PARTIAL

        event = OrderEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            timestamp=datetime.now(),
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            filled_price=fill_price,
            status=order.status,
        )

        self._add_event(event, order)

    async def order_cancelled(
        self,
        order_id: str,
        reason: str = "",
    ) -> None:
        """Record order cancellation."""
        if order_id not in self._orders:
            return

        order = self._orders[order_id]
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now()

        event = OrderEvent(
            event_id=self._generate_event_id(),
            event_type=TradeEventType.ORDER_CANCELLED,
            timestamp=datetime.now(),
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            status=OrderStatus.CANCELLED,
            message=reason,
        )

        self._add_event(event, order)

    async def order_rejected(
        self,
        order_id: str,
        reason: str = "",
    ) -> None:
        """Record order rejection."""
        if order_id not in self._orders:
            return

        order = self._orders[order_id]
        order.status = OrderStatus.REJECTED

        event = OrderEvent(
            event_id=self._generate_event_id(),
            event_type=TradeEventType.ORDER_REJECTED,
            timestamp=datetime.now(),
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            status=OrderStatus.REJECTED,
            message=reason,
        )

        self._add_event(event, order)

        logger.warning(f"Order rejected: {order_id} - {reason}")

    def _add_event(self, event: OrderEvent, order: TrackedOrder) -> None:
        """Add event and notify callbacks."""
        order.events.append(event)
        self._events.append(event)

        # Trim events
        while len(self._events) > self.config.max_events:
            self._events.pop(0)

        # Notify callbacks
        self._notify_event(event)

    def get_order(self, order_id: str) -> TrackedOrder | None:
        """Get tracked order."""
        return self._orders.get(order_id)

    def get_orders(
        self,
        status: OrderStatus | None = None,
        symbol: str | None = None,
        strategy: str | None = None,
    ) -> list[TrackedOrder]:
        """Get orders with optional filters."""
        orders = list(self._orders.values())

        if status:
            orders = [o for o in orders if o.status == status]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        if strategy:
            orders = [o for o in orders if o.strategy == strategy]

        return sorted(orders, key=lambda o: o.created_at, reverse=True)

    def get_pending_orders(self) -> list[TrackedOrder]:
        """Get all pending orders."""
        return self.get_orders(status=OrderStatus.PENDING)

    def get_events(
        self,
        minutes: int = 60,
        event_type: TradeEventType | None = None,
    ) -> list[OrderEvent]:
        """Get recent events."""
        cutoff = datetime.now() - timedelta(minutes=minutes)

        events = [e for e in self._events if e.timestamp >= cutoff]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events

    def get_statistics(self, hours: int = 24) -> TradingStats:
        """Get trading statistics."""
        cutoff = datetime.now() - timedelta(hours=hours)

        # Filter orders
        orders = [
            o for o in self._orders.values()
            if o.created_at >= cutoff
        ]

        stats = TradingStats()
        stats.total_orders = len(orders)
        stats.filled_orders = sum(1 for o in orders if o.status == OrderStatus.FILLED)
        stats.cancelled_orders = sum(1 for o in orders if o.status == OrderStatus.CANCELLED)
        stats.rejected_orders = sum(1 for o in orders if o.status == OrderStatus.REJECTED)
        stats.pending_orders = sum(
            1 for o in orders
            if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
        )

        # Calculate metrics
        filled_orders = [o for o in orders if o.status == OrderStatus.FILLED]

        if filled_orders:
            fill_times = [o.fill_time for o in filled_orders if o.fill_time > 0]
            stats.avg_fill_time = sum(fill_times) / len(fill_times) if fill_times else 0.0

            slippages = [o.slippage for o in filled_orders if o.limit_price]
            stats.avg_slippage = sum(slippages) / len(slippages) if slippages else 0.0

        stats.fill_rate = stats.filled_orders / stats.total_orders if stats.total_orders > 0 else 0.0

        # Volume calculations
        for order in filled_orders:
            volume = order.filled_quantity * order.avg_fill_price
            stats.total_volume += volume

            if order.side.lower() == "buy":
                stats.buy_volume += volume
            else:
                stats.sell_volume += volume

            # By symbol
            stats.volume_by_symbol[order.symbol] = (
                stats.volume_by_symbol.get(order.symbol, 0) + volume
            )
            stats.orders_by_symbol[order.symbol] = (
                stats.orders_by_symbol.get(order.symbol, 0) + 1
            )

            # By strategy
            if order.strategy:
                stats.orders_by_strategy[order.strategy] = (
                    stats.orders_by_strategy.get(order.strategy, 0) + 1
                )

        return stats

    def clear_old_orders(self, hours: int = 24) -> int:
        """Clear old completed orders."""
        cutoff = datetime.now() - timedelta(hours=hours)
        old_count = 0

        to_remove = []
        for order_id, order in self._orders.items():
            if (order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED] and
                order.created_at < cutoff):
                to_remove.append(order_id)

        for order_id in to_remove:
            del self._orders[order_id]
            old_count += 1

        return old_count


def create_trade_monitor(
    config: TradeMonitorConfig | None = None,
) -> TradeMonitor:
    """
    Create a trade monitor instance.

    Args:
        config: Trade monitoring configuration

    Returns:
        TradeMonitor instance
    """
    return TradeMonitor(config)


import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from .contracts import ExecutionRequest, RunContext

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order lifecycle states."""
    PENDING = "pending"
    VALIDATING = "validating"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"


class OrderType(Enum):
    """Order types supported by the system."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    """Time in force options."""
    DAY = "day"
    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    OPG = "opg"  # At the open
    CLS = "cls"  # At the close


@dataclass
class Order:
    """
    Represents a trading order with full lifecycle tracking.

    This is the internal order representation used by the order manager.
    It wraps the ExecutionRequest contract with additional metadata.
    """
    order_id: str
    broker_order_id: Optional[str]
    symbol: str
    action: str  # buy, sell, short, cover
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    time_in_force: TimeInForce
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    submitted_at: Optional[datetime]
    filled_at: Optional[datetime]
    filled_quantity: float
    filled_avg_price: float
    rejected_reason: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for logging/storage."""
        return {
            'order_id': self.order_id,
            'broker_order_id': self.broker_order_id,
            'symbol': self.symbol,
            'action': self.action,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'filled_quantity': self.filled_quantity,
            'filled_avg_price': self.filled_avg_price,
            'rejected_reason': self.rejected_reason,
            'metadata': self.metadata
        }

    def is_terminal(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        )

    def is_active(self) -> bool:
        """Check if order is active (can be cancelled/modified)."""
        return self.status in (
            OrderStatus.SUBMITTED,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIAL_FILL
        )


class OrderManagerMetrics:
    """Tracks order manager performance metrics."""

    def __init__(self) -> None:
        self.total_orders: int = 0
        self.orders_submitted: int = 0
        self.orders_filled: int = 0
        self.orders_cancelled: int = 0
        self.orders_rejected: int = 0
        self.partial_fills: int = 0
        self.total_quantity_traded: float = 0.0
        self.total_notional_value: float = 0.0
        self.avg_fill_time: float = 0.0
        self._fill_times: List[float] = []

    def record_submission(self) -> None:
        """Record an order submission."""
        self.orders_submitted += 1

    def record_fill(self, quantity: float, price: float, fill_time: float) -> None:
        """Record an order fill."""
        self.orders_filled += 1
        self.total_quantity_traded += quantity
        self.total_notional_value += quantity * price
        self._fill_times.append(fill_time)
        self.avg_fill_time = sum(self._fill_times) / len(self._fill_times)

    def record_cancel(self) -> None:
        """Record an order cancellation."""
        self.orders_cancelled += 1

    def record_reject(self) -> None:
        """Record an order rejection."""
        self.orders_rejected += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_orders': self.total_orders,
            'orders_submitted': self.orders_submitted,
            'orders_filled': self.orders_filled,
            'orders_cancelled': self.orders_cancelled,
            'orders_rejected': self.orders_rejected,
            'partial_fills': self.partial_fills,
            'fill_rate': self.orders_filled / max(1, self.orders_submitted),
            'total_quantity_traded': self.total_quantity_traded,
            'total_notional_value': self.total_notional_value,
            'avg_fill_time': self.avg_fill_time
        }


class OrderManager:
    """
    Manages the full lifecycle of trading orders.

    Responsibilities:
    - Create and track orders
    - Manage order state transitions
    - Provide order query and lookup functionality
    - Track order history and audit trail
    - Calculate order statistics and metrics
    - Validate orders before submission
    - Handle order cancellations and modifications
    """

    def __init__(self, max_order_history: int = 10000) -> None:
        """
        Initialize the order manager.

        Args:
            max_order_history: Maximum number of completed orders to keep in memory
        """
        self.orders: Dict[str, Order] = {}  # All orders indexed by order_id
        self.broker_order_map: Dict[str, str] = {}  # broker_order_id -> order_id
        self.symbol_orders: Dict[str, List[str]] = defaultdict(list)  # symbol -> order_ids
        self.active_orders: List[str] = []  # Active order IDs
        self.completed_orders: List[str] = []  # Completed order IDs
        self.max_order_history = max_order_history
        self.metrics = OrderManagerMetrics()

        logger.info("OrderManager initialized", extra={
            'max_order_history': max_order_history
        })

    def create_order(
        self,
        request: ExecutionRequest,
        order_type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.DAY,
        context: Optional[RunContext] = None
    ) -> Order:
        """
        Create a new order from an execution request.

        Args:
            request: Execution request with order details
            order_type: Type of order
            time_in_force: Time in force option
            context: Optional run context

        Returns:
            Created order object
        """
        order_id = self._generate_order_id()
        now = datetime.utcnow()

        order = Order(
            order_id=order_id,
            broker_order_id=None,
            symbol=request.symbol,
            action=request.action,
            order_type=order_type,
            quantity=request.quantity,
            price=request.price if order_type != OrderType.MARKET else None,
            stop_price=None,
            time_in_force=time_in_force,
            status=OrderStatus.PENDING,
            created_at=now,
            updated_at=now,
            submitted_at=None,
            filled_at=None,
            filled_quantity=0.0,
            filled_avg_price=0.0,
            rejected_reason=None,
            metadata=request.metadata.copy()
        )

        self.orders[order_id] = order
        self.symbol_orders[request.symbol].append(order_id)
        self.metrics.total_orders += 1

        log_extra = {'order_id': order_id, 'symbol': request.symbol}
        if context:
            log_extra['run_id'] = context.run_id

        logger.info("Order created", extra=log_extra)

        return order

    def submit(self, order: Order, context: Optional[RunContext] = None) -> None:
        """
        Mark an order as submitted.

        Args:
            order: Order to submit
            context: Optional run context
        """
        if order.status != OrderStatus.PENDING:
            logger.warning(f"Cannot submit order in status {order.status}", extra={
                'order_id': order.order_id
            })
            return

        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.utcnow()
        order.updated_at = datetime.utcnow()

        if order.order_id not in self.active_orders:
            self.active_orders.append(order.order_id)

        self.metrics.record_submission()

        log_extra = {'order_id': order.order_id, 'symbol': order.symbol}
        if context:
            log_extra['run_id'] = context.run_id

        logger.info("Order submitted", extra=log_extra)

    def acknowledge(
        self,
        order_id: str,
        broker_order_id: str,
        context: Optional[RunContext] = None
    ) -> bool:
        """
        Acknowledge an order submission from the broker.

        Args:
            order_id: Internal order ID
            broker_order_id: Broker's order ID
            context: Optional run context

        Returns:
            True if successful, False otherwise
        """
        order = self.orders.get(order_id)
        if not order:
            logger.error(f"Order not found: {order_id}")
            return False

        order.broker_order_id = broker_order_id
        order.status = OrderStatus.ACKNOWLEDGED
        order.updated_at = datetime.utcnow()

        self.broker_order_map[broker_order_id] = order_id

        log_extra = {
            'order_id': order_id,
            'broker_order_id': broker_order_id,
            'symbol': order.symbol
        }
        if context:
            log_extra['run_id'] = context.run_id

        logger.info("Order acknowledged", extra=log_extra)

        return True

    def update_fill(
        self,
        order_id: str,
        filled_quantity: float,
        filled_price: float,
        is_complete: bool = False,
        context: Optional[RunContext] = None
    ) -> bool:
        """
        Update order with fill information.

        Args:
            order_id: Order ID
            filled_quantity: Quantity filled in this update
            filled_price: Price of this fill
            is_complete: Whether this completes the order
            context: Optional run context

        Returns:
            True if successful, False otherwise
        """
        order = self.orders.get(order_id)
        if not order:
            logger.error(f"Order not found: {order_id}")
            return False

        # Update filled quantity and average price
        total_filled = order.filled_quantity + filled_quantity
        if total_filled > 0:
            order.filled_avg_price = (
                (order.filled_quantity * order.filled_avg_price +
                 filled_quantity * filled_price) / total_filled
            )
        order.filled_quantity = total_filled

        # Update status
        if is_complete or abs(order.filled_quantity - order.quantity) < 0.0001:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.utcnow()

            # Calculate fill time
            if order.submitted_at:
                fill_time = (order.filled_at - order.submitted_at).total_seconds()
                self.metrics.record_fill(order.filled_quantity, order.filled_avg_price, fill_time)

            self._move_to_completed(order_id)
        else:
            order.status = OrderStatus.PARTIAL_FILL
            self.metrics.partial_fills += 1

        order.updated_at = datetime.utcnow()

        log_extra = {
            'order_id': order_id,
            'symbol': order.symbol,
            'filled_quantity': filled_quantity,
            'filled_price': filled_price,
            'total_filled': order.filled_quantity,
            'status': order.status.value
        }
        if context:
            log_extra['run_id'] = context.run_id

        logger.info("Order fill updated", extra=log_extra)

        return True

    def cancel(self, order_id: str, context: Optional[RunContext] = None) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID
            context: Optional run context

        Returns:
            True if cancellation initiated, False otherwise
        """
        order = self.orders.get(order_id)
        if not order:
            logger.error(f"Order not found: {order_id}")
            return False

        if not order.is_active():
            logger.warning(f"Cannot cancel order in status {order.status}", extra={
                'order_id': order_id
            })
            return False

        order.status = OrderStatus.CANCELLING
        order.updated_at = datetime.utcnow()

        log_extra = {'order_id': order_id, 'symbol': order.symbol}
        if context:
            log_extra['run_id'] = context.run_id

        logger.info("Order cancel requested", extra=log_extra)

        return True

    def confirm_cancel(
        self,
        order_id: str,
        context: Optional[RunContext] = None
    ) -> bool:
        """
        Confirm an order cancellation.

        Args:
            order_id: Order ID
            context: Optional run context

        Returns:
            True if successful, False otherwise
        """
        order = self.orders.get(order_id)
        if not order:
            logger.error(f"Order not found: {order_id}")
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()

        self.metrics.record_cancel()
        self._move_to_completed(order_id)

        log_extra = {'order_id': order_id, 'symbol': order.symbol}
        if context:
            log_extra['run_id'] = context.run_id

        logger.info("Order cancelled", extra=log_extra)

        return True

    def reject(
        self,
        order_id: str,
        reason: str,
        context: Optional[RunContext] = None
    ) -> bool:
        """
        Mark an order as rejected.

        Args:
            order_id: Order ID
            reason: Rejection reason
            context: Optional run context

        Returns:
            True if successful, False otherwise
        """
        order = self.orders.get(order_id)
        if not order:
            logger.error(f"Order not found: {order_id}")
            return False

        order.status = OrderStatus.REJECTED
        order.rejected_reason = reason
        order.updated_at = datetime.utcnow()

        self.metrics.record_reject()
        self._move_to_completed(order_id)

        log_extra = {
            'order_id': order_id,
            'symbol': order.symbol,
            'reason': reason
        }
        if context:
            log_extra['run_id'] = context.run_id

        logger.warning("Order rejected", extra=log_extra)

        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID."""
        return self.orders.get(order_id)

    def get_order_by_broker_id(self, broker_order_id: str) -> Optional[Order]:
        """Get an order by broker order ID."""
        order_id = self.broker_order_map.get(broker_order_id)
        if order_id:
            return self.orders.get(order_id)
        return None

    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        order_ids = self.symbol_orders.get(symbol, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]

    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return [self.orders[oid] for oid in self.active_orders if oid in self.orders]

    def get_active_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get active orders for a symbol."""
        orders = self.get_orders_by_symbol(symbol)
        return [o for o in orders if o.is_active()]

    def get_completed_orders(self, limit: Optional[int] = None) -> List[Order]:
        """Get completed orders."""
        order_ids = self.completed_orders[-limit:] if limit else self.completed_orders
        return [self.orders[oid] for oid in order_ids if oid in self.orders]

    def get_metrics(self) -> Dict[str, Any]:
        """Get order manager metrics."""
        return self.metrics.to_dict()

    def cancel_all_for_symbol(
        self,
        symbol: str,
        context: Optional[RunContext] = None
    ) -> int:
        """
        Cancel all active orders for a symbol.

        Args:
            symbol: Symbol to cancel orders for
            context: Optional run context

        Returns:
            Number of orders cancelled
        """
        active_orders = self.get_active_orders_by_symbol(symbol)
        count = 0

        for order in active_orders:
            if self.cancel(order.order_id, context):
                count += 1

        logger.info(f"Cancelled {count} orders for {symbol}", extra={
            'symbol': symbol,
            'count': count
        })

        return count

    def cancel_all(self, context: Optional[RunContext] = None) -> int:
        """
        Cancel all active orders.

        Args:
            context: Optional run context

        Returns:
            Number of orders cancelled
        """
        active_orders = self.get_active_orders()
        count = 0

        for order in active_orders:
            if self.cancel(order.order_id, context):
                count += 1

        logger.info(f"Cancelled {count} active orders", extra={'count': count})

        return count

    def _move_to_completed(self, order_id: str) -> None:
        """Move an order from active to completed."""
        if order_id in self.active_orders:
            self.active_orders.remove(order_id)

        if order_id not in self.completed_orders:
            self.completed_orders.append(order_id)

        # Trim completed orders if needed
        if len(self.completed_orders) > self.max_order_history:
            excess = len(self.completed_orders) - self.max_order_history
            old_order_ids = self.completed_orders[:excess]

            # Remove old orders from memory
            for oid in old_order_ids:
                if oid in self.orders:
                    order = self.orders[oid]
                    # Remove from symbol index
                    if order.symbol in self.symbol_orders:
                        if oid in self.symbol_orders[order.symbol]:
                            self.symbol_orders[order.symbol].remove(oid)
                    # Remove from broker map
                    if order.broker_order_id and order.broker_order_id in self.broker_order_map:
                        del self.broker_order_map[order.broker_order_id]
                    # Remove order
                    del self.orders[oid]

            self.completed_orders = self.completed_orders[excess:]

    def _generate_order_id(self) -> str:
        """Generate a unique order ID."""
        return f"ORD-{uuid.uuid4().hex[:12].upper()}"

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on order manager.

        Returns:
            Health status dictionary
        """
        return {
            'healthy': True,
            'total_orders': len(self.orders),
            'active_orders': len(self.active_orders),
            'completed_orders': len(self.completed_orders),
            'metrics': self.metrics.to_dict()
        }

"""
Execution Engine for Ultimate Trading Bot v2.2.

This module provides the central execution engine that coordinates
order management, position tracking, and strategy execution.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field

from src.execution.base_executor import (
    BaseExecutor,
    PaperExecutor,
    ExecutorConfig,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    Fill,
    Position,
    ExecutionState,
    ExecutionResult,
)
from src.execution.order_manager import OrderManager, OrderManagerConfig, OrderPriority
from src.execution.position_manager import PositionManager, PositionManagerConfig


logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """Execution modes."""

    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    DRY_RUN = "dry_run"


class ExecutionEngineConfig(BaseModel):
    """Configuration for execution engine."""

    model_config = {"arbitrary_types_allowed": True}

    mode: ExecutionMode = Field(default=ExecutionMode.PAPER, description="Execution mode")
    enable_risk_checks: bool = Field(default=True, description="Enable pre-trade risk checks")
    enable_position_limits: bool = Field(default=True, description="Enable position limits")
    max_order_value: float = Field(default=100000.0, description="Maximum order value")
    max_position_value: float = Field(default=250000.0, description="Maximum position value")
    max_daily_trades: int = Field(default=100, description="Maximum daily trades")
    max_daily_volume: float = Field(default=500000.0, description="Maximum daily volume")
    require_confirmation: bool = Field(default=False, description="Require order confirmation")
    log_all_activity: bool = Field(default=True, description="Log all activity")
    heartbeat_interval_seconds: float = Field(default=30.0, description="Heartbeat interval")


@dataclass
class ExecutionStats:
    """Execution statistics."""

    start_time: datetime = field(default_factory=datetime.now)
    total_orders: int = 0
    filled_orders: int = 0
    canceled_orders: int = 0
    rejected_orders: int = 0
    total_fills: int = 0
    total_volume: float = 0.0
    total_commission: float = 0.0
    daily_trades: int = 0
    daily_volume: float = 0.0
    last_trade_time: datetime | None = None


@dataclass
class TradeRecord:
    """Record of a trade execution."""

    trade_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    order_type: OrderType
    timestamp: datetime
    order_id: str
    strategy_id: str | None = None
    pnl: float | None = None
    commission: float = 0.0


class ExecutionEngine:
    """
    Central execution engine for trading operations.

    Coordinates order management, position tracking, risk checks,
    and provides a unified interface for strategy execution.
    """

    def __init__(
        self,
        executor: BaseExecutor | None = None,
        config: ExecutionEngineConfig | None = None,
    ):
        """
        Initialize execution engine.

        Args:
            executor: Order executor to use
            config: Engine configuration
        """
        self.config = config or ExecutionEngineConfig()

        if executor:
            self.executor = executor
        else:
            if self.config.mode == ExecutionMode.PAPER:
                self.executor = PaperExecutor()
            else:
                self.executor = PaperExecutor()

        self.order_manager = OrderManager(
            executor=self.executor,
            config=OrderManagerConfig(),
        )

        self.position_manager = PositionManager(
            executor=self.executor,
            config=PositionManagerConfig(),
        )

        self._state = ExecutionState.IDLE
        self._stats = ExecutionStats()
        self._trade_history: list[TradeRecord] = []
        self._pending_signals: list[dict[str, Any]] = []

        self._risk_checker: Callable[[Order, dict], tuple[bool, str | None]] | None = None
        self._callbacks: dict[str, list[Callable]] = {
            "order_submitted": [],
            "order_filled": [],
            "order_canceled": [],
            "trade_executed": [],
            "position_changed": [],
            "error": [],
        }

        self._running = False
        self._heartbeat_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        self._setup_internal_callbacks()

        logger.info(f"ExecutionEngine initialized (mode: {self.config.mode.value})")

    def _setup_internal_callbacks(self) -> None:
        """Setup internal callbacks between components."""
        self.order_manager.register_callback(
            "order_filled",
            self._on_order_filled,
        )

        self.order_manager.register_callback(
            "order_submitted",
            self._on_order_submitted,
        )

        self.order_manager.register_callback(
            "order_canceled",
            self._on_order_canceled,
        )

    async def start(self) -> None:
        """Start the execution engine."""
        self._running = True
        self._state = ExecutionState.RUNNING
        self._stats = ExecutionStats()

        await self.executor.start()
        await self.order_manager.start()
        await self.position_manager.start()

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info("ExecutionEngine started")

    async def stop(self) -> None:
        """Stop the execution engine."""
        self._running = False
        self._state = ExecutionState.STOPPING

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        await self.order_manager.stop()
        await self.position_manager.stop()
        await self.executor.stop()

        self._state = ExecutionState.STOPPED
        logger.info("ExecutionEngine stopped")

    async def execute_signal(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        strategy_id: str | None = None,
        signal_id: str | None = None,
        priority: OrderPriority = OrderPriority.NORMAL,
        **kwargs: Any,
    ) -> ExecutionResult:
        """
        Execute a trading signal.

        Args:
            symbol: Asset symbol
            side: Order side
            quantity: Order quantity
            order_type: Order type
            limit_price: Limit price
            stop_price: Stop price
            strategy_id: Strategy identifier
            signal_id: Signal identifier
            priority: Execution priority
            **kwargs: Additional order parameters

        Returns:
            ExecutionResult
        """
        if self._state != ExecutionState.RUNNING:
            return ExecutionResult(
                order=Order(
                    symbol=symbol, side=side, order_type=order_type,
                    quantity=quantity, limit_price=limit_price,
                ),
                success=False,
                error_message=f"Engine not running (state: {self._state.value})",
            )

        order = await self.executor.create_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            strategy_id=strategy_id,
            signal_id=signal_id,
            **kwargs,
        )

        if self.config.enable_risk_checks:
            approved, rejection_reason = await self._check_risk(order)
            if not approved:
                order.status = OrderStatus.REJECTED
                order.error_message = rejection_reason
                return ExecutionResult(
                    order=order,
                    success=False,
                    error_message=rejection_reason,
                )

        if self.config.enable_position_limits:
            within_limits, limit_error = await self._check_position_limits(order)
            if not within_limits:
                order.status = OrderStatus.REJECTED
                order.error_message = limit_error
                return ExecutionResult(
                    order=order,
                    success=False,
                    error_message=limit_error,
                )

        if self.config.require_confirmation:
            async with self._lock:
                self._pending_signals.append({
                    "order": order,
                    "priority": priority,
                    "timestamp": datetime.now(),
                })
            return ExecutionResult(
                order=order,
                success=True,
                error_message="Order pending confirmation",
            )

        result = await self.order_manager.submit_order(order)

        if result.success:
            async with self._lock:
                self._stats.total_orders += 1
                self._stats.daily_trades += 1
                self._stats.last_trade_time = datetime.now()

        return result

    async def _check_risk(
        self,
        order: Order,
    ) -> tuple[bool, str | None]:
        """Check order against risk limits."""
        order_value = order.quantity * (order.limit_price or 100.0)

        if order_value > self.config.max_order_value:
            return False, f"Order value ${order_value:,.2f} exceeds limit ${self.config.max_order_value:,.2f}"

        if self._stats.daily_trades >= self.config.max_daily_trades:
            return False, f"Daily trade limit ({self.config.max_daily_trades}) reached"

        if self._stats.daily_volume + order_value > self.config.max_daily_volume:
            return False, f"Daily volume limit ${self.config.max_daily_volume:,.2f} would be exceeded"

        if self._risk_checker:
            try:
                account = await self.executor.get_account_info()
                return self._risk_checker(order, account)
            except Exception as e:
                logger.error(f"Risk check error: {e}")
                return False, f"Risk check failed: {str(e)}"

        return True, None

    async def _check_position_limits(
        self,
        order: Order,
    ) -> tuple[bool, str | None]:
        """Check order against position limits."""
        position = self.position_manager.get_position(order.symbol)

        if position:
            current_value = abs(position.market_value)
            order_value = order.quantity * (order.limit_price or position.current_price)

            if order.side == OrderSide.BUY and position.is_long:
                new_value = current_value + order_value
            elif order.side == OrderSide.SELL and position.is_short:
                new_value = current_value + order_value
            else:
                new_value = abs(current_value - order_value)

            if new_value > self.config.max_position_value:
                return False, f"Position value would exceed limit ${self.config.max_position_value:,.2f}"

        return True, None

    async def _on_order_submitted(self, order: Order) -> None:
        """Handle order submitted event."""
        if self.config.log_all_activity:
            logger.info(
                f"Order submitted: {order.symbol} {order.side.value} "
                f"{order.quantity} @ {order.order_type.value}"
            )

        await self._trigger_callbacks("order_submitted", order)

    async def _on_order_filled(self, order: Order) -> None:
        """Handle order filled event."""
        async with self._lock:
            self._stats.filled_orders += 1
            self._stats.total_fills += 1

        fills = self.order_manager.get_fills(order.order_id)
        for fill in fills:
            await self.position_manager.record_fill(fill)

            trade = TradeRecord(
                trade_id=fill.fill_id,
                symbol=fill.symbol,
                side=fill.side,
                quantity=fill.quantity,
                price=fill.price,
                order_type=order.order_type,
                timestamp=fill.executed_at,
                order_id=order.order_id,
                strategy_id=order.strategy_id,
                commission=fill.commission,
            )
            self._trade_history.append(trade)

            async with self._lock:
                self._stats.total_volume += fill.quantity * fill.price
                self._stats.daily_volume += fill.quantity * fill.price
                self._stats.total_commission += fill.commission

            await self._trigger_callbacks("trade_executed", trade)

        if self.config.log_all_activity:
            logger.info(
                f"Order filled: {order.symbol} {order.side.value} "
                f"{order.filled_qty} @ {order.avg_fill_price}"
            )

        await self._trigger_callbacks("order_filled", order)

    async def _on_order_canceled(self, order: Order) -> None:
        """Handle order canceled event."""
        async with self._lock:
            self._stats.canceled_orders += 1

        if self.config.log_all_activity:
            logger.info(f"Order canceled: {order.order_id}")

        await self._trigger_callbacks("order_canceled", order)

    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop for monitoring."""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval_seconds)

                now = datetime.now()
                if now.hour == 0 and now.minute < 1:
                    self._stats.daily_trades = 0
                    self._stats.daily_volume = 0.0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def buy(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """
        Execute a buy order.

        Args:
            symbol: Asset symbol
            quantity: Quantity to buy
            order_type: Order type
            limit_price: Limit price
            **kwargs: Additional parameters

        Returns:
            ExecutionResult
        """
        return await self.execute_signal(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            **kwargs,
        )

    async def sell(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """
        Execute a sell order.

        Args:
            symbol: Asset symbol
            quantity: Quantity to sell
            order_type: Order type
            limit_price: Limit price
            **kwargs: Additional parameters

        Returns:
            ExecutionResult
        """
        return await self.execute_signal(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            **kwargs,
        )

    async def close_position(
        self,
        symbol: str,
        quantity: float | None = None,
    ) -> ExecutionResult | None:
        """
        Close a position.

        Args:
            symbol: Position symbol
            quantity: Quantity to close (None for full)

        Returns:
            ExecutionResult if order submitted
        """
        position = self.position_manager.get_position(symbol)
        if not position:
            return None

        close_qty = quantity or abs(position.quantity)
        close_side = OrderSide.SELL if position.is_long else OrderSide.BUY

        return await self.execute_signal(
            symbol=symbol,
            side=close_side,
            quantity=close_qty,
            order_type=OrderType.MARKET,
        )

    async def close_all_positions(self) -> list[ExecutionResult]:
        """
        Close all positions.

        Returns:
            List of execution results
        """
        results: list[ExecutionResult] = []

        for position in self.position_manager.get_all_positions():
            result = await self.close_position(position.symbol)
            if result:
                results.append(result)

        return results

    async def cancel_order(self, order_id: str) -> ExecutionResult:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            ExecutionResult
        """
        return await self.order_manager.cancel_order(order_id)

    async def cancel_all_orders(
        self,
        symbol: str | None = None,
    ) -> int:
        """
        Cancel all orders.

        Args:
            symbol: Filter by symbol

        Returns:
            Number of orders canceled
        """
        return await self.order_manager.cancel_all_orders(symbol=symbol)

    def set_risk_checker(
        self,
        checker: Callable[[Order, dict], tuple[bool, str | None]],
    ) -> None:
        """
        Set custom risk checker.

        Args:
            checker: Risk check function
        """
        self._risk_checker = checker

    def register_callback(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """Register callback for execution events."""
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
                await self._trigger_callbacks("error", {"event": event, "error": str(e)})

    def get_state(self) -> ExecutionState:
        """Get engine state."""
        return self._state

    def get_stats(self) -> ExecutionStats:
        """Get execution statistics."""
        return self._stats

    def get_trade_history(self, limit: int = 100) -> list[TradeRecord]:
        """Get trade history."""
        return self._trade_history[-limit:]

    def get_active_orders(self) -> list[Order]:
        """Get active orders."""
        return self.order_manager.get_active_orders()

    def get_positions(self) -> list:
        """Get all positions."""
        return self.position_manager.get_all_positions()

    def get_position(self, symbol: str):
        """Get position by symbol."""
        return self.position_manager.get_position(symbol)

    async def get_account_info(self) -> dict[str, Any]:
        """Get account information."""
        return await self.executor.get_account_info()

    async def get_summary(self) -> dict[str, Any]:
        """
        Get execution engine summary.

        Returns:
            Summary dictionary
        """
        account = await self.executor.get_account_info()
        portfolio = await self.position_manager.get_portfolio_summary()

        return {
            "timestamp": datetime.now().isoformat(),
            "state": self._state.value,
            "mode": self.config.mode.value,
            "account": account,
            "portfolio": portfolio,
            "stats": {
                "start_time": self._stats.start_time.isoformat(),
                "total_orders": self._stats.total_orders,
                "filled_orders": self._stats.filled_orders,
                "canceled_orders": self._stats.canceled_orders,
                "rejected_orders": self._stats.rejected_orders,
                "total_volume": self._stats.total_volume,
                "total_commission": self._stats.total_commission,
                "daily_trades": self._stats.daily_trades,
                "daily_volume": self._stats.daily_volume,
            },
            "active_orders": len(self.order_manager.get_active_orders()),
            "pending_orders": len(self.order_manager.get_pending_orders()),
        }

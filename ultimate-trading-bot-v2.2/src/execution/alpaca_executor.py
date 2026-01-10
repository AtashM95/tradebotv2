"""
Alpaca Broker Executor for Ultimate Trading Bot v2.2.

This module provides integration with Alpaca Trading API
for real order execution and account management.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.execution.base_executor import (
    BaseExecutor,
    ExecutorConfig,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
    Position,
    Fill,
    FillType,
    ExecutionResult,
)


logger = logging.getLogger(__name__)


class AlpacaConfig(BaseModel):
    """Configuration for Alpaca executor."""

    model_config = {"arbitrary_types_allowed": True}

    api_key: str = Field(default="", description="Alpaca API key")
    api_secret: str = Field(default="", description="Alpaca API secret")
    base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca API base URL"
    )
    data_url: str = Field(
        default="https://data.alpaca.markets",
        description="Alpaca data API URL"
    )
    use_paper: bool = Field(default=True, description="Use paper trading")
    stream_trades: bool = Field(default=True, description="Stream trade updates")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout_seconds: float = Field(default=30.0, description="Request timeout")


class AlpacaExecutor(BaseExecutor):
    """
    Executor for Alpaca Trading API.

    Provides real-time order execution, position management,
    and account information through Alpaca's API.
    """

    def __init__(
        self,
        alpaca_config: AlpacaConfig | None = None,
        executor_config: ExecutorConfig | None = None,
    ):
        """
        Initialize Alpaca executor.

        Args:
            alpaca_config: Alpaca-specific configuration
            executor_config: Base executor configuration
        """
        super().__init__(executor_config)

        self.alpaca_config = alpaca_config or AlpacaConfig()
        self._trading_client: Any = None
        self._data_client: Any = None
        self._stream_client: Any = None
        self._is_connected = False
        self._stream_task: asyncio.Task | None = None

        logger.info(
            f"AlpacaExecutor initialized (paper={self.alpaca_config.use_paper})"
        )

    async def connect(self) -> bool:
        """
        Connect to Alpaca API.

        Returns:
            Whether connection was successful
        """
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            self._trading_client = TradingClient(
                api_key=self.alpaca_config.api_key,
                secret_key=self.alpaca_config.api_secret,
                paper=self.alpaca_config.use_paper,
            )

            self._data_client = StockHistoricalDataClient(
                api_key=self.alpaca_config.api_key,
                secret_key=self.alpaca_config.api_secret,
            )

            account = self._trading_client.get_account()
            logger.info(
                f"Connected to Alpaca. Account: {account.account_number}, "
                f"Equity: ${float(account.equity):,.2f}"
            )

            self._is_connected = True

            if self.alpaca_config.stream_trades:
                self._stream_task = asyncio.create_task(self._start_stream())

            return True

        except ImportError:
            logger.error("Alpaca SDK not installed. Run: pip install alpaca-py")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        self._is_connected = False
        logger.info("Disconnected from Alpaca")

    async def _start_stream(self) -> None:
        """Start streaming trade updates."""
        try:
            from alpaca.trading.stream import TradingStream

            stream = TradingStream(
                api_key=self.alpaca_config.api_key,
                secret_key=self.alpaca_config.api_secret,
                paper=self.alpaca_config.use_paper,
            )

            @stream.on("trade_updates")
            async def handle_trade_update(data: Any) -> None:
                await self._process_trade_update(data)

            await stream.run()

        except Exception as e:
            logger.error(f"Trade stream error: {e}")

    async def _process_trade_update(self, data: Any) -> None:
        """Process incoming trade update."""
        try:
            event = data.event
            order_data = data.order

            order_id = str(order_data.id)
            client_order_id = order_data.client_order_id

            order = self._active_orders.get(order_id)
            if not order:
                order = self._active_orders.get(client_order_id)

            if not order:
                logger.warning(f"Received update for unknown order: {order_id}")
                return

            if event == "fill":
                fill = Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=float(order_data.filled_qty) - order.filled_qty,
                    price=float(order_data.filled_avg_price),
                    fill_type=FillType.FULL if order_data.filled_qty == order_data.qty else FillType.PARTIAL,
                )

                order.filled_qty = float(order_data.filled_qty)
                order.avg_fill_price = float(order_data.filled_avg_price)
                order.remaining_qty = float(order_data.qty) - order.filled_qty
                order.status = self._map_alpaca_status(order_data.status)

                if order.status == OrderStatus.FILLED:
                    order.filled_at = datetime.now()

                async with self._lock:
                    self._fill_history.append(fill)

                await self._trigger_callbacks("fill_received", fill)

                if order.status == OrderStatus.FILLED:
                    await self._trigger_callbacks("order_filled", order)
                    async with self._lock:
                        if order.order_id in self._active_orders:
                            del self._active_orders[order.order_id]
                        self._order_history.append(order)

            elif event == "canceled":
                order.status = OrderStatus.CANCELED
                order.canceled_at = datetime.now()
                await self._trigger_callbacks("order_canceled", order)

                async with self._lock:
                    if order.order_id in self._active_orders:
                        del self._active_orders[order.order_id]
                    self._order_history.append(order)

            elif event == "rejected":
                order.status = OrderStatus.REJECTED
                order.error_message = getattr(order_data, "failed_at", "Unknown reason")
                await self._trigger_callbacks("order_rejected", order)

                async with self._lock:
                    if order.order_id in self._active_orders:
                        del self._active_orders[order.order_id]
                    self._order_history.append(order)

        except Exception as e:
            logger.error(f"Error processing trade update: {e}")

    def _map_alpaca_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca status to internal status."""
        status_map = {
            "new": OrderStatus.NEW,
            "accepted": OrderStatus.ACCEPTED,
            "pending_new": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "done_for_day": OrderStatus.EXPIRED,
            "canceled": OrderStatus.CANCELED,
            "expired": OrderStatus.EXPIRED,
            "replaced": OrderStatus.REPLACED,
            "pending_cancel": OrderStatus.PENDING_CANCEL,
            "pending_replace": OrderStatus.PENDING_REPLACE,
            "rejected": OrderStatus.REJECTED,
        }
        return status_map.get(str(alpaca_status).lower(), OrderStatus.PENDING)

    def _map_order_type(self, order_type: OrderType) -> str:
        """Map internal order type to Alpaca format."""
        type_map = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit",
            OrderType.TRAILING_STOP: "trailing_stop",
        }
        return type_map.get(order_type, "market")

    def _map_time_in_force(self, tif: TimeInForce) -> str:
        """Map internal TIF to Alpaca format."""
        tif_map = {
            TimeInForce.DAY: "day",
            TimeInForce.GTC: "gtc",
            TimeInForce.IOC: "ioc",
            TimeInForce.FOK: "fok",
            TimeInForce.OPG: "opg",
            TimeInForce.CLS: "cls",
        }
        return tif_map.get(tif, "day")

    async def submit_order(self, order: Order) -> ExecutionResult:
        """
        Submit order to Alpaca.

        Args:
            order: Order to submit

        Returns:
            ExecutionResult
        """
        start_time = datetime.now()

        if not self._is_connected:
            return ExecutionResult(
                order=order,
                success=False,
                error_message="Not connected to Alpaca",
            )

        if self.config.validate_orders:
            valid, error = self.validate_order(order)
            if not valid:
                order.status = OrderStatus.REJECTED
                order.error_message = error
                return ExecutionResult(order=order, success=False, error_message=error)

        try:
            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
                StopOrderRequest,
                StopLimitOrderRequest,
                TrailingStopOrderRequest,
            )
            from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce as AlpacaTIF

            side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL
            tif = AlpacaTIF(self._map_time_in_force(order.time_in_force))

            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    client_order_id=order.client_order_id,
                    extended_hours=order.extended_hours,
                )
            elif order.order_type == OrderType.LIMIT:
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    limit_price=order.limit_price,
                    client_order_id=order.client_order_id,
                    extended_hours=order.extended_hours,
                )
            elif order.order_type == OrderType.STOP:
                request = StopOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    stop_price=order.stop_price,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == OrderType.STOP_LIMIT:
                request = StopLimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    stop_price=order.stop_price,
                    limit_price=order.limit_price,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == OrderType.TRAILING_STOP:
                request = TrailingStopOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    trail_percent=order.trail_percent,
                    trail_price=order.trail_amount,
                    client_order_id=order.client_order_id,
                )
            else:
                return ExecutionResult(
                    order=order,
                    success=False,
                    error_message=f"Unsupported order type: {order.order_type}",
                )

            alpaca_order = self._trading_client.submit_order(request)

            order.broker_order_id = str(alpaca_order.id)
            order.status = self._map_alpaca_status(alpaca_order.status)
            order.submitted_at = datetime.now()

            if alpaca_order.filled_qty:
                order.filled_qty = float(alpaca_order.filled_qty)
            if alpaca_order.filled_avg_price:
                order.avg_fill_price = float(alpaca_order.filled_avg_price)

            async with self._lock:
                self._active_orders[order.order_id] = order

            await self._trigger_callbacks("order_submitted", order)

            exec_time = (datetime.now() - start_time).total_seconds() * 1000

            logger.info(
                f"Order submitted: {order.symbol} {order.side.value} {order.quantity} "
                f"@ {order.order_type.value} (ID: {order.broker_order_id})"
            )

            return ExecutionResult(
                order=order,
                success=True,
                broker_response={"alpaca_order_id": str(alpaca_order.id)},
                execution_time_ms=exec_time,
            )

        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_message = str(e)

            logger.error(f"Failed to submit order: {e}")

            await self._trigger_callbacks("error", {"order": order, "error": str(e)})

            return ExecutionResult(
                order=order,
                success=False,
                error_message=str(e),
            )

    async def cancel_order(self, order_id: str) -> ExecutionResult:
        """
        Cancel order on Alpaca.

        Args:
            order_id: Order ID to cancel

        Returns:
            ExecutionResult
        """
        if not self._is_connected:
            return ExecutionResult(
                order=Order(order_id=order_id, symbol="", side=OrderSide.BUY,
                           order_type=OrderType.MARKET, quantity=0),
                success=False,
                error_message="Not connected to Alpaca",
            )

        try:
            order = self._active_orders.get(order_id)
            if not order:
                return ExecutionResult(
                    order=Order(order_id=order_id, symbol="", side=OrderSide.BUY,
                               order_type=OrderType.MARKET, quantity=0),
                    success=False,
                    error_message="Order not found",
                )

            cancel_id = order.broker_order_id or order.client_order_id

            self._trading_client.cancel_order_by_id(cancel_id)

            order.status = OrderStatus.PENDING_CANCEL

            logger.info(f"Cancel request sent for order: {order_id}")

            return ExecutionResult(order=order, success=True)

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return ExecutionResult(
                order=self._active_orders.get(order_id, Order(
                    order_id=order_id, symbol="", side=OrderSide.BUY,
                    order_type=OrderType.MARKET, quantity=0
                )),
                success=False,
                error_message=str(e),
            )

    async def modify_order(
        self,
        order_id: str,
        new_quantity: float | None = None,
        new_limit_price: float | None = None,
        new_stop_price: float | None = None,
    ) -> ExecutionResult:
        """
        Modify order on Alpaca.

        Args:
            order_id: Order ID to modify
            new_quantity: New quantity
            new_limit_price: New limit price
            new_stop_price: New stop price

        Returns:
            ExecutionResult
        """
        if not self._is_connected:
            return ExecutionResult(
                order=Order(order_id=order_id, symbol="", side=OrderSide.BUY,
                           order_type=OrderType.MARKET, quantity=0),
                success=False,
                error_message="Not connected to Alpaca",
            )

        try:
            from alpaca.trading.requests import ReplaceOrderRequest

            order = self._active_orders.get(order_id)
            if not order:
                return ExecutionResult(
                    order=Order(order_id=order_id, symbol="", side=OrderSide.BUY,
                               order_type=OrderType.MARKET, quantity=0),
                    success=False,
                    error_message="Order not found",
                )

            replace_id = order.broker_order_id or order.client_order_id

            request = ReplaceOrderRequest(
                qty=new_quantity,
                limit_price=new_limit_price,
                stop_price=new_stop_price,
            )

            new_order = self._trading_client.replace_order_by_id(
                order_id=replace_id,
                order_data=request,
            )

            order.broker_order_id = str(new_order.id)
            if new_quantity:
                order.quantity = new_quantity
                order.remaining_qty = new_quantity - order.filled_qty
            if new_limit_price:
                order.limit_price = new_limit_price
            if new_stop_price:
                order.stop_price = new_stop_price

            order.status = OrderStatus.REPLACED

            logger.info(f"Order modified: {order_id}")

            return ExecutionResult(order=order, success=True)

        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return ExecutionResult(
                order=self._active_orders.get(order_id, Order(
                    order_id=order_id, symbol="", side=OrderSide.BUY,
                    order_type=OrderType.MARKET, quantity=0
                )),
                success=False,
                error_message=str(e),
            )

    async def get_order_status(self, order_id: str) -> Order | None:
        """
        Get order status from Alpaca.

        Args:
            order_id: Order ID

        Returns:
            Order if found
        """
        if order_id in self._active_orders:
            order = self._active_orders[order_id]

            if self._is_connected:
                try:
                    lookup_id = order.broker_order_id or order.client_order_id
                    alpaca_order = self._trading_client.get_order_by_id(lookup_id)

                    order.status = self._map_alpaca_status(alpaca_order.status)
                    if alpaca_order.filled_qty:
                        order.filled_qty = float(alpaca_order.filled_qty)
                    if alpaca_order.filled_avg_price:
                        order.avg_fill_price = float(alpaca_order.filled_avg_price)

                except Exception as e:
                    logger.error(f"Failed to get order status: {e}")

            return order

        for order in self._order_history:
            if order.order_id == order_id:
                return order

        return None

    async def get_positions(self) -> list[Position]:
        """
        Get positions from Alpaca.

        Returns:
            List of positions
        """
        if not self._is_connected:
            return []

        try:
            alpaca_positions = self._trading_client.get_all_positions()

            positions: list[Position] = []
            for pos in alpaca_positions:
                position = Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    avg_entry_price=float(pos.avg_entry_price),
                    market_value=float(pos.market_value),
                    cost_basis=float(pos.cost_basis),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc),
                    current_price=float(pos.current_price),
                    side="long" if float(pos.qty) > 0 else "short",
                    asset_class=str(pos.asset_class),
                    exchange=str(pos.exchange),
                )
                positions.append(position)

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_account_info(self) -> dict[str, Any]:
        """
        Get account information from Alpaca.

        Returns:
            Account information dictionary
        """
        if not self._is_connected:
            return {"error": "Not connected"}

        try:
            account = self._trading_client.get_account()

            return {
                "account_number": account.account_number,
                "status": str(account.status),
                "cash": float(account.cash),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "daytrading_buying_power": float(account.daytrading_buying_power),
                "portfolio_value": float(account.portfolio_value),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "last_equity": float(account.last_equity),
                "sma": float(account.sma),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked,
                "trade_suspended_by_user": account.trade_suspended_by_user,
                "is_paper": self.alpaca_config.use_paper,
            }

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {"error": str(e)}

    async def get_open_orders(self) -> list[Order]:
        """
        Get all open orders from Alpaca.

        Returns:
            List of open orders
        """
        if not self._is_connected:
            return []

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            alpaca_orders = self._trading_client.get_orders(request)

            orders: list[Order] = []
            for ao in alpaca_orders:
                order = Order(
                    order_id=str(ao.id),
                    client_order_id=ao.client_order_id,
                    broker_order_id=str(ao.id),
                    symbol=ao.symbol,
                    side=OrderSide.BUY if str(ao.side) == "buy" else OrderSide.SELL,
                    order_type=OrderType(str(ao.type)),
                    quantity=float(ao.qty),
                    limit_price=float(ao.limit_price) if ao.limit_price else None,
                    stop_price=float(ao.stop_price) if ao.stop_price else None,
                    status=self._map_alpaca_status(ao.status),
                    filled_qty=float(ao.filled_qty) if ao.filled_qty else 0,
                    avg_fill_price=float(ao.filled_avg_price) if ao.filled_avg_price else None,
                )
                order.update_remaining()
                orders.append(order)

            return orders

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    async def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.

        Returns:
            Number of orders canceled
        """
        if not self._is_connected:
            return 0

        try:
            responses = self._trading_client.cancel_orders()
            canceled_count = len(responses) if responses else 0
            logger.info(f"Canceled {canceled_count} orders")
            return canceled_count

        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 0

    async def close_all_positions(self) -> int:
        """
        Close all positions.

        Returns:
            Number of positions closed
        """
        if not self._is_connected:
            return 0

        try:
            responses = self._trading_client.close_all_positions(cancel_orders=True)
            closed_count = len(responses) if responses else 0
            logger.info(f"Closed {closed_count} positions")
            return closed_count

        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return 0

    def is_connected(self) -> bool:
        """Check if connected to Alpaca."""
        return self._is_connected

"""
Position Manager Module for Ultimate Trading Bot v2.2.

This module provides comprehensive position management including tracking,
P&L calculation, stop/target monitoring, and position lifecycle management.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
import logging
from collections import defaultdict

from pydantic import BaseModel, Field

from src.core.event_bus import EventBus, EventType, Event, get_event_bus
from src.core.models import (
    Position, ClosedPosition, PositionSide, Order, OrderSide, OrderFill
)


logger = logging.getLogger(__name__)


class PositionManagerConfig(BaseModel):
    """Position manager configuration."""

    max_positions: int = Field(default=20, ge=1, le=100, description="Max open positions")
    track_closed_positions: bool = Field(default=True, description="Track closed positions")
    max_closed_history: int = Field(default=1000, ge=100, description="Max closed position history")
    auto_update_prices: bool = Field(default=True, description="Auto-update position prices")
    price_update_interval: float = Field(default=1.0, ge=0.1, description="Price update interval")
    check_stops_interval: float = Field(default=0.5, ge=0.1, description="Stop check interval")
    default_stop_loss_pct: Optional[float] = Field(default=None, description="Default stop loss %")
    default_take_profit_pct: Optional[float] = Field(default=None, description="Default take profit %")
    enable_trailing_stops: bool = Field(default=True, description="Enable trailing stops")


class PositionMetrics(BaseModel):
    """Portfolio position metrics."""

    total_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0
    total_value: float = 0.0
    total_cost: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    winning_positions: int = 0
    losing_positions: int = 0


class PositionManager:
    """
    Manages all position-related operations.

    Provides functionality for:
    - Position tracking and lifecycle management
    - P&L calculation (unrealized and realized)
    - Stop loss and take profit monitoring
    - Trailing stop management
    - Position metrics and reporting
    """

    def __init__(
        self,
        config: Optional[PositionManagerConfig] = None,
        event_bus: Optional[EventBus] = None,
        data_provider: Optional[Any] = None
    ) -> None:
        """
        Initialize the position manager.

        Args:
            config: Position manager configuration
            event_bus: Event bus instance
            data_provider: Market data provider
        """
        self.config = config or PositionManagerConfig()
        self.event_bus = event_bus or get_event_bus()
        self._data_provider = data_provider

        # Position storage
        self._positions: Dict[str, Position] = {}
        self._positions_by_symbol: Dict[str, str] = {}  # symbol -> position_id
        self._closed_positions: List[ClosedPosition] = []

        # State
        self._lock = asyncio.Lock()
        self._price_update_task: Optional[asyncio.Task] = None
        self._stop_check_task: Optional[asyncio.Task] = None
        self._running = False
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Set up event handlers
        self._setup_event_handlers()

        logger.info("PositionManager initialized")

    def _setup_event_handlers(self) -> None:
        """Set up event handlers."""
        self.event_bus.subscribe(
            EventType.ORDER_FILLED,
            self._on_order_filled
        )
        self.event_bus.subscribe(
            EventType.QUOTE_UPDATE,
            self._on_quote_update
        )

    def set_data_provider(self, provider: Any) -> None:
        """Set the market data provider."""
        self._data_provider = provider
        logger.debug("Data provider set")

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def start(self) -> None:
        """Start the position manager."""
        if self._running:
            return

        self._running = True

        # Start price update task
        if self.config.auto_update_prices:
            self._price_update_task = asyncio.create_task(self._price_update_loop())

        # Start stop check task
        self._stop_check_task = asyncio.create_task(self._stop_check_loop())

        logger.info("PositionManager started")

    async def stop(self) -> None:
        """Stop the position manager."""
        self._running = False

        if self._price_update_task:
            self._price_update_task.cancel()
            try:
                await self._price_update_task
            except asyncio.CancelledError:
                pass

        if self._stop_check_task:
            self._stop_check_task.cancel()
            try:
                await self._stop_check_task
            except asyncio.CancelledError:
                pass

        logger.info("PositionManager stopped")

    async def _price_update_loop(self) -> None:
        """Periodically update position prices."""
        while self._running:
            try:
                await self._update_all_prices()
                await asyncio.sleep(self.config.price_update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in price update loop: {e}")
                await asyncio.sleep(1)

    async def _stop_check_loop(self) -> None:
        """Periodically check stops and targets."""
        while self._running:
            try:
                await self._check_all_stops()
                await asyncio.sleep(self.config.check_stops_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stop check loop: {e}")
                await asyncio.sleep(1)

    # =========================================================================
    # POSITION CREATION
    # =========================================================================

    async def open_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        strategy_id: Optional[str] = None,
        entry_order_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[Position]:
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            side: Position side (long/short)
            quantity: Position quantity
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            trailing_stop: Trailing stop distance
            strategy_id: Associated strategy ID
            entry_order_id: Entry order ID
            tags: Custom tags

        Returns:
            Created position or None
        """
        async with self._lock:
            # Check if position already exists for symbol
            if symbol.upper() in self._positions_by_symbol:
                existing_id = self._positions_by_symbol[symbol.upper()]
                existing = self._positions[existing_id]
                logger.warning(f"Position already exists for {symbol}, updating instead")
                return await self._update_existing_position(existing, quantity, entry_price)

            # Check position limit
            if len(self._positions) >= self.config.max_positions:
                logger.error("Maximum positions reached")
                return None

            # Apply default stop/target if configured
            if stop_loss is None and self.config.default_stop_loss_pct:
                if side == PositionSide.LONG:
                    stop_loss = entry_price * (1 - self.config.default_stop_loss_pct)
                else:
                    stop_loss = entry_price * (1 + self.config.default_stop_loss_pct)

            if take_profit is None and self.config.default_take_profit_pct:
                if side == PositionSide.LONG:
                    take_profit = entry_price * (1 + self.config.default_take_profit_pct)
                else:
                    take_profit = entry_price * (1 - self.config.default_take_profit_pct)

            # Create position
            position = Position(
                symbol=symbol.upper(),
                side=side,
                quantity=quantity,
                avg_entry_price=entry_price,
                current_price=entry_price,
                market_value=quantity * entry_price,
                cost_basis=quantity * entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=trailing_stop,
                strategy_id=strategy_id,
                entry_order_id=entry_order_id,
                tags=tags or {}
            )

            # Initialize trailing stop if enabled
            if trailing_stop and self.config.enable_trailing_stops:
                if side == PositionSide.LONG:
                    position.trailing_stop_price = entry_price - trailing_stop
                else:
                    position.trailing_stop_price = entry_price + trailing_stop

            # Store position
            self._positions[position.position_id] = position
            self._positions_by_symbol[position.symbol] = position.position_id

            # Emit event
            await self.event_bus.emit(
                EventType.POSITION_OPENED,
                data=position,
                source="position_manager"
            )

            logger.info(
                f"Position opened: {position.position_id} - "
                f"{side.value} {quantity} {symbol} @ {entry_price}"
            )
            return position

    async def _update_existing_position(
        self,
        position: Position,
        quantity: float,
        price: float
    ) -> Position:
        """Update an existing position with a new fill."""
        # Calculate new average price
        total_cost = position.cost_basis + (quantity * price)
        new_quantity = position.quantity + quantity
        new_avg_price = total_cost / new_quantity

        position.quantity = new_quantity
        position.avg_entry_price = new_avg_price
        position.cost_basis = total_cost
        position.update_price(position.current_price)
        position.update_timestamp()

        await self.event_bus.emit(
            EventType.POSITION_UPDATED,
            data=position,
            source="position_manager"
        )

        return position

    # =========================================================================
    # POSITION CLOSING
    # =========================================================================

    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_quantity: Optional[float] = None,
        exit_reason: str = "manual",
        exit_order_id: Optional[str] = None
    ) -> Optional[ClosedPosition]:
        """
        Close a position.

        Args:
            position_id: Position ID to close
            exit_price: Exit price
            exit_quantity: Quantity to close (None = all)
            exit_reason: Reason for closing
            exit_order_id: Exit order ID

        Returns:
            ClosedPosition record or None
        """
        async with self._lock:
            position = self._positions.get(position_id)
            if not position:
                logger.error(f"Position not found: {position_id}")
                return None

            # Determine quantity to close
            close_quantity = exit_quantity or position.quantity

            if close_quantity > position.quantity:
                close_quantity = position.quantity

            # Calculate P&L
            if position.side == PositionSide.LONG:
                realized_pnl = (exit_price - position.avg_entry_price) * close_quantity
            else:
                realized_pnl = (position.avg_entry_price - exit_price) * close_quantity

            realized_pnl_pct = realized_pnl / (position.avg_entry_price * close_quantity)

            # Create closed position record
            closed = ClosedPosition(
                position_id=position_id,
                symbol=position.symbol,
                side=position.side,
                quantity=close_quantity,
                entry_price=position.avg_entry_price,
                exit_price=exit_price,
                entry_time=position.created_at,
                exit_time=datetime.now(timezone.utc),
                realized_pnl=realized_pnl,
                realized_pnl_percent=realized_pnl_pct,
                strategy_id=position.strategy_id,
                exit_reason=exit_reason,
                tags=position.tags
            )

            # Update or remove position
            if close_quantity >= position.quantity:
                # Full close
                del self._positions[position_id]
                if position.symbol in self._positions_by_symbol:
                    del self._positions_by_symbol[position.symbol]
            else:
                # Partial close
                position.quantity -= close_quantity
                position.cost_basis = position.quantity * position.avg_entry_price
                position.realized_pnl += realized_pnl
                position.update_price(position.current_price)
                position.update_timestamp()

            # Store closed position
            if self.config.track_closed_positions:
                self._closed_positions.append(closed)
                if len(self._closed_positions) > self.config.max_closed_history:
                    self._closed_positions = self._closed_positions[-self.config.max_closed_history:]

            # Emit event
            await self.event_bus.emit(
                EventType.POSITION_CLOSED,
                data=closed,
                source="position_manager"
            )

            logger.info(
                f"Position closed: {position_id} - "
                f"{position.symbol} PnL: {realized_pnl:.2f} ({realized_pnl_pct:.2%})"
            )
            return closed

    async def close_all_positions(
        self,
        symbol: Optional[str] = None,
        exit_reason: str = "close_all"
    ) -> List[ClosedPosition]:
        """
        Close all positions, optionally filtered by symbol.

        Args:
            symbol: Optional symbol filter
            exit_reason: Reason for closing

        Returns:
            List of closed positions
        """
        closed = []

        positions = list(self._positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol.upper()]

        for position in positions:
            # Get current price
            price = position.current_price
            if self._data_provider:
                try:
                    quote = await self._data_provider.get_quote(position.symbol)
                    if quote:
                        price = quote.last_price or quote.mid_price
                except Exception:
                    pass

            result = await self.close_position(
                position.position_id,
                exit_price=price,
                exit_reason=exit_reason
            )
            if result:
                closed.append(result)

        return closed

    # =========================================================================
    # POSITION QUERIES
    # =========================================================================

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a position by ID."""
        return self._positions.get(position_id)

    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get a position by symbol."""
        position_id = self._positions_by_symbol.get(symbol.upper())
        if position_id:
            return self._positions.get(position_id)
        return None

    async def get_positions(
        self,
        side: Optional[PositionSide] = None,
        strategy_id: Optional[str] = None,
        profitable: Optional[bool] = None
    ) -> List[Position]:
        """
        Get positions with optional filters.

        Args:
            side: Filter by side
            strategy_id: Filter by strategy
            profitable: Filter by profitability

        Returns:
            List of matching positions
        """
        positions = list(self._positions.values())

        if side:
            positions = [p for p in positions if p.side == side]

        if strategy_id:
            positions = [p for p in positions if p.strategy_id == strategy_id]

        if profitable is not None:
            if profitable:
                positions = [p for p in positions if p.unrealized_pnl > 0]
            else:
                positions = [p for p in positions if p.unrealized_pnl <= 0]

        return positions

    def get_closed_positions(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[ClosedPosition]:
        """Get closed position history."""
        positions = self._closed_positions

        if symbol:
            positions = [p for p in positions if p.symbol == symbol.upper()]

        return positions[-limit:]

    def has_position(self, symbol: str) -> bool:
        """Check if a position exists for a symbol."""
        return symbol.upper() in self._positions_by_symbol

    def get_position_count(self) -> int:
        """Get the number of open positions."""
        return len(self._positions)

    # =========================================================================
    # PRICE UPDATES
    # =========================================================================

    async def update_price(self, symbol: str, price: float) -> None:
        """
        Update the price for a position.

        Args:
            symbol: Symbol to update
            price: New price
        """
        position_id = self._positions_by_symbol.get(symbol.upper())
        if position_id:
            position = self._positions[position_id]
            position.update_price(price)

    async def _update_all_prices(self) -> None:
        """Update prices for all positions."""
        if not self._data_provider:
            return

        symbols = list(self._positions_by_symbol.keys())
        if not symbols:
            return

        try:
            quotes = await self._data_provider.get_quotes(symbols)
            for symbol, quote in quotes.items():
                if quote and (quote.last_price or quote.mid_price):
                    price = quote.last_price or quote.mid_price
                    await self.update_price(symbol, price)
        except Exception as e:
            logger.error(f"Error updating prices: {e}")

    async def _on_quote_update(self, event: Event) -> None:
        """Handle quote update event."""
        quote = event.data
        if hasattr(quote, 'symbol') and hasattr(quote, 'last_price'):
            await self.update_price(quote.symbol, quote.last_price or quote.mid_price)

    # =========================================================================
    # STOP/TARGET MANAGEMENT
    # =========================================================================

    async def set_stop_loss(
        self,
        position_id: str,
        stop_price: float
    ) -> bool:
        """Set stop loss for a position."""
        position = self._positions.get(position_id)
        if not position:
            return False

        position.stop_loss = stop_price
        position.update_timestamp()

        logger.info(f"Stop loss set for {position.symbol}: {stop_price}")
        return True

    async def set_take_profit(
        self,
        position_id: str,
        target_price: float
    ) -> bool:
        """Set take profit for a position."""
        position = self._positions.get(position_id)
        if not position:
            return False

        position.take_profit = target_price
        position.update_timestamp()

        logger.info(f"Take profit set for {position.symbol}: {target_price}")
        return True

    async def set_trailing_stop(
        self,
        position_id: str,
        trailing_distance: float
    ) -> bool:
        """Set trailing stop for a position."""
        if not self.config.enable_trailing_stops:
            logger.warning("Trailing stops are disabled")
            return False

        position = self._positions.get(position_id)
        if not position:
            return False

        position.trailing_stop = trailing_distance

        # Initialize trailing stop price
        if position.side == PositionSide.LONG:
            position.trailing_stop_price = position.current_price - trailing_distance
        else:
            position.trailing_stop_price = position.current_price + trailing_distance

        position.update_timestamp()

        logger.info(f"Trailing stop set for {position.symbol}: {trailing_distance}")
        return True

    async def _check_all_stops(self) -> None:
        """Check all positions for stop/target triggers."""
        for position in list(self._positions.values()):
            # Check stop loss
            if position.should_stop_out():
                await self.event_bus.emit(
                    EventType.POSITION_STOP_TRIGGERED,
                    data=position,
                    source="position_manager"
                )
                logger.warning(f"Stop triggered for {position.symbol}")

            # Check take profit
            elif position.should_take_profit():
                await self.event_bus.emit(
                    EventType.POSITION_TARGET_REACHED,
                    data=position,
                    source="position_manager"
                )
                logger.info(f"Target reached for {position.symbol}")

    # =========================================================================
    # METRICS
    # =========================================================================

    def get_metrics(self) -> PositionMetrics:
        """Calculate current position metrics."""
        metrics = PositionMetrics()

        for position in self._positions.values():
            metrics.total_positions += 1
            metrics.total_value += position.market_value
            metrics.total_cost += position.cost_basis
            metrics.total_unrealized_pnl += position.unrealized_pnl

            if position.side == PositionSide.LONG:
                metrics.long_positions += 1
                metrics.long_exposure += position.market_value
            else:
                metrics.short_positions += 1
                metrics.short_exposure += position.market_value

            if position.unrealized_pnl > 0:
                metrics.winning_positions += 1
            else:
                metrics.losing_positions += 1

        metrics.net_exposure = metrics.long_exposure - metrics.short_exposure
        metrics.gross_exposure = metrics.long_exposure + metrics.short_exposure

        # Add realized P&L from closed positions
        for closed in self._closed_positions:
            metrics.total_realized_pnl += closed.realized_pnl

        return metrics

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    async def _on_order_filled(self, event: Event) -> None:
        """Handle order filled event to create/update positions."""
        order = event.data
        if not isinstance(order, Order):
            return

        # Determine position side
        if order.side == OrderSide.BUY:
            position_side = PositionSide.LONG
        else:
            position_side = PositionSide.SHORT

        # Check for existing position
        existing = self.get_position_by_symbol(order.symbol)

        if existing:
            # Check if this closes the position
            if existing.side != position_side:
                # Closing trade
                await self.close_position(
                    existing.position_id,
                    exit_price=order.avg_fill_price or order.limit_price,
                    exit_quantity=order.filled_quantity,
                    exit_reason="order_fill",
                    exit_order_id=order.order_id
                )
            else:
                # Adding to position
                await self._update_existing_position(
                    existing,
                    order.filled_quantity,
                    order.avg_fill_price or order.limit_price
                )
        else:
            # New position
            await self.open_position(
                symbol=order.symbol,
                side=position_side,
                quantity=order.filled_quantity,
                entry_price=order.avg_fill_price or order.limit_price,
                strategy_id=order.strategy_id,
                entry_order_id=order.order_id
            )

    # =========================================================================
    # SYNC
    # =========================================================================

    async def sync_positions(self) -> None:
        """Synchronize positions with broker."""
        if self._data_provider and hasattr(self._data_provider, 'get_positions'):
            try:
                broker_positions = await self._data_provider.get_positions()
                for pos_data in broker_positions:
                    symbol = pos_data.get('symbol', '').upper()
                    if symbol and symbol not in self._positions_by_symbol:
                        position = Position(**pos_data)
                        self._positions[position.position_id] = position
                        self._positions_by_symbol[position.symbol] = position.position_id

                logger.info(f"Synced {len(broker_positions)} positions from broker")
            except Exception as e:
                logger.error(f"Error syncing positions: {e}")

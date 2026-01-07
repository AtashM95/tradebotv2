"""
Trading Engine Module for Ultimate Trading Bot v2.2.

This module provides the main trading engine that orchestrates all
trading activities including strategy execution, order management,
and risk control.
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type
import logging

from pydantic import BaseModel, Field

from src.core.event_bus import (
    EventBus, EventType, Event, EventPriority, get_event_bus
)
from src.core.models import (
    Order, OrderSide, OrderType, OrderStatus, TimeInForce,
    Position, PositionSide, TradingSignal, SignalType, AccountInfo
)


logger = logging.getLogger(__name__)


class EngineState(str, Enum):
    """Trading engine state enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class TradingMode(str, Enum):
    """Trading mode enumeration."""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    SIMULATION = "simulation"


class EngineConfig(BaseModel):
    """Trading engine configuration."""

    mode: TradingMode = Field(default=TradingMode.PAPER, description="Trading mode")
    enable_trading: bool = Field(default=True, description="Enable trading")
    enable_auto_trading: bool = Field(default=False, description="Auto-execute signals")
    max_positions: int = Field(default=10, ge=1, le=100, description="Max positions")
    max_daily_trades: int = Field(default=50, ge=1, le=500, description="Max daily trades")
    default_order_type: OrderType = Field(default=OrderType.MARKET, description="Default order type")
    default_time_in_force: TimeInForce = Field(default=TimeInForce.DAY, description="Default TIF")
    min_signal_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Min signal strength")
    signal_aggregation: bool = Field(default=True, description="Aggregate signals")
    trade_extended_hours: bool = Field(default=False, description="Trade extended hours")
    cooldown_seconds: int = Field(default=60, ge=0, description="Cooldown between trades")
    require_confirmation: bool = Field(default=False, description="Require manual confirmation")
    log_level: str = Field(default="INFO", description="Log level")


class TradingEngineStats(BaseModel):
    """Trading engine statistics."""

    started_at: Optional[datetime] = None
    total_signals: int = 0
    signals_executed: int = 0
    signals_rejected: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0
    orders_rejected: int = 0
    positions_opened: int = 0
    positions_closed: int = 0
    total_pnl: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    daily_trades: int = 0
    last_trade_time: Optional[datetime] = None
    errors: int = 0

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.0
        return self.winning_trades / total


class TradingEngine:
    """
    Main trading engine that orchestrates all trading activities.

    The engine manages:
    - Strategy execution and signal processing
    - Order lifecycle management
    - Position tracking
    - Risk management
    - Event coordination
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        event_bus: Optional[EventBus] = None
    ) -> None:
        """
        Initialize the trading engine.

        Args:
            config: Engine configuration
            event_bus: Event bus instance
        """
        self.config = config or EngineConfig()
        self.event_bus = event_bus or get_event_bus()
        self.state = EngineState.STOPPED
        self.stats = TradingEngineStats()

        # Component references (injected later)
        self._order_manager: Optional[Any] = None
        self._position_manager: Optional[Any] = None
        self._risk_manager: Optional[Any] = None
        self._account_manager: Optional[Any] = None
        self._data_manager: Optional[Any] = None

        # Internal state
        self._strategies: Dict[str, Any] = {}
        self._active_signals: Dict[str, TradingSignal] = {}
        self._pending_orders: Dict[str, Order] = {}
        self._symbol_cooldowns: Dict[str, datetime] = {}
        self._running_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Event handlers
        self._setup_event_handlers()

        logger.info(f"TradingEngine initialized in {self.config.mode} mode")

    def _setup_event_handlers(self) -> None:
        """Set up event handlers."""
        self.event_bus.subscribe(
            EventType.SIGNAL_GENERATED,
            self._handle_signal,
            priority=EventPriority.HIGH
        )
        self.event_bus.subscribe(
            EventType.ORDER_FILLED,
            self._handle_order_filled,
            priority=EventPriority.HIGH
        )
        self.event_bus.subscribe(
            EventType.ORDER_REJECTED,
            self._handle_order_rejected,
            priority=EventPriority.HIGH
        )
        self.event_bus.subscribe(
            EventType.ORDER_CANCELLED,
            self._handle_order_cancelled,
            priority=EventPriority.NORMAL
        )
        self.event_bus.subscribe(
            EventType.POSITION_STOP_TRIGGERED,
            self._handle_stop_triggered,
            priority=EventPriority.CRITICAL
        )
        self.event_bus.subscribe(
            EventType.RISK_LIMIT_BREACH,
            self._handle_risk_breach,
            priority=EventPriority.CRITICAL
        )

    # =========================================================================
    # COMPONENT INJECTION
    # =========================================================================

    def set_order_manager(self, manager: Any) -> None:
        """Set the order manager component."""
        self._order_manager = manager
        logger.debug("Order manager set")

    def set_position_manager(self, manager: Any) -> None:
        """Set the position manager component."""
        self._position_manager = manager
        logger.debug("Position manager set")

    def set_risk_manager(self, manager: Any) -> None:
        """Set the risk manager component."""
        self._risk_manager = manager
        logger.debug("Risk manager set")

    def set_account_manager(self, manager: Any) -> None:
        """Set the account manager component."""
        self._account_manager = manager
        logger.debug("Account manager set")

    def set_data_manager(self, manager: Any) -> None:
        """Set the data manager component."""
        self._data_manager = manager
        logger.debug("Data manager set")

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    async def start(self) -> bool:
        """
        Start the trading engine.

        Returns:
            True if started successfully
        """
        if self.state == EngineState.RUNNING:
            logger.warning("Engine is already running")
            return True

        async with self._lock:
            try:
                self.state = EngineState.STARTING
                logger.info("Starting trading engine...")

                # Start event bus
                await self.event_bus.start()

                # Initialize components
                await self._initialize_components()

                # Start main loop
                self._running_task = asyncio.create_task(self._main_loop())

                self.state = EngineState.RUNNING
                self.stats.started_at = datetime.now(timezone.utc)

                # Emit start event
                await self.event_bus.emit(
                    EventType.SYSTEM_START,
                    data={"mode": self.config.mode.value},
                    source="trading_engine"
                )

                logger.info("Trading engine started successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to start trading engine: {e}")
                self.state = EngineState.ERROR
                self.stats.errors += 1
                return False

    async def stop(self, force: bool = False) -> bool:
        """
        Stop the trading engine.

        Args:
            force: Force stop without cleanup

        Returns:
            True if stopped successfully
        """
        if self.state == EngineState.STOPPED:
            logger.warning("Engine is already stopped")
            return True

        async with self._lock:
            try:
                self.state = EngineState.STOPPING
                logger.info("Stopping trading engine...")

                # Cancel main loop
                if self._running_task:
                    self._running_task.cancel()
                    try:
                        await self._running_task
                    except asyncio.CancelledError:
                        pass

                # Cleanup if not forced
                if not force:
                    await self._cleanup()

                # Stop event bus
                await self.event_bus.stop()

                self.state = EngineState.STOPPED

                # Emit stop event
                await self.event_bus.emit(
                    EventType.SYSTEM_STOP,
                    data={"forced": force},
                    source="trading_engine"
                )

                logger.info("Trading engine stopped successfully")
                return True

            except Exception as e:
                logger.error(f"Error stopping trading engine: {e}")
                self.state = EngineState.ERROR
                return False

    async def pause(self) -> bool:
        """
        Pause the trading engine.

        Returns:
            True if paused successfully
        """
        if self.state != EngineState.RUNNING:
            logger.warning(f"Cannot pause engine in state: {self.state}")
            return False

        self.state = EngineState.PAUSED
        logger.info("Trading engine paused")
        return True

    async def resume(self) -> bool:
        """
        Resume the trading engine.

        Returns:
            True if resumed successfully
        """
        if self.state != EngineState.PAUSED:
            logger.warning(f"Cannot resume engine in state: {self.state}")
            return False

        self.state = EngineState.RUNNING
        logger.info("Trading engine resumed")
        return True

    async def _initialize_components(self) -> None:
        """Initialize all components."""
        # Initialize each component if available
        if self._account_manager:
            await self._account_manager.refresh()

        if self._position_manager:
            await self._position_manager.sync_positions()

        if self._order_manager:
            await self._order_manager.sync_orders()

    async def _cleanup(self) -> None:
        """Clean up before stopping."""
        # Cancel pending orders if not in live mode
        if self.config.mode != TradingMode.LIVE and self._order_manager:
            pending = list(self._pending_orders.values())
            for order in pending:
                try:
                    await self._order_manager.cancel_order(order.order_id)
                except Exception as e:
                    logger.error(f"Error cancelling order {order.order_id}: {e}")

    async def _main_loop(self) -> None:
        """Main engine loop."""
        while self.state in (EngineState.RUNNING, EngineState.PAUSED):
            try:
                if self.state == EngineState.PAUSED:
                    await asyncio.sleep(1)
                    continue

                # Process pending signals
                await self._process_signals()

                # Check positions
                await self._check_positions()

                # Update statistics
                await self._update_stats()

                # Small delay to prevent busy loop
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.stats.errors += 1
                await asyncio.sleep(1)

    # =========================================================================
    # SIGNAL HANDLING
    # =========================================================================

    async def _handle_signal(self, event: Event) -> None:
        """Handle incoming trading signal."""
        signal = event.data
        if not isinstance(signal, TradingSignal):
            return

        self.stats.total_signals += 1

        # Skip if paused or not running
        if self.state != EngineState.RUNNING:
            logger.debug(f"Ignoring signal in state: {self.state}")
            return

        # Skip weak signals
        if signal.strength < self.config.min_signal_strength:
            logger.debug(f"Signal too weak: {signal.strength}")
            self.stats.signals_rejected += 1
            return

        # Check cooldown
        if not await self._check_cooldown(signal.symbol):
            logger.debug(f"Symbol {signal.symbol} on cooldown")
            self.stats.signals_rejected += 1
            return

        # Store signal
        self._active_signals[signal.signal_id] = signal

        # Auto-execute if enabled
        if self.config.enable_auto_trading and not self.config.require_confirmation:
            await self.execute_signal(signal)
        else:
            logger.info(f"Signal awaiting confirmation: {signal}")

    async def _process_signals(self) -> None:
        """Process active signals."""
        expired = []

        for signal_id, signal in self._active_signals.items():
            if signal.is_expired or not signal.is_active:
                expired.append(signal_id)
                continue

        # Remove expired signals
        for signal_id in expired:
            del self._active_signals[signal_id]

    async def execute_signal(self, signal: TradingSignal) -> Optional[Order]:
        """
        Execute a trading signal.

        Args:
            signal: Signal to execute

        Returns:
            Created order or None
        """
        if not self.config.enable_trading:
            logger.warning("Trading is disabled")
            return None

        # Validate signal
        validation = await self._validate_signal(signal)
        if not validation["valid"]:
            logger.warning(f"Signal validation failed: {validation['reason']}")
            self.stats.signals_rejected += 1
            return None

        # Create order from signal
        order = await self._signal_to_order(signal)
        if not order:
            return None

        # Submit order
        if self._order_manager:
            result = await self._order_manager.submit_order(order)
            if result:
                self.stats.signals_executed += 1
                self.stats.orders_submitted += 1
                self._pending_orders[order.order_id] = order
                self._set_cooldown(signal.symbol)

                # Deactivate signal
                signal.is_active = False

                return order

        return None

    async def _validate_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """Validate a signal before execution."""
        # Check daily trade limit
        if self.stats.daily_trades >= self.config.max_daily_trades:
            return {"valid": False, "reason": "Daily trade limit reached"}

        # Check position limit for new positions
        if signal.is_buy_signal and self._position_manager:
            positions = await self._position_manager.get_positions()
            if len(positions) >= self.config.max_positions:
                return {"valid": False, "reason": "Max positions reached"}

        # Check risk limits
        if self._risk_manager:
            risk_check = await self._risk_manager.check_signal(signal)
            if not risk_check["approved"]:
                return {"valid": False, "reason": risk_check.get("reason", "Risk check failed")}

        return {"valid": True}

    async def _signal_to_order(self, signal: TradingSignal) -> Optional[Order]:
        """Convert signal to order."""
        order_side = signal.get_order_side()
        if not order_side:
            return None

        # Determine quantity
        quantity = await self._calculate_quantity(signal)
        if quantity <= 0:
            return None

        # Determine prices
        limit_price = None
        stop_price = None

        if self.config.default_order_type == OrderType.LIMIT:
            limit_price = signal.entry_price or signal.price
        elif self.config.default_order_type == OrderType.STOP:
            stop_price = signal.entry_price or signal.price

        order = Order(
            symbol=signal.symbol,
            side=order_side,
            order_type=self.config.default_order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=self.config.default_time_in_force,
            strategy_id=signal.strategy_id,
            extended_hours=self.config.trade_extended_hours,
            tags={"signal_id": signal.signal_id}
        )

        return order

    async def _calculate_quantity(self, signal: TradingSignal) -> float:
        """Calculate order quantity based on signal and account."""
        # Default to a small quantity
        quantity = 1.0

        if self._account_manager and self._risk_manager:
            account = await self._account_manager.get_account()
            if account:
                quantity = await self._risk_manager.calculate_position_size(
                    symbol=signal.symbol,
                    price=signal.price,
                    account_value=account.portfolio_value,
                    signal_strength=signal.strength
                )

        return quantity

    async def _check_cooldown(self, symbol: str) -> bool:
        """Check if symbol is on cooldown."""
        if symbol not in self._symbol_cooldowns:
            return True

        cooldown_end = self._symbol_cooldowns[symbol]
        return datetime.now(timezone.utc) >= cooldown_end

    def _set_cooldown(self, symbol: str) -> None:
        """Set cooldown for a symbol."""
        from datetime import timedelta
        self._symbol_cooldowns[symbol] = (
            datetime.now(timezone.utc) +
            timedelta(seconds=self.config.cooldown_seconds)
        )

    # =========================================================================
    # ORDER HANDLING
    # =========================================================================

    async def _handle_order_filled(self, event: Event) -> None:
        """Handle order filled event."""
        order = event.data
        if not isinstance(order, Order):
            return

        self.stats.orders_filled += 1
        self.stats.daily_trades += 1
        self.stats.last_trade_time = datetime.now(timezone.utc)

        # Remove from pending
        if order.order_id in self._pending_orders:
            del self._pending_orders[order.order_id]

        logger.info(f"Order filled: {order.symbol} {order.side} {order.filled_quantity} @ {order.avg_fill_price}")

    async def _handle_order_rejected(self, event: Event) -> None:
        """Handle order rejected event."""
        order = event.data
        self.stats.orders_rejected += 1

        if order.order_id in self._pending_orders:
            del self._pending_orders[order.order_id]

        logger.warning(f"Order rejected: {order.symbol} - {order.reject_reason}")

    async def _handle_order_cancelled(self, event: Event) -> None:
        """Handle order cancelled event."""
        order = event.data
        self.stats.orders_cancelled += 1

        if order.order_id in self._pending_orders:
            del self._pending_orders[order.order_id]

        logger.info(f"Order cancelled: {order.symbol}")

    # =========================================================================
    # POSITION HANDLING
    # =========================================================================

    async def _check_positions(self) -> None:
        """Check positions for stop/target triggers."""
        if not self._position_manager:
            return

        positions = await self._position_manager.get_positions()
        for position in positions:
            # Check stop loss
            if position.should_stop_out():
                await self.event_bus.emit(
                    EventType.POSITION_STOP_TRIGGERED,
                    data=position,
                    source="trading_engine"
                )

            # Check take profit
            if position.should_take_profit():
                await self.event_bus.emit(
                    EventType.POSITION_TARGET_REACHED,
                    data=position,
                    source="trading_engine"
                )

    async def _handle_stop_triggered(self, event: Event) -> None:
        """Handle stop loss trigger."""
        position = event.data
        if not isinstance(position, Position):
            return

        logger.warning(f"Stop triggered for {position.symbol}")

        # Create exit order
        exit_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY

        order = Order(
            symbol=position.symbol,
            side=exit_side,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            time_in_force=TimeInForce.GTC,
            tags={"exit_reason": "stop_loss", "position_id": position.position_id}
        )

        if self._order_manager:
            await self._order_manager.submit_order(order)

    async def _handle_risk_breach(self, event: Event) -> None:
        """Handle risk limit breach."""
        logger.critical(f"Risk limit breach: {event.data}")

        # Pause engine
        await self.pause()

        # Emit warning
        await self.event_bus.emit(
            EventType.SYSTEM_WARNING,
            data={"message": "Risk limit breached, engine paused"},
            source="trading_engine"
        )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    async def _update_stats(self) -> None:
        """Update engine statistics."""
        # Reset daily stats at market open
        # This would check market hours and reset appropriately
        pass

    def get_stats(self) -> TradingEngineStats:
        """Get engine statistics."""
        return self.stats

    def get_state(self) -> EngineState:
        """Get current engine state."""
        return self.state

    # =========================================================================
    # STRATEGY MANAGEMENT
    # =========================================================================

    def register_strategy(self, strategy_id: str, strategy: Any) -> None:
        """Register a trading strategy."""
        self._strategies[strategy_id] = strategy
        logger.info(f"Strategy registered: {strategy_id}")

    def unregister_strategy(self, strategy_id: str) -> bool:
        """Unregister a trading strategy."""
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            logger.info(f"Strategy unregistered: {strategy_id}")
            return True
        return False

    def get_strategy(self, strategy_id: str) -> Optional[Any]:
        """Get a registered strategy."""
        return self._strategies.get(strategy_id)

    def list_strategies(self) -> List[str]:
        """List registered strategy IDs."""
        return list(self._strategies.keys())


# Singleton instance
_engine_instance: Optional[TradingEngine] = None


def get_trading_engine(config: Optional[EngineConfig] = None) -> TradingEngine:
    """
    Get or create the trading engine singleton.

    Args:
        config: Optional configuration for new instance

    Returns:
        TradingEngine instance
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TradingEngine(config)
    return _engine_instance


def reset_trading_engine() -> None:
    """Reset the trading engine singleton."""
    global _engine_instance
    _engine_instance = None

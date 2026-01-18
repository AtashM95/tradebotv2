"""
Core Package for Ultimate Trading Bot v2.2.

This package provides the core functionality for the trading bot including:
- Event bus for publish-subscribe messaging
- Data models for orders, positions, and signals
- Trading engine orchestration
- Order and position management
- Account management
- Session management
- Market hours tracking
- Symbol management
- State machine
- Scheduler
- Heartbeat monitoring
"""

from src.core.event_bus import (
    EventType,
    Event,
    EventBus,
    get_event_bus,
)

from src.core.models import (
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    PositionSide,
    SignalType,
    SignalStrength,
    Order,
    OrderFill,
    Position,
    ClosedPosition,
    TradingSignal,
    AccountInfo,
    Quote,
    Bar,
    Trade,
)

from src.core.trading_engine import (
    EngineState,
    TradingMode,
    TradingEngineConfig,
    TradingEngine,
)

from src.core.order_manager import (
    OrderManagerConfig,
    OrderManager,
)

from src.core.position_manager import (
    PositionManagerConfig,
    PositionMetrics,
    PositionManager,
)

from src.core.account_manager import (
    AccountStatus,
    AccountType,
    MarginStatus,
    AccountManagerConfig,
    AccountSnapshot,
    AccountInfo as AccountInfoModel,
    AccountHistory,
    AccountManager,
)

from src.core.session_manager import (
    SessionState,
    SessionType,
    SessionManagerConfig,
    SessionStats,
    TradingSession,
    SessionManager,
)

from src.core.market_hours import (
    MarketSession,
    MarketStatus,
    MarketHoursConfig,
    MarketSchedule,
    MarketHours,
    US_MARKET_HOLIDAYS,
    US_EARLY_CLOSE_DATES,
)

from src.core.symbol_manager import (
    AssetClass,
    AssetStatus,
    AssetExchange,
    SymbolManagerConfig,
    AssetInfo,
    SymbolUniverse,
    SymbolManager,
)

from src.core.state_machine import (
    TransitionResult,
    StateTransition,
    StateHistory,
    StateMachine,
    CompositeStateMachine,
)

from src.core.scheduler import (
    TaskPriority,
    TaskStatus,
    ScheduleType,
    ScheduledTask,
    TaskResult,
    SchedulerConfig,
    Scheduler,
)

from src.core.heartbeat import (
    ComponentStatus,
    HealthCheckResult,
    ComponentConfig,
    ComponentState,
    HeartbeatConfig,
    HeartbeatMonitor,
)


__all__ = [
    # Event Bus
    "EventType",
    "Event",
    "EventBus",
    "get_event_bus",
    # Models
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "PositionSide",
    "SignalType",
    "SignalStrength",
    "Order",
    "OrderFill",
    "Position",
    "ClosedPosition",
    "TradingSignal",
    "AccountInfo",
    "Quote",
    "Bar",
    "Trade",
    # Trading Engine
    "EngineState",
    "TradingMode",
    "TradingEngineConfig",
    "TradingEngine",
    # Order Manager
    "OrderManagerConfig",
    "OrderManager",
    # Position Manager
    "PositionManagerConfig",
    "PositionMetrics",
    "PositionManager",
    # Account Manager
    "AccountStatus",
    "AccountType",
    "MarginStatus",
    "AccountManagerConfig",
    "AccountSnapshot",
    "AccountInfoModel",
    "AccountHistory",
    "AccountManager",
    # Session Manager
    "SessionState",
    "SessionType",
    "SessionManagerConfig",
    "SessionStats",
    "TradingSession",
    "SessionManager",
    # Market Hours
    "MarketSession",
    "MarketStatus",
    "MarketHoursConfig",
    "MarketSchedule",
    "MarketHours",
    "US_MARKET_HOLIDAYS",
    "US_EARLY_CLOSE_DATES",
    # Symbol Manager
    "AssetClass",
    "AssetStatus",
    "AssetExchange",
    "SymbolManagerConfig",
    "AssetInfo",
    "SymbolUniverse",
    "SymbolManager",
    # State Machine
    "TransitionResult",
    "StateTransition",
    "StateHistory",
    "StateMachine",
    "CompositeStateMachine",
    # Scheduler
    "TaskPriority",
    "TaskStatus",
    "ScheduleType",
    "ScheduledTask",
    "TaskResult",
    "SchedulerConfig",
    "Scheduler",
    # Heartbeat
    "ComponentStatus",
    "HealthCheckResult",
    "ComponentConfig",
    "ComponentState",
    "HeartbeatConfig",
    "HeartbeatMonitor",
]

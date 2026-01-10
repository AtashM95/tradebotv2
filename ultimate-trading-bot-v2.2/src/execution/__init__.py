"""
Execution Package for Ultimate Trading Bot v2.2.

This package provides comprehensive order execution functionality
including order management, position tracking, and algorithmic execution.
"""

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
    FillType,
    Position,
    ExecutionContext,
    ExecutionResult,
    ExecutionState,
)
from src.execution.alpaca_executor import (
    AlpacaExecutor,
    AlpacaConfig,
)
from src.execution.order_manager import (
    OrderManager,
    OrderManagerConfig,
    OrderGroup,
    OrderGroupType,
    OrderPriority,
    OrderMetrics,
)
from src.execution.position_manager import (
    PositionManager,
    PositionManagerConfig,
    EnhancedPosition,
    PositionEntry,
    PositionAction,
    PortfolioSnapshot,
    PositionAlert,
    PnLType,
)
from src.execution.execution_engine import (
    ExecutionEngine,
    ExecutionEngineConfig,
    ExecutionMode,
    ExecutionStats,
    TradeRecord,
)
from src.execution.execution_algos import (
    AlgoManager,
    AlgoType,
    AlgoStatus,
    AlgoConfig,
    TWAPConfig,
    VWAPConfig,
    IcebergConfig,
    POVConfig,
    AlgoExecution,
    AlgoOrder,
    AlgoMetrics,
    BaseExecutionAlgo,
    TWAPAlgo,
    VWAPAlgo,
    IcebergAlgo,
    POVAlgo,
)
from src.execution.smart_router import (
    SmartOrderRouter,
    SmartRouterConfig,
    Venue,
    VenueType,
    VenueStatus,
    RoutingStrategy,
    RouteDecision,
)
from src.execution.slippage_tracker import (
    SlippageTracker,
    SlippageTrackerConfig,
    SlippageRecord,
    SlippageStats,
    SlippageType,
    SlippageSource,
    SlippageAlert,
)


__all__ = [
    # Base Executor
    "BaseExecutor",
    "PaperExecutor",
    "ExecutorConfig",
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "Fill",
    "FillType",
    "Position",
    "ExecutionContext",
    "ExecutionResult",
    "ExecutionState",
    # Alpaca Executor
    "AlpacaExecutor",
    "AlpacaConfig",
    # Order Manager
    "OrderManager",
    "OrderManagerConfig",
    "OrderGroup",
    "OrderGroupType",
    "OrderPriority",
    "OrderMetrics",
    # Position Manager
    "PositionManager",
    "PositionManagerConfig",
    "EnhancedPosition",
    "PositionEntry",
    "PositionAction",
    "PortfolioSnapshot",
    "PositionAlert",
    "PnLType",
    # Execution Engine
    "ExecutionEngine",
    "ExecutionEngineConfig",
    "ExecutionMode",
    "ExecutionStats",
    "TradeRecord",
    # Execution Algorithms
    "AlgoManager",
    "AlgoType",
    "AlgoStatus",
    "AlgoConfig",
    "TWAPConfig",
    "VWAPConfig",
    "IcebergConfig",
    "POVConfig",
    "AlgoExecution",
    "AlgoOrder",
    "AlgoMetrics",
    "BaseExecutionAlgo",
    "TWAPAlgo",
    "VWAPAlgo",
    "IcebergAlgo",
    "POVAlgo",
    # Smart Router
    "SmartOrderRouter",
    "SmartRouterConfig",
    "Venue",
    "VenueType",
    "VenueStatus",
    "RoutingStrategy",
    "RouteDecision",
    # Slippage Tracker
    "SlippageTracker",
    "SlippageTrackerConfig",
    "SlippageRecord",
    "SlippageStats",
    "SlippageType",
    "SlippageSource",
    "SlippageAlert",
]


def create_execution_engine(
    mode: ExecutionMode = ExecutionMode.PAPER,
    config: dict | None = None,
) -> ExecutionEngine:
    """
    Create a configured execution engine.

    Args:
        mode: Execution mode (paper, live, etc.)
        config: Optional configuration dictionary

    Returns:
        Configured ExecutionEngine instance
    """
    engine_config = ExecutionEngineConfig(mode=mode)
    if config:
        engine_config = ExecutionEngineConfig(mode=mode, **config)

    if mode == ExecutionMode.PAPER:
        executor = PaperExecutor()
    else:
        executor = PaperExecutor()

    return ExecutionEngine(executor=executor, config=engine_config)


def create_alpaca_executor(
    api_key: str,
    api_secret: str,
    paper: bool = True,
    config: dict | None = None,
) -> AlpacaExecutor:
    """
    Create an Alpaca executor.

    Args:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        paper: Use paper trading
        config: Optional additional configuration

    Returns:
        Configured AlpacaExecutor instance
    """
    alpaca_config = AlpacaConfig(
        api_key=api_key,
        api_secret=api_secret,
        use_paper=paper,
        **(config or {}),
    )

    return AlpacaExecutor(alpaca_config=alpaca_config)


def create_algo_manager(
    executor: BaseExecutor | None = None,
) -> AlgoManager:
    """
    Create an algorithm manager.

    Args:
        executor: Order executor (uses PaperExecutor if None)

    Returns:
        AlgoManager instance
    """
    if executor is None:
        executor = PaperExecutor()

    return AlgoManager(executor=executor)


def create_smart_router(
    config: dict | None = None,
) -> SmartOrderRouter:
    """
    Create a smart order router.

    Args:
        config: Optional configuration dictionary

    Returns:
        SmartOrderRouter instance
    """
    router_config = SmartRouterConfig()
    if config:
        router_config = SmartRouterConfig(**config)

    return SmartOrderRouter(config=router_config)


def create_comprehensive_execution_system(
    mode: ExecutionMode = ExecutionMode.PAPER,
    config: dict | None = None,
) -> dict:
    """
    Create a comprehensive execution system with all components.

    Args:
        mode: Execution mode
        config: Optional configuration dictionary

    Returns:
        Dictionary containing all execution components
    """
    exec_config = config or {}

    if mode == ExecutionMode.PAPER:
        executor = PaperExecutor(
            config=ExecutorConfig(**exec_config.get("executor", {}))
        )
    else:
        executor = PaperExecutor(
            config=ExecutorConfig(**exec_config.get("executor", {}))
        )

    engine = ExecutionEngine(
        executor=executor,
        config=ExecutionEngineConfig(
            mode=mode,
            **exec_config.get("engine", {}),
        ),
    )

    algo_manager = AlgoManager(executor=executor)

    smart_router = SmartOrderRouter(
        config=SmartRouterConfig(**exec_config.get("router", {}))
    )

    slippage_tracker = SlippageTracker(
        config=SlippageTrackerConfig(**exec_config.get("slippage", {}))
    )

    return {
        "executor": executor,
        "engine": engine,
        "order_manager": engine.order_manager,
        "position_manager": engine.position_manager,
        "algo_manager": algo_manager,
        "smart_router": smart_router,
        "slippage_tracker": slippage_tracker,
    }

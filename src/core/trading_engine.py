
import logging
import time
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
from collections import defaultdict
from .contracts import RunContext, TradeFill, MarketSnapshot, SignalIntent, RiskDecision, ExecutionRequest
from ..data.data_manager import DataManager
from ..strategies.strategy_manager import StrategyManager
from ..risk.risk_manager import RiskManager
from ..execution.execution_engine import ExecutionEngine

logger = logging.getLogger(__name__)


class EngineState(Enum):
    """Trading engine operational states."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    RECOVERING = "recovering"


class TradingEngineMetrics:
    """Tracks trading engine performance and operational metrics."""

    def __init__(self) -> None:
        self.cycles_completed: int = 0
        self.symbols_processed: int = 0
        self.signals_generated: int = 0
        self.signals_approved: int = 0
        self.signals_vetoed: int = 0
        self.orders_placed: int = 0
        self.fills_received: int = 0
        self.errors_encountered: int = 0
        self.total_processing_time: float = 0.0
        self.last_cycle_time: float = 0.0
        self.avg_cycle_time: float = 0.0
        self.data_fetch_time: float = 0.0
        self.strategy_time: float = 0.0
        self.risk_time: float = 0.0
        self.execution_time: float = 0.0
        self.start_time: datetime = datetime.utcnow()
        self.last_update: datetime = datetime.utcnow()

    def update_cycle(self, cycle_time: float) -> None:
        """Update cycle metrics."""
        self.cycles_completed += 1
        self.last_cycle_time = cycle_time
        self.total_processing_time += cycle_time
        self.avg_cycle_time = self.total_processing_time / self.cycles_completed
        self.last_update = datetime.utcnow()

    def record_signal(self, approved: bool) -> None:
        """Record signal processing result."""
        self.signals_generated += 1
        if approved:
            self.signals_approved += 1
        else:
            self.signals_vetoed += 1

    def record_fill(self) -> None:
        """Record a trade fill."""
        self.fills_received += 1

    def record_error(self) -> None:
        """Record an error occurrence."""
        self.errors_encountered += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/reporting."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        return {
            'cycles_completed': self.cycles_completed,
            'symbols_processed': self.symbols_processed,
            'signals_generated': self.signals_generated,
            'signals_approved': self.signals_approved,
            'signals_vetoed': self.signals_vetoed,
            'approval_rate': self.signals_approved / max(1, self.signals_generated),
            'orders_placed': self.orders_placed,
            'fills_received': self.fills_received,
            'errors_encountered': self.errors_encountered,
            'avg_cycle_time': self.avg_cycle_time,
            'last_cycle_time': self.last_cycle_time,
            'data_fetch_time': self.data_fetch_time,
            'strategy_time': self.strategy_time,
            'risk_time': self.risk_time,
            'execution_time': self.execution_time,
            'uptime_seconds': uptime,
            'last_update': self.last_update.isoformat()
        }


class TradingEngine:
    """
    Core trading engine that orchestrates the entire trading pipeline.

    Responsibilities:
    - Coordinate data fetching, strategy execution, risk management, and order execution
    - Manage engine state and lifecycle
    - Track performance metrics and statistics
    - Handle errors and implement recovery mechanisms
    - Process symbols in batches for efficiency
    - Provide hooks for monitoring and logging

    Architecture:
    The engine follows a strict pipeline: data → strategies → risk → execution
    Each stage is isolated and communicates via contracts (dataclasses).
    """

    def __init__(
        self,
        data: DataManager,
        strategies: StrategyManager,
        risk: RiskManager,
        execution: ExecutionEngine,
        max_symbols_per_cycle: int = 50,
        enable_parallel: bool = False
    ) -> None:
        """
        Initialize the trading engine.

        Args:
            data: Data manager for market data access
            strategies: Strategy manager for signal generation
            risk: Risk manager for trade approval
            execution: Execution engine for order placement
            max_symbols_per_cycle: Maximum symbols to process per cycle
            enable_parallel: Enable parallel processing (future enhancement)
        """
        self.data = data
        self.strategies = strategies
        self.risk = risk
        self.execution = execution
        self.max_symbols_per_cycle = max_symbols_per_cycle
        self.enable_parallel = enable_parallel

        self.state = EngineState.INITIALIZING
        self.metrics = TradingEngineMetrics()
        self.error_callbacks: List[Callable] = []
        self.pre_cycle_hooks: List[Callable] = []
        self.post_cycle_hooks: List[Callable] = []

        self._last_snapshots: Dict[str, MarketSnapshot] = {}
        self._failed_symbols: Dict[str, int] = defaultdict(int)
        self._max_symbol_failures = 3

        logger.info("TradingEngine initialized", extra={
            'max_symbols_per_cycle': max_symbols_per_cycle,
            'enable_parallel': enable_parallel
        })

    def register_error_callback(self, callback: Callable) -> None:
        """Register a callback to be invoked on errors."""
        self.error_callbacks.append(callback)

    def register_pre_cycle_hook(self, hook: Callable) -> None:
        """Register a hook to run before each cycle."""
        self.pre_cycle_hooks.append(hook)

    def register_post_cycle_hook(self, hook: Callable) -> None:
        """Register a hook to run after each cycle."""
        self.post_cycle_hooks.append(hook)

    def get_state(self) -> EngineState:
        """Get current engine state."""
        return self.state

    def get_metrics(self) -> Dict[str, Any]:
        """Get current engine metrics."""
        return self.metrics.to_dict()

    def initialize(self, context: RunContext) -> bool:
        """
        Initialize the engine for a trading session.

        Args:
            context: Run context with session information

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing trading engine", extra={'run_id': context.run_id})
            self.state = EngineState.INITIALIZING

            # Validate components
            if not self.data.watchlist:
                logger.warning("Watchlist is empty", extra={'run_id': context.run_id})

            # Initialize subsystems
            self.data.initialize(context)
            self.strategies.initialize(context)
            self.risk.initialize(context)
            self.execution.initialize(context)

            self.state = EngineState.READY
            logger.info("Trading engine initialized successfully", extra={'run_id': context.run_id})
            return True

        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}", extra={
                'run_id': context.run_id,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            self.state = EngineState.ERROR
            return False

    def shutdown(self, context: RunContext) -> None:
        """Gracefully shutdown the engine."""
        logger.info("Shutting down trading engine", extra={'run_id': context.run_id})
        self.state = EngineState.STOPPED

        # Shutdown subsystems
        try:
            self.execution.shutdown(context)
            self.risk.shutdown(context)
            self.strategies.shutdown(context)
            self.data.shutdown(context)
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", extra={
                'run_id': context.run_id,
                'error': str(e)
            })

        # Log final metrics
        logger.info("Engine shutdown complete", extra={
            'run_id': context.run_id,
            'metrics': self.metrics.to_dict()
        })

    def run_once(self, context: RunContext) -> List[TradeFill]:
        """
        Execute one complete trading cycle.

        This is the main entry point for the trading loop. It processes
        all symbols in the watchlist through the full pipeline.

        Args:
            context: Run context with session information

        Returns:
            List of trade fills from this cycle
        """
        if self.state not in (EngineState.READY, EngineState.RUNNING):
            logger.warning(f"Cannot run cycle in state: {self.state}", extra={
                'run_id': context.run_id
            })
            return []

        cycle_start = time.time()
        self.state = EngineState.RUNNING
        fills: List[TradeFill] = []

        try:
            # Execute pre-cycle hooks
            self._execute_hooks(self.pre_cycle_hooks, context)

            # Process watchlist
            fills = self._process_watchlist(context)

            # Execute post-cycle hooks
            self._execute_hooks(self.post_cycle_hooks, context)

            # Update metrics
            cycle_time = time.time() - cycle_start
            self.metrics.update_cycle(cycle_time)

            logger.info("Cycle completed", extra={
                'run_id': context.run_id,
                'fills': len(fills),
                'cycle_time': cycle_time,
                'symbols_processed': self.metrics.symbols_processed
            })

            self.state = EngineState.READY

        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Error in trading cycle: {e}", extra={
                'run_id': context.run_id,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            self._handle_error(context, e)

        return fills

    def _process_watchlist(self, context: RunContext) -> List[TradeFill]:
        """Process all symbols in the watchlist."""
        fills: List[TradeFill] = []
        watchlist = self.data.watchlist

        # Process in batches if watchlist is large
        for batch_start in range(0, len(watchlist), self.max_symbols_per_cycle):
            batch_end = min(batch_start + self.max_symbols_per_cycle, len(watchlist))
            batch = watchlist[batch_start:batch_end]

            logger.debug(f"Processing batch {batch_start}-{batch_end}", extra={
                'run_id': context.run_id,
                'batch_size': len(batch)
            })

            batch_fills = self._process_symbol_batch(batch, context)
            fills.extend(batch_fills)

        return fills

    def _process_symbol_batch(self, symbols: List[str], context: RunContext) -> List[TradeFill]:
        """Process a batch of symbols through the pipeline."""
        fills: List[TradeFill] = []

        for symbol in symbols:
            # Skip symbols with too many failures
            if self._failed_symbols[symbol] >= self._max_symbol_failures:
                logger.debug(f"Skipping {symbol} due to repeated failures", extra={
                    'run_id': context.run_id,
                    'symbol': symbol,
                    'failures': self._failed_symbols[symbol]
                })
                continue

            try:
                fill = self._process_symbol(symbol, context)
                if fill is not None:
                    fills.append(fill)
                    self.metrics.record_fill()

                # Reset failure count on success
                if symbol in self._failed_symbols:
                    del self._failed_symbols[symbol]

            except Exception as e:
                self._failed_symbols[symbol] += 1
                logger.error(f"Error processing {symbol}: {e}", extra={
                    'run_id': context.run_id,
                    'symbol': symbol,
                    'error': str(e),
                    'failures': self._failed_symbols[symbol]
                })

        return fills

    def _process_symbol(self, symbol: str, context: RunContext) -> Optional[TradeFill]:
        """
        Process a single symbol through the full pipeline.

        Pipeline stages:
        1. Fetch market data snapshot
        2. Generate trading signal
        3. Evaluate risk decision
        4. Execute order if approved

        Args:
            symbol: Symbol to process
            context: Run context

        Returns:
            TradeFill if order was executed, None otherwise
        """
        self.metrics.symbols_processed += 1

        # Stage 1: Data
        data_start = time.time()
        snapshot = self._fetch_snapshot(symbol, context)
        self.metrics.data_fetch_time += time.time() - data_start

        if snapshot is None:
            return None

        # Stage 2: Strategy
        strategy_start = time.time()
        signal = self._generate_signal(snapshot, context)
        self.metrics.strategy_time += time.time() - strategy_start

        if signal is None:
            return None

        # Stage 3: Risk
        risk_start = time.time()
        decision = self._evaluate_risk(signal, snapshot, context)
        self.metrics.risk_time += time.time() - risk_start

        if not decision.approved:
            self.metrics.record_signal(False)
            logger.debug("Trade vetoed by risk", extra={
                'run_id': context.run_id,
                'symbol': symbol,
                'reason': decision.reason
            })
            return None

        self.metrics.record_signal(True)

        # Stage 4: Execution
        exec_start = time.time()
        fill = self._execute_trade(signal, decision, snapshot, context)
        self.metrics.execution_time += time.time() - exec_start

        if fill is not None:
            self.metrics.orders_placed += 1

        return fill

    def _fetch_snapshot(self, symbol: str, context: RunContext) -> Optional[MarketSnapshot]:
        """Fetch market data snapshot for a symbol."""
        try:
            snapshot = self.data.get_snapshot(symbol)
            self._last_snapshots[symbol] = snapshot
            return snapshot
        except Exception as e:
            logger.error(f"Failed to fetch snapshot for {symbol}: {e}", extra={
                'run_id': context.run_id,
                'symbol': symbol,
                'error': str(e)
            })
            return None

    def _generate_signal(self, snapshot: MarketSnapshot, context: RunContext) -> Optional[SignalIntent]:
        """Generate trading signal from strategies."""
        try:
            signal = self.strategies.generate_signal(snapshot)
            if signal is not None:
                logger.debug("Signal generated", extra={
                    'run_id': context.run_id,
                    'symbol': snapshot.symbol,
                    'action': signal.action,
                    'confidence': signal.confidence
                })
            return signal
        except Exception as e:
            logger.error(f"Failed to generate signal for {snapshot.symbol}: {e}", extra={
                'run_id': context.run_id,
                'symbol': snapshot.symbol,
                'error': str(e)
            })
            return None

    def _evaluate_risk(
        self,
        signal: SignalIntent,
        snapshot: MarketSnapshot,
        context: RunContext
    ) -> RiskDecision:
        """Evaluate risk for a trading signal."""
        try:
            decision = self.risk.evaluate(signal, snapshot, context)
            logger.debug("Risk evaluation", extra={
                'run_id': context.run_id,
                'symbol': signal.symbol,
                'approved': decision.approved,
                'reason': decision.reason
            })
            return decision
        except Exception as e:
            logger.error(f"Risk evaluation failed for {signal.symbol}: {e}", extra={
                'run_id': context.run_id,
                'symbol': signal.symbol,
                'error': str(e)
            })
            # Default to veto on error
            return RiskDecision(
                approved=False,
                reason=f"risk_error: {str(e)}",
                adjusted_size=0.0
            )

    def _execute_trade(
        self,
        signal: SignalIntent,
        decision: RiskDecision,
        snapshot: MarketSnapshot,
        context: RunContext
    ) -> Optional[TradeFill]:
        """Execute an approved trade."""
        try:
            req = self.execution.prepare_request(signal, decision, snapshot)
            fill = self.execution.execute(req, context)

            logger.info("Trade executed", extra={
                'run_id': context.run_id,
                'symbol': fill.symbol,
                'action': fill.action,
                'quantity': fill.quantity,
                'price': fill.price
            })

            return fill

        except Exception as e:
            logger.error(f"Execution failed for {signal.symbol}: {e}", extra={
                'run_id': context.run_id,
                'symbol': signal.symbol,
                'error': str(e)
            })
            return None

    def _execute_hooks(self, hooks: List[Callable], context: RunContext) -> None:
        """Execute registered hooks."""
        for hook in hooks:
            try:
                hook(context, self)
            except Exception as e:
                logger.error(f"Hook execution failed: {e}", extra={
                    'run_id': context.run_id,
                    'hook': hook.__name__,
                    'error': str(e)
                })

    def _handle_error(self, context: RunContext, error: Exception) -> None:
        """Handle errors and invoke error callbacks."""
        self.state = EngineState.ERROR

        for callback in self.error_callbacks:
            try:
                callback(context, error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}", extra={
                    'run_id': context.run_id,
                    'error': str(e)
                })

    def pause(self) -> None:
        """Pause the engine (stops processing but maintains state)."""
        logger.info("Pausing trading engine")
        self.state = EngineState.PAUSED

    def resume(self) -> None:
        """Resume the engine from paused state."""
        if self.state == EngineState.PAUSED:
            logger.info("Resuming trading engine")
            self.state = EngineState.READY
        else:
            logger.warning(f"Cannot resume from state: {self.state}")

    def get_last_snapshot(self, symbol: str) -> Optional[MarketSnapshot]:
        """Get the last snapshot for a symbol."""
        return self._last_snapshots.get(symbol)

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        logger.info("Resetting engine metrics")
        self.metrics = TradingEngineMetrics()

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the engine and all subsystems.

        Returns:
            Dictionary with health status of all components
        """
        health = {
            'engine_state': self.state.value,
            'healthy': self.state in (EngineState.READY, EngineState.RUNNING),
            'metrics': self.metrics.to_dict(),
            'subsystems': {
                'data': self.data.health_check(),
                'strategies': self.strategies.health_check(),
                'risk': self.risk.health_check(),
                'execution': self.execution.health_check()
            }
        }
        return health

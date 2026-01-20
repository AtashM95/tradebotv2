
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from ..core.contracts import SignalIntent, RiskDecision, MarketSnapshot, ExecutionRequest, TradeFill, RunContext
from .paper_executor import PaperExecutor

logger = logging.getLogger(__name__)


class ExecutionEngineMetrics:
    """Tracks execution engine performance metrics."""

    def __init__(self) -> None:
        self.total_requests: int = 0
        self.successful_executions: int = 0
        self.failed_executions: int = 0
        self.total_quantity_executed: float = 0.0
        self.total_notional_executed: float = 0.0
        self.avg_execution_time: float = 0.0
        self._execution_times: List[float] = []

    def record_execution(self, success: bool, quantity: float, price: float, exec_time: float) -> None:
        """Record an execution."""
        self.total_requests += 1
        if success:
            self.successful_executions += 1
            self.total_quantity_executed += quantity
            self.total_notional_executed += quantity * price
            self._execution_times.append(exec_time)
            if self._execution_times:
                self.avg_execution_time = sum(self._execution_times) / len(self._execution_times)
        else:
            self.failed_executions += 1

    def get_success_rate(self) -> float:
        """Get execution success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_executions / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_requests': self.total_requests,
            'successful_executions': self.successful_executions,
            'failed_executions': self.failed_executions,
            'success_rate': self.get_success_rate(),
            'total_quantity_executed': self.total_quantity_executed,
            'total_notional_executed': self.total_notional_executed,
            'avg_execution_time': self.avg_execution_time
        }


class ExecutionEngine:
    """
    Manages order execution across multiple brokers.

    Responsibilities:
    - Prepare execution requests from signals and risk decisions
    - Route orders to appropriate executors (paper, live, etc.)
    - Track execution performance and metrics
    - Handle execution errors and retries
    - Provide execution analytics
    - Support multiple execution venues
    """

    def __init__(self, mode: str = "paper") -> None:
        """
        Initialize execution engine.

        Args:
            mode: Execution mode (paper, live, etc.)
        """
        self.mode = mode
        self.metrics = ExecutionEngineMetrics()

        # Initialize executors based on mode
        if mode == "paper":
            self.executor = PaperExecutor()
        elif mode == "live":
            # In a real system, this would be AlpacaExecutor or similar
            logger.warning("Live mode not fully implemented, using paper executor")
            self.executor = PaperExecutor()
        else:
            logger.warning(f"Unknown mode {mode}, defaulting to paper")
            self.executor = PaperExecutor()

        logger.info("ExecutionEngine initialized", extra={'mode': mode})

    def initialize(self, context: RunContext) -> None:
        """
        Initialize execution engine for trading session.

        Args:
            context: Run context
        """
        logger.info("Initializing ExecutionEngine", extra={
            'run_id': context.run_id,
            'mode': self.mode
        })

        # Initialize executor
        if hasattr(self.executor, 'initialize'):
            self.executor.initialize(context)

        logger.info("ExecutionEngine initialized successfully", extra={'run_id': context.run_id})

    def shutdown(self, context: RunContext) -> None:
        """
        Shutdown execution engine.

        Args:
            context: Run context
        """
        logger.info("Shutting down ExecutionEngine", extra={
            'run_id': context.run_id,
            'metrics': self.metrics.to_dict()
        })

        # Shutdown executor
        if hasattr(self.executor, 'shutdown'):
            self.executor.shutdown(context)

        logger.info("ExecutionEngine shutdown complete", extra={'run_id': context.run_id})

    def prepare_request(
        self,
        signal: SignalIntent,
        decision: RiskDecision,
        snapshot: MarketSnapshot
    ) -> ExecutionRequest:
        """
        Prepare an execution request from signal and risk decision.

        Args:
            signal: Trading signal
            decision: Risk decision with adjusted size
            snapshot: Market snapshot

        Returns:
            ExecutionRequest ready for execution
        """
        # Determine action
        action = signal.action

        # Calculate quantity based on adjusted size
        quantity = decision.adjusted_size

        # Use current market price
        price = snapshot.price

        # Build metadata
        metadata = {
            'signal_confidence': signal.confidence,
            'risk_adjusted': True,
            'original_size': 1.0,
            'adjusted_size': decision.adjusted_size,
            'risk_reason': decision.reason
        }
        metadata.update(signal.metadata)
        metadata.update(decision.metadata)

        request = ExecutionRequest(
            symbol=signal.symbol,
            action=action,
            quantity=quantity,
            price=price,
            metadata=metadata
        )

        logger.debug("Execution request prepared", extra={
            'symbol': request.symbol,
            'action': request.action,
            'quantity': request.quantity,
            'price': request.price
        })

        return request

    def execute(self, request: ExecutionRequest, context: RunContext) -> TradeFill:
        """
        Execute an order.

        Args:
            request: Execution request
            context: Run context

        Returns:
            TradeFill with execution details
        """
        import time
        exec_start = time.time()

        try:
            # Route to appropriate executor
            fill = self.executor.execute(request, context)

            exec_time = time.time() - exec_start

            # Record metrics
            self.metrics.record_execution(
                True,
                fill.quantity,
                fill.price,
                exec_time
            )

            log_extra = {
                'run_id': context.run_id,
                'symbol': fill.symbol,
                'action': fill.action,
                'quantity': fill.quantity,
                'price': fill.price,
                'exec_time': exec_time
            }
            logger.info("Order executed successfully", extra=log_extra)

            return fill

        except Exception as e:
            exec_time = time.time() - exec_start
            self.metrics.record_execution(False, 0.0, 0.0, exec_time)

            log_extra = {
                'run_id': context.run_id,
                'symbol': request.symbol,
                'error': str(e)
            }
            logger.error(f"Execution failed: {e}", extra=log_extra)

            raise

    def execute_batch(
        self,
        requests: List[ExecutionRequest],
        context: RunContext
    ) -> List[TradeFill]:
        """
        Execute multiple orders.

        Args:
            requests: List of execution requests
            context: Run context

        Returns:
            List of trade fills
        """
        fills = []

        for request in requests:
            try:
                fill = self.execute(request, context)
                fills.append(fill)
            except Exception as e:
                logger.error(f"Failed to execute {request.symbol}: {e}", extra={
                    'run_id': context.run_id,
                    'symbol': request.symbol
                })

        logger.info(f"Batch execution complete: {len(fills)}/{len(requests)} successful", extra={
            'run_id': context.run_id,
            'total': len(requests),
            'successful': len(fills)
        })

        return fills

    def cancel_order(self, order_id: str, context: RunContext) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            context: Run context

        Returns:
            True if successful, False otherwise
        """
        if hasattr(self.executor, 'cancel_order'):
            return self.executor.cancel_order(order_id, context)

        logger.warning("Executor does not support order cancellation")
        return False

    def get_order_status(self, order_id: str, context: RunContext) -> Optional[Dict[str, Any]]:
        """
        Get order status.

        Args:
            order_id: Order ID
            context: Run context

        Returns:
            Order status dictionary or None
        """
        if hasattr(self.executor, 'get_order_status'):
            return self.executor.get_order_status(order_id, context)

        logger.warning("Executor does not support order status queries")
        return None

    def validate_request(self, request: ExecutionRequest) -> bool:
        """
        Validate an execution request.

        Args:
            request: Execution request to validate

        Returns:
            True if valid, False otherwise
        """
        # Basic validation
        if not request.symbol:
            logger.error("Invalid request: missing symbol")
            return False

        if not request.action:
            logger.error("Invalid request: missing action")
            return False

        if request.quantity <= 0:
            logger.error(f"Invalid request: invalid quantity {request.quantity}")
            return False

        if request.price <= 0:
            logger.error(f"Invalid request: invalid price {request.price}")
            return False

        return True

    def estimate_cost(self, request: ExecutionRequest) -> float:
        """
        Estimate execution cost for a request.

        Args:
            request: Execution request

        Returns:
            Estimated cost
        """
        notional = request.quantity * request.price

        # Estimate commission (example: $0 for paper, real brokers would have actual commission)
        commission = 0.0

        # Estimate slippage (example: 0.05% of notional)
        slippage = notional * 0.0005

        total_cost = commission + slippage

        logger.debug(f"Estimated execution cost for {request.symbol}: ${total_cost:.2f}", extra={
            'symbol': request.symbol,
            'notional': notional,
            'commission': commission,
            'slippage': slippage
        })

        return total_cost

    def get_execution_summary(
        self,
        fills: List[TradeFill]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for a list of fills.

        Args:
            fills: List of trade fills

        Returns:
            Summary dictionary
        """
        if not fills:
            return {
                'total_fills': 0,
                'total_quantity': 0.0,
                'total_notional': 0.0,
                'avg_price': 0.0,
                'symbols': []
            }

        total_quantity = sum(f.quantity for f in fills)
        total_notional = sum(f.quantity * f.price for f in fills)
        avg_price = total_notional / total_quantity if total_quantity > 0 else 0.0
        symbols = list(set(f.symbol for f in fills))

        return {
            'total_fills': len(fills),
            'total_quantity': total_quantity,
            'total_notional': total_notional,
            'avg_price': avg_price,
            'symbols': symbols
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution engine metrics."""
        return self.metrics.to_dict()

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on execution engine.

        Returns:
            Health status dictionary
        """
        # Check executor health
        executor_healthy = True
        if hasattr(self.executor, 'health_check'):
            executor_health = self.executor.health_check()
            executor_healthy = executor_health.get('healthy', True)

        return {
            'healthy': executor_healthy,
            'mode': self.mode,
            'executor_healthy': executor_healthy,
            'metrics': self.metrics.to_dict()
        }

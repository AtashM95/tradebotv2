
import logging
import random
from datetime import datetime
from typing import Dict, Optional, Any
from ..core.contracts import ExecutionRequest, TradeFill, RunContext

logger = logging.getLogger(__name__)


class PaperExecutor:
    """
    Simulates order execution for paper trading.

    Responsibilities:
    - Simulate instant fills at market price
    - Add realistic slippage and delays
    - Track simulated order history
    - Provide execution statistics
    - Support various order types
    """

    def __init__(
        self,
        slippage_bps: float = 0.5,
        fill_probability: float = 1.0,
        add_latency: bool = False
    ) -> None:
        """
        Initialize paper executor.

        Args:
            slippage_bps: Slippage in basis points (0.5 = 0.05%)
            fill_probability: Probability of fill (0.0-1.0)
            add_latency: Whether to simulate execution latency
        """
        self.slippage_bps = slippage_bps
        self.fill_probability = fill_probability
        self.add_latency = add_latency

        # Track execution history
        self.execution_count: int = 0
        self.total_slippage: float = 0.0
        self.fills_history: list[TradeFill] = []

        logger.info("PaperExecutor initialized", extra={
            'slippage_bps': slippage_bps,
            'fill_probability': fill_probability,
            'add_latency': add_latency
        })

    def initialize(self, context: RunContext) -> None:
        """
        Initialize paper executor for trading session.

        Args:
            context: Run context
        """
        logger.info("PaperExecutor initialized", extra={'run_id': context.run_id})

    def shutdown(self, context: RunContext) -> None:
        """
        Shutdown paper executor.

        Args:
            context: Run context
        """
        logger.info("PaperExecutor shutdown", extra={
            'run_id': context.run_id,
            'executions': self.execution_count,
            'avg_slippage': self.total_slippage / max(1, self.execution_count)
        })

    def execute(self, request: ExecutionRequest, context: RunContext) -> TradeFill:
        """
        Execute an order in paper trading mode.

        Args:
            request: Execution request
            context: Run context

        Returns:
            TradeFill with simulated execution
        """
        # Simulate fill probability
        if random.random() > self.fill_probability:
            raise Exception(f"Simulated order rejection for {request.symbol}")

        # Simulate execution latency
        if self.add_latency:
            import time
            time.sleep(random.uniform(0.01, 0.05))

        # Calculate slippage
        slippage_multiplier = 1.0 + (self.slippage_bps / 10000.0)

        # Apply slippage based on action
        action = request.action.lower()
        if action in ('buy', 'cover'):
            # Buys slip up
            fill_price = request.price * slippage_multiplier
        elif action in ('sell', 'short'):
            # Sells slip down
            fill_price = request.price / slippage_multiplier
        else:
            fill_price = request.price

        # Add some randomness to price
        price_noise = random.uniform(-0.001, 0.001)
        fill_price = fill_price * (1.0 + price_noise)

        # Create trade fill
        fill = TradeFill(
            symbol=request.symbol,
            action=request.action,
            quantity=request.quantity,
            price=fill_price,
            timestamp=datetime.utcnow(),
            metadata={
                **request.metadata,
                'paper_trade': True,
                'requested_price': request.price,
                'slippage': fill_price - request.price,
                'slippage_pct': ((fill_price - request.price) / request.price) * 100
            }
        )

        # Track metrics
        self.execution_count += 1
        self.total_slippage += abs(fill_price - request.price)
        self.fills_history.append(fill)

        # Limit history size
        if len(self.fills_history) > 1000:
            self.fills_history = self.fills_history[-1000:]

        log_extra = {
            'run_id': context.run_id,
            'symbol': fill.symbol,
            'action': fill.action,
            'quantity': fill.quantity,
            'requested_price': request.price,
            'fill_price': fill_price,
            'slippage': fill_price - request.price
        }
        logger.info("Paper trade executed", extra=log_extra)

        return fill

    def cancel_order(self, order_id: str, context: RunContext) -> bool:
        """
        Simulate order cancellation.

        Args:
            order_id: Order ID to cancel
            context: Run context

        Returns:
            True (always successful in paper trading)
        """
        logger.info(f"Paper order cancelled: {order_id}", extra={
            'run_id': context.run_id,
            'order_id': order_id
        })
        return True

    def get_order_status(self, order_id: str, context: RunContext) -> Dict[str, Any]:
        """
        Get simulated order status.

        Args:
            order_id: Order ID
            context: Run context

        Returns:
            Order status dictionary
        """
        # In paper trading, orders are filled immediately
        return {
            'order_id': order_id,
            'status': 'filled',
            'filled_quantity': 0.0,
            'remaining_quantity': 0.0,
            'avg_fill_price': 0.0
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get paper executor statistics.

        Returns:
            Statistics dictionary
        """
        avg_slippage = self.total_slippage / max(1, self.execution_count)

        return {
            'execution_count': self.execution_count,
            'total_slippage': self.total_slippage,
            'avg_slippage': avg_slippage,
            'fills_in_history': len(self.fills_history)
        }

    def get_recent_fills(self, limit: int = 10) -> list[TradeFill]:
        """
        Get recent fills.

        Args:
            limit: Number of recent fills to return

        Returns:
            List of recent trade fills
        """
        return self.fills_history[-limit:]

    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self.execution_count = 0
        self.total_slippage = 0.0
        logger.info("Paper executor statistics reset")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on paper executor.

        Returns:
            Health status dictionary
        """
        return {
            'healthy': True,
            'mode': 'paper',
            'executions': self.execution_count,
            'avg_slippage': self.total_slippage / max(1, self.execution_count),
            'statistics': self.get_statistics()
        }

    def simulate_partial_fill(
        self,
        request: ExecutionRequest,
        fill_percentage: float,
        context: RunContext
    ) -> TradeFill:
        """
        Simulate a partial fill.

        Args:
            request: Execution request
            fill_percentage: Percentage to fill (0.0-1.0)
            context: Run context

        Returns:
            TradeFill with partial quantity
        """
        # Adjust quantity
        partial_request = ExecutionRequest(
            symbol=request.symbol,
            action=request.action,
            quantity=request.quantity * fill_percentage,
            price=request.price,
            metadata={**request.metadata, 'partial_fill': True}
        )

        return self.execute(partial_request, context)

    def simulate_reject(self, request: ExecutionRequest, reason: str) -> None:
        """
        Simulate an order rejection.

        Args:
            request: Execution request
            reason: Rejection reason
        """
        logger.warning(f"Simulated order rejection: {reason}", extra={
            'symbol': request.symbol,
            'reason': reason
        })
        raise Exception(f"Order rejected: {reason}")

    def set_slippage(self, slippage_bps: float) -> None:
        """
        Update slippage setting.

        Args:
            slippage_bps: New slippage in basis points
        """
        old_slippage = self.slippage_bps
        self.slippage_bps = slippage_bps
        logger.info(f"Slippage updated: {old_slippage} -> {slippage_bps} bps")

    def set_fill_probability(self, probability: float) -> None:
        """
        Update fill probability.

        Args:
            probability: New fill probability (0.0-1.0)
        """
        old_prob = self.fill_probability
        self.fill_probability = max(0.0, min(1.0, probability))
        logger.info(f"Fill probability updated: {old_prob} -> {self.fill_probability}")

    def enable_latency(self, enable: bool) -> None:
        """
        Enable or disable execution latency simulation.

        Args:
            enable: Whether to enable latency
        """
        self.add_latency = enable
        logger.info(f"Execution latency {'enabled' if enable else 'disabled'}")

    def get_fill_by_symbol(self, symbol: str) -> list[TradeFill]:
        """
        Get all fills for a specific symbol.

        Args:
            symbol: Symbol to filter by

        Returns:
            List of fills for the symbol
        """
        return [f for f in self.fills_history if f.symbol == symbol]

    def get_total_volume_by_symbol(self, symbol: str) -> float:
        """
        Get total volume executed for a symbol.

        Args:
            symbol: Symbol to check

        Returns:
            Total quantity executed
        """
        fills = self.get_fill_by_symbol(symbol)
        return sum(f.quantity for f in fills)

    def clear_history(self) -> None:
        """Clear fills history."""
        self.fills_history.clear()
        logger.info("Paper executor history cleared")

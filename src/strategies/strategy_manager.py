
import logging
from typing import Dict, Any
from .base_strategy import BaseStrategy
from .signal_generator import SignalGenerator
from ..core.contracts import MarketSnapshot, SignalIntent, RunContext

logger = logging.getLogger(__name__)


class StrategyManager:
    def __init__(self, strategies: list[BaseStrategy]) -> None:
        self.strategies = strategies
        self.generator = SignalGenerator()

    def initialize(self, context: RunContext) -> None:
        """Initialize strategy manager for trading session."""
        logger.info("StrategyManager initialized", extra={
            'run_id': context.run_id,
            'strategy_count': len(self.strategies)
        })

    def shutdown(self, context: RunContext) -> None:
        """Shutdown strategy manager."""
        logger.info("StrategyManager shutdown", extra={'run_id': context.run_id})

    def generate_signal(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        signals = [s.generate(snapshot) for s in self.strategies]
        signals = [s for s in signals if s is not None]
        return self.generator.combine(signals)

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on strategy manager."""
        return {
            'healthy': True,
            'strategy_count': len(self.strategies),
            'active_strategies': len([s for s in self.strategies if hasattr(s, 'is_active') and s.is_active])
        }

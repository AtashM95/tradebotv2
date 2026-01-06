
from .base_strategy import BaseStrategy
from .signal_generator import SignalGenerator
from ..core.contracts import MarketSnapshot, SignalIntent

class StrategyManager:
    def __init__(self, strategies: list[BaseStrategy]) -> None:
        self.strategies = strategies
        self.generator = SignalGenerator()

    def generate_signal(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        signals = [s.generate(snapshot) for s in self.strategies]
        signals = [s for s in signals if s is not None]
        return self.generator.combine(signals)

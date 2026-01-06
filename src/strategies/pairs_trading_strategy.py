
from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy

class PairsTradingStrategy(BaseStrategy):
    name = 'pairs_trading'

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        if snapshot.price <= 0:
            return None
        action = 'buy' if snapshot.price < sum(snapshot.history)/len(snapshot.history) else 'sell'
        return SignalIntent(symbol=snapshot.symbol, action=action, confidence=0.6, metadata={'strategy': self.name})

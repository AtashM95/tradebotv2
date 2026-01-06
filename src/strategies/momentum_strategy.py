
from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    name = 'momentum'

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        if snapshot.price <= 0:
            return None
        action = 'buy' if snapshot.price < sum(snapshot.history)/len(snapshot.history) else 'sell'
        return SignalIntent(symbol=snapshot.symbol, action=action, confidence=0.6, metadata={'strategy': self.name})


from src.core.contracts import MarketSnapshot
from src.strategies.rsi_strategy import RsiStrategy
from datetime import datetime

def test_strategy_signal():
    snap = MarketSnapshot(symbol='AAPL', price=100.0, history=[99.0, 101.0], timestamp=datetime.utcnow())
    sig = RsiStrategy().generate(snap)
    assert sig is not None

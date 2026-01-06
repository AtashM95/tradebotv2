
from src.analysis.technical_analyzer import analyze
from src.core.contracts import MarketSnapshot
from datetime import datetime

def test_analysis():
    out = analyze(MarketSnapshot('AAPL', 100.0, [100.0], datetime.utcnow()))
    assert out['symbol'] == 'AAPL'


from src.core.symbol_manager import SymbolManager

def test_watchlist_load(sample_watchlist):
    symbols = SymbolManager(sample_watchlist).load()
    assert symbols == ['AAPL']

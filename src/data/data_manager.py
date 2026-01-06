
from datetime import datetime
from .market_data import MarketData
from .cache_manager import CacheManager
from ..core.contracts import MarketSnapshot

class DataManager:
    def __init__(self, watchlist: list[str]) -> None:
        self.watchlist = watchlist
        self.market_data = MarketData()
        self.cache = CacheManager()

    def get_snapshot(self, symbol: str) -> MarketSnapshot:
        price, history = self.market_data.get_price(symbol)
        return MarketSnapshot(symbol=symbol, price=price, history=history, timestamp=datetime.utcnow())

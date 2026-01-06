
from ..core.contracts import MarketSnapshot

def analyze(snapshot: MarketSnapshot) -> dict:
    return {'symbol': snapshot.symbol, 'price': snapshot.price}

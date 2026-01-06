
from typing import Dict

class PositionManager:
    def __init__(self) -> None:
        self.positions: Dict[str, float] = {}

    def update(self, symbol: str, qty: float) -> None:
        self.positions[symbol] = self.positions.get(symbol, 0.0) + qty

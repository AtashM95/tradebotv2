
from abc import ABC, abstractmethod
from ..core.contracts import MarketSnapshot, SignalIntent


class BaseStrategy(ABC):
    name = 'base'

    @abstractmethod
    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        return None

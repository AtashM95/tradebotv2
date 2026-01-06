
import random

class MarketData:
    def get_price(self, symbol: str) -> tuple[float, list[float]]:
        base = random.uniform(100, 200)
        history = [base + random.uniform(-1, 1) for _ in range(20)]
        return history[-1], history

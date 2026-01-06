
class PositionSizer:
    def size(self, equity: float, risk_per_trade: float, price: float) -> float:
        risk_amount = equity * risk_per_trade
        return max(risk_amount / max(price, 1.0), 1.0)

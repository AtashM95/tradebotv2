
class CostTracker:
    def __init__(self, daily_cap: float = 5.0) -> None:
        self.daily_cap = daily_cap
        self.spent = 0.0

    def add(self, cost: float) -> bool:
        if self.spent + cost > self.daily_cap:
            return False
        self.spent += cost
        return True

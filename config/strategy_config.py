
from dataclasses import dataclass

@dataclass
class StrategyConfig:
    max_positions: int = 5
    risk_per_trade: float = 0.01

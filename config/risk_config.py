
from dataclasses import dataclass

@dataclass
class RiskConfig:
    max_daily_loss: float = 0.05
    max_position_size: float = 0.1

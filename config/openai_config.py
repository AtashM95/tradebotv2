
from dataclasses import dataclass

@dataclass
class OpenAIConfig:
    model: str = 'gpt-4o-mini'
    timeout_s: int = 20
    daily_cost_cap: float = 5.0

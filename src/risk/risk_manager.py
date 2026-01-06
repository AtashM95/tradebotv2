
from config.risk_config import RiskConfig
from ..core.contracts import SignalIntent, RiskDecision

class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def evaluate(self, signal: SignalIntent) -> RiskDecision:
        approved = abs(signal.confidence) > 0.1
        return RiskDecision(approved=approved, reason='ok' if approved else 'low_conf', adjusted_size=1.0)

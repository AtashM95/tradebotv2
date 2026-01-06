
from src.risk.risk_manager import RiskManager
from config.risk_config import RiskConfig
from src.core.contracts import SignalIntent

def test_risk_decision():
    decision = RiskManager(RiskConfig()).evaluate(SignalIntent('AAPL','buy',0.5))
    assert decision.approved

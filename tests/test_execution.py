
from src.execution.execution_engine import ExecutionEngine
from src.core.contracts import SignalIntent, RiskDecision, MarketSnapshot
from datetime import datetime

def test_execution():
    eng = ExecutionEngine()
    req = eng.prepare_request(SignalIntent('AAPL','buy',0.5), RiskDecision(True,'ok',1.0), MarketSnapshot('AAPL',100.0,[100.0],datetime.utcnow()))
    fill = eng.execute(req, type('C', (), {'dry_run': True})())
    assert fill.symbol == 'AAPL'


from datetime import datetime
from ..core.contracts import SignalIntent, RiskDecision, MarketSnapshot, ExecutionRequest, TradeFill, RunContext
from .paper_executor import PaperExecutor

class ExecutionEngine:
    def __init__(self) -> None:
        self.executor = PaperExecutor()

    def prepare_request(self, signal: SignalIntent, decision: RiskDecision, snapshot: MarketSnapshot) -> ExecutionRequest:
        return ExecutionRequest(symbol=signal.symbol, action=signal.action, quantity=decision.adjusted_size, price=snapshot.price)

    def execute(self, request: ExecutionRequest, context: RunContext) -> TradeFill:
        return self.executor.execute(request, context)


from datetime import datetime
from ..core.contracts import ExecutionRequest, TradeFill, RunContext

class PaperExecutor:
    def execute(self, request: ExecutionRequest, context: RunContext) -> TradeFill:
        return TradeFill(symbol=request.symbol, action=request.action, quantity=request.quantity, price=request.price, timestamp=datetime.utcnow())

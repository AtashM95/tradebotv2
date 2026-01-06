
import logging
from datetime import datetime
from .contracts import RunContext, TradeFill
from ..data.data_manager import DataManager
from ..strategies.strategy_manager import StrategyManager
from ..risk.risk_manager import RiskManager
from ..execution.execution_engine import ExecutionEngine

logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, data: DataManager, strategies: StrategyManager, risk: RiskManager, execution: ExecutionEngine) -> None:
        self.data = data
        self.strategies = strategies
        self.risk = risk
        self.execution = execution

    def run_once(self, context: RunContext) -> list[TradeFill]:
        fills: list[TradeFill] = []
        for symbol in self.data.watchlist:
            snapshot = self.data.get_snapshot(symbol)
            signal = self.strategies.generate_signal(snapshot)
            if signal is None:
                continue
            decision = self.risk.evaluate(signal)
            if not decision.approved:
                logger.info('risk veto', extra={'run_id': context.run_id})
                continue
            req = self.execution.prepare_request(signal, decision, snapshot)
            fill = self.execution.execute(req, context)
            fills.append(fill)
        return fills

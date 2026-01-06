
from src.core.trading_engine import TradingEngine
from src.data.data_manager import DataManager
from src.strategies.strategy_manager import StrategyManager
from src.strategies.rsi_strategy import RsiStrategy
from src.risk.risk_manager import RiskManager
from src.execution.execution_engine import ExecutionEngine
from config.risk_config import RiskConfig
from src.core.session_manager import SessionManager

def test_trading_engine():
    engine = TradingEngine(DataManager(['AAPL']), StrategyManager([RsiStrategy()]), RiskManager(RiskConfig()), ExecutionEngine())
    ctx = SessionManager('paper', True).start()
    fills = engine.run_once(ctx)
    assert isinstance(fills, list)

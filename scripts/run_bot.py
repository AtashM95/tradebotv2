
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config.settings import Settings
from config.logging_config import configure_logging
from src.core.session_manager import SessionManager
from src.core.trading_engine import TradingEngine
from src.data.data_manager import DataManager
from src.core.symbol_manager import SymbolManager
from src.strategies.strategy_manager import StrategyManager
from src.strategies.rsi_strategy import RsiStrategy
from src.risk.risk_manager import RiskManager
from src.execution.execution_engine import ExecutionEngine
from config.risk_config import RiskConfig
from src.data.database_manager import DatabaseManager


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='paper')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--watchlist', default='data/watchlist.csv')
    args, _ = parser.parse_known_args(argv)
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    session = SessionManager(mode=args.mode, dry_run=args.dry_run)
    context = session.start()
    symbols = SymbolManager(args.watchlist).load()
    data = DataManager(symbols)
    strategies = StrategyManager([RsiStrategy()])
    risk = RiskManager(RiskConfig())
    execution = ExecutionEngine()
    engine = TradingEngine(data, strategies, risk, execution)
    engine.initialize(context)
    fills = engine.run_once(context)
    engine.shutdown(context)
    db = DatabaseManager()
    for fill in fills:
        db.save_trade(fill)
    print(f'fills: {len(fills)}')

if __name__ == '__main__':
    main()

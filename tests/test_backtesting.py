
from src.backtesting.backtest_engine import BacktestEngine

def test_backtest_engine():
    assert BacktestEngine().run()['status'] == 'ok'

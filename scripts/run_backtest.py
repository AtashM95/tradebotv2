
from src.backtesting.backtest_engine import BacktestEngine

def main():
    engine = BacktestEngine()
    print(engine.run())

if __name__ == '__main__':
    main()


from src.ai.trading_agent import TradingAgent

def test_ai_agent():
    assert TradingAgent().run('plan')['status'] == 'ok'

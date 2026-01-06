
from src.ai.strategy_advisor import StrategyAdvisor

def test_ai_integration():
    assert StrategyAdvisor().run('advice')['status'] == 'ok'

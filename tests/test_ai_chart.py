
from src.ai.chart_analyzer import ChartAnalyzer

def test_ai_chart():
    assert ChartAnalyzer().run('chart')['status'] == 'ok'

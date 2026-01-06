
from src.ai.sentiment_analyzer import SentimentAnalyzer

def test_ai_sentiment():
    assert SentimentAnalyzer().run('x')['status'] == 'ok'

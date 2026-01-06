
from src.sentiment.sentiment_engine import SentimentEngine

def test_sentiment_engine():
    assert SentimentEngine().run('x')['status'] == 'ok'

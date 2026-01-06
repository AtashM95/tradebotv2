
from .openai_client import OpenAIClient

class ChartAnalyzer:
    def __init__(self) -> None:
        self.client = OpenAIClient()

    def run(self, text: str) -> dict:
        result = self.client.analyze(text)
        return {'status': 'ok', 'result': result}

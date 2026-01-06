
import os
import time


class OpenAIClient:
    def __init__(self) -> None:
        self.api_key = os.getenv('OPENAI_API_KEY', '')

    def analyze(self, prompt: str) -> str:
        for _ in range(2):
            if not self.api_key:
                return 'no-key'
            time.sleep(0.1)
            return f'analysis: {prompt[:20]}'
        return 'no-key'

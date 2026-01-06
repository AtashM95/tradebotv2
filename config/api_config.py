
import os

class APIConfig:
    def __init__(self) -> None:
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY', '')
        self.alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY', '')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')

    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

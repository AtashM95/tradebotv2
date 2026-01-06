
from config.prompts_config import PROMPTS

class PromptManager:
    def get(self, key: str) -> str:
        return PROMPTS.get(key, '')


from dataclasses import dataclass
import os
from .constants import DEFAULT_MODE

@dataclass
class Settings:
    mode: str = DEFAULT_MODE
    dry_run: bool = True
    database_url: str = 'sqlite:///tradebot.db'
    log_level: str = 'INFO'

    @classmethod
    def from_env(cls) -> 'Settings':
        mode = os.getenv('TRADEBOT_MODE', DEFAULT_MODE)
        dry_run = os.getenv('TRADEBOT_DRY_RUN', 'true').lower() == 'true'
        database_url = os.getenv('DATABASE_URL', 'sqlite:///tradebot.db')
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        return cls(mode=mode, dry_run=dry_run, database_url=database_url, log_level=log_level)

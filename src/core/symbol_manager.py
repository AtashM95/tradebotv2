
from pathlib import Path

class SymbolManager:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def load(self) -> list[str]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding='utf-8').splitlines()
        return [l.strip() for l in lines[1:] if l.strip()]

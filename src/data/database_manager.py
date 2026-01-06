
import sqlite3
from pathlib import Path
from datetime import datetime
from ..core.contracts import TradeFill

class DatabaseManager:
    def __init__(self, path: str = 'tradebot.db') -> None:
        self.path = Path(path)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.path)
        conn.execute('CREATE TABLE IF NOT EXISTS trades (symbol TEXT, action TEXT, quantity REAL, price REAL, timestamp TEXT)')
        conn.commit()
        conn.close()

    def save_trade(self, fill: TradeFill) -> None:
        conn = sqlite3.connect(self.path)
        conn.execute('INSERT INTO trades VALUES (?, ?, ?, ?, ?)', (fill.symbol, fill.action, fill.quantity, fill.price, fill.timestamp.isoformat()))
        conn.commit()
        conn.close()

    def list_trades(self) -> list[tuple]:
        conn = sqlite3.connect(self.path)
        rows = conn.execute('SELECT symbol, action, quantity, price, timestamp FROM trades').fetchall()
        conn.close()
        return rows
